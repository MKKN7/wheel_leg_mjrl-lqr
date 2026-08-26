"""Gymnasium environment for safe residual control of the wheeled-leg robot.

The policy never replaces the physical LQR controller.  Its six normalized
actions are bounded torque residuals that are added to ``PhysicalLqr.command``.
The low-centre closed-chain stance is projected once and then restored with
``mj_resetData`` on subsequent episode resets, so model and data objects are
not rebuilt in the simulation loop.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

import lqr_deploy as lqr
from lqr_deploy import HeadingCommand, PhysicalLqr, wrap_to_pi
from terrain_curriculum import (
    TerrainCurriculumConfig,
    TerrainRoute,
    TerrainTask,
    validate_scene_contract,
)


DEFAULT_EPISODE_SECONDS = 8.0
DEFAULT_CONTROL_DECIMATION = 10
MAX_ATTITUDE_ERROR_RAD = 1.0
# A complete loss of wheel support needs immediate braking.  A single wheel
# can briefly unload on a grade or a step while the other one remains a valid
# support, so it gets a short transition window.  Geometry, linkage, attitude
# and leg-length hard checks remain active throughout both windows.
CONTACT_RECOVERY_BOTH_WHEELS_TRIGGER_SECONDS = 0.004
CONTACT_RECOVERY_SINGLE_WHEEL_TRIGGER_SECONDS = 0.025
CONTACT_RECOVERY_TIMEOUT_SECONDS = 0.180
CONTACT_RECOVERY_STABLE_SECONDS = 0.080
# At high speed a compliant hfield contact manifold can omit a few 1 kHz
# contact reports while the wheel bottom remains within the measured support
# tolerance. This is only a confidence filter for recovery bookkeeping; a
# wheel that separates beyond this clearance still uses the normal recovery
# and hard-safety path.
WHEEL_NEAR_SUPPORT_CLEARANCE_M = 0.005
TERRAIN_HIGH_SPEED_COMMAND_MIN_MPS = 1.50
TERRAIN_HIGH_SPEED_COMMAND_MIN_YAW_RATE_RAD_S = 0.20
TERRAIN_HIGH_SPEED_ACCELERATION_MPS2 = 0.30
TERRAIN_HIGH_SPEED_YAW_ACCELERATION_RAD_S2 = 0.15
TERRAIN_DIRECT_YAW_RATE_MIN_RAD_S = 1e-6
RL_LEG_LENGTH_COMMAND_RATE_MPS = 0.05
RL_COMMAND_SPEED_FRACTION = 0.65
LOCOMOTION_COMMAND_SCHEMA = "speed_yaw_rate_jump_terrain_v11"
RESIDUAL_AUTHORITY_SCHEMA = "phase_masked_residual_v7"
LEG_DIFF_STRAIGHT_WEIGHT = 2.0
LEG_DIFF_TURN_WEIGHT = 0.5
YAW_RATE_TRACKING_WEIGHT = 0.05
JUMP_PHASE_NAMES = ("prepare", "crouch", "thrust", "flight", "landing")
DEFAULT_JUMP_AT_S = 0.80
JUMP_TARGET_CLEARANCE_M = 0.20
JUMP_MAX_REWARD_CLEARANCE_M = 0.25
JUMP_MIN_CLEARANCE_SUCCESS_M = 0.18
JUMP_PEAK_CLEARANCE_SCALE_M = 0.10
JUMP_HEIGHT_REWARD_WEIGHT = 8.0
JUMP_SUCCESS_REWARD = 15.0
JUMP_ABORT_PENALTY = 10.0
JUMP_LANDING_FALL_PENALTY = 60.0
JUMP_LANDING_GUARD_SECONDS = 0.50
JUMP_STABLE_ATTITUDE_LIMIT_RAD = 0.35
JUMP_STABLE_VERTICAL_SPEED_MPS = 1.0
JUMP_STABLE_ANGULAR_SPEED_RAD_S = 2.5
# Keep jump residual authority below the LQR's torque reserve.  The command
# trajectory remains the source of the common leg-length target; the policy
# can only make small per-hip corrections while those phases are in contact.
JUMP_PREPARE_CROUCH_RESIDUAL_SCALE = 0.10
JUMP_THRUST_RESIDUAL_SCALE = 0.05
# During flight the policy receives proprioceptive contact/leg feedback and
# may make a small differential hip correction.  Wheels and common leg length
# remain owned by the jump trajectory; hard linkage checks still run every
# physics step.
JUMP_FLIGHT_RESIDUAL_SCALE = 0.10
JUMP_LANDING_RESIDUAL_SCALE = 0.05
# The local terrain patch is expressed in the body-heading frame.  It gives a
# residual locomotion policy enough preview to prepare its legs for a grade or
# step without exposing a route, waypoint, or global map as an input.
TERRAIN_LOOKAHEAD_DISTANCES_M = (0.10, 0.28, 0.50, 0.76)
TERRAIN_LOOKAHEAD_LATERAL_OFFSETS_M = (-0.18, 0.0, 0.18)
TERRAIN_HEIGHT_NORMALIZATION_M = 0.25
TERRAIN_SLOPE_NORMALIZATION = 0.50
STATIC_SUPPORT_FOOTPRINT_TOLERANCE_M = 1e-6
TERRAIN_OBSERVATION_SIZE = (
    len(TERRAIN_LOOKAHEAD_DISTANCES_M) * len(TERRAIN_LOOKAHEAD_LATERAL_OFFSETS_M)
    + 4
)
# Terrain routes are reset/evaluation metadata, not a hidden navigation
# target.  Convert the normal per-control-step tracking reward into a rate
# reward for terrain episodes, then give a terminal task-completion bonus that
# dominates waiting for the deadline.
TERRAIN_DENSE_REWARD_RATE_SCALE = 1.0
TERRAIN_PROGRESS_REWARD_PER_M = 0.0
TERRAIN_COMPLETION_REWARD = 100.0
TERRAIN_CORRIDOR_PENALTY_PER_M = 0.0
TERRAIN_TASK_TIMEOUT_PENALTY = 10.0


@dataclass(frozen=True)
class LocomotionCommand:
    """High-level command consumed by the residual locomotion controller.

    ``jump_request`` is edge-triggered by :meth:`set_locomotion_command`.
    Generic commands use the LQR's validated low-speed rolling launch
    reference and retain ``forward_speed_mps`` as the post-landing request.
    Terrain tasks that require a calibrated feature-crossing launch may supply
    their own supervisor-only launch reference.  The policy never emits this
    command; it only emits bounded residual motor actions while the supervisor
    grants the relevant joints authority.
    """

    forward_speed_mps: float
    yaw_rate_rad_s: float
    jump_request: bool = False


def locomotion_command_conditioning_config() -> dict[str, float]:
    """Return terrain command-conditioning values used for checkpoint identity."""
    return {
        "high_speed_command_min_mps": TERRAIN_HIGH_SPEED_COMMAND_MIN_MPS,
        "high_speed_command_min_yaw_rate_rad_s": TERRAIN_HIGH_SPEED_COMMAND_MIN_YAW_RATE_RAD_S,
        "high_speed_acceleration_mps2": TERRAIN_HIGH_SPEED_ACCELERATION_MPS2,
        "high_speed_yaw_acceleration_rad_s2": TERRAIN_HIGH_SPEED_YAW_ACCELERATION_RAD_S2,
        "high_speed_direct_yaw_rate_tracking": 1.0,
        "terrain_direct_yaw_rate_min_rad_s": TERRAIN_DIRECT_YAW_RATE_MIN_RAD_S,
        "wheel_near_support_clearance_m": WHEEL_NEAR_SUPPORT_CLEARANCE_M,
    }


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Episode-constant simulation-to-real perturbations for residual training.

    Physical parameters are sampled before each LQR working-point projection.
    The ranges are deliberately bounded so the relinearized safety controller
    remains the primary stabilizer rather than turning every reset into a fall.
    """

    enabled: bool = False
    mass_global_range: tuple[float, float] = (0.90, 1.10)
    mass_body_range: tuple[float, float] = (0.95, 1.05)
    inertia_range: tuple[float, float] = (0.85, 1.15)
    friction_sliding_range: tuple[float, float] = (0.70, 1.30)
    randomize_terrain_friction: bool = True
    damping_range: tuple[float, float] = (0.70, 1.35)
    hip_strength_range: tuple[float, float] = (0.82, 1.15)
    wheel_strength_range: tuple[float, float] = (0.85, 1.15)
    sensor_noise_scale_range: tuple[float, float] = (0.50, 1.00)
    # The current 1 kHz LQR has only a few milliseconds of phase margin.
    # Keep the default trainable; larger delays can be selected explicitly
    # after controller retuning.
    control_delay_steps_range: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        for name in (
            "mass_global_range",
            "mass_body_range",
            "inertia_range",
            "friction_sliding_range",
            "damping_range",
            "hip_strength_range",
            "wheel_strength_range",
            "sensor_noise_scale_range",
        ):
            lower, upper = getattr(self, name)
            minimum = 0.0 if name == "sensor_noise_scale_range" else 1e-9
            if not np.isfinite(lower) or not np.isfinite(upper) or lower < minimum or lower > upper:
                raise ValueError(f"{name} must be a finite ordered range with lower bound >= {minimum:g}")
        delay_lower, delay_upper = self.control_delay_steps_range
        if (
            not isinstance(delay_lower, (int, np.integer))
            or not isinstance(delay_upper, (int, np.integer))
            or delay_lower < 0
            or delay_lower > delay_upper
        ):
            raise ValueError("control_delay_steps_range must be an ordered non-negative integer range")
        if not isinstance(self.randomize_terrain_friction, (bool, np.bool_)):
            raise ValueError("randomize_terrain_friction must be boolean")

    @classmethod
    def training_defaults(cls) -> "DomainRandomizationConfig":
        return cls(enabled=True)

    @classmethod
    def vehicle_only_defaults(cls) -> "DomainRandomizationConfig":
        """Randomize the robot while keeping a terrain's geometry and friction fixed."""
        return cls(enabled=True, randomize_terrain_friction=False)

    @classmethod
    def terrain_vehicle_only_defaults(cls) -> "DomainRandomizationConfig":
        """Trainable first-stage RMUC vehicle randomization.

        The hfield and support friction stay fixed.  Wheel friction remains a
        vehicle parameter, while the physical ranges cover enough mass,
        inertia, damping, and actuator variation to prevent a nominal-only
        policy.  Sensor noise is introduced gently and the normal walking
        controller predicts through the one-step actuator FIFO, so the profile
        also covers a 0-1 ms total-control delay.
        """
        return cls(
            enabled=True,
            mass_global_range=(0.96, 1.04),
            mass_body_range=(0.97, 1.03),
            inertia_range=(0.90, 1.10),
            friction_sliding_range=(0.90, 1.10),
            randomize_terrain_friction=False,
            damping_range=(0.88, 1.12),
            hip_strength_range=(0.93, 1.07),
            wheel_strength_range=(0.93, 1.07),
            sensor_noise_scale_range=(0.0, 0.02),
            control_delay_steps_range=(0, 1),
        )

    @classmethod
    def jump_training_defaults(cls) -> "DomainRandomizationConfig":
        """Symmetric physical curriculum for a jump that runs at motor saturation.

        The jump controller is already near the actuator limit.  Keep
        symmetric physical perturbations, while reserving sensor noise and
        control delay for the walking curriculum until a landing observer is
        tuned for those disturbances.
        """
        return cls(
            enabled=True,
            mass_global_range=(0.98, 1.02),
            mass_body_range=(0.98, 1.02),
            inertia_range=(0.90, 1.10),
            friction_sliding_range=(0.90, 1.10),
            damping_range=(0.90, 1.10),
            hip_strength_range=(0.98, 1.02),
            wheel_strength_range=(0.95, 1.05),
            sensor_noise_scale_range=(0.0, 0.0),
            control_delay_steps_range=(0, 0),
        )

    @classmethod
    def jump_vehicle_only_defaults(cls) -> "DomainRandomizationConfig":
        """Use the conservative jump curriculum without altering terrain friction."""
        return cls(
            enabled=True,
            mass_global_range=(0.98, 1.02),
            mass_body_range=(0.98, 1.02),
            inertia_range=(0.90, 1.10),
            friction_sliding_range=(0.90, 1.10),
            randomize_terrain_friction=False,
            damping_range=(0.90, 1.10),
            hip_strength_range=(0.98, 1.02),
            wheel_strength_range=(0.95, 1.05),
            sensor_noise_scale_range=(0.0, 0.0),
            control_delay_steps_range=(0, 0),
        )

    @classmethod
    def disabled(cls) -> "DomainRandomizationConfig":
        return cls(enabled=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 5,
            "enabled": self.enabled,
            "mass_global_range": self.mass_global_range,
            "mass_body_range": self.mass_body_range,
            "inertia_range": self.inertia_range,
            "friction_sliding_range": self.friction_sliding_range,
            "randomize_terrain_friction": self.randomize_terrain_friction,
            "damping_range": self.damping_range,
            "hip_strength_range": self.hip_strength_range,
            "wheel_strength_range": self.wheel_strength_range,
            "sensor_noise_scale_range": self.sensor_noise_scale_range,
            "control_delay_steps_range": self.control_delay_steps_range,
        }


class WheelLegResidualEnv(gym.Env):
    """Command-conditioned residual locomotion controller on top of PhysicalLqr.

    A :class:`LocomotionCommand` supplies forward speed, yaw rate and a jump
    edge. The first six policy actions are residual torques in XML actuator
    order; the seventh is a rate-limited common leg-length command. The LQR
    remains the nominal safety layer, while phase masks expose only the motor
    authority that is physically safe during a jump.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        render_mode: str | None = None,
        episode_seconds: float = DEFAULT_EPISODE_SECONDS,
        control_decimation: int = DEFAULT_CONTROL_DECIMATION,
        randomize_command: bool = True,
        randomize_leg_length: bool = True,
        max_forward_speed: float | None = None,
        command_speed_limit_mps: float | None = None,
        command_resample_seconds: float | None = None,
        command_speed_fraction: float = RL_COMMAND_SPEED_FRACTION,
        max_command_yaw_delta_rad: float = 0.0,
        max_command_yaw_rate_rad_s: float = lqr.MAX_YAW_RATE_RAD_S,
        jump_probability: float = 0.0,
        jump_at: float = DEFAULT_JUMP_AT_S,
        domain_randomization: DomainRandomizationConfig | None = None,
        jump_domain_randomization: DomainRandomizationConfig | None = None,
        terrain_curriculum: TerrainCurriculumConfig | None = None,
        terrain_stage_id: str | None = None,
        terrain_evaluation: bool = False,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if episode_seconds <= 0.0:
            raise ValueError("episode_seconds must be positive")
        if control_decimation < 1:
            raise ValueError("control_decimation must be at least one")
        if command_speed_limit_mps is not None and (
            not np.isfinite(command_speed_limit_mps) or command_speed_limit_mps <= 0.0
        ):
            raise ValueError("command_speed_limit_mps must be positive and finite when supplied")
        if command_resample_seconds is not None and (
            not np.isfinite(command_resample_seconds) or command_resample_seconds <= 0.0
        ):
            raise ValueError("command_resample_seconds must be positive and finite when supplied")
        if not np.isfinite(command_speed_fraction) or not 0.0 < command_speed_fraction <= 1.0:
            raise ValueError("command_speed_fraction must be within (0, 1]")
        if not np.isfinite(max_command_yaw_delta_rad) or not 0.0 <= max_command_yaw_delta_rad <= np.pi:
            raise ValueError("max_command_yaw_delta_rad must be within 0..pi")
        if (
            not np.isfinite(max_command_yaw_rate_rad_s)
            or not 0.0 <= max_command_yaw_rate_rad_s <= lqr.MAX_YAW_RATE_RAD_S
        ):
            raise ValueError(
                f"max_command_yaw_rate_rad_s must be within 0..{lqr.MAX_YAW_RATE_RAD_S:.3f}"
            )
        if not np.isfinite(jump_probability) or not 0.0 <= jump_probability <= 1.0:
            raise ValueError("jump_probability must be within 0..1")
        if not np.isfinite(jump_at) or not 0.0 <= jump_at < episode_seconds:
            raise ValueError("jump_at must be within the episode")
        if terrain_curriculum is None and terrain_stage_id is not None:
            raise ValueError("terrain_stage_id requires terrain_curriculum")
        if terrain_curriculum is not None and terrain_stage_id is None:
            raise ValueError("terrain_curriculum requires terrain_stage_id")
        if terrain_evaluation and terrain_curriculum is None:
            raise ValueError("terrain_evaluation requires terrain_curriculum")

        self.xml_path = Path(xml_path) if xml_path is not None else lqr.XML_PATH
        if terrain_curriculum is not None:
            # Training scripts validate the same v4 contract before startup;
            # keep direct Gymnasium construction equally fail-closed.
            validate_scene_contract(terrain_curriculum, self.xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.refs = lqr.build_refs(self.model)
        self._sensor_refs = dict(self.refs.sensor_refs)
        velocity_sensor = lqr.object_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "world_horizontal_velocity_xy",
        )
        self._sensor_refs["world_horizontal_velocity_xy"] = (
            int(self.model.sensor_adr[velocity_sensor]),
            int(self.model.sensor_dim[velocity_sensor]),
        )

        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None
        self._render_camera: mujoco.MjvCamera | None = None
        self.episode_seconds = float(episode_seconds)
        self.control_decimation = int(control_decimation)
        self.randomize_command = bool(randomize_command)
        self.randomize_leg_length = bool(randomize_leg_length)
        self.max_command_yaw_delta_rad = float(max_command_yaw_delta_rad)
        self.max_command_yaw_rate_rad_s = float(max_command_yaw_rate_rad_s)
        self.jump_probability = float(jump_probability)
        self.jump_at = float(jump_at)
        self.domain_randomization = (
            DomainRandomizationConfig.disabled()
            if domain_randomization is None
            else domain_randomization
        )
        self.jump_domain_randomization = jump_domain_randomization
        self._active_domain_randomization = self.domain_randomization
        self._active_domain_randomization_profile = "walking"
        configured_forward_limit = (
            getattr(lqr, "DEFAULT_FORWARD_SPEED_LIMIT_MPS", getattr(lqr, "MAX_FORWARD_SPEED_MPS", 0.25))
            if max_forward_speed is None
            else max_forward_speed
        )
        self.max_forward_speed = lqr.validate_forward_speed_limit(configured_forward_limit)
        self.max_reverse_speed = lqr.validate_reverse_speed_limit(
            getattr(lqr, "MAX_REVERSE_SPEED_MPS", self.max_forward_speed)
        )
        requested_command_limit = (
            self.max_forward_speed
            if command_speed_limit_mps is None
            else float(command_speed_limit_mps)
        )
        if requested_command_limit > self.max_forward_speed:
            raise ValueError(
                "command_speed_limit_mps cannot exceed the physical max_forward_speed"
            )
        self.command_speed_limit_mps = requested_command_limit
        self.command_resample_seconds = command_resample_seconds
        self.command_speed_fraction = float(command_speed_fraction)
        self.terrain_curriculum = terrain_curriculum
        self.terrain_stage_id = terrain_stage_id
        self.terrain_evaluation = bool(terrain_evaluation)
        self._terrain_stage = (
            None if terrain_curriculum is None else terrain_curriculum.stage(terrain_stage_id)
        )
        if terrain_curriculum is not None:
            if terrain_curriculum.limits.max_forward_speed_mps > self.max_forward_speed + 1e-9:
                raise ValueError(
                    "terrain curriculum speed limit exceeds the environment physical forward-speed limit"
                )
            if terrain_curriculum.limits.max_yaw_rate_rad_s > self.max_command_yaw_rate_rad_s + 1e-9:
                raise ValueError(
                    "terrain curriculum yaw-rate limit exceeds max_command_yaw_rate_rad_s"
                )
        # Retained for existing callers that need a single maximum magnitude.
        self.max_speed = self.max_forward_speed
        self.acceleration_limit = float(getattr(lqr, "DEFAULT_ACCELERATION_MPS2", 0.60))

        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()
        self._nominal_geom_contype = self.model.geom_contype.copy()
        self._nominal_geom_conaffinity = self.model.geom_conaffinity.copy()
        self._nominal_dof_damping = self.model.dof_damping.copy()
        self._nominal_actuator_gainprm = self.model.actuator_gainprm.copy()
        self._nominal_actuator_forcerange = self.model.actuator_forcerange.copy()
        self._control_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._control_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._control_scale = np.maximum(np.abs(self._control_low), np.abs(self._control_high))
        self._terrain_support_geoms = tuple(
            geom_id for geom_id in self.refs.ground_geoms if geom_id != self.refs.ground_geom
        )
        self._terrain_hfield_geom = next(
            (
                geom_id
                for geom_id in self._terrain_support_geoms
                if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_HFIELD
            ),
            None,
        )
        self._terrain_static_box_geoms = tuple(
            geom_id
            for geom_id in self._terrain_support_geoms
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        )
        self._terrain_hfield_id: int | None = None
        self._terrain_hfield_size: np.ndarray | None = None
        self._terrain_hfield_rows = 0
        self._terrain_hfield_columns = 0
        self._terrain_hfield_samples: np.ndarray | None = None
        if self._terrain_hfield_geom is not None:
            hfield_id = int(self.model.geom_dataid[self._terrain_hfield_geom])
            rows = int(self.model.hfield_nrow[hfield_id])
            columns = int(self.model.hfield_ncol[hfield_id])
            address = int(self.model.hfield_adr[hfield_id])
            self._terrain_hfield_id = hfield_id
            self._terrain_hfield_size = self.model.hfield_size[hfield_id].copy()
            self._terrain_hfield_rows = rows
            self._terrain_hfield_columns = columns
            self._terrain_hfield_samples = self.model.hfield_data[
                address : address + rows * columns
            ].reshape(rows, columns).copy()
        requested_residual_limits = np.array((8.0, 8.0, 0.75, 8.0, 8.0, 0.75))
        self.residual_limits = np.minimum(
            requested_residual_limits,
            0.25 * (self._control_high - self._control_low),
        )
        self.action_space = spaces.Box(
            low=-np.ones(self.model.nu + 1, dtype=np.float32),
            high=np.ones(self.model.nu + 1, dtype=np.float32),
            dtype=np.float32,
        )

        self._hip_qpos_addresses = np.array(
            [
                self.model.jnt_qposadr[
                    lqr.object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in (
                    "left_hip_pitch",
                    "left_active_link_pitch",
                    "right_hip_pitch",
                    "right_active_link_pitch",
                )
            ],
            dtype=np.int32,
        )
        self._hip_dof_addresses = np.array(
            [
                self.model.jnt_dofadr[
                    lqr.object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in (
                    "left_hip_pitch",
                    "left_active_link_pitch",
                    "right_hip_pitch",
                    "right_active_link_pitch",
                )
            ],
            dtype=np.int32,
        )
        self._robot_body = self.refs.robot_body

        # 3 tilt + 3 velocity + 3 angular velocity + 4 hip positions +
        # 4 hip velocities + 2 wheel velocities + 2 leg lengths + 2 length
        # rates + speed/leg/yaw-rate/jump commands + 3 heading features + jump task/countdown/phase
        # state + 5 jump-height/vertical-motion features + 16 terrain-preview
        # features + 2 contacts + 7 previous actions.
        self._observation_size = 51 + TERRAIN_OBSERVATION_SIZE
        self.observation_space = spaces.Box(
            low=np.full(self._observation_size, -10.0, dtype=np.float32),
            high=np.full(self._observation_size, 10.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._controller: PhysicalLqr | None = None
        self._stance_qpos: np.ndarray | None = None
        self._stance_qvel: np.ndarray | None = None
        self._stance_ctrl: np.ndarray | None = None
        self._stance_leg_lengths: np.ndarray | None = None
        self._default_leg_length = 0.0
        self._reference_quaternion: np.ndarray | None = None
        self._reference_heading_yaw = 0.0
        self._reference_body_height = 0.0
        self._command_speed = 0.0
        self._lqr_speed_reference_scale = 1.0
        self._command_yaw_rate_rad_s = 0.0
        self._conditioned_yaw_rate_rad_s = 0.0
        self._jump_command_request = False
        self._jump_request_latched = False
        self._next_command_resample_s = np.inf
        self._command_leg_length = 0.0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self._last_lqr_control = np.zeros(self.model.nu, dtype=np.float64)
        self._last_requested_control = np.zeros(self.model.nu, dtype=np.float64)
        self._last_control = np.zeros(self.model.nu, dtype=np.float64)
        self._last_control_duration_s = (
            self.control_decimation * self.model.opt.timestep
        )
        self._wheel_contact_loss_steps = np.zeros(2, dtype=np.int32)
        self._contact_recovery_active = False
        self._contact_recovery_started_s = -np.inf
        self._contact_recovery_stable_steps = 0
        self._contact_recovery_count = 0
        self._contact_recovery_saved_speed = 0.0
        self._contact_recovery_saved_yaw = 0.0
        self._contact_recovery_saved_leg_length = 0.0
        self._terrain_drop_latched = False
        self._terrain_drop_last_detected_height_m = 0.0
        self._jump_scheduled = False
        self._jump_at: float | None = None
        self._jump_triggered = False
        self._jump_succeeded = False
        self._jump_landing_stable = False
        self._jump_landing_pending = False
        self._jump_failed = False
        self._jump_failure_reason = ""
        self._jump_success_this_step = False
        self._jump_failure_this_step = False
        self._jump_landing_fall_this_step = False
        self._jump_landing_fall_penalized = False
        self._jump_has_been_airborne = False
        self._jump_peak_body_rise_m = 0.0
        self._jump_peak_clearance_m = 0.0
        self._jump_peak_clearance_mean_m = 0.0
        self._jump_rewarded_peak_clearance_mean_m = 0.0
        self._jump_height_target_reached = False
        self._jump_landing_guard_until_s = -np.inf
        self._active_terrain_task: TerrainTask | None = None
        self._active_terrain_route: TerrainRoute | None = None
        self._terrain_task_deadline_s = self.episode_seconds
        self._terrain_start_xy = np.zeros(2, dtype=np.float64)
        self._terrain_progress_m = 0.0
        self._terrain_previous_progress_m = 0.0
        self._terrain_progress_delta_this_step_m = 0.0
        self._terrain_max_progress_m = 0.0
        self._terrain_lateral_error_m = 0.0
        self._terrain_task_completed = False
        self._terrain_task_failed = False
        self._terrain_completion_this_step = False
        self._terrain_command_tracking_hold_s = 0.0
        self._terrain_command_tracking_ready = False
        self._terrain_progress_jump_triggered = False
        self._terrain_task_has_progress_jump = False
        self._terrain_spawn_surface_height_m = 0.0
        self._terrain_confirmed_support_height_m = 0.0
        self._terrain_projection_support_height_m = float(
            self.model.geom_pos[self.refs.ground_geom, 2]
        )
        self._terrain_drop_latched = False
        self._terrain_drop_last_detected_height_m = 0.0
        self._sensor_noise_standard_deviation = np.zeros(self.model.nsensordata, dtype=np.float64)
        self._sensor_noise_bias = np.zeros(self.model.nsensordata, dtype=np.float64)
        self._sensor_noise_scale = 0.0
        self._state_position_noise = np.zeros(self.model.nq, dtype=np.float64)
        self._state_velocity_noise = np.zeros(self.model.nv, dtype=np.float64)
        self._root_qpos_address = int(self.model.jnt_qposadr[self.refs.root_joint])
        self._root_dof_address = int(self.model.jnt_dofadr[self.refs.root_joint])
        self._control_delay_steps = 0
        self._control_delay_buffer: deque[np.ndarray] = deque()
        self._control_delay_bypassed = False
        self._domain_randomization_sample: dict[str, Any] = self._nominal_domain_randomization_sample()

    def _nominal_domain_randomization_sample(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "profile": "nominal",
            "mass_global_scale": 1.0,
            "body_mass_scale_min": 1.0,
            "body_mass_scale_max": 1.0,
            "inertia_scale_min": 1.0,
            "inertia_scale_max": 1.0,
            "sliding_friction_scale": 1.0,
            "terrain_friction_randomized": False,
            "damping_scale": 1.0,
            "hip_strength_scale": 1.0,
            "wheel_strength_scale": 1.0,
            "sensor_noise_scale": 0.0,
            "control_delay_steps": 0,
            "control_delay_ms": 0.0,
        }

    def _select_episode_domain_randomization(self) -> None:
        if (
            (self._jump_scheduled or self._terrain_task_has_progress_jump)
            and self.jump_domain_randomization is not None
        ):
            self._active_domain_randomization = self.jump_domain_randomization
            self._active_domain_randomization_profile = "jump_safe"
        else:
            self._active_domain_randomization = self.domain_randomization
            self._active_domain_randomization_profile = "walking"

    @staticmethod
    def _sample_uniform(
        rng: np.random.Generator,
        bounds: tuple[float, float],
        size: int | tuple[int, ...] | None = None,
    ) -> float | np.ndarray:
        return rng.uniform(bounds[0], bounds[1], size=size)

    def _sample_mirrored_body_scales(
        self,
        dynamic_bodies: np.ndarray,
        bounds: tuple[float, float],
    ) -> np.ndarray:
        """Sample physical-body scales while preserving left/right linkage symmetry."""
        scales = np.ones(self.model.nbody, dtype=np.float64)
        dynamic_body_ids = {int(body_id) for body_id in dynamic_bodies}
        sampled: set[int] = set()
        for body_id in dynamic_bodies:
            body_id = int(body_id)
            if body_id in sampled:
                continue
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            paired_body_id = -1
            if body_name is not None and body_name.startswith("left_"):
                paired_name = f"right_{body_name[len('left_'):]}"
                paired_body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, paired_name
                )
            scale = float(self._sample_uniform(self.np_random, bounds))
            scales[body_id] = scale
            sampled.add(body_id)
            if paired_body_id in dynamic_body_ids:
                scales[paired_body_id] = scale
                sampled.add(paired_body_id)
        return scales

    def _restore_nominal_model_parameters(self) -> None:
        self.model.body_mass[:] = self._nominal_body_mass
        self.model.body_inertia[:] = self._nominal_body_inertia
        self.model.geom_friction[:] = self._nominal_geom_friction
        self.model.geom_contype[:] = self._nominal_geom_contype
        self.model.geom_conaffinity[:] = self._nominal_geom_conaffinity
        self.model.dof_damping[:] = self._nominal_dof_damping
        self.model.actuator_gainprm[:] = self._nominal_actuator_gainprm
        self.model.actuator_forcerange[:] = self._nominal_actuator_forcerange

    def _apply_domain_randomization(self) -> None:
        """Restore nominal dynamics, then sample one episode-constant model."""
        self._restore_nominal_model_parameters()
        config = self._active_domain_randomization
        sample = self._nominal_domain_randomization_sample()
        sample["profile"] = self._active_domain_randomization_profile
        if not config.enabled:
            mujoco.mj_setConst(self.model, self.data)
            self._domain_randomization_sample = sample
            self._reset_sensor_noise()
            return

        dynamic_bodies = np.flatnonzero(self._nominal_body_mass > 0.0)
        global_mass_scale = float(self._sample_uniform(self.np_random, config.mass_global_range))
        local_mass_scales = self._sample_mirrored_body_scales(
            dynamic_bodies, config.mass_body_range
        )
        body_mass_scales = global_mass_scale * local_mass_scales
        self.model.body_mass[dynamic_bodies] = (
            self._nominal_body_mass[dynamic_bodies] * body_mass_scales[dynamic_bodies]
        )

        inertia_scales = self._sample_mirrored_body_scales(
            dynamic_bodies, config.inertia_range
        )
        self.model.body_inertia[dynamic_bodies] = (
            self._nominal_body_inertia[dynamic_bodies] * inertia_scales[dynamic_bodies, None]
        )

        sliding_friction_scale = float(self._sample_uniform(
            self.np_random, config.friction_sliding_range,
        ))
        friction_geoms = list(self.refs.wheel_geoms)
        if config.randomize_terrain_friction:
            friction_geoms = [*self.refs.ground_geoms, *friction_geoms]
        self.model.geom_friction[list(friction_geoms), 0] = (
            self._nominal_geom_friction[list(friction_geoms), 0] * sliding_friction_scale
        )

        damping_scale = float(self._sample_uniform(self.np_random, config.damping_range))
        damped_dofs = self._nominal_dof_damping > 0.0
        self.model.dof_damping[damped_dofs] = self._nominal_dof_damping[damped_dofs] * damping_scale

        hip_strength_scale = float(self._sample_uniform(self.np_random, config.hip_strength_range))
        wheel_strength_scale = float(self._sample_uniform(self.np_random, config.wheel_strength_range))
        hip_actuators = np.array((0, 1, 3, 4), dtype=np.int32)
        wheel_actuators = np.array((2, 5), dtype=np.int32)
        for actuator_ids, strength_scale in (
            (hip_actuators, hip_strength_scale),
            (wheel_actuators, wheel_strength_scale),
        ):
            self.model.actuator_gainprm[actuator_ids, 0] = (
                self._nominal_actuator_gainprm[actuator_ids, 0] * strength_scale
            )
            self.model.actuator_forcerange[actuator_ids] = (
                self._nominal_actuator_forcerange[actuator_ids] * strength_scale
            )

        mujoco.mj_setConst(self.model, self.data)
        self._sensor_noise_scale = float(self._sample_uniform(
            self.np_random, config.sensor_noise_scale_range,
        ))
        self._control_delay_steps = int(self.np_random.integers(
            config.control_delay_steps_range[0], config.control_delay_steps_range[1] + 1,
        ))
        sample.update({
            "enabled": True,
            "mass_global_scale": global_mass_scale,
            "body_mass_scale_min": float(np.min(body_mass_scales[dynamic_bodies])),
            "body_mass_scale_max": float(np.max(body_mass_scales[dynamic_bodies])),
            "inertia_scale_min": float(np.min(inertia_scales[dynamic_bodies])),
            "inertia_scale_max": float(np.max(inertia_scales[dynamic_bodies])),
            "sliding_friction_scale": sliding_friction_scale,
            "terrain_friction_randomized": bool(config.randomize_terrain_friction),
            "damping_scale": damping_scale,
            "hip_strength_scale": hip_strength_scale,
            "wheel_strength_scale": wheel_strength_scale,
            "sensor_noise_scale": self._sensor_noise_scale,
            "control_delay_steps": self._control_delay_steps,
            "control_delay_ms": 1000.0 * self._control_delay_steps * self.model.opt.timestep,
        })
        self._domain_randomization_sample = sample
        self._reset_sensor_noise()

    def _reset_sensor_noise(self) -> None:
        self._sensor_noise_standard_deviation.fill(0.0)
        self._sensor_noise_bias.fill(0.0)
        self._sensor_noise_sample = np.zeros(self.model.nsensordata, dtype=np.float64)
        self._state_position_noise.fill(0.0)
        self._state_velocity_noise.fill(0.0)
        self._orientation_measurement_noise = np.zeros(3, dtype=np.float64)
        if not self._active_domain_randomization.enabled:
            self._sensor_noise_scale = 0.0
            return

        # Standard deviations use physical units: m/s, rad/s, m, m/s and rad.
        per_sensor_standard_deviation = {
            "world_horizontal_velocity_xy": 0.040,
            "left_wheel_angular_velocity": 0.35,
            "right_wheel_angular_velocity": 0.35,
            "left_leg_length": 0.0015,
            "right_leg_length": 0.0015,
            "left_leg_length_velocity": 0.030,
            "right_leg_length_velocity": 0.030,
            "body_angular_velocity": 0.035,
            "imu_gyroscope": 0.035,
            "imu_linear_accelerometer": 0.15,
            "imu_linear_velocity": 0.040,
        }
        for name, standard_deviation in per_sensor_standard_deviation.items():
            address, dimension = self._sensor_refs[name]
            self._sensor_noise_standard_deviation[address : address + dimension] = (
                standard_deviation * self._sensor_noise_scale
            )
        self._sensor_noise_bias[:] = self.np_random.normal(
            0.0,
            0.25 * self._sensor_noise_standard_deviation,
        )
        self._sample_sensor_noise()

    def _sample_sensor_noise(self) -> None:
        if not self._active_domain_randomization.enabled:
            return
        self._sensor_noise_sample[:] = self.np_random.normal(
            0.0,
            self._sensor_noise_standard_deviation,
        )
        # The policy's attitude/yaw features are IMU-like measurements.  Keep
        # this sample-and-hold at the 100 Hz policy rate instead of 1 kHz white noise.
        self._orientation_measurement_noise[:] = self.np_random.normal(
            0.0,
            0.010 * self._sensor_noise_scale,
            size=3,
        )

        root_qpos = self._root_qpos_address
        root_dof = self._root_dof_address
        self._state_position_noise[root_qpos : root_qpos + 3] = self.np_random.normal(
            0.0,
            0.003 * self._sensor_noise_scale,
            size=3,
        )
        self._state_position_noise[self._hip_qpos_addresses] = self.np_random.normal(
            0.0,
            0.003 * self._sensor_noise_scale,
            size=self._hip_qpos_addresses.size,
        )
        self._state_velocity_noise[root_dof : root_dof + 3] = self.np_random.normal(
            0.0,
            0.040 * self._sensor_noise_scale,
            size=3,
        )
        self._state_velocity_noise[root_dof + 3 : root_dof + 6] = self.np_random.normal(
            0.0,
            0.035 * self._sensor_noise_scale,
            size=3,
        )
        self._state_velocity_noise[self._hip_dof_addresses] = self.np_random.normal(
            0.0,
            0.040 * self._sensor_noise_scale,
            size=self._hip_dof_addresses.size,
        )

    def _prepare_noisy_controller_measurement(self) -> tuple[np.ndarray, np.ndarray]:
        """Temporarily expose a noisy state estimate to the LQR command path."""
        true_qpos = self.data.qpos.copy()
        true_qvel = self.data.qvel.copy()
        if not self._active_domain_randomization.enabled:
            return true_qpos, true_qvel

        self.data.qpos[:] += self._state_position_noise
        root_quaternion = self.data.qpos[self._root_qpos_address + 3 : self._root_qpos_address + 7]
        mujoco.mju_quatIntegrate(root_quaternion, self._orientation_measurement_noise, 1.0)
        self.data.qvel[:] += self._state_velocity_noise
        mujoco.mj_forward(self.model, self.data)
        self.data.sensordata[:] += self._sensor_noise_bias + self._sensor_noise_sample
        return true_qpos, true_qvel

    def _restore_true_controller_state(self, true_qpos: np.ndarray, true_qvel: np.ndarray) -> None:
        """Restore physical state while retaining the LQR's gas-spring force."""
        applied_force = self.data.qfrc_applied.copy()
        self.data.qpos[:] = true_qpos
        self.data.qvel[:] = true_qvel
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:] = applied_force

    def _apply_sensor_noise_to_data(self) -> None:
        """Expose one noisy 100 Hz sensor sample to the controller and policy.

        This runs only after safety checks.  The following ``mj_step`` refreshes
        MuJoCo's raw sensors before the next safety check, so safety guards
        always evaluate the true mechanism rather than noisy measurements.
        """
        if not self._active_domain_randomization.enabled:
            return
        # Recompute the raw sensor vector first; otherwise adding samples at
        # the policy boundary would accumulate noise across control steps.
        mujoco.mj_forward(self.model, self.data)
        self.data.sensordata[:] += self._sensor_noise_bias + self._sensor_noise_sample

    def _measured_heading_noise_rad(self) -> float:
        return float(self._orientation_measurement_noise[2])

    def _reset_control_delay(self) -> None:
        self._control_delay_buffer.clear()
        self._control_delay_bypassed = False
        initial_control = (
            np.zeros(self.model.nu, dtype=np.float64)
            if self._stance_ctrl is None
            else self._stance_ctrl
        )
        for _ in range(self._control_delay_steps):
            self._control_delay_buffer.append(initial_control.copy())

    def _apply_control_delay(self, requested_control: np.ndarray, *, bypass: bool = False) -> np.ndarray:
        if self._control_delay_steps <= 0:
            return requested_control.copy()
        if bypass:
            self._control_delay_buffer.clear()
            self._control_delay_bypassed = True
            return requested_control.copy()
        if self._control_delay_bypassed:
            self._control_delay_buffer.clear()
            for _ in range(self._control_delay_steps):
                self._control_delay_buffer.append(self._last_control.copy())
            self._control_delay_bypassed = False
        self._control_delay_buffer.append(requested_control.copy())
        return self._control_delay_buffer.popleft()

    def _project_low_centre_stance(self) -> None:
        """Create one validated LQR working point and cache its physical state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._controller = lqr.settle_and_relinearize(
            self.model,
            self.data,
            self.refs,
            speed=0.0,
            acceleration_limit=self.acceleration_limit,
            max_forward_speed=self.max_forward_speed,
            max_reverse_speed=self.max_reverse_speed,
        )
        self._stance_qpos = self.data.qpos.copy()
        self._stance_qvel = self.data.qvel.copy()
        self._stance_ctrl = self._controller.control_equilibrium.copy()
        self._stance_leg_lengths = np.array(
            (
                self._sensor("left_leg_length")[0],
                self._sensor("right_leg_length")[0],
            ),
            dtype=np.float64,
        )
        self._default_leg_length = float(np.mean(self._stance_leg_lengths))

    def _restore_stance(self) -> None:
        if self._stance_qpos is None or self._stance_qvel is None or self._stance_ctrl is None:
            raise RuntimeError("low-centre stance has not been projected")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._stance_qpos
        self.data.qvel[:] = self._stance_qvel
        self.data.ctrl[:] = self._stance_ctrl
        self.data.qfrc_applied[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _prepare_lqr_projection_support(self) -> None:
        """Use the inherited plane while building an LQR trim for a terrain scene."""
        if not self._terrain_support_geoms:
            return
        self.model.geom_contype[self.refs.ground_geom] = self._nominal_geom_contype[self.refs.ground_geom]
        self.model.geom_conaffinity[self.refs.ground_geom] = self._nominal_geom_conaffinity[self.refs.ground_geom]
        for geom_id in self._terrain_support_geoms:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
        mujoco.mj_forward(self.model, self.data)
        # The cached LQR stance is projected against this plane, not against
        # whichever static support happens to overlap its world XY coordinate.
        self._terrain_projection_support_height_m = float(
            self.data.geom_xpos[self.refs.ground_geom, 2]
        )

    def _activate_terrain_support(self) -> None:
        """Replace the temporary LQR projection plane with the scene supports."""
        if not self._terrain_support_geoms:
            return
        self.model.geom_contype[self.refs.ground_geom] = 0
        self.model.geom_conaffinity[self.refs.ground_geom] = 0
        for geom_id in self._terrain_support_geoms:
            self.model.geom_contype[geom_id] = self._nominal_geom_contype[geom_id]
            self.model.geom_conaffinity[geom_id] = self._nominal_geom_conaffinity[geom_id]
        mujoco.mj_forward(self.model, self.data)

    def _reset_lqr_command(self, command_yaw: float | None = None) -> None:
        if self._controller is None:
            raise RuntimeError("LQR controller is not available")
        self._controller.reset_commands(
            self.data,
            self._lqr_speed_reference_for_command(self._command_speed),
            leg_length=self._command_leg_length,
            yaw=command_yaw,
        )

    def sample_command_speed(
        self,
        rng: np.random.Generator,
        fraction: float | None = None,
    ) -> float:
        """Sample a training command inside the configured high-level speed envelope."""
        if fraction is None:
            fraction = self.command_speed_fraction
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be within (0, 1]")
        forward_limit = self.command_speed_limit_mps
        reverse_limit = min(self.max_reverse_speed, forward_limit)
        if float(rng.uniform()) < 0.5:
            return float(rng.uniform(-fraction * reverse_limit, 0.0))
        return float(rng.uniform(0.0, fraction * forward_limit))

    def sample_command_yaw_rate(self, rng: np.random.Generator) -> float:
        """Sample a signed high-level yaw-rate command for locomotion training."""
        if self.max_command_yaw_rate_rad_s <= 0.0:
            return 0.0
        return float(rng.uniform(
            -self.max_command_yaw_rate_rad_s,
            self.max_command_yaw_rate_rad_s,
        ))

    def _select_terrain_task(
        self,
        options: dict[str, Any],
    ) -> tuple[TerrainTask | None, TerrainRoute | None]:
        """Choose one fixed reset corridor without exposing it to the policy.

        A terrain task supplies only the high-level command and the initial
        pose.  The route endpoint is retained for reward and evaluation
        bookkeeping; it is never translated into a navigation target.
        """
        requested_task_id = options.get("terrain_task_id")
        requested_route_index = options.get("terrain_route_index")
        if self.terrain_curriculum is None:
            if requested_task_id is not None or requested_route_index is not None:
                raise ValueError("terrain_task_id and terrain_route_index require terrain_curriculum")
            return None, None
        if requested_task_id is None:
            if self._terrain_stage is None:
                raise RuntimeError("terrain curriculum has no active stage")
            candidates = tuple(
                self.terrain_curriculum.task(task_id) for task_id in self._terrain_stage.task_ids
            )
            weights = np.asarray([task.sampling_weight for task in candidates], dtype=np.float64)
            task = candidates[int(self.np_random.choice(len(candidates), p=weights / np.sum(weights)))]
        else:
            if not isinstance(requested_task_id, str):
                raise ValueError("terrain_task_id must be a string")
            task = self.terrain_curriculum.task(requested_task_id)
            if self._terrain_stage is not None and task.task_id not in self._terrain_stage.task_ids:
                raise ValueError(
                    f"terrain task {task.task_id!r} is not enabled by stage {self._terrain_stage.stage_id!r}"
                )
        if requested_route_index is None:
            route_index = int(self.np_random.integers(len(task.routes)))
        elif isinstance(requested_route_index, (int, np.integer)) and requested_route_index >= 0:
            route_index = int(requested_route_index)
        else:
            raise ValueError("terrain_route_index must be a non-negative integer")
        return task, task.route_at(route_index)

    def _terrain_task_command(self, task: TerrainTask | None) -> LocomotionCommand | None:
        if task is None:
            return None
        if self._terrain_stage is None:
            raise RuntimeError("terrain task selected without an active terrain stage")
        command = self._terrain_stage.command_for(task)
        return LocomotionCommand(
            forward_speed_mps=command.forward_speed_mps,
            yaw_rate_rad_s=command.yaw_rate_rad_s,
            jump_request=command.jump_request,
        )

    def _set_root_heading(self, heading_yaw: float) -> None:
        """Rotate the free root so LQR's rolling-direction yaw matches a task pose."""
        if self._controller is None:
            raise RuntimeError("LQR controller is not available")
        current_heading = self._controller.heading_yaw(self.data)
        delta = wrap_to_pi(float(heading_yaw) - current_heading)
        yaw_rotation = np.array((np.cos(0.5 * delta), 0.0, 0.0, np.sin(0.5 * delta)))
        root_quaternion = self.data.qpos[
            self._root_qpos_address + 3 : self._root_qpos_address + 7
        ].copy()
        rotated = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(rotated, yaw_rotation, root_quaternion)
        self.data.qpos[self._root_qpos_address + 3 : self._root_qpos_address + 7] = rotated

    def _apply_terrain_route_spawn(self, route: TerrainRoute) -> None:
        """Translate a validated flat LQR stance to one fixed terrain start.

        The physical root is moved only during reset.  During simulation the
        LQR receives a filtered support-height reference, never an injected
        root-position update.
        """
        if self._controller is None or self._stance_qpos is None or self._stance_qvel is None:
            raise RuntimeError("low-centre stance has not been projected")
        source_height = self._terrain_projection_support_height_m
        spawn_xy = np.asarray(route.spawn.xy(), dtype=np.float64)
        spawn_height = self.terrain_surface_height_m(spawn_xy)
        self.data.qpos[:] = self._stance_qpos
        self.data.qvel[:] = self._stance_qvel
        if self._stance_ctrl is not None:
            self.data.ctrl[:] = self._stance_ctrl
        self.data.qfrc_applied[:] = 0.0
        self.data.qpos[self._root_qpos_address : self._root_qpos_address + 2] = spawn_xy
        self.data.qpos[self._root_qpos_address + 2] += spawn_height - source_height
        mujoco.mj_forward(self.model, self.data)
        self._set_root_heading(route.spawn.yaw_rad)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._terrain_spawn_surface_height_m = float(spawn_height)
        self._terrain_start_xy = spawn_xy.copy()

    def _validate_terrain_route_spawn(self, route: TerrainRoute) -> None:
        """Reject an invalid declared spawn before it becomes a training sample."""
        support_height, valid = self._terrain_wheel_support_state()
        if not valid:
            raise RuntimeError(
                f"terrain route {route.route_id!r} places a wheel outside the active hfield"
            )
        try:
            lqr.validate_standing_contact(self.data, self.refs)
        except RuntimeError as error:
            contacts = self._wheel_contacts()
            raise RuntimeError(
                f"terrain route {route.route_id!r} has invalid initial support: "
                f"contacts={contacts}, support_height={support_height:.4f}m"
            ) from error
        self._terrain_confirmed_support_height_m = support_height

    def _clear_terrain_route_state(self) -> None:
        self._active_terrain_task = None
        self._active_terrain_route = None
        self._lqr_speed_reference_scale = 1.0
        self._terrain_task_deadline_s = self.episode_seconds
        self._terrain_start_xy.fill(0.0)
        self._terrain_progress_m = 0.0
        self._terrain_previous_progress_m = 0.0
        self._terrain_progress_delta_this_step_m = 0.0
        self._terrain_max_progress_m = 0.0
        self._terrain_lateral_error_m = 0.0
        self._terrain_task_completed = False
        self._terrain_task_failed = False
        self._terrain_completion_this_step = False
        self._terrain_command_tracking_hold_s = 0.0
        self._terrain_command_tracking_ready = False
        self._terrain_progress_jump_triggered = False
        self._terrain_task_has_progress_jump = False
        self._terrain_spawn_surface_height_m = 0.0
        self._terrain_confirmed_support_height_m = 0.0

    def _terrain_route_state(self) -> tuple[float, float, bool]:
        """Return signed progress, lateral displacement, and corridor validity."""
        route = self._active_terrain_route
        if route is None:
            return 0.0, 0.0, True
        start = np.asarray(route.spawn.xy(), dtype=np.float64)
        goal = np.asarray(route.goal.xy(), dtype=np.float64)
        direction = goal - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            raise RuntimeError("terrain route has zero planar length")
        direction /= length
        displacement = self.data.xpos[self._robot_body, :2] - start
        progress = float(np.dot(displacement, direction))
        lateral = float(abs(direction[0] * displacement[1] - direction[1] * displacement[0]))
        return progress, lateral, lateral <= route.corridor_half_width_m

    def _update_terrain_task_progress(self) -> None:
        if self._active_terrain_task is None:
            return
        progress, lateral, corridor_valid = self._terrain_route_state()
        self._terrain_previous_progress_m = self._terrain_progress_m
        self._terrain_progress_m = progress
        self._terrain_max_progress_m = max(self._terrain_max_progress_m, progress)
        self._terrain_lateral_error_m = lateral
        task = self._active_terrain_task
        if task.uses_command_tracking_completion:
            speed_tolerance = task.speed_tracking_tolerance_mps
            yaw_tolerance = task.yaw_rate_tracking_tolerance_rad_s
            hold_seconds = task.command_tracking_hold_seconds
            if speed_tolerance is None or yaw_tolerance is None or hold_seconds is None:
                raise RuntimeError("command-tracking terrain task is missing validated tolerances")
            yaw_state = self.yaw_state()
            speed_matches = abs(self._forward_speed() - self._command_speed) <= speed_tolerance
            yaw_rate_matches = (
                abs(yaw_state["yaw_rate_rad_s"] - self._command_yaw_rate_rad_s)
                <= yaw_tolerance
            )
            self._terrain_command_tracking_ready = bool(speed_matches and yaw_rate_matches)
            if self._terrain_command_tracking_ready:
                self._terrain_command_tracking_hold_s += self.model.opt.timestep
            else:
                self._terrain_command_tracking_hold_s = 0.0
            complete = self._terrain_command_tracking_hold_s >= hold_seconds
        else:
            self._terrain_command_tracking_ready = False
            self._terrain_command_tracking_hold_s = 0.0
            complete = (
                progress >= task.required_distance_m - task.completion_tolerance_m
                and corridor_valid
            )
        if complete and not self._terrain_task_completed:
            self._terrain_task_completed = True
            self._terrain_completion_this_step = True

        if task.jump_edge_due(
            progress,
            already_triggered=self._terrain_progress_jump_triggered,
        ):
            # This is a task-level high-level command edge, emitted once at a
            # declared corridor progress.  It is neither a policy output nor a
            # navigation command.  request_lqr_jump preserves the current
            # travel request for its validated low-speed rolling launch.
            self._terrain_progress_jump_triggered = True
            self._jump_command_request = True
            self._jump_request_latched = True
            self.request_lqr_jump(launch_speed_mps=task.jump_launch_speed_mps)

    def _clamp_command_speed(self, speed: float) -> float:
        forward_limit = min(self.max_forward_speed, self.command_speed_limit_mps)
        reverse_limit = min(self.max_reverse_speed, forward_limit)
        value = float(speed)
        if not np.isfinite(value):
            raise ValueError("command_speed must be finite")
        # ``lqr.validate_forward_speed_limit`` describes the physical model
        # envelope (currently >=1 m/s), whereas a curriculum command limit may
        # intentionally be a conservative value such as 0.20 m/s.  Apply the
        # high-level envelope locally and leave the physical LQR limit intact.
        return float(np.clip(value, -reverse_limit, forward_limit))

    def _lqr_speed_reference_for_command(self, high_level_speed_mps: float) -> float:
        """Map a terrain task's high-level speed into its LQR reference."""
        return float(high_level_speed_mps) * self._lqr_speed_reference_scale

    def _clamp_command_yaw_rate(self, yaw_rate_rad_s: float) -> float:
        value = float(yaw_rate_rad_s)
        if not np.isfinite(value):
            raise ValueError("command_yaw_rate_rad_s must be finite")
        return float(np.clip(
            value,
            -self.max_command_yaw_rate_rad_s,
            self.max_command_yaw_rate_rad_s,
        ))

    def _uses_high_speed_command_conditioning(self) -> bool:
        """Whether the current terrain command needs a gradual high-speed turn-in."""
        return bool(
            self._terrain_hfield_geom is not None
            and abs(self._command_speed) >= TERRAIN_HIGH_SPEED_COMMAND_MIN_MPS
            and abs(self._command_yaw_rate_rad_s)
            >= TERRAIN_HIGH_SPEED_COMMAND_MIN_YAW_RATE_RAD_S
        )

    def _uses_terrain_direct_yaw_rate_tracking(self) -> bool:
        """Whether a routed terrain task has an explicit nonzero yaw-rate command."""
        return bool(
            self._controller is not None
            and self._controller.terrain_heading_stabilization_enabled
            and abs(self._command_yaw_rate_rad_s) >= TERRAIN_DIRECT_YAW_RATE_MIN_RAD_S
        )

    def _apply_command_conditioning(self) -> None:
        """Keep high-speed command entry inside the verified terrain envelope."""
        if self._controller is None:
            return
        self._controller.motion.acceleration_limit = (
            TERRAIN_HIGH_SPEED_ACCELERATION_MPS2
            if self._uses_high_speed_command_conditioning()
            else self.acceleration_limit
        )

    def _sync_terrain_yaw_rate_command(self) -> None:
        """Route the conditioned terrain yaw request to the LQR."""
        if self._controller is None:
            return
        if self._uses_terrain_direct_yaw_rate_tracking():
            self._controller.set_terrain_yaw_rate_command(
                self._conditioned_yaw_rate_rad_s
            )
        else:
            self._controller.set_terrain_yaw_rate_command(None)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self._clear_terrain_route_state()
        terrain_task, terrain_route = self._select_terrain_task(options)
        terrain_command = self._terrain_task_command(terrain_task)
        self._lqr_speed_reference_scale = (
            1.0 if terrain_task is None else terrain_task.lqr_speed_reference_scale
        )
        if terrain_command is not None and any(
            key in options
            for key in (
                "locomotion_command",
                "command_speed",
                "command_yaw_rate_rad_s",
                "jump_request",
                "jump_at",
            )
        ):
            raise ValueError(
                "terrain_task_id owns the high-level locomotion command; do not override it in reset options"
            )
        command_option = terrain_command if terrain_command is not None else options.get("locomotion_command")
        if command_option is None:
            locomotion_command: LocomotionCommand | None = None
        elif isinstance(command_option, LocomotionCommand):
            locomotion_command = command_option
        elif isinstance(command_option, dict):
            try:
                locomotion_command = LocomotionCommand(
                    forward_speed_mps=float(command_option["forward_speed_mps"]),
                    yaw_rate_rad_s=float(command_option["yaw_rate_rad_s"]),
                    jump_request=bool(command_option.get("jump_request", False)),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "locomotion_command must provide finite forward_speed_mps and yaw_rate_rad_s"
                ) from error
        else:
            raise ValueError("locomotion_command must be a LocomotionCommand or mapping")
        command_yaw = options.get("command_yaw")
        command_yaw_delta = options.get("command_yaw_delta_rad")
        command_yaw_rate = options.get("command_yaw_rate_rad_s")
        if command_yaw is not None:
            command_yaw = float(command_yaw)
            if not np.isfinite(command_yaw):
                raise ValueError("command_yaw must be finite")
        if command_yaw_delta is not None:
            command_yaw_delta = float(command_yaw_delta)
            if not np.isfinite(command_yaw_delta) or abs(command_yaw_delta) > np.pi:
                raise ValueError("command_yaw_delta_rad must be within -pi..pi")
            if command_yaw is not None:
                raise ValueError("provide only one of command_yaw and command_yaw_delta_rad")
        if command_yaw_rate is not None:
            command_yaw_rate = self._clamp_command_yaw_rate(command_yaw_rate)
        if locomotion_command is not None and command_yaw_rate is not None:
            raise ValueError("provide command_yaw_rate_rad_s either directly or through locomotion_command")

        jump_request = (
            bool(locomotion_command.jump_request)
            if locomotion_command is not None
            else bool(options.get("jump_request", False))
        )
        if not isinstance(options.get("jump_request", False), (bool, np.bool_)):
            raise ValueError("jump_request must be boolean")
        if jump_request and "jump_at" in options:
            raise ValueError("provide jump_request or jump_at, not both")

        if jump_request:
            requested_jump_at = 0.0
        elif "jump_at" in options:
            requested_jump_at = options["jump_at"]
            if requested_jump_at is not None:
                requested_jump_at = float(requested_jump_at)
                if not np.isfinite(requested_jump_at) or not 0.0 <= requested_jump_at < self.episode_seconds:
                    raise ValueError("jump_at must be within the episode")
        elif (
            terrain_task is None
            and self.jump_probability > 0.0
            and float(self.np_random.uniform()) < self.jump_probability
        ):
            requested_jump_at = self.jump_at
        else:
            requested_jump_at = None
        self._jump_at = requested_jump_at
        self._jump_scheduled = requested_jump_at is not None

        if locomotion_command is not None:
            command_speed = float(locomotion_command.forward_speed_mps)
        elif "command_speed" in options:
            command_speed = float(options["command_speed"])
        elif self.randomize_command:
            command_speed = self.sample_command_speed(self.np_random)
        else:
            command_speed = 0.0
        self._command_speed = self._clamp_command_speed(command_speed)
        if locomotion_command is not None:
            self._command_yaw_rate_rad_s = self._clamp_command_yaw_rate(
                locomotion_command.yaw_rate_rad_s
            )
        elif command_yaw_rate is not None:
            self._command_yaw_rate_rad_s = command_yaw_rate
        elif self.randomize_command:
            self._command_yaw_rate_rad_s = self.sample_command_yaw_rate(self.np_random)
        else:
            self._command_yaw_rate_rad_s = 0.0
        self._conditioned_yaw_rate_rad_s = 0.0
        if "command_leg_length" in options:
            command_leg_length = float(options["command_leg_length"])
        elif self.randomize_leg_length:
            command_leg_length = float(self.np_random.uniform(
                lqr.WALK_LEG_LENGTH_MIN_M + 0.01,
                lqr.WALK_LEG_LENGTH_MAX_M - 0.01,
            ))
        else:
            command_leg_length = self._default_leg_length or 0.5 * (
                lqr.WALK_LEG_LENGTH_MIN_M + lqr.WALK_LEG_LENGTH_MAX_M
            )
        if not lqr.WALK_LEG_LENGTH_MIN_M <= command_leg_length <= lqr.WALK_LEG_LENGTH_MAX_M:
            raise ValueError(
                f"command_leg_length must be within {lqr.WALK_LEG_LENGTH_MIN_M:.3f}.."
                f"{lqr.WALK_LEG_LENGTH_MAX_M:.3f}m"
            )
        self._command_leg_length = command_leg_length
        self._terrain_task_has_progress_jump = bool(
            terrain_task is not None and terrain_task.has_progress_jump_trigger
        )

        # Domains are episode-constant.  Rebuilding the working point after a
        # physical change is required because equilibrium, trims and LQR gains
        # all depend on the sampled model.
        self._select_episode_domain_randomization()
        self._apply_domain_randomization()
        self._prepare_lqr_projection_support()
        if self._controller is None or self._active_domain_randomization.enabled:
            self._project_low_centre_stance()
        else:
            self._restore_stance()
        self._activate_terrain_support()
        if terrain_route is not None:
            self._apply_terrain_route_spawn(terrain_route)
        if not self.randomize_leg_length and "command_leg_length" not in options:
            self._command_leg_length = self._default_leg_length

        current_yaw = self._controller.heading_yaw(self.data)
        if terrain_route is not None:
            command_yaw = terrain_route.spawn.yaw_rad
        elif command_yaw is None:
            if command_yaw_delta is not None:
                command_yaw = wrap_to_pi(current_yaw + command_yaw_delta)
            elif self._command_yaw_rate_rad_s == 0.0 and self.max_command_yaw_delta_rad > 0.0:
                command_yaw = wrap_to_pi(current_yaw + float(self.np_random.uniform(
                    -self.max_command_yaw_delta_rad,
                    self.max_command_yaw_delta_rad,
                )))
        self._reset_lqr_command(command_yaw)
        self._apply_command_conditioning()
        if terrain_route is not None:
            self._validate_terrain_route_spawn(terrain_route)
            self._controller.configure_terrain_support_reference(
                self.data,
                self._terrain_spawn_surface_height_m,
            )
            self._controller.configure_terrain_heading_stabilization(True)
            self._active_terrain_task = terrain_task
            self._active_terrain_route = terrain_route
            self._terrain_task_deadline_s = min(
                self.episode_seconds,
                float(terrain_task.max_episode_seconds),
            )
            self._update_terrain_task_progress()
        else:
            self._controller.disable_terrain_support_reference()
            self._controller.configure_terrain_heading_stabilization(False)
        self._sync_terrain_yaw_rate_command()

        self._reference_quaternion = self.data.xquat[self._robot_body].copy()
        self._reference_heading_yaw = self._controller.heading_yaw(self.data)
        self._reference_body_height = float(self.data.xpos[self._robot_body, 2])
        self._previous_action.fill(0.0)
        self._last_lqr_control[:] = self._stance_ctrl
        self._last_requested_control[:] = self._stance_ctrl
        self._last_control[:] = self._stance_ctrl
        self._reset_control_delay()
        self._sample_sensor_noise()
        self._wheel_contact_loss_steps.fill(0)
        self._contact_recovery_active = False
        self._contact_recovery_started_s = -np.inf
        self._contact_recovery_stable_steps = 0
        self._contact_recovery_count = 0
        self._contact_recovery_saved_speed = self._command_speed
        self._contact_recovery_saved_yaw = self._controller.command_yaw
        self._contact_recovery_saved_leg_length = self._command_leg_length
        self._terrain_drop_latched = False
        self._terrain_drop_last_detected_height_m = 0.0
        # Scheduled training jumps remain visible through the countdown features,
        # but the high-level jump bit changes only when the command is issued.
        self._jump_command_request = jump_request
        self._jump_request_latched = False
        self._next_command_resample_s = (
            float(self.command_resample_seconds)
            if (
                terrain_task is None
                and self.randomize_command
                and self.command_resample_seconds is not None
            )
            else np.inf
        )
        self._jump_triggered = False
        self._jump_succeeded = False
        self._jump_landing_stable = False
        self._jump_landing_pending = False
        self._jump_failed = False
        self._jump_failure_reason = ""
        self._jump_success_this_step = False
        self._jump_failure_this_step = False
        self._jump_landing_fall_this_step = False
        self._jump_landing_fall_penalized = False
        self._jump_has_been_airborne = False
        self._jump_peak_body_rise_m = 0.0
        self._jump_peak_clearance_m = 0.0
        self._jump_peak_clearance_mean_m = 0.0
        self._jump_rewarded_peak_clearance_mean_m = 0.0
        self._jump_height_target_reached = False
        self._jump_landing_guard_until_s = -np.inf
        info = self._info("reset")
        self._apply_sensor_noise_to_data()
        return self._observation(), info

    def _sensor(self, name: str) -> np.ndarray:
        address, dimension = self._sensor_refs[name]
        return self.data.sensordata[address : address + dimension].copy()

    def _orientation_error(self) -> np.ndarray:
        if self._reference_quaternion is None:
            return np.zeros(3, dtype=np.float64)
        reference_quaternion = self._reference_quaternion
        if self._controller is not None:
            try:
                yaw_delta = wrap_to_pi(
                    self._controller.heading_yaw(self.data) - self._reference_heading_yaw
                )
            except RuntimeError:
                yaw_delta = 0.0
            yaw_rotation = np.array(
                (np.cos(0.5 * yaw_delta), 0.0, 0.0, np.sin(0.5 * yaw_delta))
            )
            heading_aligned_reference = np.empty(4)
            mujoco.mju_mulQuat(heading_aligned_reference, yaw_rotation, reference_quaternion)
            reference_quaternion = heading_aligned_reference
        error = np.empty(3, dtype=np.float64)
        mujoco.mju_subQuat(
            error,
            self.data.xquat[self._robot_body],
            reference_quaternion,
        )
        return error

    def _wheel_contacts(self) -> tuple[int, int]:
        return tuple(
            lqr.wheel_ground_contacts(self.data, self.refs, geom_id)
            for geom_id in self.refs.wheel_geoms
        )

    def _wheel_support_confidence(self) -> np.ndarray:
        """Return per-wheel support confidence without changing raw telemetry.

        MuJoCo's hfield contact manifold can be momentarily absent under a
        high-speed rolling wheel even though its bottom remains within a few
        millimetres of the sampled support. Keep the geometry exception scoped
        to the explicitly conditioned high-speed task: ordinary walking,
        jump, and drop phases retain raw-contact safety semantics.
        """
        contacts = np.asarray(self._wheel_contacts(), dtype=bool)
        if self._terrain_hfield_geom is None or not self._uses_high_speed_command_conditioning():
            return contacts
        near_support = self._wheel_ground_clearances_m() <= WHEEL_NEAR_SUPPORT_CLEARANCE_M
        return np.logical_or(contacts, near_support)

    def _contact_loss_duration_s(self) -> float:
        """Return the longest current per-wheel contact-loss duration."""
        return float(np.max(self._wheel_contact_loss_steps)) * self.model.opt.timestep

    def _contact_recovery_duration_s(self) -> float:
        if not self._contact_recovery_active:
            return 0.0
        return max(0.0, float(self.data.time) - self._contact_recovery_started_s)

    def _start_contact_recovery(self) -> None:
        """Temporarily hand a persistent wheel unload back to the LQR alone."""
        if self._controller is None or self._contact_recovery_active:
            return
        self._contact_recovery_active = True
        self._contact_recovery_started_s = float(self.data.time)
        self._contact_recovery_stable_steps = 0
        self._contact_recovery_count += 1
        self._contact_recovery_saved_speed = self._command_speed
        self._contact_recovery_saved_yaw = self._controller.command_yaw
        self._contact_recovery_saved_leg_length = self._command_leg_length

        # The recovery path uses a noise/delay-free LQR command on subsequent
        # physics steps.  It does not weaken any geometry or linkage limit.
        self._controller.begin_contact_recovery(self.data, self._default_leg_length)

    def _cancel_contact_recovery(self) -> None:
        """Clear a pending wheel-contact recovery before handing control to a jump."""
        if self._controller is not None:
            self._controller.end_contact_recovery()
        self._contact_recovery_active = False
        self._contact_recovery_started_s = -np.inf
        self._contact_recovery_stable_steps = 0
        self._wheel_contact_loss_steps.fill(0)

    def _finish_contact_recovery(self) -> None:
        """Restore the operator/task command after sustained two-wheel support."""
        if self._controller is None or not self._contact_recovery_active:
            return
        self._cancel_contact_recovery()
        self._controller.set_target_leg_length(self._contact_recovery_saved_leg_length)
        self._controller.set_command_yaw(self._contact_recovery_saved_yaw)
        self._controller.set_target_speed(
            self._lqr_speed_reference_for_command(self._contact_recovery_saved_speed)
        )
        # begin_contact_recovery deliberately freezes the direct yaw-rate
        # loop while braking.  Once the two-wheel stability gate passes,
        # restore the still-active high-level rate request in this same
        # control interval instead of waiting for the next policy step.
        self._sync_terrain_yaw_rate_command()

    def _update_wheel_contact_recovery(self) -> str | None:
        """Track wheel contacts independently and recover before declaring a fall."""
        if self._controller is not None and self._controller.drop.active:
            # A full-width terrain drop expects a short, bounded loss of both
            # wheel contacts.  Its dedicated controller owns this interval;
            # the ordinary recovery timeout resumes only after handoff.
            self._wheel_contact_loss_steps.fill(0)
            return None
        supports = self._wheel_support_confidence()
        self._wheel_contact_loss_steps = np.where(
            supports,
            0,
            self._wheel_contact_loss_steps + 1,
        ).astype(np.int32)

        if not self._contact_recovery_active:
            recovery_trigger = (
                CONTACT_RECOVERY_BOTH_WHEELS_TRIGGER_SECONDS
                if not bool(np.any(supports))
                else CONTACT_RECOVERY_SINGLE_WHEEL_TRIGGER_SECONDS
            )
            if self._contact_loss_duration_s() >= recovery_trigger:
                self._start_contact_recovery()
            return None

        if bool(np.all(supports)):
            self._contact_recovery_stable_steps += 1
            stable_duration_s = (
                self._contact_recovery_stable_steps * self.model.opt.timestep
            )
            if stable_duration_s >= CONTACT_RECOVERY_STABLE_SECONDS:
                self._finish_contact_recovery()
                return None
        else:
            self._contact_recovery_stable_steps = 0

        if self._contact_recovery_duration_s() >= CONTACT_RECOVERY_TIMEOUT_SECONDS:
            return "wheel_contact_loss_timeout"
        return None

    def _static_box_support_height_m(
        self,
        geom_id: int,
        world_xy: np.ndarray,
    ) -> float | None:
        """Return the upward-facing top height of one static support box at XY."""
        center = self.data.geom_xpos[geom_id]
        rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
        half_x, half_y, half_z = self.model.geom_size[geom_id, :3]

        # The world-XY projection of the local top face is invertible whenever
        # that face has a nonzero vertical normal.  Select its upward-facing
        # side so this also remains correct for an accidentally inverted box.
        face_sign = 1.0 if rotation[2, 2] >= 0.0 else -1.0
        face_local_z = face_sign * half_z
        xy_basis = rotation[:2, :2]
        if abs(float(np.linalg.det(xy_basis))) <= STATIC_SUPPORT_FOOTPRINT_TOLERANCE_M:
            return None
        top_origin_xy = center[:2] + rotation[:2, 2] * face_local_z
        local_xy = np.linalg.solve(xy_basis, world_xy - top_origin_xy)
        if (
            abs(float(local_xy[0])) > half_x + STATIC_SUPPORT_FOOTPRINT_TOLERANCE_M
            or abs(float(local_xy[1])) > half_y + STATIC_SUPPORT_FOOTPRINT_TOLERANCE_M
        ):
            return None
        return float(
            center[2]
            + rotation[2, 0] * local_xy[0]
            + rotation[2, 1] * local_xy[1]
            + rotation[2, 2] * face_local_z
        )

    def _terrain_surface_height_and_validity(
        self,
        world_xy: np.ndarray | tuple[float, float],
    ) -> tuple[float, bool]:
        """Return the highest active support surface and its footprint validity."""
        xy = np.asarray(world_xy, dtype=np.float64)
        if xy.shape != (2,):
            raise ValueError(f"expected world XY shape (2,), got {xy.shape}")
        ground_height = float(self.data.geom_xpos[self.refs.ground_geom, 2])
        hfield_height = ground_height
        hfield_valid = False
        hfield_available = (
            self._terrain_hfield_geom is not None
            and self._terrain_hfield_size is not None
            and self._terrain_hfield_samples is not None
        )
        if hfield_available:
            geom_id = self._terrain_hfield_geom
            center = self.data.geom_xpos[geom_id]
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            local = rotation.T @ np.array((xy[0] - center[0], xy[1] - center[1], 0.0))
            half_x, half_y, height_scale, _ = self._terrain_hfield_size
            normalized_x = (local[0] + half_x) / (2.0 * half_x)
            normalized_y = (local[1] + half_y) / (2.0 * half_y)
            if 0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0:
                column = normalized_x * (self._terrain_hfield_columns - 1)
                row = normalized_y * (self._terrain_hfield_rows - 1)
                column0 = int(np.floor(column))
                row0 = int(np.floor(row))
                column1 = min(column0 + 1, self._terrain_hfield_columns - 1)
                row1 = min(row0 + 1, self._terrain_hfield_rows - 1)
                row_fraction = row - row0
                column_fraction = column - column0
                samples = self._terrain_hfield_samples
                lower = (
                    (1.0 - column_fraction) * samples[row0, column0]
                    + column_fraction * samples[row0, column1]
                )
                upper = (
                    (1.0 - column_fraction) * samples[row1, column0]
                    + column_fraction * samples[row1, column1]
                )
                local_height = (
                    (1.0 - row_fraction) * lower + row_fraction * upper
                ) * height_scale
                hfield_height = float(center[2] + local_height)
                hfield_valid = True
            else:
                hfield_height = float(center[2] - self._terrain_hfield_size[3])

        # Preserve the established RMUC hfield path byte-for-byte in behavior
        # when no static terrain has been registered.
        if not self._terrain_static_box_geoms:
            if hfield_available:
                return hfield_height, hfield_valid
            return ground_height, True

        candidate_heights: list[float] = [hfield_height] if hfield_valid else []
        for geom_id in self._terrain_static_box_geoms:
            height = self._static_box_support_height_m(geom_id, xy)
            if height is not None:
                candidate_heights.append(height)
        if candidate_heights:
            return float(max(candidate_heights)), True
        # Terrain routes deliberately disable the flat projection plane.  An
        # XY point outside every declared support must remain invalid even
        # though a numerical fallback height is still useful to callers.
        return hfield_height if self._terrain_hfield_geom is not None else ground_height, False

    def terrain_surface_height_m(self, world_xy: np.ndarray | tuple[float, float]) -> float:
        """Return the active support height under a world XY point."""
        height, _ = self._terrain_surface_height_and_validity(world_xy)
        return height

    def _local_full_width_drop_ahead_m(self) -> float | None:
        """Return a locally observed down-step only when both wheel tracks agree.

        This intentionally consumes only hfield samples in the robot heading
        frame.  It never reads a route goal or changes the high-level command,
        so the resulting supervisor remains a local full-body reflex rather
        than a navigation primitive.
        """
        controller = self._controller
        if (
            controller is None
            or self._terrain_hfield_geom is None
            or not controller.terrain_support_reference_enabled
            or controller.forward_speed(self.data) < lqr.TERRAIN_DROP_ARM_MIN_FORWARD_SPEED_MPS
        ):
            return None
        heading = controller.heading_yaw(self.data)
        forward = np.array((np.cos(heading), np.sin(heading)), dtype=np.float64)
        support_heights: list[float] = []
        ahead_heights: list[float] = []
        valid = True
        for geom_id in self.refs.wheel_geoms:
            wheel_xy = self.data.geom_xpos[geom_id, :2]
            support_height, support_valid = self._terrain_surface_height_and_validity(wheel_xy)
            ahead_height, ahead_valid = self._terrain_surface_height_and_validity(
                wheel_xy + lqr.TERRAIN_DROP_LOOKAHEAD_M * forward
            )
            support_heights.append(support_height)
            ahead_heights.append(ahead_height)
            valid = valid and support_valid and ahead_valid
        if not valid:
            return None
        drops = np.asarray(support_heights, dtype=np.float64) - np.asarray(
            ahead_heights, dtype=np.float64
        )
        if (
            float(np.min(drops)) < lqr.TERRAIN_DROP_MIN_HEIGHT_M
            or float(np.max(drops)) > lqr.TERRAIN_DROP_MAX_HEIGHT_M
            or float(np.ptp(drops)) > lqr.TERRAIN_DROP_FULL_WIDTH_TOLERANCE_M
            or float(np.ptp(support_heights)) > lqr.TERRAIN_DROP_FULL_WIDTH_TOLERANCE_M
            or float(np.ptp(ahead_heights)) > lqr.TERRAIN_DROP_FULL_WIDTH_TOLERANCE_M
        ):
            return None
        return float(np.mean(drops))

    def _maybe_arm_terrain_drop(self) -> None:
        """Hand a verified local down-step to the bounded LQR supervisor."""
        controller = self._controller
        if controller is None:
            return
        if controller.drop.active:
            return
        drop_height = self._local_full_width_drop_ahead_m()
        if drop_height is None:
            self._terrain_drop_latched = False
            return
        if self._terrain_drop_latched:
            return
        if controller.jump.active or controller.jump_pending or self._jump_landing_pending:
            return
        self._cancel_contact_recovery()
        if controller.request_terrain_drop(self.data, drop_height):
            self._terrain_drop_latched = True
            self._terrain_drop_last_detected_height_m = drop_height

    def _terrain_observation(self) -> np.ndarray:
        """Return a local height preview in the current body-heading frame."""
        if self._controller is None:
            return np.zeros(TERRAIN_OBSERVATION_SIZE, dtype=np.float64)
        root_xy = self.data.xpos[self._robot_body, :2]
        support_height, valid = self._terrain_surface_height_and_validity(root_xy)
        heading = self._controller.heading_yaw(self.data)
        forward = np.array((np.cos(heading), np.sin(heading)), dtype=np.float64)
        lateral = np.array((-forward[1], forward[0]), dtype=np.float64)
        patch: list[float] = []
        for distance in TERRAIN_LOOKAHEAD_DISTANCES_M:
            for offset in TERRAIN_LOOKAHEAD_LATERAL_OFFSETS_M:
                height, sample_valid = self._terrain_surface_height_and_validity(
                    root_xy + distance * forward + offset * lateral
                )
                valid = valid and sample_valid
                patch.append(float(np.clip(
                    (height - support_height) / TERRAIN_HEIGHT_NORMALIZATION_M,
                    -2.0,
                    2.0,
                )))
        forward_near, forward_valid = self._terrain_surface_height_and_validity(
            root_xy + 0.20 * forward
        )
        backward_near, backward_valid = self._terrain_surface_height_and_validity(
            root_xy - 0.20 * forward
        )
        left_near, left_valid = self._terrain_surface_height_and_validity(root_xy + 0.16 * lateral)
        right_near, right_valid = self._terrain_surface_height_and_validity(root_xy - 0.16 * lateral)
        valid = valid and forward_valid and backward_valid and left_valid and right_valid
        wheel_heights = np.array(
            [
                self.terrain_surface_height_m(self.data.geom_xpos[geom_id, :2])
                for geom_id in self.refs.wheel_geoms
            ],
            dtype=np.float64,
        )
        forward_grade = (forward_near - backward_near) / (0.40 * TERRAIN_SLOPE_NORMALIZATION)
        lateral_grade = (left_near - right_near) / (0.32 * TERRAIN_SLOPE_NORMALIZATION)
        wheel_support_difference = (wheel_heights[0] - wheel_heights[1]) / TERRAIN_HEIGHT_NORMALIZATION_M
        return np.asarray(
            (
                *patch,
                float(np.clip(forward_grade, -2.0, 2.0)),
                float(np.clip(lateral_grade, -2.0, 2.0)),
                float(np.clip(wheel_support_difference, -2.0, 2.0)),
                1.0 if valid else 0.0,
            ),
            dtype=np.float64,
        )

    def _terrain_wheel_support_state(self) -> tuple[float, bool]:
        """Return wheel support height and terrain-footprint validity without contact force."""
        heights: list[float] = []
        valid = True
        for geom_id in self.refs.wheel_geoms:
            height, wheel_valid = self._terrain_surface_height_and_validity(
                self.data.geom_xpos[geom_id, :2]
            )
            heights.append(height)
            valid = valid and wheel_valid
        return float(np.mean(heights)), bool(valid)

    def _terrain_support_height_m(self) -> float:
        """Average the two local wheel supports for the LQR vertical reference."""
        height, _ = self._terrain_wheel_support_state()
        return height

    def _update_confirmed_terrain_support(self) -> bool:
        """Advance the fall-reference only while both wheels have valid support."""
        support_height, valid = self._terrain_wheel_support_state()
        if valid and all(self._wheel_contacts()):
            self._terrain_confirmed_support_height_m = support_height
        return valid

    def _body_height_reference_m(self) -> float:
        """Return the support-relative body-height floor for a terrain task.

        The physical safety check must follow a verified terrain support when a
        route descends from a platform.  This does not alter the physical root
        pose; it only prevents a legitimate terrain height change from being
        classified as a fall against the spawn-height reference.
        """
        if self._active_terrain_route is None:
            return self._reference_body_height
        return float(
            self._reference_body_height
            + self._terrain_confirmed_support_height_m
            - self._terrain_spawn_surface_height_m
        )

    def _rebase_body_height_after_drop(self) -> None:
        """Rebase fall detection on a validated lower two-wheel support."""
        if self._active_terrain_route is None or self._controller is None:
            return
        contacts = self._wheel_contacts()
        support_height, valid = self._terrain_wheel_support_state()
        if not valid or not all(contacts):
            return
        if abs(support_height - self._terrain_spawn_surface_height_m) <= 0.05:
            return
        self._terrain_confirmed_support_height_m = support_height
        self._reference_body_height = (
            float(self.data.xpos[self._robot_body, 2])
            - support_height
            + self._terrain_spawn_surface_height_m
        )

    def _wheel_ground_clearances_m(self) -> np.ndarray:
        """Return physical left/right wheel-bottom clearance above support terrain."""
        clearances: list[float] = []
        for geom_id in self.refs.wheel_geoms:
            radius = float(self.model.geom_size[geom_id, 0])
            half_width = float(self.model.geom_size[geom_id, 1])
            wheel_axis_z = float(self.data.geom_xmat[geom_id].reshape(3, 3)[2, 2])
            vertical_extent = radius * np.sqrt(max(0.0, 1.0 - wheel_axis_z * wheel_axis_z))
            vertical_extent += half_width * abs(wheel_axis_z)
            wheel_center = self.data.geom_xpos[geom_id]
            ground_height = self.terrain_surface_height_m(wheel_center[:2])
            clearances.append(float(wheel_center[2] - vertical_extent - ground_height))
        return np.maximum(0.0, np.asarray(clearances, dtype=np.float64))

    def _wheel_ground_clearance_m(self) -> float:
        """Return the lower wheel clearance, the hard-safe jump-height metric."""
        return float(np.min(self._wheel_ground_clearances_m()))

    def _ground_has_nonwheel_contact(self) -> bool:
        """Return whether a body/link hit a support or declared obstacle."""
        support_contacts, obstacle_contacts = lqr.nonwheel_static_contact_counts(
            self.data, self.refs
        )
        return bool(support_contacts or obstacle_contacts)

    def _jump_observation(self) -> np.ndarray:
        phase_features = np.zeros(len(JUMP_PHASE_NAMES) + 2, dtype=np.float64)
        if self._jump_scheduled and not (self._jump_succeeded or self._jump_failed):
            phase_features[0] = 1.0
            if not self._jump_triggered and self._jump_at is not None:
                phase_features[1] = float(np.clip(
                    (self._jump_at - self.data.time) / max(self._jump_at, self.model.opt.timestep),
                    0.0,
                    1.0,
                ))
        if self._controller is not None and self._controller.jump.phase_name in JUMP_PHASE_NAMES:
            phase_features[2 + JUMP_PHASE_NAMES.index(self._controller.jump.phase_name)] = 1.0
        return phase_features

    def _jump_height_observation(self) -> np.ndarray:
        """Expose current height and vertical motion for the scheduled jump task."""
        clearances = self._wheel_ground_clearances_m()
        body_rise = max(
            0.0,
            float(self.data.xpos[self._robot_body, 2]) - self._body_height_reference_m(),
        )
        return np.array((
            clearances[0] / JUMP_MAX_REWARD_CLEARANCE_M,
            clearances[1] / JUMP_MAX_REWARD_CLEARANCE_M,
            float(np.mean(clearances)) / JUMP_MAX_REWARD_CLEARANCE_M,
            body_rise / JUMP_MAX_REWARD_CLEARANCE_M,
            float(self.data.qvel[self._root_dof_address + 2]) / 3.0,
        ), dtype=np.float64)

    def _schedule_jump_if_due(self) -> None:
        if (
            self._controller is not None
            and self._jump_at is not None
            and not self._jump_triggered
            and self.data.time >= self._jump_at
        ):
            self._jump_command_request = True
            self._jump_request_latched = True
            self.request_lqr_jump()

    def _jump_landing_is_stable(self) -> bool:
        """Validate the first landed state before declaring a jump successful."""
        if self._controller is None or not all(self._wheel_contacts()):
            return False
        if self._ground_has_nonwheel_contact():
            return False
        attitude_error = float(np.linalg.norm(self._orientation_error()))
        root_dof = self._root_dof_address
        vertical_speed = abs(float(self.data.qvel[root_dof + 2]))
        angular_speed = float(np.linalg.norm(self.data.qvel[root_dof + 3 : root_dof + 6]))
        leg_difference = abs(
            self._sensor("left_leg_length")[0] - self._sensor("right_leg_length")[0]
        )
        return bool(
            attitude_error <= JUMP_STABLE_ATTITUDE_LIMIT_RAD
            and vertical_speed <= JUMP_STABLE_VERTICAL_SPEED_MPS
            and angular_speed <= JUMP_STABLE_ANGULAR_SPEED_RAD_S
            and leg_difference <= lqr.MAX_LEG_LENGTH_DIFFERENCE_M
        )

    def _update_jump_peak_height(self) -> None:
        if self._controller is None or not self._jump_triggered or not self._controller.jump.active:
            return
        airborne = not any(self._wheel_contacts())
        clearances = self._wheel_ground_clearances_m()
        if not airborne:
            return
        self._jump_has_been_airborne = True
        self._jump_peak_clearance_m = max(self._jump_peak_clearance_m, float(np.min(clearances)))
        self._jump_peak_clearance_mean_m = max(
            self._jump_peak_clearance_mean_m,
            float(np.mean(clearances)),
        )
        self._jump_height_target_reached = bool(
            self._jump_peak_clearance_mean_m >= JUMP_TARGET_CLEARANCE_M
            and self._jump_peak_clearance_m >= JUMP_MIN_CLEARANCE_SUCCESS_M
        )
        body_rise = max(
            0.0,
            float(self.data.xpos[self._robot_body, 2]) - self._body_height_reference_m(),
        )
        self._jump_peak_body_rise_m = max(self._jump_peak_body_rise_m, body_rise)

    def _record_jump_landing_guard(self, safety_reason: str | None) -> None:
        if safety_reason is None or self._jump_landing_fall_penalized:
            return
        # A stable but low jump is a task miss, not a physical landing fall.
        # An unstable landing or timeout remains subject to the severe penalty.
        if safety_reason == "jump_aborted_jump_height_target_not_reached":
            return
        before_stable_landing = self._jump_has_been_airborne and not self._jump_succeeded
        within_landing_guard = self._jump_landing_pending
        if before_stable_landing or within_landing_guard:
            self._jump_landing_fall_penalized = True
            self._jump_landing_fall_this_step = True
            self._jump_succeeded = False
            self._jump_landing_stable = False
            self._jump_landing_pending = False
            self._jump_failed = True
            self._jump_failure_reason = f"post-landing fall: {safety_reason}"

    def _record_jump_transition(self, was_active: bool) -> None:
        if self._controller is None or not was_active or self._controller.jump.active:
            return
        abort_reason = self._controller.jump.abort_reason
        if abort_reason:
            self._jump_failed = True
            self._jump_failure_reason = abort_reason
            self._jump_failure_this_step = True
        else:
            if not self._jump_has_been_airborne:
                self._jump_failed = True
                self._jump_failure_reason = "insufficient two-wheel liftoff"
                self._jump_failure_this_step = True
            else:
                self._jump_landing_stable = self._jump_landing_is_stable()
                if self._jump_landing_stable:
                    # Success is confirmed only after the full guard period.
                    # This prevents a single quiet touchdown sample from
                    # earning the landing reward before the body settles.
                    self._jump_landing_pending = True
                    self._jump_landing_guard_until_s = float(self.data.time) + JUMP_LANDING_GUARD_SECONDS
                else:
                    self._jump_failed = True
                    self._jump_failure_reason = "unstable landing"
                    self._jump_failure_this_step = True

    def _update_jump_landing_guard(self) -> None:
        """Confirm stable touchdown continuously before awarding jump success."""
        if not self._jump_landing_pending:
            return
        if not self._jump_landing_is_stable():
            self._jump_landing_pending = False
            self._jump_landing_stable = False
            self._jump_failed = True
            self._jump_failure_reason = "unstable landing during guard"
            self._jump_failure_this_step = True
            self._jump_landing_fall_penalized = True
            self._jump_landing_fall_this_step = True
            return
        if float(self.data.time) < self._jump_landing_guard_until_s:
            return
        self._jump_landing_pending = False
        if self._jump_height_target_reached:
            self._jump_succeeded = True
            self._jump_success_this_step = True
        else:
            self._jump_failed = True
            self._jump_failure_reason = "jump height target not reached"
            self._jump_failure_this_step = True

    def _safety_reason(self) -> str | None:
        if self._active_domain_randomization.enabled:
            # Noise is presented to the controller/policy measurement path,
            # never to physical safety checks or contact geometry validation.
            mujoco.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return "non_finite_state"
        jump_active = self._controller is not None and self._controller.jump.active
        drop_active = self._controller is not None and self._controller.drop.active
        protected_transition = jump_active or self._jump_landing_pending or drop_active
        if protected_transition:
            try:
                lqr.validate_jump_contacts(self.data, self.refs)
            except RuntimeError as error:
                prefix = "terrain_drop" if drop_active else "jump"
                return f"{prefix}_{str(error).lower().replace(' ', '_')}"
        else:
            if self._ground_has_nonwheel_contact():
                return "nonwheel_ground_contact"
            try:
                lqr.validate_linkage_clearance(self.data, self.refs)
            except RuntimeError:
                return "linkage_self_contact"
            try:
                lqr.validate_closed_loop_error(self.data, self.refs)
            except RuntimeError:
                return "closed_linkage_residual"
            try:
                lqr.validate_leg_length_state(self.data, self.refs)
            except RuntimeError:
                return "leg_length_limit"

            contact_reason = self._update_wheel_contact_recovery()
            if contact_reason is not None:
                return contact_reason

        if self._controller is not None and self._controller.drop.abort_reason:
            return "terrain_drop_aborted_" + self._controller.drop.abort_reason.replace(" ", "_")

        if self._jump_failed:
            return f"jump_aborted_{self._jump_failure_reason.replace(' ', '_')}"

        attitude_error = float(np.linalg.norm(self._orientation_error()))
        if attitude_error > MAX_ATTITUDE_ERROR_RAD:
            return "attitude_limit"
        body_height = float(self.data.xpos[self._robot_body, 2])
        if body_height < self._body_height_reference_m() - 0.22:
            return "body_height_limit"
        return None

    def _observation(self) -> np.ndarray:
        if self._stance_leg_lengths is None:
            raise RuntimeError("low-centre stance has not been projected")
        world_velocity = self._sensor("world_horizontal_velocity_xy")
        body_angular_velocity = self._sensor("body_angular_velocity")
        wheel_velocity = np.array(
            (
                self._sensor("left_wheel_angular_velocity")[0],
                self._sensor("right_wheel_angular_velocity")[0],
            )
        )
        leg_lengths = np.array(
            (
                self._sensor("left_leg_length")[0],
                self._sensor("right_leg_length")[0],
            )
        )
        leg_length_velocity = np.array(
            (
                self._sensor("left_leg_length_velocity")[0],
                self._sensor("right_leg_length_velocity")[0],
            )
        )
        yaw_state = self.yaw_state()
        contacts = np.asarray(self._wheel_contacts(), dtype=np.float64)
        observation = np.concatenate(
            (
                (self._orientation_error() + self._orientation_measurement_noise)
                / MAX_ATTITUDE_ERROR_RAD,
                world_velocity / 2.0,
                body_angular_velocity / 10.0,
                self.data.qpos[self._hip_qpos_addresses] / 2.5,
                self.data.qvel[self._hip_dof_addresses] / 15.0,
                wheel_velocity / 50.0,
                (leg_lengths - self._stance_leg_lengths) / 0.15,
                leg_length_velocity / 3.0,
                np.array((self._command_speed / self.command_speed_limit_mps,)),
                np.array(((self._command_leg_length - self._default_leg_length) / 0.10,)),
                np.array((
                    0.0
                    if self.max_command_yaw_rate_rad_s <= 0.0
                    else self._command_yaw_rate_rad_s / self.max_command_yaw_rate_rad_s,
                )),
                np.array((float(self._jump_command_request),)),
                np.array((
                    yaw_state["yaw_error_rad"] / np.pi,
                    yaw_state["pending_yaw_error_rad"] / np.pi,
                    yaw_state["yaw_rate_normalized"],
                )),
                self._jump_observation(),
                self._jump_height_observation(),
                self._terrain_observation(),
                contacts,
                self._previous_action,
            )
        )
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(observation, -10.0, 10.0).astype(np.float32)

    def _forward_speed(self) -> float:
        if self._controller is None:
            return 0.0
        return self._controller.forward_speed(self.data)

    def _speed_status(self, forward_speed: float, ramped_speed: float, *, jump_active: bool) -> str:
        """Report high-level tracking while retaining the internal LQR ramp state."""
        lqr_target = self._lqr_speed_reference_for_command(self._command_speed)
        if jump_active:
            return "JUMP"
        if abs(lqr_target - ramped_speed) > lqr.speed_tracking_tolerance(lqr_target):
            return "RAMPING"
        return lqr.speed_tracking_status(
            self._command_speed,
            self._command_speed,
            forward_speed,
        )

    @property
    def lqr_controller(self) -> PhysicalLqr:
        """Expose the active LQR instance for environment-side heading control."""
        if self._controller is None:
            raise RuntimeError("call reset before accessing the LQR controller")
        return self._controller

    @property
    def heading_command(self) -> HeadingCommand:
        """Expose the LQR's command_yaw/reference_yaw state."""
        return self.lqr_controller.heading_command

    @property
    def command_yaw(self) -> float:
        return self.yaw_state()["command_yaw_rad"]

    @property
    def command_speed(self) -> float:
        """Current high-level forward-speed request before LQR rate limiting."""
        return float(self._command_speed)

    @property
    def locomotion_command(self) -> LocomotionCommand:
        """Return the current speed/yaw-rate/jump command seen by the policy."""
        return LocomotionCommand(
            forward_speed_mps=float(self._command_speed),
            yaw_rate_rad_s=float(self._command_yaw_rate_rad_s),
            jump_request=bool(self._jump_command_request),
        )

    @property
    def reference_yaw(self) -> float:
        return self.yaw_state()["reference_yaw_rad"]

    @property
    def true_yaw(self) -> float:
        return self.yaw_state()["true_yaw_rad"]

    @property
    def yaw_error(self) -> float:
        return self.yaw_state()["yaw_error_rad"]

    def set_command_yaw(self, yaw: float) -> float:
        """Set the LQR world-frame heading command from the environment."""
        target = wrap_to_pi(float(yaw))
        if self._contact_recovery_active:
            self._contact_recovery_saved_yaw = target
            return target
        return self.lqr_controller.set_command_yaw(target)

    def set_command_speed(self, speed: float) -> float:
        """Set the high-level speed request without interrupting a jump gate."""
        target = self._clamp_command_speed(speed)
        if self._contact_recovery_active:
            self._command_speed = target
            self._contact_recovery_saved_speed = target
            return target
        controller = self.lqr_controller
        self._command_speed = target
        controller.set_jump_resume_speed(self._lqr_speed_reference_for_command(target))
        self._apply_command_conditioning()
        self._sync_terrain_yaw_rate_command()
        return self._command_speed

    def set_locomotion_command(self, command: LocomotionCommand) -> LocomotionCommand:
        """Set the high-level command without exposing actuator-level control.

        A true ``jump_request`` is accepted only on its rising edge.  The LQR
        then handles the low-speed launch gate and later restores the requested
        forward speed, while this method continues to carry the yaw-rate task.
        """
        if not isinstance(command, LocomotionCommand):
            raise TypeError("command must be a LocomotionCommand")
        speed = self.set_command_speed(command.forward_speed_mps)
        yaw_rate = self._clamp_command_yaw_rate(command.yaw_rate_rad_s)
        jump_request = bool(command.jump_request)
        rising_edge = jump_request and not self._jump_request_latched
        self._command_yaw_rate_rad_s = yaw_rate
        self._jump_command_request = jump_request
        if not jump_request:
            self._jump_request_latched = False
        elif rising_edge:
            self._jump_request_latched = True
            # A generic high-level command has no terrain-specific launch
            # trajectory.  Do not map its walking request directly to the
            # launch reference: that creates asymmetric landing impulses under
            # vehicle randomization.  The calibrated LQR reference remains a
            # controlled rolling launch on RMUC, while ``speed`` is retained
            # as the post-landing resume command by set_jump_resume_speed().
            self.request_lqr_jump()
        self._apply_command_conditioning()
        self._sync_terrain_yaw_rate_command()
        return LocomotionCommand(speed, yaw_rate, jump_request)

    def _advance_locomotion_command(self) -> None:
        """Integrate the high-level yaw-rate command at policy-rate boundaries."""
        if (
            self._controller is None
            or self._controller.jump.active
            or self._controller.jump_pending
            or self._controller.drop.active
            or self._jump_landing_pending
            or self._contact_recovery_active
        ):
            return
        elapsed = self.control_decimation * self.model.opt.timestep
        if self._uses_high_speed_command_conditioning():
            maximum_change = TERRAIN_HIGH_SPEED_YAW_ACCELERATION_RAD_S2 * elapsed
            self._conditioned_yaw_rate_rad_s += float(np.clip(
                self._command_yaw_rate_rad_s - self._conditioned_yaw_rate_rad_s,
                -maximum_change,
                maximum_change,
            ))
        else:
            self._conditioned_yaw_rate_rad_s = self._command_yaw_rate_rad_s
        self._sync_terrain_yaw_rate_command()
        if abs(self._conditioned_yaw_rate_rad_s) <= 1e-12:
            return
        self.set_command_yaw(
            self._controller.command_yaw + self._conditioned_yaw_rate_rad_s * elapsed
        )

    def _maybe_resample_locomotion_command(self) -> None:
        """Draw the next high-level training command when the controller is free.

        Commands are held for a finite interval so PPO observes a genuine
        command-following problem rather than a reset-only target.  A jump keeps
        its speed/yaw request frozen from launch through landing recovery.
        """
        if (
            self._controller is None
            or self._active_terrain_task is not None
            or not self.randomize_command
            or self.command_resample_seconds is None
            or not np.isfinite(self._next_command_resample_s)
            or float(self.data.time) + 1e-12 < self._next_command_resample_s
        ):
            return
        jump_in_progress = self._jump_scheduled and not (
            self._jump_succeeded or self._jump_failed
        )
        if (
            jump_in_progress
            or self._controller.jump.active
            or self._controller.jump_pending
            or self._controller.drop.active
            or self._jump_landing_pending
            or self._contact_recovery_active
        ):
            return
        self.set_locomotion_command(
            LocomotionCommand(
                forward_speed_mps=self.sample_command_speed(self.np_random),
                yaw_rate_rad_s=self.sample_command_yaw_rate(self.np_random),
                jump_request=False,
            )
        )
        while self._next_command_resample_s <= float(self.data.time) + 1e-12:
            self._next_command_resample_s += float(self.command_resample_seconds)

    def adjust_command_speed(self, delta: float) -> float:
        """Increment the manual LQR speed command and keep telemetry in sync."""
        return self.set_command_speed(self._command_speed + float(delta))

    def adjust_command_yaw(self, delta: float) -> float:
        """Increment the manual world-frame heading command."""
        source_yaw = (
            self._contact_recovery_saved_yaw
            if self._contact_recovery_active
            else self.lqr_controller.command_yaw
        )
        return self.set_command_yaw(source_yaw + float(delta))

    def hold_current_yaw(self) -> float:
        """Make the current measured heading the manual LQR heading target."""
        self._command_yaw_rate_rad_s = 0.0
        return self.set_command_yaw(self.lqr_controller.heading_yaw(self.data))

    def adjust_command_leg_length(self, delta: float) -> float:
        """Increment the common LQR leg-length command and update telemetry."""
        if self._contact_recovery_active:
            lower, upper = self.lqr_controller.leg_length_limits()
            target = float(np.clip(
                self._contact_recovery_saved_leg_length + float(delta), lower, upper,
            ))
            self._command_leg_length = target
            self._contact_recovery_saved_leg_length = target
            return target
        target = self.lqr_controller.adjust_target_leg_length(delta)
        self._command_leg_length = float(target)
        return target

    def request_lqr_jump(self, *, launch_speed_mps: float | None = None) -> bool:
        """Queue an operator-requested LQR jump with environment safety tracking."""
        controller = self.lqr_controller
        if controller.jump.active or controller.jump_pending or controller.drop.active:
            self._jump_command_request = True
            self._jump_request_latched = True
            return True
        self._cancel_contact_recovery()
        controller.request_jump(self.data, launch_speed_mps=launch_speed_mps)
        # Keep the high-level travel request for post-landing recovery and
        # truthful telemetry.  The LQR independently regulates its temporary
        # low-speed launch reference while the request is pending.
        self._command_leg_length = float(controller.leg_command.target_length)
        self._jump_scheduled = True
        self._jump_at = None
        self._jump_triggered = True
        self._jump_command_request = True
        self._jump_request_latched = True
        self._jump_succeeded = False
        self._jump_landing_stable = False
        self._jump_landing_pending = False
        self._jump_failed = False
        self._jump_failure_reason = ""
        self._jump_success_this_step = False
        self._jump_failure_this_step = False
        self._jump_landing_fall_this_step = False
        self._jump_landing_fall_penalized = False
        self._jump_has_been_airborne = False
        self._jump_peak_body_rise_m = 0.0
        self._jump_peak_clearance_m = 0.0
        self._jump_peak_clearance_mean_m = 0.0
        self._jump_rewarded_peak_clearance_mean_m = 0.0
        self._jump_height_target_reached = False
        self._jump_landing_guard_until_s = -np.inf
        return True

    def resume_after_nonphysical_jump_task_miss(self) -> bool:
        """Allow an interactive viewer to continue after a stable low jump.

        Training keeps the task miss terminal so PPO can learn the height
        objective.  A manual RMUC session must not freeze a physically stable
        robot merely because it did not reach the configured training height.
        """
        if (
            self._controller is None
            or self._controller.jump.active
            or self._jump_failure_reason != "jump height target not reached"
        ):
            return False
        self._jump_failed = False
        self._jump_failure_reason = ""
        self._jump_failure_this_step = False
        self._jump_scheduled = False
        self._jump_at = None
        self._jump_triggered = False
        self._jump_command_request = False
        self._jump_request_latched = False
        return True

    def yaw_state(self) -> dict[str, float]:
        """Return the LQR heading variables without changing the policy interface."""
        if self._controller is None:
            return {
                "command_yaw_rad": 0.0,
                "reference_yaw_rad": 0.0,
                "true_yaw_rad": 0.0,
                "yaw_error_rad": 0.0,
                "pending_yaw_error_rad": 0.0,
                "yaw_rate_rad_s": 0.0,
                "target_yaw_rate_rad_s": 0.0,
                "yaw_rate_normalized": 0.0,
                "turn_intensity": 0.0,
            }
        heading_command = self.heading_command
        true_yaw = self._controller.heading_yaw(self.data)
        command_yaw = float(heading_command.command_yaw)
        reference_yaw = float(heading_command.reference_yaw)
        yaw_error = wrap_to_pi(reference_yaw - true_yaw)
        pending_yaw_error = wrap_to_pi(command_yaw - reference_yaw)
        target_yaw_rate = self._controller.yaw_rate_target(self.data)
        measured_yaw_rate = float(self._controller._measured_yaw_rate)
        measured_yaw_rate_normalized = float(np.clip(
            measured_yaw_rate / lqr.MAX_YAW_RATE_RAD_S,
            -1.0,
            1.0,
        ))
        command_turn_intensity = (
            0.0
            if self.max_command_yaw_rate_rad_s <= 0.0
            else abs(self._command_yaw_rate_rad_s) / self.max_command_yaw_rate_rad_s
        )
        turn_intensity = float(np.clip(max(
            abs(target_yaw_rate) / lqr.MAX_YAW_RATE_RAD_S,
            abs(measured_yaw_rate_normalized),
            abs(pending_yaw_error) / np.pi,
            command_turn_intensity,
        ), 0.0, 1.0))
        return {
            "command_yaw_rad": command_yaw,
            "reference_yaw_rad": reference_yaw,
            "true_yaw_rad": true_yaw,
            "yaw_error_rad": yaw_error,
            "pending_yaw_error_rad": pending_yaw_error,
            "yaw_rate_rad_s": measured_yaw_rate,
            "target_yaw_rate_rad_s": target_yaw_rate,
            "yaw_rate_normalized": measured_yaw_rate_normalized,
            "turn_intensity": turn_intensity,
        }

#速度跟踪作为最大权重，紧跟陆地——腾空腿长跟踪和位姿跟踪，加入偏航跟踪，能量和地接作为小权重，为跳跃状态机设计两套奖励模式
    def _reward(
        self,
        control: np.ndarray,
        action: np.ndarray,
        unsafe: bool,
        *,
        task_completed_safely: bool,
    ) -> float:
        forward_speed = self._forward_speed()
        ramped_speed = float(self._controller.motion.current_speed)
        # Terrain tasks are explicit high-level locomotion commands.  Their
        # completion gate already compares against ``_command_speed``; use the
        # same target for the dense policy reward so PPO cannot optimize the
        # LQR's internal acceleration ramp while missing the requested speed.
        tracking_speed = (
            self._command_speed
            if self._active_terrain_task is not None
            else ramped_speed
        )
        speed_error = forward_speed - tracking_speed
        jump_active = self._controller.jump.active
        yaw_state = self.yaw_state()
        attitude_cost = float(np.dot(self._orientation_error(), self._orientation_error()))
        self.leg_diff = abs(
            self._sensor("left_leg_length")[0] - self._sensor("right_leg_length")[0]
        )
        jump_peak_increment = 0.0
        if jump_active:
            # The LQR keeps the wheel/leg trajectory authority during a jump.
            # Phase-gated hip residuals are still regularized through the
            # effective action passed to this function.
            # 防止reward hacking
            tracking = 0.0
            leg_tracking = 0.0
            attitude_tracking = 0.0
            yaw_tracking = 0.0
            yaw_rate_error = 0.0
            energy_cost = 0.0
            residual_cost = float(np.mean(action * action))
            contact_bonus = 0.0
            # A new peak is rewarded once and capped at 25 cm.  This promotes
            # useful clearance rather than accumulating reward by hovering.
            leg_diff_weight = LEG_DIFF_TURN_WEIGHT
            capped_peak = min(self._jump_peak_clearance_mean_m, JUMP_MAX_REWARD_CLEARANCE_M)
            new_peak = max(0.0, capped_peak - self._jump_rewarded_peak_clearance_mean_m)
            if new_peak > 0.0:
                self._jump_rewarded_peak_clearance_mean_m = capped_peak
            jump_peak_increment = new_peak / JUMP_PEAK_CLEARANCE_SCALE_M
        else:
            tracking = float(np.exp(-((speed_error / 0.20) ** 2)))
            leg_error = float(np.mean((
                self._sensor("left_leg_length")[0],
                self._sensor("right_leg_length")[0],
            )) - self._command_leg_length)
            leg_tracking = float(np.exp(-((leg_error / 0.05) ** 2)))
            attitude_tracking = float(np.exp(-attitude_cost))
            yaw_error = yaw_state["yaw_error_rad"]
            yaw_tracking = float(np.exp(-((yaw_error / 0.10) ** 2)))
            yaw_rate_error = float(np.clip(
                (yaw_state["yaw_rate_rad_s"] - yaw_state["target_yaw_rate_rad_s"])
                / lqr.MAX_YAW_RATE_RAD_S,
                -1.0,
                1.0,
            ))
            energy_cost = float(np.mean((control / self._control_scale) ** 2))
            residual_cost = float(np.mean(action * action))
            contact_bonus = 0.10 if all(self._wheel_contacts()) else -0.15
            turn_intensity = yaw_state["turn_intensity"]
            leg_diff_weight = LEG_DIFF_TURN_WEIGHT + (
                LEG_DIFF_STRAIGHT_WEIGHT - LEG_DIFF_TURN_WEIGHT
            ) * (1.0 - turn_intensity)
        reward = (
            0.65 * tracking
            + 0.30 * leg_tracking
            + 0.30 * attitude_tracking
            + contact_bonus
            + 0.25 * yaw_tracking
            + JUMP_HEIGHT_REWARD_WEIGHT * jump_peak_increment
        )
        reward -= (
            0.03 * energy_cost
            + 0.02 * residual_cost
            + YAW_RATE_TRACKING_WEIGHT * yaw_rate_error * yaw_rate_error
            + leg_diff_weight * self.leg_diff
        )
        if self._active_terrain_task is not None:
            # The base locomotion terms were originally expressed per 10 ms
            # policy step.  On a 14--40 s terrain task that rewards waiting
            # for thousands of steps more than finishing.  Treat them as a
            # physical reward rate instead; all sparse jump/success events
            # below intentionally remain unscaled.
            reward *= (
                TERRAIN_DENSE_REWARD_RATE_SCALE
                * self._last_control_duration_s
            )
        if self._jump_success_this_step:
            reward += JUMP_SUCCESS_REWARD
        if self._jump_failure_this_step:
            reward -= JUMP_ABORT_PENALTY
        if self._jump_landing_fall_this_step:
            reward -= JUMP_LANDING_FALL_PENALTY
        if self._active_terrain_task is not None:
            # Route geometry remains evaluation metadata.  It does not shape
            # the policy toward an unobserved waypoint or corridor during
            # training.  Reaching the route completion threshold is a
            # terminal task outcome in both training and evaluation.
            if self._terrain_completion_this_step and task_completed_safely:
                reward += TERRAIN_COMPLETION_REWARD
            if self._terrain_task_failed:
                reward -= TERRAIN_TASK_TIMEOUT_PENALTY
        if unsafe:
            reward -= 30.0
        self._jump_success_this_step = False
        self._jump_failure_this_step = False
        self._jump_landing_fall_this_step = False
        self._terrain_completion_this_step = False
        self._terrain_progress_delta_this_step_m = 0.0
        return float(reward)
#速度跟随最高权重，腿长误差和关节位姿次权重 额外加入双腿相对误差 跳跃时给予各项指标相应缓冲带宽
    def _info(self, safety_reason: str | None = None) -> dict[str, Any]:
        forward_speed = self._forward_speed()
        yaw_state = self.yaw_state()
        ramped_speed = 0.0 if self._controller is None else float(self._controller.motion.current_speed)
        jump_active = self._controller is not None and self._controller.jump.active
        jump_pending = self._controller is not None and self._controller.jump_pending
        jump_phase = None if self._controller is None else self._controller.jump.phase_name
        jump_elapsed = 0.0 if self._controller is None else self._controller.jump.elapsed(float(self.data.time))
        drop_active = self._controller is not None and self._controller.drop.active
        drop_phase = None if self._controller is None else self._controller.drop.phase_name
        drop_elapsed = (
            0.0
            if self._controller is None
            else self._controller.drop.elapsed(float(self.data.time))
        )
        drop_height = (
            self._terrain_drop_last_detected_height_m
            if self._controller is None or not self._controller.drop.active
            else self._controller.drop.drop_height_m
        )
        jump_brake_elapsed = (
            0.0
            if self._controller is None
            else self._controller.jump_pending_elapsed(float(self.data.time))
        )
        locomotion_command_deferred = bool(
            jump_pending
            or jump_active
            or drop_active
            or self._jump_landing_pending
            or self._contact_recovery_active
        )
        speed_status = (
            "JUMP_BRAKING"
            if jump_pending
            else "TERRAIN_DROP"
            if drop_active
            else self._speed_status(
                forward_speed,
                ramped_speed,
                jump_active=jump_active,
            )
        )
        terrain_task = self._active_terrain_task
        terrain_route = self._active_terrain_route
        terrain_support_height = (
            self._terrain_support_height_m() if terrain_route is not None else 0.0
        )
        return {
            "locomotion_command_schema": LOCOMOTION_COMMAND_SCHEMA,
            "residual_authority_schema": RESIDUAL_AUTHORITY_SCHEMA,
            "command_speed_mps": self._command_speed,
            "lqr_nominal_speed_reference_mps": self._lqr_speed_reference_for_command(
                self._command_speed
            ),
            "lqr_speed_reference_scale": self._lqr_speed_reference_scale,
            "command_speed_limit_mps": self.command_speed_limit_mps,
            "command_resample_seconds": self.command_resample_seconds,
            "command_yaw_rate_rad_s": self._command_yaw_rate_rad_s,
            "conditioned_yaw_rate_rad_s": self._conditioned_yaw_rate_rad_s,
            "locomotion_command_conditioning": locomotion_command_conditioning_config(),
            "command_speed_deferred": locomotion_command_deferred,
            "command_yaw_rate_deferred": locomotion_command_deferred,
            "jump_request": self._jump_command_request,
            "ramped_command_speed_mps": ramped_speed,
            "forward_speed_mps": forward_speed,
            "speed_error_mps": forward_speed - self._command_speed,
            "speed_status": speed_status,
            **yaw_state,
            "jump_scheduled": self._jump_scheduled,
            "jump_at_s": self._jump_at,
            "jump_triggered": self._jump_triggered,
            "jump_pending": jump_pending,
            "jump_brake_elapsed_s": jump_brake_elapsed,
            "jump_active": jump_active,
            "jump_phase": jump_phase,
            "jump_elapsed_s": jump_elapsed,
            "jump_impact_active": (
                False if self._controller is None else self._controller.jump.impact_active
            ),
            "jump_impact_elapsed_s": (
                0.0
                if self._controller is None
                or not self._controller.jump.impact_active
                or self._controller.jump.impact_start_time is None
                else max(0.0, float(self.data.time) - self._controller.jump.impact_start_time)
            ),
            "jump_impact_first_contact": (
                np.zeros(2, dtype=np.int32)
                if self._controller is None
                else np.asarray(self._controller.jump.impact_first_contact, dtype=np.int32)
            ),
            "jump_impact_max_leg_difference_m": (
                0.0
                if self._controller is None
                else self._controller.jump.impact_max_leg_difference_m
            ),
            "jump_succeeded": self._jump_succeeded,
            "jump_landing_stable": self._jump_landing_stable,
            "jump_landing_pending": self._jump_landing_pending,
            "jump_failed": self._jump_failed,
            "jump_abort_reason": self._jump_failure_reason,
            "jump_has_been_airborne": self._jump_has_been_airborne,
            "jump_landing_fall_penalized": self._jump_landing_fall_penalized,
            "wheel_ground_clearances_m": self._wheel_ground_clearances_m().astype(np.float32),
            "wheel_ground_clearance_m": self._wheel_ground_clearance_m(),
            "jump_peak_body_rise_m": self._jump_peak_body_rise_m,
            "jump_peak_wheel_clearance_m": self._jump_peak_clearance_m,
            "jump_peak_mean_wheel_clearance_m": self._jump_peak_clearance_mean_m,
            "jump_target_clearance_m": JUMP_TARGET_CLEARANCE_M,
            "jump_max_reward_clearance_m": JUMP_MAX_REWARD_CLEARANCE_M,
            "jump_min_clearance_success_m": JUMP_MIN_CLEARANCE_SUCCESS_M,
            "jump_height_target_reached": self._jump_height_target_reached,
            "jump_landing_guard_remaining_s": max(
                0.0, self._jump_landing_guard_until_s - float(self.data.time)
            ),
            "jump_observation": self._jump_observation().astype(np.float32),
            "terrain_drop_active": drop_active,
            "terrain_drop_phase": drop_phase,
            "terrain_drop_elapsed_s": drop_elapsed,
            "terrain_drop_detected_height_m": drop_height,
            "terrain_drop_latched": self._terrain_drop_latched,
            "terrain_drop_abort_reason": (
                "" if self._controller is None else self._controller.drop.abort_reason
            ),
            "terrain_drop_config": lqr.terrain_controller_config(),
            "terrain_task_id": None if terrain_task is None else terrain_task.task_id,
            "terrain_route_id": None if terrain_route is None else terrain_route.route_id,
            "terrain_stage_id": self.terrain_stage_id,
            "terrain_evaluation": self.terrain_evaluation,
            "terrain_heading_stabilization": (
                False
                if self._controller is None
                else self._controller.terrain_heading_stabilization_enabled
            ),
            "terrain_direct_yaw_rate_tracking": (
                False
                if self._controller is None
                else self._controller.terrain_yaw_rate_command_active
            ),
            "terrain_progress_m": self._terrain_progress_m,
            "terrain_max_progress_m": self._terrain_max_progress_m,
            "terrain_required_distance_m": (
                0.0 if terrain_task is None else terrain_task.required_distance_m
            ),
            "terrain_completion_tolerance_m": (
                0.0 if terrain_task is None else terrain_task.completion_tolerance_m
            ),
            "terrain_lateral_error_m": self._terrain_lateral_error_m,
            "terrain_corridor_half_width_m": (
                0.0 if terrain_route is None else terrain_route.corridor_half_width_m
            ),
            "terrain_task_deadline_s": self._terrain_task_deadline_s,
            "terrain_task_completed": self._terrain_task_completed,
            "terrain_task_failed": self._terrain_task_failed,
            "terrain_feature_crossed": self._terrain_task_completed,
            "terrain_completion_mode": (
                None if terrain_task is None else terrain_task.completion_mode
            ),
            "terrain_command_tracking_hold_s": self._terrain_command_tracking_hold_s,
            "terrain_command_tracking_ready": self._terrain_command_tracking_ready,
            "terrain_command_tracking_required_hold_s": (
                0.0
                if terrain_task is None or terrain_task.command_tracking_hold_seconds is None
                else terrain_task.command_tracking_hold_seconds
            ),
            "terrain_speed_tracking_tolerance_mps": (
                0.0
                if terrain_task is None or terrain_task.speed_tracking_tolerance_mps is None
                else terrain_task.speed_tracking_tolerance_mps
            ),
            "terrain_yaw_rate_tracking_tolerance_rad_s": (
                0.0
                if terrain_task is None or terrain_task.yaw_rate_tracking_tolerance_rad_s is None
                else terrain_task.yaw_rate_tracking_tolerance_rad_s
            ),
            "terrain_jump_trigger_progress_m": (
                None if terrain_task is None else terrain_task.jump_trigger_progress_m
            ),
            "terrain_progress_jump_triggered": self._terrain_progress_jump_triggered,
            "terrain_support_height_m": terrain_support_height,
            "terrain_confirmed_support_height_m": self._terrain_confirmed_support_height_m,
            "terrain_observation": self._terrain_observation().astype(np.float32),
            "domain_randomization": dict(self._domain_randomization_sample),
            "domain_randomization_profile": self._active_domain_randomization_profile,
            "control_delay_steps": self._control_delay_steps,
            "control_delay_ms": 1000.0 * self._control_delay_steps * self.model.opt.timestep,
            "sensor_noise_scale": self._sensor_noise_scale,
            "speed_limit_mps": self.max_forward_speed,
            "reverse_speed_limit_mps": self.max_reverse_speed,
            "command_leg_length_m": self._command_leg_length,
            "leg_lengths_m": np.array(
                (self._sensor("left_leg_length")[0], self._sensor("right_leg_length")[0]),
                dtype=np.float32,
            ),
            "wheel_contacts": self._wheel_contacts(),
            "wheel_contact_loss_steps": self._wheel_contact_loss_steps.astype(np.int32).copy(),
            "contact_loss_duration_s": self._contact_loss_duration_s(),
            "contact_recovery_active": self._contact_recovery_active,
            "contact_recovery_duration_s": self._contact_recovery_duration_s(),
            "contact_recovery_count": self._contact_recovery_count,
            "leg_diff": abs(self._sensor("left_leg_length")[0] - self._sensor("right_leg_length")[0]),
            "lqr_control": self._last_lqr_control.astype(np.float32).copy(),
            "requested_control": self._last_requested_control.astype(np.float32).copy(),
            "applied_control": self._last_control.astype(np.float32).copy(),
            "residual_limits": self.residual_limits.astype(np.float32).copy(),
            "safety_reason": safety_reason,
        }

    def expert_action(self, *, residual: bool = True) -> np.ndarray:
        """Return the safe residual action that preserves the LQR baseline."""
        if self._controller is None:
            raise RuntimeError("call reset before requesting an expert action")
        if not residual:
            raise ValueError(
                "WheelLegResidualEnv only accepts residual actions; "
                "absolute LQR torques cannot be used as policy actions"
            )
        return np.zeros(self.action_space.shape, dtype=np.float32)

    def _policy_action_authority(self) -> tuple[np.ndarray, np.ndarray]:
        """Return binary action authority and physical residual scales.

        The policy action layout follows the MJCF actuator layout plus the
        common leg-length rate.  During contact-critical jump phases, only
        small per-hip residuals are admitted.  Wheels and the common leg
        command stay owned by the validated LQR jump trajectory.
        """
        mask = np.zeros(self.action_space.shape, dtype=np.float64)
        scale = np.zeros(self.action_space.shape, dtype=np.float64)
        if self._controller is None:
            return mask, scale
        if (
            self._contact_recovery_active
            or self._controller.jump_pending
            or self._controller.drop.active
            or self._jump_landing_pending
        ):
            return mask, scale
        phase = self._controller.jump.phase_name
        if phase is None:
            mask[:] = 1.0
            scale[:] = 1.0
            return mask, scale
        if phase in ("prepare", "crouch"):
            hip_scale = JUMP_PREPARE_CROUCH_RESIDUAL_SCALE
        elif phase == "thrust":
            hip_scale = JUMP_THRUST_RESIDUAL_SCALE
        elif phase == "flight":
            if self._controller.jump.impact_active:
                return mask, scale
            hip_scale = JUMP_FLIGHT_RESIDUAL_SCALE
        elif phase == "landing":
            hip_scale = JUMP_LANDING_RESIDUAL_SCALE
        else:
            return mask, scale
        hip_actuators = tuple(self._controller.hip_actuator_ids)
        mask[list(hip_actuators)] = 1.0
        scale[list(hip_actuators)] = hip_scale
        return mask, scale

    def policy_action_mask(self) -> np.ndarray:
        """Return the current 7-D binary residual-action authority mask."""
        mask, _ = self._policy_action_authority()
        return mask.astype(np.float32)

    def _policy_action_authority_phase(self) -> str:
        """Name the supervisor that owns the residual channels this tick."""
        if self._controller is None:
            return "uninitialized"
        if self._contact_recovery_active:
            return "contact_recovery"
        if self._controller.jump_pending:
            return "jump_pending"
        if self._controller.jump.impact_active:
            return "jump_impact"
        if self._controller.drop.active:
            return "terrain_drop_" + str(self._controller.drop.phase_name)
        if self._jump_landing_pending:
            return "landing_guard"
        return self._controller.jump.phase_name or "walking"

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._controller is None:
            raise RuntimeError("call reset before step")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action_array.shape}")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action contains non-finite values")
        action_array = np.clip(action_array, -1.0, 1.0)
        self._jump_success_this_step = False
        self._jump_failure_this_step = False
        self._jump_landing_fall_this_step = False
        self._maybe_resample_locomotion_command()
        self._advance_locomotion_command()
        self._schedule_jump_if_due()
        self._sample_sensor_noise()
        effective_action_sum = np.zeros(self.action_space.shape, dtype=np.float64)
        applied_action_mask = np.zeros(self.action_space.shape, dtype=np.float64)
        applied_action_scale = np.zeros(self.action_space.shape, dtype=np.float64)
        authority_phases: set[str] = set()
        physics_steps = 0
        terrain_progress_before_step = self._terrain_progress_m

        safety_reason: str | None = None
        for _ in range(self.control_decimation):
            if (
                self._active_terrain_route is not None
                and self._controller.terrain_support_reference_enabled
            ):
                # Feed the true local support to the LQR before it builds the
                # next command.  The controller filters and rate-limits its
                # root-Z reference; this never injects a physical qpos change.
                self._controller.update_terrain_support_reference(
                    self._terrain_support_height_m()
                )
            self._maybe_arm_terrain_drop()
            # The leg command is part of the reference consumed by LQR on
            # this physics tick, so take its authority before calling command.
            reference_mask, reference_scale = self._policy_action_authority()
            reference_phase = self._policy_action_authority_phase()
            leg_effective_action = 0.0
            if reference_mask[-1] <= 0.0:
                if (
                    self._controller.jump.active
                    or self._controller.drop.active
                    or self._jump_landing_pending
                ):
                    # Protected transitions own their wider, physically
                    # validated leg range. Do not re-clamp it through normal
                    # walking limits.
                    self._command_leg_length = float(self._controller.leg_command.target_length)
                else:
                    # Keep external telemetry on the requested command while
                    # contact recovery temporarily holds a safer target.
                    self._command_leg_length = self._contact_recovery_saved_leg_length
            else:
                leg_effective_action = float(action_array[-1] * reference_scale[-1])
                self._command_leg_length = self._controller.adjust_target_leg_length(
                    leg_effective_action
                    * RL_LEG_LENGTH_COMMAND_RATE_MPS
                    * self.model.opt.timestep
                )
            was_jump_active = self._controller.jump.active
            was_jump_pending = self._controller.jump_pending
            was_drop_active = self._controller.drop.active
            recovery_active = self._contact_recovery_active
            lqr_owns_control = (
                was_jump_active
                or was_jump_pending
                or was_drop_active
                or self._jump_landing_pending
                or recovery_active
            )
            if was_jump_active or was_jump_pending or was_drop_active or recovery_active:
                # Keep sensor noise in the policy observation, but do not
                # corrupt the LQR jump/recovery state estimate with an
                # unobservable disturbance.  The phase mask above still
                # permits the explicitly bounded hip residual channels.
                mujoco.mj_forward(self.model, self.data)
                lqr_control = self._controller.command(self.data)
                true_qpos = true_qvel = None
            else:
                true_qpos, true_qvel = self._prepare_noisy_controller_measurement()
                # The FIFO holds controls that will be applied before this
                # newly requested command.  Predict through that queue so the
                # LQR sees the state at its actual actuation time.  The queue
                # is copied before _apply_control_delay appends this request.
                if self._control_delay_bypassed and self._control_delay_steps > 0:
                    # _apply_control_delay will rebuild this same queue below
                    # before returning the first delayed post-bypass control.
                    # Predict it here as well so the handoff has no one-tick
                    # uncompensated control gap.
                    delayed_controls = tuple(
                        self._last_control.copy()
                        for _ in range(self._control_delay_steps)
                    )
                else:
                    delayed_controls = tuple(
                        control.copy() for control in self._control_delay_buffer
                    )
                lqr_control = (
                    self._controller.command_with_delay_prediction(
                        self.data,
                        delayed_controls,
                    )
                    if delayed_controls
                    else self._controller.command(self.data)
                )
            current_jump_phase = self._controller.jump.phase_name
            current_drop_phase = self._controller.drop.phase_name
            # command() may promote a pending jump or cross a phase boundary.
            # Re-evaluate torque authority after that transition so an action
            # never leaks from thrust into flight or recovery.
            jump_finished_this_tick = was_jump_active and current_jump_phase is None
            if jump_finished_this_tick:
                # The controller has just completed landing recovery, but the
                # environment records the post-landing guard after mj_step.
                # Keep this handoff tick LQR-owned so a landing-phase action
                # cannot briefly regain all six residual torque channels.
                torque_mask = np.zeros(self.action_space.shape, dtype=np.float64)
                torque_scale = np.zeros(self.action_space.shape, dtype=np.float64)
                torque_phase = "jump_handoff"
            else:
                torque_mask, torque_scale = self._policy_action_authority()
                torque_phase = self._policy_action_authority_phase()
            effective_substep_action = np.zeros(self.action_space.shape, dtype=np.float64)
            effective_substep_action[: self.model.nu] = (
                action_array[: self.model.nu]
                * torque_mask[: self.model.nu]
                * torque_scale[: self.model.nu]
            )
            effective_substep_action[-1] = leg_effective_action
            executed_mask = torque_mask.copy()
            executed_scale = torque_scale.copy()
            executed_mask[-1] = reference_mask[-1]
            executed_scale[-1] = reference_scale[-1]
            effective_action_sum += effective_substep_action
            applied_action_mask = np.maximum(applied_action_mask, executed_mask)
            applied_action_scale = np.maximum(applied_action_scale, executed_scale)
            authority_phases.update((reference_phase, torque_phase))
            residual_action = effective_substep_action[: self.model.nu]
            requested_control = np.clip(
                lqr_control + residual_action * self.residual_limits,
                self._control_low,
                self._control_high,
            )
            if true_qpos is not None and true_qvel is not None:
                self._restore_true_controller_state(true_qpos, true_qvel)
            control = self._apply_control_delay(
                requested_control,
                bypass=(
                    lqr_owns_control
                    or current_jump_phase is not None
                    or current_drop_phase is not None
                ),
            )
            self.data.ctrl[:] = control
            mujoco.mj_step(self.model, self.data)
            self._last_lqr_control[:] = lqr_control
            self._last_requested_control[:] = requested_control
            self._last_control[:] = control
            physics_steps += 1
            if was_drop_active and not self._controller.drop.active:
                self._rebase_body_height_after_drop()
            terrain_support_valid = (
                self._update_confirmed_terrain_support()
                if self._active_terrain_route is not None
                else True
            )
            self._record_jump_transition(was_jump_active)
            self._update_jump_peak_height()
            self._update_jump_landing_guard()
            self._update_terrain_task_progress()
            safety_reason = (
                None if terrain_support_valid else "terrain_support_out_of_bounds"
            )
            if safety_reason is None:
                safety_reason = self._safety_reason()
            self._record_jump_landing_guard(safety_reason)
            if safety_reason is not None:
                break

        self._terrain_progress_delta_this_step_m = (
            self._terrain_progress_m - terrain_progress_before_step
        )
        self._last_control_duration_s = physics_steps * self.model.opt.timestep
        physical_safety_reason = safety_reason
        terrain_deadline_reached = (
            self._active_terrain_task is not None
            and self.data.time >= self._terrain_task_deadline_s
        )
        if (
            safety_reason is None
            and terrain_deadline_reached
            and not self._terrain_task_completed
        ):
            self._terrain_task_failed = True
        task_completion_terminal = bool(
            self._active_terrain_task is not None and self._terrain_task_completed
        )
        terminated = safety_reason is not None or task_completion_terminal
        truncated = bool(
            not terminated
            and (
                self.data.time >= self.episode_seconds
                or terrain_deadline_reached
            )
        )
        effective_action = effective_action_sum / max(physics_steps, 1)
        policy_action_applied = bool(np.any(applied_action_mask > 0.0))
        policy_action_authority_phase = (
            "|".join(sorted(authority_phases)) if authority_phases else "uninitialized"
        )
        self._previous_action[:] = effective_action if policy_action_applied else 0.0
        # Missing the height target after an otherwise stable touchdown is a
        # task failure, not a physical fall.  It receives the explicit abort
        # cost but not the separate unsafe-state penalty.
        physical_unsafe = bool(
            physical_safety_reason is not None
            and physical_safety_reason != "jump_aborted_jump_height_target_not_reached"
        )
        reward = self._reward(
            self._last_control,
            effective_action,
            physical_unsafe,
            task_completed_safely=bool(task_completion_terminal and safety_reason is None),
        )
        info = self._info(safety_reason)
        if task_completion_terminal:
            info["truncation_reason"] = None
            info["termination_reason"] = "terrain_task_completed"
        elif terrain_deadline_reached:
            info["truncation_reason"] = "terrain_task_deadline"
            info["termination_reason"] = None
        elif self.data.time >= self.episode_seconds:
            info["truncation_reason"] = "episode_time_limit"
            info["termination_reason"] = None
        else:
            info["truncation_reason"] = None
            info["termination_reason"] = None
        info["physical_unsafe"] = physical_unsafe
        info["policy_action_applied"] = policy_action_applied
        info["policy_action_mask"] = applied_action_mask.astype(np.float32)
        info["policy_action_scale"] = applied_action_scale.astype(np.float32)
        info["policy_action_authority_phase"] = policy_action_authority_phase
        info["effective_policy_action"] = effective_action.astype(np.float32)
        self._apply_sensor_noise_to_data()
        observation = self._observation()
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._render_camera = mujoco.MjvCamera()
            self._render_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._render_camera.trackbodyid = self.refs.robot_body
            self._render_camera.distance = 3.2
            self._render_camera.azimuth = 135.0
            self._render_camera.elevation = -20.0
        self._renderer.update_scene(self.data, camera=self._render_camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._render_camera = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the low-centre residual Gymnasium environment.")
    parser.add_argument("--steps", type=int, default=10, help="Number of zero-residual environment steps.")
    parser.add_argument("--smoke", action="store_true", help="Run a short deterministic reset/step test.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    steps = min(args.steps, 3) if args.smoke else args.steps

    environment = WheelLegResidualEnv(randomize_command=False, randomize_leg_length=False)
    try:
        observation, info = environment.reset(
            seed=0,
            options={
                "command_speed": 0.0,
                "command_leg_length": 0.244,
            },
        )
        print(f"reset: observation={observation.shape}, action={environment.action_space.shape}, contacts={info['wheel_contacts']}")
        reward = 0.0
        for _ in range(steps):
            observation, reward, terminated, truncated, info = environment.step(
                np.zeros(environment.action_space.shape, dtype=np.float32)
            )
            if terminated or truncated:
                break
        print(
            f"smoke: t={environment.data.time:.3f}s reward={reward:.3f} "
            f"terminated={terminated} truncated={truncated} reason={info['safety_reason']}"
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
