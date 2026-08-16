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


DEFAULT_EPISODE_SECONDS = 8.0
DEFAULT_CONTROL_DECIMATION = 10
MAX_ATTITUDE_ERROR_RAD = 1.0
MAX_CONTACT_LOSS_STEPS = 3
RL_LEG_LENGTH_COMMAND_RATE_MPS = 0.05
RL_COMMAND_SPEED_FRACTION = 0.65
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

    @classmethod
    def training_defaults(cls) -> "DomainRandomizationConfig":
        return cls(enabled=True)

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
    def disabled(cls) -> "DomainRandomizationConfig":
        return cls(enabled=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "enabled": self.enabled,
            "mass_global_range": self.mass_global_range,
            "mass_body_range": self.mass_body_range,
            "inertia_range": self.inertia_range,
            "friction_sliding_range": self.friction_sliding_range,
            "damping_range": self.damping_range,
            "hip_strength_range": self.hip_strength_range,
            "wheel_strength_range": self.wheel_strength_range,
            "sensor_noise_scale_range": self.sensor_noise_scale_range,
            "control_delay_steps_range": self.control_delay_steps_range,
        }


class WheelLegResidualEnv(gym.Env):
    """Low-centre LQR locomotion with a bounded six-actuator residual policy.

    The first six actions are residual torques in XML actuator order. The
    seventh action is a rate-limited common leg-length command, which lets the
    policy use the LQR's closed-chain extension controller without directly
    exciting the passive links. The observation includes current physical
    state, speed and leg-length commands, contacts, and the previous action.
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
        max_command_yaw_delta_rad: float = 0.0,
        jump_probability: float = 0.0,
        jump_at: float = DEFAULT_JUMP_AT_S,
        domain_randomization: DomainRandomizationConfig | None = None,
        jump_domain_randomization: DomainRandomizationConfig | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if episode_seconds <= 0.0:
            raise ValueError("episode_seconds must be positive")
        if control_decimation < 1:
            raise ValueError("control_decimation must be at least one")
        if not np.isfinite(max_command_yaw_delta_rad) or not 0.0 <= max_command_yaw_delta_rad <= np.pi:
            raise ValueError("max_command_yaw_delta_rad must be within 0..pi")
        if not np.isfinite(jump_probability) or not 0.0 <= jump_probability <= 1.0:
            raise ValueError("jump_probability must be within 0..1")
        if not np.isfinite(jump_at) or not 0.0 <= jump_at < episode_seconds:
            raise ValueError("jump_at must be within the episode")

        self.xml_path = Path(xml_path) if xml_path is not None else lqr.XML_PATH
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
        # rates + speed/leg commands + 3 heading features + jump task/countdown/phase
        # state + 5 jump-height/vertical-motion features + 2 contacts + 7 previous actions.
        self._observation_size = 49
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
        self._command_leg_length = 0.0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self._last_lqr_control = np.zeros(self.model.nu, dtype=np.float64)
        self._last_requested_control = np.zeros(self.model.nu, dtype=np.float64)
        self._last_control = np.zeros(self.model.nu, dtype=np.float64)
        self._contact_loss_steps = 0
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
            "damping_scale": 1.0,
            "hip_strength_scale": 1.0,
            "wheel_strength_scale": 1.0,
            "sensor_noise_scale": 0.0,
            "control_delay_steps": 0,
            "control_delay_ms": 0.0,
        }

    def _select_episode_domain_randomization(self) -> None:
        if self._jump_scheduled and self.jump_domain_randomization is not None:
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
        friction_geoms = (*self.refs.ground_geoms, *self.refs.wheel_geoms)
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
        """Use the inherited plane while building an LQR trim for an hfield scene."""
        if not self._terrain_support_geoms:
            return
        self.model.geom_contype[self.refs.ground_geom] = self._nominal_geom_contype[self.refs.ground_geom]
        self.model.geom_conaffinity[self.refs.ground_geom] = self._nominal_geom_conaffinity[self.refs.ground_geom]
        for geom_id in self._terrain_support_geoms:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
        mujoco.mj_forward(self.model, self.data)

    def _activate_terrain_support(self) -> None:
        """Replace the temporary LQR projection plane with the scene hfield."""
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
            self._command_speed,
            leg_length=self._command_leg_length,
            yaw=command_yaw,
        )

    def sample_command_speed(
        self,
        rng: np.random.Generator,
        fraction: float = RL_COMMAND_SPEED_FRACTION,
    ) -> float:
        """Sample both directions without exceeding their validated LQR ranges."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be within (0, 1]")
        if float(rng.uniform()) < 0.5:
            return float(rng.uniform(-fraction * self.max_reverse_speed, 0.0))
        return float(rng.uniform(0.0, fraction * self.max_forward_speed))

    def _clamp_command_speed(self, speed: float) -> float:
        return float(
            lqr.clamp_speed_command(
                speed,
                forward_limit=self.max_forward_speed,
                reverse_limit=self.max_reverse_speed,
            )
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        command_yaw = options.get("command_yaw")
        command_yaw_delta = options.get("command_yaw_delta_rad")
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

        if "jump_at" in options:
            requested_jump_at = options["jump_at"]
            if requested_jump_at is not None:
                requested_jump_at = float(requested_jump_at)
                if not np.isfinite(requested_jump_at) or not 0.0 <= requested_jump_at < self.episode_seconds:
                    raise ValueError("jump_at must be within the episode")
        elif self.jump_probability > 0.0 and float(self.np_random.uniform()) < self.jump_probability:
            requested_jump_at = self.jump_at
        else:
            requested_jump_at = None
        self._jump_at = requested_jump_at
        self._jump_scheduled = requested_jump_at is not None

        if "command_speed" in options:
            command_speed = float(options["command_speed"])
        elif self.randomize_command:
            command_speed = self.sample_command_speed(self.np_random)
        else:
            command_speed = 0.0
        self._command_speed = self._clamp_command_speed(command_speed)
        if self._jump_scheduled:
            self._command_speed = 0.0
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
        if not self.randomize_leg_length and "command_leg_length" not in options:
            self._command_leg_length = self._default_leg_length

        current_yaw = self._controller.heading_yaw(self.data)
        if command_yaw is None:
            if command_yaw_delta is not None:
                command_yaw = wrap_to_pi(current_yaw + command_yaw_delta)
            elif self.max_command_yaw_delta_rad > 0.0:
                command_yaw = wrap_to_pi(current_yaw + float(self.np_random.uniform(
                    -self.max_command_yaw_delta_rad,
                    self.max_command_yaw_delta_rad,
                )))
        self._reset_lqr_command(command_yaw)

        self._reference_quaternion = self.data.xquat[self._robot_body].copy()
        self._reference_heading_yaw = self._controller.heading_yaw(self.data)
        self._reference_body_height = float(self.data.xpos[self._robot_body, 2])
        self._previous_action.fill(0.0)
        self._last_lqr_control[:] = self._stance_ctrl
        self._last_requested_control[:] = self._stance_ctrl
        self._last_control[:] = self._stance_ctrl
        self._reset_control_delay()
        self._sample_sensor_noise()
        self._contact_loss_steps = 0
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

    def _wheel_ground_clearances_m(self) -> np.ndarray:
        """Return physical left/right wheel-bottom clearance above the ground."""
        clearances: list[float] = []
        for geom_id in self.refs.wheel_geoms:
            radius = float(self.model.geom_size[geom_id, 0])
            half_width = float(self.model.geom_size[geom_id, 1])
            wheel_axis_z = float(self.data.geom_xmat[geom_id].reshape(3, 3)[2, 2])
            vertical_extent = radius * np.sqrt(max(0.0, 1.0 - wheel_axis_z * wheel_axis_z))
            vertical_extent += half_width * abs(wheel_axis_z)
            clearances.append(float(self.data.geom_xpos[geom_id, 2] - vertical_extent))
        return np.maximum(0.0, np.asarray(clearances, dtype=np.float64))

    def _wheel_ground_clearance_m(self) -> float:
        """Return the lower wheel clearance, the hard-safe jump-height metric."""
        return float(np.min(self._wheel_ground_clearances_m()))

    def _ground_has_nonwheel_contact(self) -> bool:
        support_geoms = set(self.refs.ground_geoms)
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 in support_geoms:
                other_geom = contact.geom2
            elif contact.geom2 in support_geoms:
                other_geom = contact.geom1
            else:
                continue
            if other_geom not in self.refs.wheel_geoms:
                return True
        return False

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
            float(self.data.xpos[self._robot_body, 2]) - self._reference_body_height,
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
            self._controller.request_jump(self.data)
            self._jump_triggered = True
            self._command_speed = 0.0

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
        body_rise = max(0.0, float(self.data.xpos[self._robot_body, 2]) - self._reference_body_height)
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
        if jump_active:
            try:
                lqr.validate_jump_contacts(self.data, self.refs)
            except RuntimeError as error:
                return f"jump_{str(error).lower().replace(' ', '_')}"
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

            contacts = self._wheel_contacts()
            if all(contacts):
                self._contact_loss_steps = 0
            else:
                self._contact_loss_steps += 1
                if self._contact_loss_steps >= MAX_CONTACT_LOSS_STEPS:
                    return "wheel_contact_lost"

        if self._jump_failed:
            return f"jump_aborted_{self._jump_failure_reason.replace(' ', '_')}"

        attitude_error = float(np.linalg.norm(self._orientation_error()))
        if attitude_error > MAX_ATTITUDE_ERROR_RAD:
            return "attitude_limit"
        body_height = float(self.data.xpos[self._robot_body, 2])
        if body_height < self._reference_body_height - 0.22:
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
                # Keep the legacy 3 m/s observation scale so existing policies
                # remain numerically compatible while reverse commands are clipped.
                np.array((self._command_speed / self.max_speed,)),
                np.array(((self._command_leg_length - self._default_leg_length) / 0.10,)),
                np.array((
                    yaw_state["yaw_error_rad"] / np.pi,
                    yaw_state["pending_yaw_error_rad"] / np.pi,
                    yaw_state["yaw_rate_normalized"],
                )),
                self._jump_observation(),
                self._jump_height_observation(),
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
        return self.lqr_controller.set_command_yaw(yaw)

    def set_command_speed(self, speed: float) -> float:
        """Set a manual LQR speed command and keep telemetry in sync."""
        target = self.lqr_controller.set_target_speed(speed)
        self._command_speed = float(target)
        return target

    def adjust_command_speed(self, delta: float) -> float:
        """Increment the manual LQR speed command and keep telemetry in sync."""
        target = self.lqr_controller.adjust_target_speed(delta)
        self._command_speed = float(target)
        return target

    def adjust_command_yaw(self, delta: float) -> float:
        """Increment the manual world-frame heading command."""
        return self.lqr_controller.adjust_command_yaw(delta)

    def hold_current_yaw(self) -> float:
        """Make the current measured heading the manual LQR heading target."""
        return self.lqr_controller.hold_current_yaw(self.data)

    def adjust_command_leg_length(self, delta: float) -> float:
        """Increment the common LQR leg-length command and update telemetry."""
        target = self.lqr_controller.adjust_target_leg_length(delta)
        self._command_leg_length = float(target)
        return target

    def request_lqr_jump(self) -> bool:
        """Start an operator-requested LQR jump with environment safety tracking."""
        controller = self.lqr_controller
        controller.request_jump(self.data)
        self._command_speed = 0.0
        self._command_leg_length = float(controller.leg_command.target_length)
        self._jump_scheduled = True
        self._jump_at = None
        self._jump_triggered = True
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
        target_yaw_rate = float(np.clip(
            lqr.YAW_HEADING_KP_RAD_S_PER_RAD * yaw_error,
            -lqr.MAX_YAW_RATE_RAD_S,
            lqr.MAX_YAW_RATE_RAD_S,
        ))
        measured_yaw_rate = float(self._controller._measured_yaw_rate)
        measured_yaw_rate_normalized = float(np.clip(
            measured_yaw_rate / lqr.MAX_YAW_RATE_RAD_S,
            -1.0,
            1.0,
        ))
        turn_intensity = float(np.clip(max(
            abs(target_yaw_rate) / lqr.MAX_YAW_RATE_RAD_S,
            abs(measured_yaw_rate_normalized),
            abs(pending_yaw_error) / np.pi,
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
    def _reward(self, control: np.ndarray, action: np.ndarray, unsafe: bool) -> float:
        forward_speed = self._forward_speed()
        ramped_speed = float(self._controller.motion.current_speed)
        speed_error = forward_speed - ramped_speed
        jump_active = self._controller.jump.active
        yaw_state = self.yaw_state()
        attitude_cost = float(np.dot(self._orientation_error(), self._orientation_error()))
        self.leg_diff = abs(
            self._sensor("left_leg_length")[0] - self._sensor("right_leg_length")[0]
        )
        jump_peak_increment = 0.0
        if jump_active:
            # LQR owns all actions in the jump sequence, so only reward the
            # observable flight result instead of unreachable speed/yaw costs.
            # 防止reward hacking
            tracking = 0.0
            leg_tracking = 0.0
            attitude_tracking = 0.0
            yaw_tracking = 0.0
            yaw_rate_error = 0.0
            energy_cost = 0.0
            residual_cost = 0.0
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
        if self._jump_success_this_step:
            reward += JUMP_SUCCESS_REWARD
        if self._jump_failure_this_step:
            reward -= JUMP_ABORT_PENALTY
        if self._jump_landing_fall_this_step:
            reward -= JUMP_LANDING_FALL_PENALTY
        if unsafe:
            reward -= 30.0
        self._jump_success_this_step = False
        self._jump_failure_this_step = False
        self._jump_landing_fall_this_step = False
        return float(reward)
#速度跟随最高权重，腿长误差和关节位姿次权重 额外加入双腿相对误差 跳跃时给予各项指标相应缓冲带宽
    def _info(self, safety_reason: str | None = None) -> dict[str, Any]:
        forward_speed = self._forward_speed()
        yaw_state = self.yaw_state()
        ramped_speed = 0.0 if self._controller is None else float(self._controller.motion.current_speed)
        jump_active = self._controller is not None and self._controller.jump.active
        jump_phase = None if self._controller is None else self._controller.jump.phase_name
        jump_elapsed = 0.0 if self._controller is None else self._controller.jump.elapsed(float(self.data.time))
        speed_status = lqr.speed_tracking_status(
            self._command_speed,
            ramped_speed,
            forward_speed,
            jump_active=jump_active,
        )
        return {
            "command_speed_mps": self._command_speed,
            "ramped_command_speed_mps": ramped_speed,
            "forward_speed_mps": forward_speed,
            "speed_error_mps": forward_speed - self._command_speed,
            "speed_status": speed_status,
            **yaw_state,
            "jump_scheduled": self._jump_scheduled,
            "jump_at_s": self._jump_at,
            "jump_triggered": self._jump_triggered,
            "jump_active": jump_active,
            "jump_phase": jump_phase,
            "jump_elapsed_s": jump_elapsed,
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
        self._schedule_jump_if_due()
        self._sample_sensor_noise()
        # The jump state machine owns all six torques and the common leg
        # command.  Report this to PPO so action-independent jump rewards do
        # not create actor-gradient noise.
        policy_action_applied = not (
            self._controller.jump.active or self._jump_landing_pending
        )
        if not policy_action_applied:
            leg_length_action = 0.0
            # The jump state machine owns its own wider, jump-safe leg range.
            # Do not re-clamp it through the normal walking leg limit here.
            self._command_leg_length = float(self._controller.leg_command.target_length)
        else:
            leg_length_action = float(action_array[-1])
            self._command_leg_length = self._controller.adjust_target_leg_length(
                leg_length_action
                * RL_LEG_LENGTH_COMMAND_RATE_MPS
                * self.control_decimation
                * self.model.opt.timestep
            )

        safety_reason: str | None = None
        for _ in range(self.control_decimation):
            was_jump_active = self._controller.jump.active
            lqr_owns_control = was_jump_active or self._jump_landing_pending
            if was_jump_active:
                # The scheduled jump has no residual action authority.  Keep
                # sensor noise in the policy observation, but do not corrupt
                # the contact-critical LQR phase/length state machine with an
                # unobservable disturbance it cannot reject.
                mujoco.mj_forward(self.model, self.data)
                lqr_control = self._controller.command(self.data)
                true_qpos = true_qvel = None
            else:
                true_qpos, true_qvel = self._prepare_noisy_controller_measurement()
                lqr_control = self._controller.command(self.data)
            current_jump_phase = self._controller.jump.phase_name
            residual_action = (
                np.zeros(self.model.nu, dtype=np.float64)
                if lqr_owns_control
                else action_array[: self.model.nu]
            )
            requested_control = np.clip(
                lqr_control + residual_action * self.residual_limits,
                self._control_low,
                self._control_high,
            )
            if true_qpos is not None and true_qvel is not None:
                self._restore_true_controller_state(true_qpos, true_qvel)
            control = self._apply_control_delay(
                requested_control,
                bypass=lqr_owns_control or current_jump_phase is not None,
            )
            self.data.ctrl[:] = control
            mujoco.mj_step(self.model, self.data)
            self._last_lqr_control[:] = lqr_control
            self._last_requested_control[:] = requested_control
            self._last_control[:] = control
            self._record_jump_transition(was_jump_active)
            self._update_jump_peak_height()
            self._update_jump_landing_guard()
            safety_reason = self._safety_reason()
            self._record_jump_landing_guard(safety_reason)
            if safety_reason is not None:
                break

        terminated = safety_reason is not None
        truncated = bool(self.data.time >= self.episode_seconds and not terminated)
        self._previous_action[:] = action_array if policy_action_applied else 0.0
        # Missing the height target after an otherwise stable touchdown is a
        # task failure, not a physical fall.  It receives the explicit abort
        # cost but not the separate unsafe-state penalty.
        physical_unsafe = terminated and safety_reason != "jump_aborted_jump_height_target_not_reached"
        reward = self._reward(self._last_control, action_array, physical_unsafe)
        info = self._info(safety_reason)
        info["policy_action_applied"] = policy_action_applied
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
