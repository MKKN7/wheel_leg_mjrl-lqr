"""GPU-native flat walking task built on :mod:`warp_env`.

The module is deliberately a task layer, rather than another physics backend.
``WarpPhysicsBatch`` owns MuJoCo-Warp model/data allocation and the independent
per-world estop.  ``WarpFlatWalkingTask`` keeps commands, references, reward,
observation and episode state in CUDA tensors so a future PPO rollout does not
copy policy data through the host.

The public action/observation contract matches ``WheelLegResidualEnv``:

* action: ``[N, 7]`` normalized values (six actuator channels plus common leg
  length command);
* observation: ``[N, 67]`` in the same order as the CPU environment;
* PPO uses six bounded residual controls around a GPU feedback-controller
  output.  A full-range direct action mode remains available only for isolated
  diagnostic smoke tests; it is never the default training path.

Flat mode intentionally has no jump supervisor and no route geometry.  Its
seven jump features are zero, and its sixteen terrain features are zero with a
final ``valid`` flag of one.  This preserves checkpoint shape while making the
approximation explicit; terrain/jump tasks must provide their own GPU feature
provider before being enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from warp_env import WarpBatchError, WarpBatchStep, WarpPhysicsBatch
from warp_safety import (
    SAFETY_REASON_NONFINITE_CONTROL,
    WarpSafetyLimits,
    WarpSafetyResult,
    WarpSafetyScratch,
    evaluate_safety,
)


OBSERVATION_SIZE = 67
ACTION_SIZE = 7
TERRAIN_FEATURE_SIZE = 16
JUMP_FEATURE_SIZE = 7
JUMP_HEIGHT_FEATURE_SIZE = 5
MAX_ATTITUDE_ERROR_RAD = 1.0
MAX_YAW_RATE_RAD_S = 0.45
DEFAULT_LEG_LENGTH_MIN_M = 0.205
DEFAULT_LEG_LENGTH_MAX_M = 0.300
DEFAULT_LEG_COMMAND_RATE_MPS = 0.05
DEFAULT_EPISODE_SECONDS = 8.0
DEFAULT_CONTACT_LOSS_TIMEOUT_SECONDS = 0.18
DEFAULT_CONTACT_CLEARANCE_M = 0.010
DEFAULT_WHEEL_CLEARANCE_NORMALIZATION_M = 0.25
DEFAULT_BODY_RISE_NORMALIZATION_M = 0.25
DEFAULT_RESIDUAL_LIMITS = (8.0, 8.0, 0.75, 8.0, 8.0, 0.75, 1.0)

# All four names are a single MJCF contract. The order is front-left,
# front-right, rear-left, rear-right; public observations still expose only
# the two active-wheel contacts.
GUIDE_WHEEL_CONTACT_GEOM_NAMES = (
    "guide_wheel_front_left_contact",
    "guide_wheel_front_right_contact",
    "guide_wheel_rear_left_contact",
    "guide_wheel_rear_right_contact",
)


@dataclass(frozen=True)
class WarpFlatWalkingConfig:
    """GPU task parameters; callers should construct this from YAML."""

    command_speed_mps: float = 0.0
    command_yaw_rate_rad_s: float = 0.0
    command_leg_length_m: float | None = None
    command_speed_limit_mps: float = 3.0
    command_yaw_rate_limit_rad_s: float = MAX_YAW_RATE_RAD_S
    leg_length_min_m: float = DEFAULT_LEG_LENGTH_MIN_M
    leg_length_max_m: float = DEFAULT_LEG_LENGTH_MAX_M
    leg_command_rate_mps: float = DEFAULT_LEG_COMMAND_RATE_MPS
    episode_seconds: float = DEFAULT_EPISODE_SECONDS
    contact_loss_timeout_seconds: float = DEFAULT_CONTACT_LOSS_TIMEOUT_SECONDS
    contact_clearance_m: float = DEFAULT_CONTACT_CLEARANCE_M
    safety_leg_length_min_m: float = 0.180
    safety_leg_length_max_m: float = 0.400
    max_leg_length_difference_m: float = 0.015
    domain_randomization_enabled: bool = False
    sensor_noise_std: float = 0.0
    control_delay_steps: int = 0
    domain_randomization_seed: int = 0
    # Direct full-range controls are retained only for diagnostic smoke runs.
    # PPO must use a validated GPU baseline plus bounded residuals.
    direct_control_mode: bool = False
    residual_limits: tuple[float, float, float, float, float, float, float] = DEFAULT_RESIDUAL_LIMITS
    leg_action_enabled: bool = False

    def __post_init__(self) -> None:
        values = (
            ("command_speed_mps", self.command_speed_mps),
            ("command_yaw_rate_rad_s", self.command_yaw_rate_rad_s),
            ("command_speed_limit_mps", self.command_speed_limit_mps),
            ("command_yaw_rate_limit_rad_s", self.command_yaw_rate_limit_rad_s),
            ("leg_length_min_m", self.leg_length_min_m),
            ("leg_length_max_m", self.leg_length_max_m),
            ("leg_command_rate_mps", self.leg_command_rate_mps),
            ("episode_seconds", self.episode_seconds),
            ("contact_loss_timeout_seconds", self.contact_loss_timeout_seconds),
            ("contact_clearance_m", self.contact_clearance_m),
            ("safety_leg_length_min_m", self.safety_leg_length_min_m),
            ("safety_leg_length_max_m", self.safety_leg_length_max_m),
            ("max_leg_length_difference_m", self.max_leg_length_difference_m),
            ("sensor_noise_std", self.sensor_noise_std),
        )
        for name, value in values:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.command_speed_limit_mps <= 0.0:
            raise ValueError("command_speed_limit_mps must be positive")
        if self.command_yaw_rate_limit_rad_s <= 0.0:
            raise ValueError("command_yaw_rate_limit_rad_s must be positive")
        if not 0.0 < self.leg_length_min_m <= self.leg_length_max_m:
            raise ValueError("leg length limits are invalid")
        if self.episode_seconds <= 0.0 or self.contact_loss_timeout_seconds <= 0.0:
            raise ValueError("episode and contact timeout must be positive")
        if self.contact_clearance_m <= 0.0:
            raise ValueError("contact_clearance_m must be positive")
        if not 0.0 < self.safety_leg_length_min_m <= self.safety_leg_length_max_m:
            raise ValueError("safety leg length limits are invalid")
        if self.max_leg_length_difference_m <= 0.0:
            raise ValueError("max_leg_length_difference_m must be positive")
        if self.sensor_noise_std < 0.0:
            raise ValueError("sensor_noise_std must be non-negative")
        if isinstance(self.control_delay_steps, bool) or not isinstance(self.control_delay_steps, int) or self.control_delay_steps < 0:
            raise ValueError("control_delay_steps must be a non-negative integer")
        if isinstance(self.domain_randomization_seed, bool) or not isinstance(self.domain_randomization_seed, int) or self.domain_randomization_seed < 0:
            raise ValueError("domain_randomization_seed must be a non-negative integer")
        if not isinstance(self.domain_randomization_enabled, bool):
            raise ValueError("domain_randomization_enabled must be boolean")
        if abs(self.command_speed_mps) > self.command_speed_limit_mps + 1.0e-9:
            raise ValueError("command_speed_mps exceeds command_speed_limit_mps")
        if abs(self.command_yaw_rate_rad_s) > self.command_yaw_rate_limit_rad_s + 1.0e-9:
            raise ValueError("command_yaw_rate_rad_s exceeds command_yaw_rate_limit_rad_s")
        if self.command_leg_length_m is not None and not (
            self.leg_length_min_m <= self.command_leg_length_m <= self.leg_length_max_m
        ):
            raise ValueError("command_leg_length_m is outside configured leg length limits")
        if not isinstance(self.direct_control_mode, bool) or not isinstance(self.leg_action_enabled, bool):
            raise ValueError("direct_control_mode and leg_action_enabled must be boolean")
        limits = tuple(self.residual_limits)
        if len(limits) != ACTION_SIZE:
            raise ValueError(f"residual_limits must have {ACTION_SIZE} entries")
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0 for value in limits):
            raise ValueError("residual_limits must be finite and non-negative")
        if float(limits[-1]) > 1.0:
            raise ValueError("the leg-rate residual limit cannot exceed 1.0")
        object.__setattr__(self, "residual_limits", tuple(float(value) for value in limits))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WarpFlatWalkingConfig":
        """Build task parameters from a YAML mapping without hidden defaults.

        Missing keys use the conservative dataclass defaults; unknown keys are
        rejected so a typo cannot silently change a training experiment.
        """

        if not isinstance(raw, Mapping):
            raise ValueError("flat walking task config must be a mapping")
        allowed = {
            "command_speed_mps",
            "command_yaw_rate_rad_s",
            "command_leg_length_m",
            "command_speed_limit_mps",
            "command_yaw_rate_limit_rad_s",
            "leg_length_min_m",
            "leg_length_max_m",
            "leg_command_rate_mps",
            "episode_seconds",
            "contact_loss_timeout_seconds",
            "contact_clearance_m",
            "safety_leg_length_min_m",
            "safety_leg_length_max_m",
            "max_leg_length_difference_m",
            "domain_randomization_enabled",
            "sensor_noise_std",
            "control_delay_steps",
            "domain_randomization_seed",
            "direct_control_mode",
            "residual_limits",
            "leg_action_enabled",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown flat walking task config keys: {unknown}")
        return cls(**dict(raw))


def load_flat_walking_config(path: str | Path, *, section: str = "flat_walking") -> WarpFlatWalkingConfig:
    """Load the task section from a YAML file."""

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to read flat walking config {source}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("flat walking YAML root must be a mapping")
    section_raw = raw.get(section, raw)
    return WarpFlatWalkingConfig.from_mapping(section_raw)


@dataclass(frozen=True)
class WarpObservationLayout:
    """Stable slices for the 67-D CPU-compatible observation."""

    orientation: slice = slice(0, 3)
    world_velocity: slice = slice(3, 6)
    body_angular_velocity: slice = slice(6, 9)
    hip_position: slice = slice(9, 13)
    hip_velocity: slice = slice(13, 17)
    wheel_velocity: slice = slice(17, 19)
    leg_length: slice = slice(19, 21)
    leg_length_velocity: slice = slice(21, 23)
    command_speed: slice = slice(23, 24)
    command_leg_length: slice = slice(24, 25)
    command_yaw_rate: slice = slice(25, 26)
    jump_request: slice = slice(26, 27)
    yaw_state: slice = slice(27, 30)
    jump_phase: slice = slice(30, 37)
    jump_height: slice = slice(37, 42)
    terrain: slice = slice(42, 58)
    contacts: slice = slice(58, 60)
    previous_action: slice = slice(60, 67)


OBS_LAYOUT = WarpObservationLayout()


@dataclass(frozen=True)
class WarpTaskStep:
    """GPU-resident task outputs."""

    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    done: Any
    physics: WarpBatchStep


@dataclass(frozen=True)
class WarpFlatStanceCalibration:
    """One CPU-generated, immutable initialization for GPU flat rollouts.

    It is constructed once before the vector task starts.  The task uploads
    these arrays once to CUDA and every later reset writes those preallocated
    CUDA buffers through ``WarpPhysicsBatch.reset_to_state``.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    nominal_control: np.ndarray

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float32)
        qvel = np.asarray(self.qvel, dtype=np.float32)
        nominal_control = np.asarray(self.nominal_control, dtype=np.float32)
        if qpos.ndim != 1 or qvel.ndim != 1 or nominal_control.ndim != 1:
            raise ValueError("calibration qpos, qvel, and nominal_control must be vectors")
        if not (np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(nominal_control).all()):
            raise ValueError("calibration state and controls must be finite")
        object.__setattr__(self, "qpos", qpos.copy())
        object.__setattr__(self, "qvel", qvel.copy())
        object.__setattr__(self, "nominal_control", nominal_control.copy())


def _find_sensor(model: Any, name: str) -> tuple[int, int]:
    for sensor_id in range(int(model.nsensor)):
        sensor = model.sensor(sensor_id)
        if sensor.name == name:
            return int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id])
    raise WarpBatchError(f"required sensor {name!r} is missing from the MJCF")


def _find_joint(model: Any, name: str) -> tuple[int, int]:
    for joint_id in range(int(model.njnt)):
        joint = model.joint(joint_id)
        if joint.name == name:
            return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    raise WarpBatchError(f"required joint {name!r} is missing from the MJCF")


def _find_geom(model: Any, names: tuple[str, ...]) -> int:
    for geom_id in range(int(model.ngeom)):
        if model.geom(geom_id).name in names:
            return geom_id
    raise WarpBatchError(f"none of the required geoms {names!r} are present in the MJCF")


def _find_required_geoms(model: Any, names: tuple[str, ...]) -> np.ndarray:
    """Resolve an all-or-none named geom set before CUDA allocation."""

    ids_by_name = {
        model.geom(geom_id).name: geom_id
        for geom_id in range(int(model.ngeom))
        if model.geom(geom_id).name
    }
    missing = tuple(name for name in names if name not in ids_by_name)
    if missing:
        raise WarpBatchError(
            "all guide-wheel collision geoms are required by the GPU task; missing "
            + ", ".join(missing)
        )
    return np.asarray([ids_by_name[name] for name in names], dtype=np.int64)


def calibrate_flat_stance(batch: WarpPhysicsBatch) -> WarpFlatStanceCalibration:
    """Build one low-centre LQR working point through the CPU Gym environment.

    This is deliberately outside the rollout/reset hot path.  It uses the
    same ``WheelLegResidualEnv.reset`` stance projection as the CPU trainer,
    then retains only qpos/qvel/control-equilibrium.  The CPU controller is
    never called again by the GPU task.
    """

    try:
        from env import WheelLegResidualEnv
    except ImportError as error:  # pragma: no cover - project dependency
        raise WarpBatchError("flat stance calibration requires the local CPU environment") from error

    environment = WheelLegResidualEnv(
        xml_path=batch.config.xml_path,
        randomize_command=False,
        randomize_leg_length=False,
    )
    try:
        environment.reset(
            seed=0,
            options={"command_speed": 0.0, "command_yaw_rate_rad_s": 0.0},
        )
        qpos = np.asarray(environment.data.qpos, dtype=np.float32)
        qvel = np.asarray(environment.data.qvel, dtype=np.float32)
        controls = np.asarray(environment.lqr_controller.control_equilibrium, dtype=np.float32)
    finally:
        environment.close()
    if qpos.shape != (batch.host_model.nq,) or qvel.shape != (batch.host_model.nv,):
        raise WarpBatchError("CPU calibration state dimensions do not match the GPU model")
    if controls.shape != (batch.num_actuators,):
        raise WarpBatchError("CPU calibration actuator dimensions do not match the GPU model")
    return WarpFlatStanceCalibration(qpos=qpos, qvel=qvel, nominal_control=controls)


def _normalize_quaternion(torch: Any, quaternion: Any) -> Any:
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-7)
    return quaternion / norm


def _quat_multiply(torch: Any, lhs: Any, rhs: Any) -> Any:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_conjugate(torch: Any, quaternion: Any) -> Any:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _yaw_from_quaternion(torch: Any, quaternion: Any) -> Any:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_pi(torch: Any, angle: Any) -> Any:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _rotation_vector_from_quaternion(torch: Any, quaternion: Any) -> Any:
    quaternion = _normalize_quaternion(torch, quaternion)
    # q and -q represent the same attitude; choose the shortest branch.
    sign = torch.where(quaternion[..., :1] < 0.0, -1.0, 1.0)
    quaternion = quaternion * sign
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quaternion[..., :1].clamp_min(0.0))
    scale = torch.where(vector_norm > 1.0e-6, angle / vector_norm, 2.0)
    return vector * scale


def combine_side_support_contacts(
    torch: Any,
    active_contacts: Any,
    guide_contacts: Any,
    guide_side_out: Any,
    support_out: Any,
) -> Any:
    """Write per-side active-or-guide support into resident caller buffers."""

    torch.logical_or(guide_contacts[:, 0], guide_contacts[:, 2], out=guide_side_out[:, 0])
    torch.logical_or(guide_contacts[:, 1], guide_contacts[:, 3], out=guide_side_out[:, 1])
    torch.logical_or(active_contacts[:, 0], guide_side_out[:, 0], out=support_out[:, 0])
    torch.logical_or(active_contacts[:, 1], guide_side_out[:, 1], out=support_out[:, 1])
    return support_out


class WarpFlatWalkingTask:
    """67-D/7-D GPU vector task for flat walking rollouts.

    No host tensor is read in ``step``.  Model metadata and derated limits are
    captured once during construction; all episode bookkeeping is allocated
    at ``[num_worlds]`` on the configured CUDA device.
    """

    observation_size = OBSERVATION_SIZE
    action_size = ACTION_SIZE

    def __init__(
        self,
        batch: WarpPhysicsBatch,
        config: WarpFlatWalkingConfig | None = None,
        *,
        calibration: WarpFlatStanceCalibration | None = None,
        controller: Any | None = None,
    ) -> None:
        self.batch = batch
        self.config = WarpFlatWalkingConfig() if config is None else config
        torch = batch._torch
        self.torch = torch
        self.device = batch.device
        self.num_worlds = batch.num_worlds
        self.layout = OBS_LAYOUT
        self._controller = controller
        if batch.num_actuators != ACTION_SIZE - 1:
            raise WarpBatchError(
                f"flat task expects 6 actuators and one leg channel; got nu={batch.num_actuators}"
            )

        model = batch.host_model
        self._sensor = {
            name: _find_sensor(model, name)
            for name in (
                "world_horizontal_velocity_xy",
                "left_wheel_angular_velocity",
                "right_wheel_angular_velocity",
                "left_leg_length",
                "left_leg_length_velocity",
                "right_leg_length",
                "right_leg_length_velocity",
                "body_angular_velocity",
            )
        }
        self._hip_qpos = np.asarray(
            [_find_joint(model, name)[0] for name in (
                "left_hip_pitch",
                "left_active_link_pitch",
                "right_hip_pitch",
                "right_active_link_pitch",
            )],
            dtype=np.int64,
        )
        self._hip_qvel = np.asarray(
            [_find_joint(model, name)[1] for name in (
                "left_hip_pitch",
                "left_active_link_pitch",
                "right_hip_pitch",
                "right_active_link_pitch",
            )],
            dtype=np.int64,
        )
        free_joints = np.flatnonzero(np.asarray(model.jnt_type, dtype=np.int32) == 0)
        if free_joints.size != 1:
            raise WarpBatchError("flat task requires exactly one free-root joint")
        root_joint = int(free_joints[0])
        self.root_qpos_address = int(model.jnt_qposadr[root_joint])
        self.root_dof_address = int(model.jnt_dofadr[root_joint])
        self._hip_qpos_gpu = torch.as_tensor(self._hip_qpos, dtype=torch.long, device=self.device)
        self._hip_qvel_gpu = torch.as_tensor(self._hip_qvel, dtype=torch.long, device=self.device)
        # P0 joint-limit checks cover every scalar, range-limited hinge in the
        # model.  These are immutable model metadata copied once to CUDA.
        limited_hinges = np.flatnonzero(
            (np.asarray(model.jnt_type, dtype=np.int32) == 3)
            & np.asarray(model.jnt_limited, dtype=bool)
        )
        limited_qpos = np.asarray(model.jnt_qposadr[limited_hinges], dtype=np.int64)
        limited_ranges = np.asarray(model.jnt_range[limited_hinges], dtype=np.float32)
        self._limited_joint_qpos_gpu = torch.as_tensor(limited_qpos, dtype=torch.long, device=self.device)
        self._limited_joint_lower = torch.as_tensor(limited_ranges[:, 0], dtype=torch.float32, device=self.device)
        self._limited_joint_upper = torch.as_tensor(limited_ranges[:, 1], dtype=torch.float32, device=self.device)

        self._wheel_geom_ids = np.asarray(
            [
                _find_geom(model, ("left_wheel_contact", "left_wheel")),
                _find_geom(model, ("right_wheel_contact", "right_wheel")),
            ],
            dtype=np.int64,
        )
        self._wheel_geom_gpu = torch.as_tensor(self._wheel_geom_ids, dtype=torch.long, device=self.device)
        self._wheel_radius = torch.as_tensor(
            np.maximum(np.asarray(model.geom_size[self._wheel_geom_ids, 0], dtype=np.float32), 1.0e-5),
            dtype=torch.float32,
            device=self.device,
        )
        self._guide_wheel_geom_ids = _find_required_geoms(model, GUIDE_WHEEL_CONTACT_GEOM_NAMES)
        self._guide_wheel_geom_gpu = torch.as_tensor(
            self._guide_wheel_geom_ids, dtype=torch.long, device=self.device
        )
        self._guide_wheel_radius = torch.as_tensor(
            np.maximum(
                np.asarray(model.geom_size[self._guide_wheel_geom_ids, 0], dtype=np.float32),
                1.0e-5,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        ground_geom = _find_geom(model, ("ground", "rmuc_terrain"))
        self._ground_geom_id = ground_geom
        # The flat mode uses the plane support height.  A hfield route must
        # provide a terrain feature provider instead of this approximation.
        ground_position = np.asarray(batch.host_model.geom_pos[ground_geom], dtype=np.float32)
        self._ground_height = torch.tensor(float(ground_position[2]), dtype=torch.float32, device=self.device)
        # Cache the zero-copy view once.  ``observe`` is on the hot path and
        # must not repeatedly wrap a Warp array or allocate a new data view.
        self._geom_xpos = batch._warp.to_torch(batch.data.geom_xpos)
        # Geometry/contact workspaces remain resident for the rollout.  The
        # flat contact proxy is sampled at every physics substep by both the
        # safety gate and the reward/observation paths.
        self._wheel_positions = torch.empty(
            (self.num_worlds, 2, 3), dtype=torch.float32, device=self.device
        )
        self._wheel_clearances = torch.empty(
            (self.num_worlds, 2), dtype=torch.float32, device=self.device
        )
        self._wheel_contacts = torch.empty(
            (self.num_worlds, 2), dtype=torch.bool, device=self.device
        )
        self._guide_wheel_positions = torch.empty(
            (self.num_worlds, len(GUIDE_WHEEL_CONTACT_GEOM_NAMES), 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._guide_wheel_clearances = torch.empty(
            (self.num_worlds, len(GUIDE_WHEEL_CONTACT_GEOM_NAMES)),
            dtype=torch.float32,
            device=self.device,
        )
        self._guide_wheel_contacts = torch.empty(
            (self.num_worlds, len(GUIDE_WHEEL_CONTACT_GEOM_NAMES)),
            dtype=torch.bool,
            device=self.device,
        )
        self._guide_side_contacts = torch.empty((self.num_worlds, 2), dtype=torch.bool, device=self.device)
        self._support_contacts = torch.empty((self.num_worlds, 2), dtype=torch.bool, device=self.device)

        def index(name: str, offset: int = 0) -> int:
            address, dimension = self._sensor[name]
            if offset < 0 or offset >= dimension:
                raise WarpBatchError(f"sensor offset {name}[{offset}] is outside dimension {dimension}")
            return address + offset

        self._velocity_slice = slice(self._sensor["world_horizontal_velocity_xy"][0], self._sensor["world_horizontal_velocity_xy"][0] + 3)
        self._angular_slice = slice(self._sensor["body_angular_velocity"][0], self._sensor["body_angular_velocity"][0] + 3)
        self._wheel_velocity_indices = torch.as_tensor(
            [index("left_wheel_angular_velocity"), index("right_wheel_angular_velocity")],
            dtype=torch.long,
            device=self.device,
        )
        self._leg_length_indices = torch.as_tensor(
            [index("left_leg_length"), index("right_leg_length")], dtype=torch.long, device=self.device
        )
        self._leg_velocity_indices = torch.as_tensor(
            [index("left_leg_length_velocity"), index("right_leg_length_velocity")],
            dtype=torch.long,
            device=self.device,
        )

        self._control_low = batch._control_low
        self._control_high = batch._control_high
        # Construction-only actuator span and direct-control scratch buffer.
        # Diagnostic direct mode maps actions into this resident tensor at
        # every physics substep; it must not allocate an intermediate command
        # tensor in the rollout hot path.
        self._control_span = self._control_high - self._control_low
        self._control_scale = torch.maximum(self._control_low.abs(), self._control_high.abs()).clamp_min(1.0e-6)
        calibration = calibrate_flat_stance(batch) if calibration is None else calibration
        if calibration.qpos.shape != (int(model.nq),) or calibration.qvel.shape != (int(model.nv),):
            raise WarpBatchError("calibrated qpos/qvel dimensions do not match the MuJoCo-Warp model")
        if calibration.nominal_control.shape != (batch.num_actuators,):
            raise WarpBatchError("calibrated nominal-control dimension does not match the MuJoCo-Warp model")
        # ``repeat`` is construction-only; reset subsequently reuses these
        # contiguous device buffers without CPU state construction or copies.
        self._reset_qpos = torch.as_tensor(calibration.qpos, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_worlds, 1)
        self._reset_qvel = torch.as_tensor(calibration.qvel, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_worlds, 1)
        self._calibrated_nominal_controls = torch.as_tensor(
            calibration.nominal_control, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_worlds, 1)
        self._residual_limits = torch.as_tensor(
            self.config.residual_limits, dtype=torch.float32, device=self.device
        )
        self._policy_action_enabled = torch.ones(ACTION_SIZE, dtype=torch.float32, device=self.device)
        if not self.config.leg_action_enabled:
            # The seventh channel only changes a reference, not a physical
            # actuator.  Keep its PPO authority disabled until a safe GPU leg
            # controller consumes it.
            self._policy_action_enabled[-1] = 0.0
        # WarpPhysicsBatch owns an independent substep fall guard.  Its fixed
        # reference is set once from the calibrated stance, never per reset.
        batch.set_fall_guard_reference(
            self._reset_qpos[0, self.root_qpos_address + 3 : self.root_qpos_address + 7].contiguous(),
            float(calibration.qpos[self.root_qpos_address + 2]),
        )
        self._command_speed = torch.full((self.num_worlds,), float(self.config.command_speed_mps), dtype=torch.float32, device=self.device)
        self._command_yaw_rate = torch.full((self.num_worlds,), float(self.config.command_yaw_rate_rad_s), dtype=torch.float32, device=self.device)
        default_leg = self.config.command_leg_length_m
        if default_leg is None:
            default_leg = 0.5 * (self.config.leg_length_min_m + self.config.leg_length_max_m)
        self._command_leg_length = torch.full((self.num_worlds,), float(default_leg), dtype=torch.float32, device=self.device)
        self._reference_quaternion = torch.zeros((self.num_worlds, 4), dtype=torch.float32, device=self.device)
        self._reference_yaw = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        self._reference_height = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        # The robot's rolling direction is perpendicular to its wheel axle,
        # not necessarily world +X.  Keep persistent CUDA buffers so commands,
        # reward, and route adapters share the same physical heading.
        self._forward_direction = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._forward_speed = torch.empty(self.num_worlds, dtype=torch.float32, device=self.device)
        self._stance_leg_lengths = torch.zeros((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._previous_action = torch.zeros((self.num_worlds, ACTION_SIZE), dtype=torch.float32, device=self.device)
        self._action_delay_buffer = torch.zeros(
            (self.num_worlds, max(1, self.config.control_delay_steps + 1), ACTION_SIZE),
            dtype=torch.float32,
            device=self.device,
        )
        self._delayed_action = torch.empty_like(self._previous_action)
        self._contact_loss_steps = torch.zeros((self.num_worlds, 2), dtype=torch.int32, device=self.device)
        self._episode_done = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._jump_request = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        self._all_world_mask = torch.ones(self.num_worlds, dtype=torch.bool, device=self.device)
        self._last_unsafe = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._action_nonfinite = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._safety_terminated = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._safety_reason_code = torch.zeros(self.num_worlds, dtype=torch.int64, device=self.device)
        self._safe_requested_controls = torch.zeros(
            (self.num_worlds, batch.num_actuators), dtype=torch.float32, device=self.device
        )
        self._safe_action = torch.empty(
            (self.num_worlds, ACTION_SIZE), dtype=torch.float32, device=self.device
        )
        self._effective_action = torch.empty(
            (self.num_worlds, ACTION_SIZE), dtype=torch.float32, device=self.device
        )
        self._direct_control_buffer = torch.zeros(
            (self.num_worlds, batch.num_actuators), dtype=torch.float32, device=self.device
        )
        self._residual_control_buffer = torch.empty(
            (self.num_worlds, batch.num_actuators), dtype=torch.float32, device=self.device
        )
        self._safe_applied_forces = torch.zeros(
            (self.num_worlds, int(model.nv)), dtype=torch.float32, device=self.device
        )
        self._last_observation = torch.zeros((self.num_worlds, OBSERVATION_SIZE), dtype=torch.float32, device=self.device)
        self._observation_noise = torch.zeros_like(self._last_observation)
        self._observation_noise_mask = torch.zeros(OBSERVATION_SIZE, dtype=torch.float32, device=self.device)
        for feature_slice in (
            self.layout.orientation,
            self.layout.world_velocity,
            self.layout.body_angular_velocity,
            self.layout.hip_position,
            self.layout.hip_velocity,
            self.layout.wheel_velocity,
            self.layout.leg_length,
            self.layout.leg_length_velocity,
            self.layout.yaw_state,
            self.layout.jump_height,
            self.layout.contacts,
        ):
            self._observation_noise_mask[feature_slice] = 1.0
        self._noise_generator = torch.Generator(device=self.device)
        self._noise_generator.manual_seed(int(self.config.domain_randomization_seed))
        self._reward_terms: dict[str, Any] = {}
        self._terrain_features = torch.zeros((self.num_worlds, TERRAIN_FEATURE_SIZE), dtype=torch.float32, device=self.device)
        self._terrain_features[:, -1] = 1.0
        self._physics_time_step = float(batch.host_model.opt.timestep)
        self._time_step = self._physics_time_step * batch.config.physics_substeps_per_action
        # The contact timeout is a physical safety gate, so it is counted at
        # the one-millisecond physics cadence rather than policy cadence.
        self._contact_loss_limit_steps = max(
            1,
            int(math.ceil(self.config.contact_loss_timeout_seconds / self._physics_time_step)),
        )
        self._safety_limits = WarpSafetyLimits(
            torque_fraction_of_rated=batch.config.safety.torque_fraction_of_rated,
            estop_on_nonfinite_control=batch.config.safety.estop_on_nonfinite_control,
            estop_on_nonfinite_state=batch.config.safety.estop_on_nonfinite_state,
            estop_on_overflow=batch.config.safety.estop_on_overflow,
            fall_guard_enabled=batch.config.fall_guard.enabled,
            max_attitude_error_rad=batch.config.fall_guard.max_attitude_error_rad,
            max_root_height_drop_m=batch.config.fall_guard.max_root_height_drop_m,
            max_leg_length_difference_m=self.config.max_leg_length_difference_m,
            min_leg_length_m=self.config.safety_leg_length_min_m,
            max_leg_length_m=self.config.safety_leg_length_max_m,
            max_contact_loss_steps=self._contact_loss_limit_steps,
        )
        # All task-level P0 outputs are reused for each pre/post-substep
        # evaluation.  ``_latch_safety`` copies the values before the next
        # evaluation, so result aliases are safe within the current substep.
        self._safety_scratch = WarpSafetyScratch(
            worlds=self.num_worlds,
            action_dim=batch.num_actuators,
            device=self.device,
            dtype=torch.float32,
        )
        self._previous_estopped = torch.empty(self.num_worlds, dtype=torch.bool, device=self.device)
        self._safety_leg_lengths = torch.empty(
            (self.num_worlds, self._leg_length_indices.numel()),
            dtype=torch.float32,
            device=self.device,
        )
        self._safety_joint_positions = torch.empty(
            (self.num_worlds, self._limited_joint_qpos_gpu.numel()),
            dtype=torch.float32,
            device=self.device,
        )
        self.reset()

    @property
    def reward_terms(self) -> Mapping[str, Any]:
        """Individually inspectable, GPU-resident reward components."""

        return self._reward_terms

    def _apply_domain_randomization_reset(self, mask: Any) -> None:
        """Commit/reset episode-constant model DR outside the substep loop."""

        if self.config.domain_randomization_enabled:
            if not self.batch.domain_randomization_enabled:
                raise WarpBatchError(
                    "task domain_randomization_enabled requires a batch domain_randomization block"
                )
            self.batch.sample_domain_randomization(mask)
        elif self.batch.domain_randomization_enabled:
            self.batch.reset_domain_randomization(mask)

    def _delay_action(self, action: Any) -> Any:
        """Apply the configured CUDA-resident policy-period delay."""

        delay = int(self.config.control_delay_steps)
        if delay <= 0:
            self._delayed_action.copy_(action)
            return self._delayed_action
        self._action_delay_buffer[:, 1:].copy_(self._action_delay_buffer[:, :-1])
        self._action_delay_buffer[:, 0].copy_(action)
        self._delayed_action.copy_(self._action_delay_buffer[:, delay])
        return self._delayed_action

    def _require_action(self, action: Any) -> Any:
        torch = self.torch
        if not isinstance(action, torch.Tensor):
            raise TypeError("action must be a CUDA torch.Tensor")
        if action.shape != (self.num_worlds, ACTION_SIZE):
            raise ValueError(f"action must have shape {(self.num_worlds, ACTION_SIZE)}, got {tuple(action.shape)}")
        if action.device != self.device or action.dtype != torch.float32 or not action.is_contiguous():
            raise ValueError("action must be contiguous float32 on the configured CUDA device")
        finite = torch.isfinite(action).all(dim=1)
        self._action_nonfinite.copy_(~finite)
        torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0, out=self._safe_action)
        torch.clamp(self._safe_action, -1.0, 1.0, out=self._safe_action)
        return self._safe_action

    def _forward_direction_and_speed(self, quaternion: Any | None = None) -> tuple[Any, Any]:
        """Return GPU rolling direction and velocity projected onto it.

        The CPU controller defines forward as ``world_up x wheel_axle``.  A
        root quaternion's local X axis is the wheel axle in this MJCF, so the
        horizontal projection is ``(-axle_y, axle_x)``.  Treating sensor X as
        forward hid command-tracking errors whenever a route started at a
        non-zero yaw.
        """

        torch = self.torch
        if quaternion is None:
            quaternion = _normalize_quaternion(
                torch,
                self.batch.qpos[:, self.root_qpos_address + 3 : self.root_qpos_address + 7],
            )
        w, x, y, z = quaternion.unbind(dim=-1)
        axle_x = 1.0 - 2.0 * (y.square() + z.square())
        axle_y = 2.0 * (x * y + z * w)
        self._forward_direction[:, 0].copy_(-axle_y)
        self._forward_direction[:, 1].copy_(axle_x)
        self._forward_direction.div_(
            torch.linalg.vector_norm(self._forward_direction, dim=1, keepdim=True).clamp_min(1.0e-7)
        )
        velocity = self.batch.sensordata[:, self._velocity_slice]
        self._forward_speed.copy_((velocity[:, :2] * self._forward_direction).sum(dim=1))
        return self._forward_direction, self._forward_speed

    def forward_direction(self) -> Any:
        """Expose the current horizontal rolling direction on CUDA."""

        direction, _ = self._forward_direction_and_speed()
        return direction

    def forward_speed(self) -> Any:
        """Expose the current signed rolling speed on CUDA."""

        _, speed = self._forward_direction_and_speed()
        return speed

    def _require_mask(self, world_mask: Any | None) -> Any:
        torch = self.torch
        if world_mask is None:
            return self._all_world_mask
        if not isinstance(world_mask, torch.Tensor) or world_mask.shape != (self.num_worlds,):
            raise ValueError(f"world_mask must have shape {(self.num_worlds,)}")
        if world_mask.device != self.device or world_mask.dtype != torch.bool or not world_mask.is_contiguous():
            raise ValueError("world_mask must be contiguous CUDA bool")
        return world_mask

    def set_commands(self, command_speed: Any, command_yaw_rate: Any, world_mask: Any | None = None) -> None:
        """Update high-level commands on selected worlds without host transfer."""

        torch = self.torch
        mask = self._require_mask(world_mask)
        for value, name, limit in (
            (command_speed, "command_speed", self.config.command_speed_limit_mps),
            (command_yaw_rate, "command_yaw_rate", self.config.command_yaw_rate_limit_rad_s),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_worlds,):
                raise ValueError(f"{name} must be a CUDA tensor with shape {(self.num_worlds,)}")
            if value.device != self.device or value.dtype != torch.float32 or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous float32 on CUDA")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains non-finite values")
            if bool((value.abs() > limit).any().item()):
                raise ValueError(f"{name} exceeds configured limit {limit}")
        self._command_speed.masked_scatter_(mask, command_speed[mask])
        self._command_yaw_rate.masked_scatter_(mask, command_yaw_rate[mask])

    def set_terrain_features(self, terrain_features: Any) -> None:
        """Copy a verified GPU terrain feature batch into the observation buffer.

        Terrain providers live outside this flat task so the task itself never
        infers unsupported geometry.  The provider must use the same fixed
        16-feature contract and CUDA device; callers can then invoke
        :meth:`observe` to publish the updated features.
        """

        torch = self.torch
        if not isinstance(terrain_features, torch.Tensor):
            raise TypeError("terrain_features must be a torch.Tensor")
        expected = (self.num_worlds, TERRAIN_FEATURE_SIZE)
        if terrain_features.shape != expected:
            raise ValueError(f"terrain_features must have shape {expected}")
        if (
            terrain_features.device != self.device
            or terrain_features.dtype != torch.float32
            or not terrain_features.is_contiguous()
        ):
            raise ValueError("terrain_features must be contiguous float32 on the task CUDA device")
        torch.nan_to_num(terrain_features, nan=0.0, posinf=0.0, neginf=0.0, out=self._terrain_features)
        torch.clamp(self._terrain_features, -10.0, 10.0, out=self._terrain_features)

    def set_feedback_controller(self, controller: Any) -> None:
        """Attach the CUDA feedback controller used by residual PPO.

        Construction is intentionally two-phase: the task owns model/sensor
        metadata, while a controller binds to that task's resident state.
        The controller is checked on the first substep as part of the normal
        fail-closed output contract.
        """

        if self.config.direct_control_mode:
            raise WarpBatchError("a feedback controller is not used in diagnostic direct-control mode")
        if controller is None:
            raise ValueError("controller must not be None in residual mode")
        self._controller = controller

    def reset(self, world_mask: Any | None = None) -> Any:
        """Reset selected worlds and return their current 67-D observation."""

        torch = self.torch
        mask = self._require_mask(world_mask)
        # This deliberately avoids ``set_integration_state`` and its first-use
        # Windows NVRTC kernel.  ``reset_to_state`` only writes preallocated
        # CUDA views, clears the batch estop, and forwards the existing model.
        self.batch.reset_to_state(
            self._reset_qpos,
            self._reset_qvel,
            self._calibrated_nominal_controls,
            mask,
        )
        self._apply_domain_randomization_reset(mask)
        qpos = self.batch.qpos
        sensordata = self.batch.sensordata
        quaternion = _normalize_quaternion(torch, qpos[:, self.root_qpos_address + 3 : self.root_qpos_address + 7])
        self._reference_quaternion[mask] = quaternion[mask]
        reference_forward, _ = self._forward_direction_and_speed(quaternion)
        self._reference_yaw[mask] = torch.atan2(reference_forward[mask, 1], reference_forward[mask, 0])
        self._reference_height[mask] = qpos[mask, self.root_qpos_address + 2]
        leg_lengths = sensordata[:, self._leg_length_indices]
        self._stance_leg_lengths[mask] = leg_lengths[mask]
        if self.config.command_leg_length_m is None:
            self._command_leg_length[mask] = leg_lengths[mask].mean(dim=1)
        else:
            self._command_leg_length[mask] = float(self.config.command_leg_length_m)
        self._previous_action[mask] = 0.0
        self._action_delay_buffer[mask] = 0.0
        self._delayed_action[mask] = 0.0
        self._observation_noise[mask] = 0.0
        self._contact_loss_steps[mask] = 0
        self._episode_done[mask] = False
        self._last_unsafe[mask] = False
        self._action_nonfinite[mask] = False
        self._safety_terminated[mask] = False
        self._safety_reason_code[mask] = 0
        self._safe_requested_controls[mask] = 0.0
        self._safe_applied_forces[mask] = 0.0
        self._jump_request[mask] = 0.0
        reset_controller = getattr(self._controller, "reset", None)
        if callable(reset_controller):
            reset_controller(mask)
        return self.observe()

    def _sensor_values(self) -> tuple[Any, Any, Any, Any, Any]:
        sensor = self.batch.sensordata
        return (
            sensor[:, self._velocity_slice],
            sensor[:, self._angular_slice],
            sensor.index_select(1, self._wheel_velocity_indices),
            sensor.index_select(1, self._leg_length_indices),
            sensor.index_select(1, self._leg_velocity_indices),
        )

    def _orientation_error(self) -> tuple[Any, Any, Any]:
        torch = self.torch
        qpos = self.batch.qpos
        current = _normalize_quaternion(torch, qpos[:, self.root_qpos_address + 3 : self.root_qpos_address + 7])
        forward, _ = self._forward_direction_and_speed(current)
        current_yaw = torch.atan2(forward[:, 1], forward[:, 0])
        yaw_delta = _wrap_pi(torch, current_yaw - self._reference_yaw)
        half = 0.5 * yaw_delta
        yaw_rotation = torch.stack((torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)), dim=-1)
        aligned_reference = _quat_multiply(torch, yaw_rotation, self._reference_quaternion)
        relative = _quat_multiply(torch, current, _quat_conjugate(torch, aligned_reference))
        return _rotation_vector_from_quaternion(torch, relative), current_yaw, current

    def _wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        torch = self.torch
        torch.index_select(self._geom_xpos, 1, self._wheel_geom_gpu, out=self._wheel_positions)
        torch.sub(
            self._wheel_positions[..., 2],
            self._wheel_radius,
            out=self._wheel_clearances,
        )
        self._wheel_clearances.sub_(self._ground_height).clamp_(min=0.0)
        torch.le(
            self._wheel_clearances,
            self.config.contact_clearance_m,
            out=self._wheel_contacts,
        )
        return self._wheel_clearances, self._wheel_contacts

    def _guide_wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        """Return private guide-wheel plane clearances/contact indicators."""

        torch = self.torch
        torch.index_select(
            self._geom_xpos,
            1,
            self._guide_wheel_geom_gpu,
            out=self._guide_wheel_positions,
        )
        torch.sub(
            self._guide_wheel_positions[..., 2],
            self._guide_wheel_radius,
            out=self._guide_wheel_clearances,
        )
        self._guide_wheel_clearances.sub_(self._ground_height).clamp_(min=0.0)
        torch.le(
            self._guide_wheel_clearances,
            self.config.contact_clearance_m,
            out=self._guide_wheel_contacts,
        )
        return self._guide_wheel_clearances, self._guide_wheel_contacts

    def _side_support_contacts(self) -> Any:
        """Return private per-side support while preserving public active contacts."""

        _, active_contacts = self._wheel_clearances_and_contacts()
        _, guide_contacts = self._guide_wheel_clearances_and_contacts()
        return combine_side_support_contacts(
            self.torch,
            active_contacts,
            guide_contacts,
            self._guide_side_contacts,
            self._support_contacts,
        )

    def observe(self) -> Any:
        """Build the full 67-D observation entirely on CUDA."""

        torch = self.torch
        qpos = self.batch.qpos
        qvel = self.batch.qvel
        velocity, angular_velocity, wheel_velocity, leg_lengths, leg_velocity = self._sensor_values()
        orientation_error, current_yaw, _ = self._orientation_error()
        yaw_error = _wrap_pi(torch, self._reference_yaw - current_yaw)
        root_yaw_rate = qvel[:, self.root_dof_address + 5]
        clearances, contacts = self._wheel_clearances_and_contacts()
        body_height = qpos[:, self.root_qpos_address + 2]
        body_rise = torch.clamp(body_height - self._reference_height, min=0.0)
        terrain = self._terrain_features

        # ``_last_observation`` is the public CUDA observation buffer.  Every
        # slice is overwritten below (including disabled jump fields), so it
        # can be safely reused without a per-observe ``torch.zeros``.
        obs = self._last_observation
        obs[:, self.layout.orientation] = orientation_error / MAX_ATTITUDE_ERROR_RAD
        obs[:, self.layout.world_velocity] = velocity / 2.0
        obs[:, self.layout.body_angular_velocity] = angular_velocity / 10.0
        obs[:, self.layout.hip_position] = qpos.index_select(1, self._hip_qpos_gpu) / 2.5
        obs[:, self.layout.hip_velocity] = qvel.index_select(1, self._hip_qvel_gpu) / 15.0
        obs[:, self.layout.wheel_velocity] = wheel_velocity / 50.0
        obs[:, self.layout.leg_length] = (leg_lengths - self._stance_leg_lengths) / 0.15
        obs[:, self.layout.leg_length_velocity] = leg_velocity / 3.0
        obs[:, self.layout.command_speed] = (self._command_speed / self.config.command_speed_limit_mps).unsqueeze(1)
        default_leg = self._stance_leg_lengths.mean(dim=1)
        obs[:, self.layout.command_leg_length] = ((self._command_leg_length - default_leg) / 0.10).unsqueeze(1)
        obs[:, self.layout.command_yaw_rate] = (self._command_yaw_rate / self.config.command_yaw_rate_limit_rad_s).unsqueeze(1)
        obs[:, self.layout.jump_request] = self._jump_request.unsqueeze(1)
        yaw_state = obs[:, self.layout.yaw_state]
        yaw_state.zero_()
        yaw_state[:, 0] = yaw_error / math.pi
        yaw_state[:, 2] = torch.clamp(root_yaw_rate / MAX_YAW_RATE_RAD_S, -1.0, 1.0)
        # Jump phase/countdown are intentionally inactive in flat walking.
        obs[:, self.layout.jump_phase] = 0.0
        jump_height = obs[:, self.layout.jump_height]
        jump_height[:, 0] = clearances[:, 0] / DEFAULT_WHEEL_CLEARANCE_NORMALIZATION_M
        jump_height[:, 1] = clearances[:, 1] / DEFAULT_WHEEL_CLEARANCE_NORMALIZATION_M
        jump_height[:, 2] = clearances.mean(dim=1) / DEFAULT_WHEEL_CLEARANCE_NORMALIZATION_M
        jump_height[:, 3] = body_rise / DEFAULT_BODY_RISE_NORMALIZATION_M
        jump_height[:, 4] = qvel[:, self.root_dof_address + 2] / 3.0
        obs[:, self.layout.terrain] = terrain
        obs[:, self.layout.contacts] = contacts.to(torch.float32)
        obs[:, self.layout.previous_action] = self._previous_action
        torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0, out=obs)
        torch.clamp(obs, -10.0, 10.0, out=obs)
        self._observation_noise.zero_()
        if self.config.domain_randomization_enabled and self.config.sensor_noise_std > 0.0:
            self._observation_noise.normal_(
                0.0,
                float(self.config.sensor_noise_std),
                generator=self._noise_generator,
            )
            self._observation_noise.mul_(self._observation_noise_mask.unsqueeze(0))
            obs.add_(self._observation_noise).clamp_(-10.0, 10.0)
        return obs

    def _direct_controls(self, action: Any) -> Any:
        controls = self._direct_control_buffer
        controls.copy_(action[:, : self.batch.num_actuators])
        controls.add_(1.0).mul_(0.5)
        controls.mul_(self._control_span.unsqueeze(0)).add_(self._control_low.unsqueeze(0))
        return controls

    def _controller_nominal_controls(self) -> Any:
        """Read a GPU feedback baseline without defining controller policy here.

        The injected controller must be callable as ``controller(task)`` or
        expose ``compute_controls(task)`` and return contiguous ``[N, 6]``
        float32 CUDA actuator controls.  This narrow interface leaves the
        task responsible for residual bounds and safety while the controller
        owns its state-feedback law.
        """

        controller = self._controller
        if controller is None:
            raise WarpBatchError(
                "flat PPO residual mode requires an injected GPU feedback controller; "
                "static calibrated equilibrium is reset-only, not a walking baseline"
            )
        compute = getattr(controller, "compute_controls", controller)
        if not callable(compute):
            raise TypeError("controller must be callable or expose compute_controls(task)")
        controls = compute(self)
        if not isinstance(controls, self.torch.Tensor) or controls.shape != (self.num_worlds, self.batch.num_actuators):
            raise ValueError(
                f"controller output must have shape {(self.num_worlds, self.batch.num_actuators)}"
            )
        if controls.device != self.device or controls.dtype != self.torch.float32 or not controls.is_contiguous():
            raise ValueError("controller output must be contiguous float32 on the configured CUDA device")
        return controls

    def _controller_applied_forces(self, safe_controls: Any) -> Any:
        """Return the controller's force buffer after actuator safety clipping.

        Residual walking requires a feedback controller that explicitly owns
        the gas-spring generalized forces.  Passing ``None`` would cause the
        batch to clear that force buffer, so missing support fails closed. The
        final actuator command is supplied so the controller can reserve
        combined joint-torque headroom for the generalized gas-spring force.
        """

        controller = self._controller
        compute = getattr(controller, "applied_generalized_forces", None)
        if not callable(compute):
            raise WarpBatchError(
                "flat PPO feedback controller must provide applied_generalized_forces(task)"
            )
        forces = compute(self, safe_controls=safe_controls)
        expected = (self.num_worlds, int(self.batch.host_model.nv))
        if not isinstance(forces, self.torch.Tensor) or forces.shape != expected:
            raise ValueError(f"controller generalized force output must have shape {expected}")
        if forces.device != self.device or forces.dtype != self.torch.float32 or not forces.is_contiguous():
            raise ValueError("controller generalized forces must be contiguous float32 on CUDA")
        self._safe_applied_forces.copy_(forces)
        self._safe_applied_forces.masked_fill_(self._safety_terminated.unsqueeze(1), 0.0)
        return self._safe_applied_forces

    def _residual_controls(self, action: Any, nominal_controls: Any) -> Any:
        """Apply the six configured normalized residual caps on CUDA."""

        torch = self.torch
        torch.mul(
            action[:, : self.batch.num_actuators],
            self._residual_limits[: self.batch.num_actuators],
            out=self._residual_control_buffer,
        )
        torch.add(nominal_controls, self._residual_control_buffer, out=self._residual_control_buffer)
        return self._residual_control_buffer

    def _update_contact_loss(self, contacts: Any) -> None:
        self._contact_loss_steps.masked_fill_(contacts, 0)
        self._contact_loss_steps.add_((~contacts).to(dtype=self.torch.int32))

    def _evaluate_safety(self, controls: Any) -> WarpSafetyResult:
        """Run task-level P0 checks using only resident CUDA tensors."""

        contacts = self._side_support_contacts()
        torch = self.torch
        torch.index_select(
            self.batch.sensordata,
            1,
            self._leg_length_indices,
            out=self._safety_leg_lengths,
        )
        joint_positions = None
        joint_lower = None
        joint_upper = None
        if self._limited_joint_qpos_gpu.numel() > 0:
            torch.index_select(
                self.batch.qpos,
                1,
                self._limited_joint_qpos_gpu,
                out=self._safety_joint_positions,
            )
            joint_positions = self._safety_joint_positions
            joint_lower = self._limited_joint_lower
            joint_upper = self._limited_joint_upper
        torch.logical_or(
            self._safety_terminated,
            self.batch.estopped,
            out=self._previous_estopped,
        )
        return evaluate_safety(
            self.batch.qpos,
            self.batch.qvel,
            controls,
            self.batch.overflow,
            root_qpos_address=self.root_qpos_address,
            reference_quaternion=self._reference_quaternion,
            reference_root_height_m=self._reference_height,
            control_low=self._control_low,
            control_high=self._control_high,
            limits=self._safety_limits,
            previous_estopped=self._previous_estopped,
            previous_reason_code=self._safety_reason_code,
            sensordata=self.batch.sensordata,
            joint_positions=joint_positions,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            leg_lengths=self._safety_leg_lengths,
            wheel_contact=contacts,
            contact_loss_steps=self._contact_loss_steps,
            safe_controls_out=self._safe_requested_controls,
            scratch=self._safety_scratch,
        )

    def _latch_safety(self, result: WarpSafetyResult, *, action_nonfinite: Any | None = None) -> Any:
        """Persist task safety state and zero controls for malformed actions."""

        terminated = result.terminated
        reason_code = result.reason_code
        if action_nonfinite is not None:
            terminated = terminated | action_nonfinite
            self._safe_requested_controls.masked_fill_(action_nonfinite.unsqueeze(1), 0.0)
            reason_code = self.torch.where(
                action_nonfinite,
                self.torch.full_like(reason_code, SAFETY_REASON_NONFINITE_CONTROL),
                reason_code,
            )
        self._safety_terminated.copy_(terminated)
        self._safety_reason_code.copy_(reason_code)
        self._safe_requested_controls.masked_fill_(self._safety_terminated.unsqueeze(1), 0.0)
        self._safe_applied_forces.masked_fill_(self._safety_terminated.unsqueeze(1), 0.0)
        # Task checks include legs and wheel support, which the generic batch
        # does not know.  Latch them into the independent batch estop now so
        # its persistent MuJoCo control/force views are torque-free even when
        # this was the final substep of a policy interval.
        self.batch.latch_estop(self._safety_terminated)
        return terminated

    def _termination(self) -> tuple[Any, Any, Any]:
        unsafe = self._safety_terminated | self.batch.estopped
        self._last_unsafe.copy_(unsafe)
        terminated = self._episode_done | unsafe
        elapsed = self.batch.time >= self.config.episode_seconds
        truncated = elapsed & ~terminated
        done = terminated | truncated
        self._episode_done |= done
        return terminated, truncated, done

    def _reward(self, controls: Any, action: Any, unsafe: Any) -> Any:
        torch = self.torch
        velocity, _, _, leg_lengths, _ = self._sensor_values()
        orientation_error, current_yaw, _ = self._orientation_error()
        yaw_error = _wrap_pi(torch, self._reference_yaw - current_yaw)
        measured_yaw_rate = self.batch.qvel[:, self.root_dof_address + 5]
        _, forward_speed = self._forward_direction_and_speed()
        speed_error = forward_speed - self._command_speed
        tracking = torch.exp(-torch.square(speed_error / 0.20))
        leg_error = leg_lengths.mean(dim=1) - self._command_leg_length
        leg_tracking = torch.exp(-torch.square(leg_error / 0.05))
        attitude_tracking = torch.exp(-torch.square(orientation_error).sum(dim=1))
        yaw_tracking = torch.exp(-torch.square(yaw_error / 0.10))
        yaw_rate_error = torch.clamp(
            (measured_yaw_rate - self._command_yaw_rate) / MAX_YAW_RATE_RAD_S,
            -1.0,
            1.0,
        )
        energy_cost = torch.mean(torch.square(controls / self._control_scale.unsqueeze(0)), dim=1)
        residual_cost = torch.mean(torch.square(action), dim=1)
        contacts = self._side_support_contacts()
        contact_bonus = torch.where(contacts.all(dim=1), 0.10, -0.15)
        turn_intensity = torch.clamp(
            torch.maximum(
                (self._command_yaw_rate.abs() / MAX_YAW_RATE_RAD_S),
                (measured_yaw_rate.abs() / MAX_YAW_RATE_RAD_S),
            ),
            0.0,
            1.0,
        )
        leg_diff_weight = 0.5 + (2.0 - 0.5) * (1.0 - turn_intensity)
        leg_diff = torch.abs(leg_lengths[:, 0] - leg_lengths[:, 1])
        reward = (
            0.65 * tracking
            + 0.30 * leg_tracking
            + 0.30 * attitude_tracking
            + contact_bonus
            + 0.25 * yaw_tracking
        )
        reward = reward - (
            0.03 * energy_cost
            + 0.02 * residual_cost
            + 0.05 * torch.square(yaw_rate_error)
            + leg_diff_weight * leg_diff
        )
        unsafe_penalty = torch.where(
            unsafe,
            torch.full_like(reward, 30.0),
            torch.zeros_like(reward),
        )
        self._reward_terms.update({
            "speed_tracking": 0.65 * tracking,
            "leg_tracking": 0.30 * leg_tracking,
            "attitude_tracking": 0.30 * attitude_tracking,
            "contact": contact_bonus,
            "yaw_tracking": 0.25 * yaw_tracking,
            "energy_cost": -0.03 * energy_cost,
            "residual_cost": -0.02 * residual_cost,
            "yaw_rate_cost": -0.05 * torch.square(yaw_rate_error),
            "leg_symmetry_cost": -leg_diff_weight * leg_diff,
            "unsafe_penalty": -unsafe_penalty,
        })
        return reward - unsafe_penalty

    def step(self, action: Any, controls: Any | None = None) -> WarpTaskStep:
        """Advance physics and return CUDA observation/reward/done tensors.

        In residual mode ``controls`` is a GPU feedback-controller *nominal*
        in MuJoCo actuator units; the first six normalized policy channels are
        clipped by ``residual_limits`` around it.  Omitting it invokes the
        injected controller.  In explicitly enabled diagnostic direct mode it
        is instead a raw physical command and omitted controls map the action
        through the derated actuator range.
        """

        torch = self.torch
        action = self._require_action(action)
        torch.mul(action, self._policy_action_enabled, out=self._effective_action)
        effective_action = self._delay_action(self._effective_action)
        supplied_controls = controls
        if supplied_controls is not None:
            if not isinstance(supplied_controls, torch.Tensor) or supplied_controls.shape != (self.num_worlds, 6):
                raise ValueError(f"controls must have shape {(self.num_worlds, 6)}")
            if (
                supplied_controls.device != self.device
                or supplied_controls.dtype != torch.float32
                or not supplied_controls.is_contiguous()
            ):
                raise ValueError("controls must be contiguous float32 on CUDA")
        if self.config.leg_action_enabled:
            self._command_leg_length.add_(
                effective_action[:, 6]
                * self._residual_limits[6]
                * self.config.leg_command_rate_mps
                * self._time_step
            ).clamp_(self.config.leg_length_min_m, self.config.leg_length_max_m)

        # Recompute feedback and P0 checks at every physics tick.  Holding a
        # nominal torque for the whole policy interval destabilizes this
        # closed-chain model and would hide a substep contact/fall event.
        physics = None
        for _ in range(self.batch.config.physics_substeps_per_action):
            controller_owns_forces = False
            if supplied_controls is not None:
                nominal_controls = supplied_controls
            elif self.config.direct_control_mode:
                nominal_controls = self._direct_controls(effective_action)
            else:
                nominal_controls = self._controller_nominal_controls()
                controller_owns_forces = True

            requested_controls = (
                nominal_controls
                if self.config.direct_control_mode
                else self._residual_controls(effective_action, nominal_controls)
            )
            # Clamp/estop before the current physical substep.  A malformed
            # policy action is separately latched because ``_require_action``
            # sanitizes it before residual arithmetic can contaminate CUDA.
            pre_safety = self._evaluate_safety(requested_controls)
            self._latch_safety(pre_safety, action_nonfinite=self._action_nonfinite)
            if controller_owns_forces:
                applied_forces = self._controller_applied_forces(self._safe_requested_controls)
                applied_forces.masked_fill_(self._safety_terminated.unsqueeze(1), 0.0)
            else:
                self._safe_applied_forces.zero_()
                applied_forces = self._safe_applied_forces
            physics = self.batch.step(
                self._safe_requested_controls,
                physics_substeps=1,
                applied_forces=applied_forces,
            )

            # MuJoCo-Warp integration does not promise fresh sensors/geometry.
            # Refresh the existing GPU model/data views before every post-step
            # contact and leg safety evaluation; this does not construct data.
            self.batch.forward()
            # Contact is intentionally evaluated after every forward step.
            # This flat proxy is clearance-based; terrain tasks must replace
            # it with a verified contact-manifold provider.
            contacts = self._side_support_contacts()
            self._update_contact_loss(contacts)
            post_safety = self._evaluate_safety(self.batch._safe_controls)
            self._latch_safety(post_safety)

        if physics is None:  # pragma: no cover - config rejects zero substeps
            raise RuntimeError("flat task did not execute a physics substep")
        terminated, truncated, done = self._termination()
        # MuJoCo-Warp sanitizes/derates controls in ``_safe_controls`` before
        # writing them to the model.  Score that physical command, never a
        # malformed raw tensor that may contain NaN/Inf.
        reward = self._reward(self.batch._safe_controls, effective_action, self._last_unsafe)
        self._previous_action.copy_(effective_action)
        observation = self.observe()
        return WarpTaskStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            done=done,
            physics=physics,
        )

    def step_policy(self, action: Any) -> Any:
        """Adapt ``step`` to the CUDA PPO collector's vector-env contract."""

        from warp_ppo import WarpVectorStep

        result = self.step(action)
        # A finite physical terminal transition is still a valid actor sample:
        # its action led to the safety outcome and PPO must retain the six
        # actuator channels.  The collector owns episode-boundary handling
        # through its continuation mask.  Only a malformed input action loses
        # actor authority here; the seventh leg-reference channel stays
        # disabled by ``_policy_action_enabled``.
        policy_action_masks = (
            (~self._action_nonfinite).to(dtype=self.torch.float32).unsqueeze(1)
            * self._policy_action_enabled.unsqueeze(0)
        ).contiguous()
        return WarpVectorStep(
            observations=result.observation,
            rewards=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            policy_action_masks=policy_action_masks,
        )

    def tensors(self) -> Mapping[str, Any]:
        """Expose rollout tensors without copying them to CPU."""

        return {
            "observation": self._last_observation,
            "previous_action": self._previous_action,
            "command_speed": self._command_speed,
            "command_yaw_rate": self._command_yaw_rate,
            "command_leg_length": self._command_leg_length,
            "terminated": self._episode_done,
            "safety_terminated": self._safety_terminated,
            "safety_reason_code": self._safety_reason_code,
            "calibrated_nominal_controls": self._calibrated_nominal_controls,
            "guide_wheel_contacts": self._guide_wheel_contacts,
            "support_contacts": self._support_contacts,
        }


__all__ = [
    "ACTION_SIZE",
    "GUIDE_WHEEL_CONTACT_GEOM_NAMES",
    "OBSERVATION_SIZE",
    "OBS_LAYOUT",
    "WarpFlatWalkingConfig",
    "WarpFlatStanceCalibration",
    "WarpFlatWalkingTask",
    "WarpObservationLayout",
    "WarpTaskStep",
    "calibrate_flat_stance",
    "combine_side_support_contacts",
    "load_flat_walking_config",
]
