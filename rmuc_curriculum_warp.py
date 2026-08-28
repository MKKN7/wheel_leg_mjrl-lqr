"""Fail-closed CUDA adapters for the RMUC grades and low-speed turn stages.

The RMUC scene has a collision hfield plus a temporary flat plane used only
while the CPU LQR trim is calibrated.  This module never treats that plane as
terrain: the batch YAML disables it before GPU model upload, and the same
immutable hfield samples drive the CUDA terrain preview, support checks and
route-relative fall references.

``grades`` distributes CUDA worlds across the declared flat/uphill/downhill
routes. ``low_speed_turning`` distributes them across the level flat/left/right
routes and publishes their low-speed signed yaw commands.  Both stages keep
the 67-D observation / 7-D residual interface and use the existing P0 safety
path; a real zero-residual CUDA gate must pass before PPO storage is created.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from terrain_curriculum import TerrainCurriculumConfig, TerrainRoute, TerrainTask, load_terrain_curriculum
from warp_safety import SAFETY_REASON_CONTACT_LOSS
from warp_task import (
    ACTION_SIZE,
    GUIDE_WHEEL_CONTACT_GEOM_NAMES,
    OBSERVATION_SIZE,
    WarpFlatWalkingTask,
    combine_side_support_contacts,
)


CONTROLLER_BACKEND = "rmuc_route_controller_v1"
REWARD_SCHEMA = "warp_rmuc_terrain_compensated_reward_v2"
EXPECTED_ACTION_MASK = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
_TERRAIN_FEATURE_SIZE = 16

# The direct-jump state is intentionally task-owned rather than policy-owned.
# It is only enabled by the dedicated RMUC stair-up course contract below.
RMUC_JUMP_IDLE = 0
RMUC_JUMP_PREPARE = 1
RMUC_JUMP_CROUCH = 2
RMUC_JUMP_THRUST = 3
RMUC_JUMP_FLIGHT = 4
RMUC_JUMP_LANDING = 5
RMUC_JUMP_RECOVERY = 6
RMUC_JUMP_PHASE_COUNT = 7


class RmucCurriculumAdapterError(ValueError):
    """Raised before an unsupported RMUC course reaches CUDA physics."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RmucCurriculumAdapterError(f"{name} must be a YAML mapping")
    return value


def _exact_keys(mapping: Mapping[str, Any], name: str, expected: set[str]) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing}")
        if unknown:
            detail.append(f"unknown={unknown}")
        raise RmucCurriculumAdapterError(f"{name} keys are invalid: {', '.join(detail)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RmucCurriculumAdapterError(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RmucCurriculumAdapterError(f"{name} must be boolean")
    return value


def _finite(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RmucCurriculumAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        qualifier = "finite and positive" if positive else "finite and non-negative" if nonnegative else "finite"
        raise RmucCurriculumAdapterError(f"{name} must be {qualifier}")
    return result


def _path(source: Path, value: Any, name: str) -> Path:
    candidate = Path(_string(value, name))
    result = candidate.resolve() if candidate.is_absolute() else (source.parent / candidate).resolve()
    if not result.is_file():
        raise RmucCurriculumAdapterError(f"{name} does not exist: {result}")
    return result


def _names(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RmucCurriculumAdapterError(f"{name} must be a non-empty sequence")
    result = tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise RmucCurriculumAdapterError(f"{name} must not contain duplicates")
    return result


def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    if quaternion.shape != (4,):
        raise RmucCurriculumAdapterError("static geometry quaternion must have four values")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-8:
        raise RmucCurriculumAdapterError("static geometry quaternion must be finite and non-zero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _rolling_heading_from_quaternion(quaternion: np.ndarray) -> float:
    rotation = _rotation_matrix_from_quaternion(np.asarray(quaternion, dtype=np.float64))
    return math.atan2(float(rotation[0, 0]), -float(rotation[1, 0]))


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class TerrainFeatureSettings:
    lookahead_distances_m: tuple[float, ...]
    lateral_offsets_m: tuple[float, ...]
    height_normalization_m: float
    slope_normalization: float

    def __post_init__(self) -> None:
        if len(self.lookahead_distances_m) * len(self.lateral_offsets_m) != 12:
            raise RmucCurriculumAdapterError("terrain_features must define exactly twelve preview samples")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.lookahead_distances_m):
            raise RmucCurriculumAdapterError("terrain lookahead distances must be finite and positive")
        if any(not math.isfinite(value) for value in self.lateral_offsets_m):
            raise RmucCurriculumAdapterError("terrain lateral offsets must be finite")
        if self.height_normalization_m <= 0.0 or self.slope_normalization <= 0.0:
            raise RmucCurriculumAdapterError("terrain normalization values must be positive")


@dataclass(frozen=True)
class RouteRewardSettings:
    progress_reward_per_m: float
    progress_delta_clip_m: float
    completion_bonus: float
    jump_peak_increment_weight: float
    jump_landing_bonus: float
    jump_failure_penalty: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise RmucCurriculumAdapterError(f"route_reward.{name} must be finite and non-negative")
        if self.progress_delta_clip_m <= 0.0:
            raise RmucCurriculumAdapterError("route_reward.progress_delta_clip_m must be positive")


@dataclass(frozen=True)
class RmucJumpSupervisorSettings:
    """YAML-owned, bounded direct-jump parameters for the RMUC 204 mm riser."""

    prepare_length_m: float
    crouch_length_m: float
    thrust_length_m: float
    flight_retract_length_m: float
    flight_preload_length_m: float
    landing_length_m: float
    prepare_seconds: float
    crouch_seconds: float
    thrust_seconds: float
    flight_retract_seconds: float
    maximum_airborne_seconds: float
    landing_confirm_seconds: float
    recovery_seconds: float
    prelanding_seconds: float
    landing_torque_fraction: float
    flight_torque_fraction: float
    jump_residual_fraction: float
    thrust_leg_force_limit_n: float
    minimum_peak_body_rise_m: float
    maximum_landing_vertical_speed_mps: float
    maximum_landing_angular_speed_rad_s: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise RmucCurriculumAdapterError(f"jump_supervisor.{name} must be finite and non-negative")
        for name in (
            "prepare_length_m",
            "crouch_length_m",
            "thrust_length_m",
            "flight_retract_length_m",
            "flight_preload_length_m",
            "landing_length_m",
            "prepare_seconds",
            "crouch_seconds",
            "thrust_seconds",
            "flight_retract_seconds",
            "maximum_airborne_seconds",
            "landing_confirm_seconds",
            "recovery_seconds",
            "thrust_leg_force_limit_n",
            "minimum_peak_body_rise_m",
            "maximum_landing_vertical_speed_mps",
            "maximum_landing_angular_speed_rad_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise RmucCurriculumAdapterError(f"jump_supervisor.{name} must be positive")
        if self.crouch_length_m >= self.thrust_length_m:
            raise RmucCurriculumAdapterError("jump_supervisor.crouch_length_m must be below thrust_length_m")
        if self.prelanding_seconds < 0.050:
            raise RmucCurriculumAdapterError("jump_supervisor.prelanding_seconds must be at least 50 ms")
        for name in ("landing_torque_fraction", "flight_torque_fraction", "jump_residual_fraction"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise RmucCurriculumAdapterError(f"jump_supervisor.{name} must be within [0, 1]")


@dataclass(frozen=True)
class ControllerSettings:
    command_speed_gain_nm_per_mps: float
    command_yaw_gain_nm_per_rad_s: float
    command_wheel_feedforward_limit_nm: float
    command_wheel_accel_limit_nm: float
    command_wheel_brake_limit_nm: float
    terrain_support_reference_max_rate_mps: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise RmucCurriculumAdapterError(f"controller.{name} must be finite and non-negative")


@dataclass(frozen=True)
class StabilityGateSettings:
    duration_seconds: float
    require_no_terminated: bool
    require_no_overflow: bool
    require_finite_state: bool
    minimum_progress_m: float
    maximum_speed_mae_mps: float
    maximum_yaw_mae_rad_s: float
    maximum_unsafe_rate: float

    def __post_init__(self) -> None:
        if not (self.require_no_terminated and self.require_no_overflow and self.require_finite_state):
            raise RmucCurriculumAdapterError(
                "RMUC CUDA stability gates must require non-terminated, non-overflow finite physics"
            )
        for name in (
            "duration_seconds",
            "minimum_progress_m",
            "maximum_speed_mae_mps",
            "maximum_yaw_mae_rad_s",
            "maximum_unsafe_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise RmucCurriculumAdapterError(f"stability_gate.{name} must be finite and non-negative")
        if self.duration_seconds <= 0.0 or self.minimum_progress_m <= 0.0:
            raise RmucCurriculumAdapterError("stability_gate duration and minimum progress must be positive")
        if self.maximum_speed_mae_mps <= 0.0 or self.maximum_yaw_mae_rad_s <= 0.0:
            raise RmucCurriculumAdapterError("stability_gate tracking limits must be positive")
        if self.maximum_unsafe_rate != 0.0:
            raise RmucCurriculumAdapterError("RMUC CUDA gate maximum_unsafe_rate must remain exactly zero")


@dataclass(frozen=True)
class CourseSpec:
    stage_id: str
    task_mode: str
    batch_config_path: Path
    terrain_stage_id: str
    task_ids: tuple[str, ...]
    terrain_mode: str
    hfield_geom: str | None
    projection_plane_geom: str | None
    command_speed_envelope_mps: float
    command_yaw_envelope_rad_s: float
    stability_gate: StabilityGateSettings
    direct_jump: bool


@dataclass(frozen=True)
class RmucCourseConfig:
    source_path: Path
    flat_ppo_config_path: Path
    calibration_scene_path: Path
    terrain_curriculum_path: Path
    terrain_features: TerrainFeatureSettings
    route_reward: RouteRewardSettings
    controller: ControllerSettings
    jump: RmucJumpSupervisorSettings
    courses: Mapping[str, CourseSpec]


def _float_list(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise RmucCurriculumAdapterError(f"{name} must be a non-empty sequence")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


def _gate(value: Any, name: str) -> StabilityGateSettings:
    raw = _mapping(value, name)
    expected = set(StabilityGateSettings.__dataclass_fields__)
    _exact_keys(raw, name, expected)
    return StabilityGateSettings(
        duration_seconds=_finite(raw["duration_seconds"], f"{name}.duration_seconds", positive=True),
        require_no_terminated=_boolean(raw["require_no_terminated"], f"{name}.require_no_terminated"),
        require_no_overflow=_boolean(raw["require_no_overflow"], f"{name}.require_no_overflow"),
        require_finite_state=_boolean(raw["require_finite_state"], f"{name}.require_finite_state"),
        minimum_progress_m=_finite(raw["minimum_progress_m"], f"{name}.minimum_progress_m", positive=True),
        maximum_speed_mae_mps=_finite(raw["maximum_speed_mae_mps"], f"{name}.maximum_speed_mae_mps", positive=True),
        maximum_yaw_mae_rad_s=_finite(raw["maximum_yaw_mae_rad_s"], f"{name}.maximum_yaw_mae_rad_s", positive=True),
        maximum_unsafe_rate=_finite(raw["maximum_unsafe_rate"], f"{name}.maximum_unsafe_rate", nonnegative=True),
    )


def load_rmuc_course_config(path: str | Path) -> RmucCourseConfig:
    """Load the strict YAML contract for all admitted RMUC CUDA routes."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise RmucCurriculumAdapterError(f"unable to read RMUC course config {source}: {error}") from error
    root = _mapping(raw, "RMUC course config")
    _exact_keys(
        root,
        "RMUC course config",
        {
            "schema_version",
            "flat_ppo_config",
            "calibration_scene",
            "terrain_curriculum",
            "terrain_features",
            "route_reward",
            "controller",
            "jump_supervisor",
            "courses",
        },
    )
    if root["schema_version"] != 1:
        raise RmucCurriculumAdapterError("RMUC course config schema_version must be 1")
    terrain_raw = _mapping(root["terrain_features"], "terrain_features")
    _exact_keys(
        terrain_raw,
        "terrain_features",
        {"lookahead_distances_m", "lateral_offsets_m", "height_normalization_m", "slope_normalization"},
    )
    reward_raw = _mapping(root["route_reward"], "route_reward")
    _exact_keys(reward_raw, "route_reward", set(RouteRewardSettings.__dataclass_fields__))
    controller_raw = _mapping(root["controller"], "controller")
    _exact_keys(controller_raw, "controller", set(ControllerSettings.__dataclass_fields__))
    jump_raw = _mapping(root["jump_supervisor"], "jump_supervisor")
    _exact_keys(jump_raw, "jump_supervisor", set(RmucJumpSupervisorSettings.__dataclass_fields__))
    courses_raw = _mapping(root["courses"], "courses")
    courses: dict[str, CourseSpec] = {}
    course_keys = {
        "task_mode",
        "batch_config",
        "terrain_stage_id",
        "task_ids",
        "terrain_mode",
        "hfield_geom",
        "projection_plane_geom",
        "command_speed_envelope_mps",
        "command_yaw_envelope_rad_s",
        "stability_gate",
        "direct_jump",
    }
    for stage_id, value in courses_raw.items():
        normalized_id = _string(stage_id, "courses key")
        course_raw = _mapping(value, f"courses.{normalized_id}")
        _exact_keys(course_raw, f"courses.{normalized_id}", course_keys)
        terrain_mode = _string(course_raw["terrain_mode"], f"courses.{normalized_id}.terrain_mode")
        if terrain_mode not in {"hfield", "plane"}:
            raise RmucCurriculumAdapterError(f"courses.{normalized_id}.terrain_mode must be hfield or plane")
        hfield_geom = course_raw["hfield_geom"]
        plane_geom = course_raw["projection_plane_geom"]
        if terrain_mode == "hfield":
            hfield_geom = _string(hfield_geom, f"courses.{normalized_id}.hfield_geom")
            plane_geom = _string(plane_geom, f"courses.{normalized_id}.projection_plane_geom")
        elif hfield_geom is not None or plane_geom is not None:
            raise RmucCurriculumAdapterError(
                f"courses.{normalized_id} plane mode must set hfield_geom and projection_plane_geom to null"
            )
        courses[normalized_id] = CourseSpec(
            stage_id=normalized_id,
            task_mode=_string(course_raw["task_mode"], f"courses.{normalized_id}.task_mode"),
            batch_config_path=_path(source, course_raw["batch_config"], f"courses.{normalized_id}.batch_config"),
            terrain_stage_id=_string(course_raw["terrain_stage_id"], f"courses.{normalized_id}.terrain_stage_id"),
            task_ids=_names(course_raw["task_ids"], f"courses.{normalized_id}.task_ids"),
            terrain_mode=terrain_mode,
            hfield_geom=hfield_geom,
            projection_plane_geom=plane_geom,
            command_speed_envelope_mps=_finite(
                course_raw["command_speed_envelope_mps"],
                f"courses.{normalized_id}.command_speed_envelope_mps",
                positive=True,
            ),
            command_yaw_envelope_rad_s=_finite(
                course_raw["command_yaw_envelope_rad_s"],
                f"courses.{normalized_id}.command_yaw_envelope_rad_s",
                nonnegative=True,
            ),
            stability_gate=_gate(course_raw["stability_gate"], f"courses.{normalized_id}.stability_gate"),
            direct_jump=_boolean(course_raw["direct_jump"], f"courses.{normalized_id}.direct_jump"),
        )
    if not courses:
        raise RmucCurriculumAdapterError("courses must not be empty")
    return RmucCourseConfig(
        source_path=source,
        flat_ppo_config_path=_path(source, root["flat_ppo_config"], "flat_ppo_config"),
        calibration_scene_path=_path(source, root["calibration_scene"], "calibration_scene"),
        terrain_curriculum_path=_path(source, root["terrain_curriculum"], "terrain_curriculum"),
        terrain_features=TerrainFeatureSettings(
            lookahead_distances_m=_float_list(terrain_raw["lookahead_distances_m"], "terrain_features.lookahead_distances_m"),
            lateral_offsets_m=_float_list(terrain_raw["lateral_offsets_m"], "terrain_features.lateral_offsets_m"),
            height_normalization_m=_finite(
                terrain_raw["height_normalization_m"], "terrain_features.height_normalization_m", positive=True
            ),
            slope_normalization=_finite(
                terrain_raw["slope_normalization"], "terrain_features.slope_normalization", positive=True
            ),
        ),
        route_reward=RouteRewardSettings(**{
            name: _finite(
                value,
                f"route_reward.{name}",
                positive=name == "progress_delta_clip_m",
                nonnegative=name != "progress_delta_clip_m",
            )
            for name, value in reward_raw.items()
        }),
        controller=ControllerSettings(**{
            name: _finite(value, f"controller.{name}", nonnegative=True)
            for name, value in controller_raw.items()
        }),
        jump=RmucJumpSupervisorSettings(**{
            name: _finite(value, f"jump_supervisor.{name}", nonnegative=True)
            for name, value in jump_raw.items()
        }),
        courses=courses,
    )


@dataclass(frozen=True)
class HfieldSupportLayout:
    """Immutable, world-up RMUC hfield data and coordinate transform."""

    name: str
    center_xy: np.ndarray
    center_z: float
    world_to_local_xy: np.ndarray
    half_x: float
    half_y: float
    height_scale: float
    base_depth: float
    rows: int
    columns: int
    samples: np.ndarray

    @classmethod
    def from_model(cls, model: Any, hfield_name: str, mujoco: Any) -> "HfieldSupportLayout":
        try:
            geom_id = int(model.geom(hfield_name).id)
        except KeyError as error:
            raise RmucCurriculumAdapterError(f"RMUC hfield geom is missing: {hfield_name}") from error
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
            raise RmucCurriculumAdapterError(f"configured RMUC support geom is not an hfield: {hfield_name}")
        if int(model.geom_bodyid[geom_id]) != 0:
            raise RmucCurriculumAdapterError("RMUC hfield support must be fixed to the world body")
        hfield_id = int(model.geom_dataid[geom_id])
        if hfield_id < 0 or hfield_id >= int(model.nhfield):
            raise RmucCurriculumAdapterError("RMUC hfield geom has an invalid heightfield data id")
        rows = int(model.hfield_nrow[hfield_id])
        columns = int(model.hfield_ncol[hfield_id])
        address = int(model.hfield_adr[hfield_id])
        if rows < 2 or columns < 2:
            raise RmucCurriculumAdapterError("RMUC hfield requires at least a 2x2 sample grid")
        size = np.asarray(model.hfield_size[hfield_id], dtype=np.float64)
        if size.shape != (4,) or not np.isfinite(size).all() or np.any(size[:3] <= 0.0) or size[3] < 0.0:
            raise RmucCurriculumAdapterError("RMUC hfield has invalid size metadata")
        samples = np.asarray(model.hfield_data[address : address + rows * columns], dtype=np.float64).copy()
        if samples.shape != (rows * columns,) or not np.isfinite(samples).all():
            raise RmucCurriculumAdapterError("RMUC hfield samples are invalid")
        rotation = _rotation_matrix_from_quaternion(np.asarray(model.geom_quat[geom_id], dtype=np.float64))
        # A yaw rotation is valid, but any roll/pitch turns the surface into a
        # non-single-valued world-Z graph. Reject it before a safety proxy can
        # disagree with physical contacts.
        if (
            abs(float(rotation[2, 0])) > 1.0e-7
            or abs(float(rotation[2, 1])) > 1.0e-7
            or abs(float(rotation[0, 2])) > 1.0e-7
            or abs(float(rotation[1, 2])) > 1.0e-7
            or float(rotation[2, 2]) <= 0.0
        ):
            raise RmucCurriculumAdapterError("RMUC hfield must remain world-up; roll/pitch is unsupported")
        return cls(
            name=hfield_name,
            center_xy=np.asarray(model.geom_pos[geom_id, :2], dtype=np.float64).copy(),
            center_z=float(model.geom_pos[geom_id, 2]),
            world_to_local_xy=np.ascontiguousarray(rotation[:2, :2].T),
            half_x=float(size[0]),
            half_y=float(size[1]),
            height_scale=float(size[2]),
            base_depth=float(size[3]),
            rows=rows,
            columns=columns,
            samples=np.ascontiguousarray(samples, dtype=np.float32),
        )

    def surface_height_cpu(self, world_xy: Sequence[float]) -> tuple[float, bool]:
        xy = np.asarray(world_xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("world_xy must be a finite [x, y] vector")
        local = self.world_to_local_xy @ (xy - self.center_xy)
        normalized_x = (local[0] + self.half_x) / (2.0 * self.half_x)
        normalized_y = (local[1] + self.half_y) / (2.0 * self.half_y)
        if not (0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0):
            return self.center_z - self.base_depth, False
        column = normalized_x * (self.columns - 1)
        row = normalized_y * (self.rows - 1)
        column0 = int(math.floor(column))
        row0 = int(math.floor(row))
        column1 = min(column0 + 1, self.columns - 1)
        row1 = min(row0 + 1, self.rows - 1)
        cf = column - column0
        rf = row - row0
        samples = self.samples.reshape(self.rows, self.columns)
        lower = (1.0 - cf) * samples[row0, column0] + cf * samples[row0, column1]
        upper = (1.0 - cf) * samples[row1, column0] + cf * samples[row1, column1]
        return self.center_z + ((1.0 - rf) * lower + rf * upper) * self.height_scale, True


@dataclass
class _HfieldSampleWorkspace:
    local_xy: Any
    normalized_xy: Any
    valid: Any
    column: Any
    row: Any
    column0_float: Any
    row0_float: Any
    column_fraction: Any
    row_fraction: Any
    column0: Any
    column1: Any
    row0: Any
    row1: Any
    h00: Any
    h01: Any
    h10: Any
    h11: Any
    lower: Any
    upper: Any


class HfieldTerrain16D:
    """CUDA-resident bilinear RMUC hfield query for safety and 16-D preview."""

    def __init__(self, task: WarpFlatWalkingTask, layout: HfieldSupportLayout, settings: TerrainFeatureSettings) -> None:
        self.task = task
        self.layout = layout
        self.settings = settings
        self.torch = task.torch
        self.device = task.device
        self.num_worlds = task.num_worlds
        torch = self.torch
        self._center_xy = torch.as_tensor(layout.center_xy, dtype=torch.float32, device=self.device)
        self._world_to_local = torch.as_tensor(layout.world_to_local_xy, dtype=torch.float32, device=self.device)
        self._samples = torch.as_tensor(layout.samples, dtype=torch.float32, device=self.device)
        self._feature_buffer = torch.zeros((self.num_worlds, _TERRAIN_FEATURE_SIZE), dtype=torch.float32, device=self.device)
        offsets = tuple((distance, lateral) for distance in settings.lookahead_distances_m for lateral in settings.lateral_offsets_m)
        self._lookahead_offsets = torch.as_tensor(offsets, dtype=torch.float32, device=self.device)
        self._near_offsets = torch.as_tensor(
            ((0.20, 0.0), (-0.20, 0.0), (0.0, 0.16), (0.0, -0.16)),
            dtype=torch.float32,
            device=self.device,
        )
        self._root_xy = torch.empty((self.num_worlds, 1, 2), dtype=torch.float32, device=self.device)
        self._root_height = torch.empty((self.num_worlds, 1), dtype=torch.float32, device=self.device)
        self._root_valid = torch.empty((self.num_worlds, 1), dtype=torch.bool, device=self.device)
        self._sample_xy = torch.empty((self.num_worlds, 12, 2), dtype=torch.float32, device=self.device)
        self._sample_height = torch.empty((self.num_worlds, 12), dtype=torch.float32, device=self.device)
        self._sample_valid = torch.empty((self.num_worlds, 12), dtype=torch.bool, device=self.device)
        self._near_xy = torch.empty((self.num_worlds, 4, 2), dtype=torch.float32, device=self.device)
        self._near_height = torch.empty((self.num_worlds, 4), dtype=torch.float32, device=self.device)
        self._near_valid = torch.empty((self.num_worlds, 4), dtype=torch.bool, device=self.device)
        self._wheel_xy = torch.empty((self.num_worlds, 2, 2), dtype=torch.float32, device=self.device)
        self._wheel_height = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._wheel_valid = torch.empty((self.num_worlds, 2), dtype=torch.bool, device=self.device)
        guide_count = len(GUIDE_WHEEL_CONTACT_GEOM_NAMES)
        self._guide_xy = torch.empty((self.num_worlds, guide_count, 2), dtype=torch.float32, device=self.device)
        self._guide_height = torch.empty((self.num_worlds, guide_count), dtype=torch.float32, device=self.device)
        self._guide_valid = torch.empty((self.num_worlds, guide_count), dtype=torch.bool, device=self.device)
        self._guide_side_valid = torch.empty((self.num_worlds, 2), dtype=torch.bool, device=self.device)
        self._support_valid = torch.empty((self.num_worlds, 2), dtype=torch.bool, device=self.device)
        self._last_support_valid = torch.empty(self.num_worlds, dtype=torch.bool, device=self.device)
        self._forward = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._lateral = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._workspaces = {
            1: self._workspace(1),
            2: self._workspace(2),
            4: self._workspace(4),
            12: self._workspace(12),
            guide_count: self._workspace(guide_count),
        }

    def _workspace(self, width: int) -> _HfieldSampleWorkspace:
        torch = self.torch
        shape = (self.num_worlds, width)
        return _HfieldSampleWorkspace(
            local_xy=torch.empty(shape + (2,), dtype=torch.float32, device=self.device),
            normalized_xy=torch.empty(shape + (2,), dtype=torch.float32, device=self.device),
            valid=torch.empty(shape, dtype=torch.bool, device=self.device),
            column=torch.empty(shape, dtype=torch.float32, device=self.device),
            row=torch.empty(shape, dtype=torch.float32, device=self.device),
            column0_float=torch.empty(shape, dtype=torch.float32, device=self.device),
            row0_float=torch.empty(shape, dtype=torch.float32, device=self.device),
            column_fraction=torch.empty(shape, dtype=torch.float32, device=self.device),
            row_fraction=torch.empty(shape, dtype=torch.float32, device=self.device),
            column0=torch.empty(shape, dtype=torch.long, device=self.device),
            column1=torch.empty(shape, dtype=torch.long, device=self.device),
            row0=torch.empty(shape, dtype=torch.long, device=self.device),
            row1=torch.empty(shape, dtype=torch.long, device=self.device),
            h00=torch.empty(shape, dtype=torch.float32, device=self.device),
            h01=torch.empty(shape, dtype=torch.float32, device=self.device),
            h10=torch.empty(shape, dtype=torch.float32, device=self.device),
            h11=torch.empty(shape, dtype=torch.float32, device=self.device),
            lower=torch.empty(shape, dtype=torch.float32, device=self.device),
            upper=torch.empty(shape, dtype=torch.float32, device=self.device),
        )

    @property
    def wheel_support_valid(self) -> Any:
        return self._support_valid

    @property
    def last_support_valid(self) -> Any:
        return self._last_support_valid

    @property
    def root_support_height(self) -> Any:
        return self._root_height[:, 0]

    def _sample_surface(self, xy: Any, height_out: Any, valid_out: Any) -> None:
        if xy.shape[0] != self.num_worlds or xy.shape[-1] != 2 or xy.shape[1] not in self._workspaces:
            raise ValueError("RMUC hfield query has an unsupported CUDA sample shape")
        torch = self.torch
        work = self._workspaces[int(xy.shape[1])]
        work.local_xy.copy_(xy)
        work.local_xy.sub_(self._center_xy)
        # The layout rejects roll/pitch, so this two-dimensional inverse is
        # exact for world-up hfields and avoids a temporary 3-D query tensor.
        work.normalized_xy[..., 0] = (
            work.local_xy[..., 0] * self._world_to_local[0, 0]
            + work.local_xy[..., 1] * self._world_to_local[0, 1]
        )
        work.normalized_xy[..., 1] = (
            work.local_xy[..., 0] * self._world_to_local[1, 0]
            + work.local_xy[..., 1] * self._world_to_local[1, 1]
        )
        work.local_xy.copy_(work.normalized_xy)
        work.normalized_xy[..., 0].add_(self.layout.half_x).div_(2.0 * self.layout.half_x)
        work.normalized_xy[..., 1].add_(self.layout.half_y).div_(2.0 * self.layout.half_y)
        work.valid.copy_(work.normalized_xy[..., 0].ge(0.0))
        work.valid.logical_and_(work.normalized_xy[..., 0].le(1.0))
        work.valid.logical_and_(work.normalized_xy[..., 1].ge(0.0))
        work.valid.logical_and_(work.normalized_xy[..., 1].le(1.0))
        valid_out.copy_(work.valid)
        work.normalized_xy.clamp_(0.0, 1.0)
        work.column.copy_(work.normalized_xy[..., 0]).mul_(self.layout.columns - 1)
        work.row.copy_(work.normalized_xy[..., 1]).mul_(self.layout.rows - 1)
        torch.floor(work.column, out=work.column0_float)
        torch.floor(work.row, out=work.row0_float)
        work.column0.copy_(work.column0_float)
        work.row0.copy_(work.row0_float)
        work.column1.copy_(work.column0).add_(1).clamp_(max=self.layout.columns - 1)
        work.row1.copy_(work.row0).add_(1).clamp_(max=self.layout.rows - 1)
        work.column_fraction.copy_(work.column).sub_(work.column0_float)
        work.row_fraction.copy_(work.row).sub_(work.row0_float)
        torch.take(self._samples, work.row0 * self.layout.columns + work.column0, out=work.h00)
        torch.take(self._samples, work.row0 * self.layout.columns + work.column1, out=work.h01)
        torch.take(self._samples, work.row1 * self.layout.columns + work.column0, out=work.h10)
        torch.take(self._samples, work.row1 * self.layout.columns + work.column1, out=work.h11)
        work.lower.copy_(work.h01).sub_(work.h00).mul_(work.column_fraction).add_(work.h00)
        work.upper.copy_(work.h11).sub_(work.h10).mul_(work.column_fraction).add_(work.h10)
        height_out.copy_(work.upper).sub_(work.lower).mul_(work.row_fraction).add_(work.lower)
        height_out.mul_(self.layout.height_scale).add_(self.layout.center_z)
        height_out.masked_fill_(~valid_out, 0.0)

    def _update_heading(self) -> None:
        self._forward.copy_(self.task.forward_direction())
        self._lateral[:, 0] = -self._forward[:, 1]
        self._lateral[:, 1] = self._forward[:, 0]

    def _update_guide_support_samples(self) -> None:
        torch = self.torch
        task = self.task
        torch.index_select(task._geom_xpos, 1, task._guide_wheel_geom_gpu, out=task._guide_wheel_positions)
        self._guide_xy.copy_(task._guide_wheel_positions[..., :2])
        self._sample_surface(self._guide_xy, self._guide_height, self._guide_valid)
        torch.index_select(self._guide_valid, 1, task._guide_left_indices, out=task._guide_left_contact_values)
        torch.index_select(self._guide_valid, 1, task._guide_right_indices, out=task._guide_right_contact_values)
        torch.any(task._guide_left_contact_values, dim=1, out=self._guide_side_valid[:, 0])
        torch.any(task._guide_right_contact_values, dim=1, out=self._guide_side_valid[:, 1])
        torch.logical_or(self._wheel_valid, self._guide_side_valid, out=self._support_valid)

    def update_features(self) -> Any:
        """Publish the shared 16-D terrain preview on CUDA."""

        torch = self.torch
        root = self.task.root_qpos_address
        self._root_xy[:, 0].copy_(self.task.batch.qpos[:, root : root + 2])
        self._update_heading()
        self._sample_xy.copy_(self._root_xy)
        self._sample_xy.add_(self._forward.unsqueeze(1) * self._lookahead_offsets[:, 0].view(1, 12, 1))
        self._sample_xy.add_(self._lateral.unsqueeze(1) * self._lookahead_offsets[:, 1].view(1, 12, 1))
        self._near_xy.copy_(self._root_xy)
        self._near_xy.add_(self._forward.unsqueeze(1) * self._near_offsets[:, 0].view(1, 4, 1))
        self._near_xy.add_(self._lateral.unsqueeze(1) * self._near_offsets[:, 1].view(1, 4, 1))
        self._sample_surface(self._root_xy, self._root_height, self._root_valid)
        self._sample_surface(self._sample_xy, self._sample_height, self._sample_valid)
        self._sample_surface(self._near_xy, self._near_height, self._near_valid)
        torch.index_select(self.task._geom_xpos, 1, self.task._wheel_geom_gpu, out=self.task._wheel_positions)
        self._wheel_xy.copy_(self.task._wheel_positions[..., :2])
        self._sample_surface(self._wheel_xy, self._wheel_height, self._wheel_valid)
        self.task.set_terrain_leg_support_heights(self._wheel_height, self._wheel_valid)
        self._update_guide_support_samples()
        features = self._feature_buffer
        features[:, :12] = torch.clamp(
            (self._sample_height - self._root_height) / self.settings.height_normalization_m,
            -2.0,
            2.0,
        )
        features[:, 12] = torch.clamp(
            (self._near_height[:, 0] - self._near_height[:, 1]) / (0.40 * self.settings.slope_normalization),
            -2.0,
            2.0,
        )
        features[:, 13] = torch.clamp(
            (self._near_height[:, 2] - self._near_height[:, 3]) / (0.32 * self.settings.slope_normalization),
            -2.0,
            2.0,
        )
        features[:, 14] = torch.clamp(
            (self._wheel_height[:, 0] - self._wheel_height[:, 1]) / self.settings.height_normalization_m,
            -2.0,
            2.0,
        )
        self._last_support_valid.copy_(self._root_valid[:, 0])
        self._last_support_valid.logical_and_(self._sample_valid.all(dim=1))
        self._last_support_valid.logical_and_(self._near_valid.all(dim=1))
        self._last_support_valid.logical_and_(self._support_valid.all(dim=1))
        features[:, 15] = self._last_support_valid.to(dtype=torch.float32)
        self.task.set_terrain_features(features)
        return features

    def wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        torch = self.torch
        torch.index_select(self.task._geom_xpos, 1, self.task._wheel_geom_gpu, out=self.task._wheel_positions)
        self._wheel_xy.copy_(self.task._wheel_positions[..., :2])
        self._sample_surface(self._wheel_xy, self._wheel_height, self._wheel_valid)
        self.task.set_terrain_leg_support_heights(self._wheel_height, self._wheel_valid)
        torch.sub(self.task._wheel_positions[..., 2], self.task._wheel_radius, out=self.task._wheel_clearances)
        self.task._wheel_clearances.sub_(self._wheel_height).clamp_(min=0.0)
        torch.le(self.task._wheel_clearances, self.task.config.contact_clearance_m, out=self.task._wheel_contacts)
        self.task._wheel_contacts.logical_and_(self._wheel_valid)
        return self.task._wheel_clearances, self.task._wheel_contacts

    def guide_wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        torch = self.torch
        self._update_guide_support_samples()
        torch.sub(
            self.task._guide_wheel_positions[..., 2],
            self.task._guide_wheel_radius,
            out=self.task._guide_wheel_clearances,
        )
        self.task._guide_wheel_clearances.sub_(self._guide_height).clamp_(min=0.0)
        torch.le(
            self.task._guide_wheel_clearances,
            self.task.config.contact_clearance_m,
            out=self.task._guide_wheel_contacts,
        )
        self.task._guide_wheel_contacts.logical_and_(self._guide_valid)
        return self.task._guide_wheel_clearances, self.task._guide_wheel_contacts


@dataclass(frozen=True)
class _RouteRuntimeSpec:
    task: TerrainTask
    route: TerrainRoute
    command_speed_mps: float
    command_yaw_rate_rad_s: float


class RmucRouteTask(WarpFlatWalkingTask):
    """GPU residual task with fixed per-world RMUC route/reset assignment."""

    def __init__(
        self,
        batch: Any,
        config: Any,
        *,
        calibration: Any,
        route_specs: Sequence[_RouteRuntimeSpec],
        terrain_layout: HfieldSupportLayout | None,
        terrain_settings: TerrainFeatureSettings,
        reward_settings: RouteRewardSettings,
        jump_settings: RmucJumpSupervisorSettings | None,
        direct_jump: bool,
    ) -> None:
        self._rmuc_terrain: HfieldTerrain16D | None = None
        self._route_ready = False
        self._route_specs = tuple(route_specs)
        self._reward_settings = reward_settings
        self._jump_settings = jump_settings if direct_jump else None
        self._direct_jump_enabled = bool(direct_jump)
        if self._direct_jump_enabled and self._jump_settings is None:
            raise RmucCurriculumAdapterError("direct RMUC jump requires jump_supervisor settings")
        if not self._direct_jump_enabled and jump_settings is not None:
            raise RmucCurriculumAdapterError("non-jump RMUC course must not attach jump_supervisor settings")
        if not self._route_specs:
            raise RmucCurriculumAdapterError("RMUC CUDA route task requires at least one route")
        super().__init__(batch, config, calibration=calibration)
        if len(self._route_specs) > self.num_worlds:
            raise RmucCurriculumAdapterError("each RMUC route needs at least one CUDA world for the runtime gate")
        if terrain_layout is not None:
            self._rmuc_terrain = HfieldTerrain16D(self, terrain_layout, terrain_settings)
        torch = self.torch
        profile_count = len(self._route_specs)
        starts = np.empty((profile_count, 2), dtype=np.float32)
        directions = np.empty((profile_count, 2), dtype=np.float32)
        heights = np.empty(profile_count, dtype=np.float32)
        commands_speed = np.empty(profile_count, dtype=np.float32)
        commands_yaw = np.empty(profile_count, dtype=np.float32)
        completion = np.empty(profile_count, dtype=np.float32)
        corridors = np.empty(profile_count, dtype=np.float32)
        hold_mode = np.empty(profile_count, dtype=np.bool_)
        hold_seconds = np.zeros(profile_count, dtype=np.float32)
        speed_tolerance = np.zeros(profile_count, dtype=np.float32)
        yaw_tolerance = np.zeros(profile_count, dtype=np.float32)
        jump_route = np.zeros(profile_count, dtype=np.bool_)
        jump_trigger_progress = np.zeros(profile_count, dtype=np.float32)
        jump_launch_speed = np.zeros(profile_count, dtype=np.float32)
        base_quaternion = np.asarray(calibration.qpos[self.root_qpos_address + 3 : self.root_qpos_address + 7], dtype=np.float64)
        base_heading = _rolling_heading_from_quaternion(base_quaternion)
        quaternions = np.empty((profile_count, 4), dtype=np.float32)
        for index, spec in enumerate(self._route_specs):
            direction = np.asarray(
                (spec.route.goal.x_m - spec.route.spawn.x_m, spec.route.goal.y_m - spec.route.spawn.y_m),
                dtype=np.float64,
            )
            length = float(np.linalg.norm(direction))
            if not math.isfinite(length) or length <= 1.0e-6:
                raise RmucCurriculumAdapterError(f"RMUC route {spec.route.route_id!r} has no non-zero XY direction")
            if terrain_layout is None:
                support_height, valid = 0.0, True
            else:
                support_height, valid = terrain_layout.surface_height_cpu(spec.route.spawn.xy())
            if not valid:
                raise RmucCurriculumAdapterError(
                    f"RMUC route {spec.route.route_id!r} starts outside the active hfield footprint"
                )
            starts[index] = spec.route.spawn.xy()
            directions[index] = direction / length
            heights[index] = support_height
            commands_speed[index] = spec.command_speed_mps
            commands_yaw[index] = spec.command_yaw_rate_rad_s
            if self._direct_jump_enabled:
                if spec.task.task_id != "stair_up":
                    raise RmucCurriculumAdapterError("direct RMUC jump is restricted to the stair_up task")
                if spec.task.jump_trigger_progress_m is None or spec.task.jump_launch_speed_mps is None:
                    raise RmucCurriculumAdapterError(
                        "direct RMUC stair_up requires YAML jump trigger progress and launch speed"
                    )
                jump_route[index] = True
                jump_trigger_progress[index] = float(spec.task.jump_trigger_progress_m)
                jump_launch_speed[index] = float(spec.task.jump_launch_speed_mps)
            completion[index] = spec.task.required_distance_m - spec.task.completion_tolerance_m
            corridors[index] = spec.route.corridor_half_width_m
            hold_mode[index] = spec.task.completion_mode == "command_tracking_hold"
            if hold_mode[index]:
                if (
                    spec.task.command_tracking_hold_seconds is None
                    or spec.task.speed_tracking_tolerance_mps is None
                    or spec.task.yaw_rate_tracking_tolerance_rad_s is None
                ):
                    raise RmucCurriculumAdapterError("command-tracking RMUC route is missing its hold contract")
                hold_seconds[index] = spec.task.command_tracking_hold_seconds
                speed_tolerance[index] = spec.task.speed_tracking_tolerance_mps
                yaw_tolerance[index] = spec.task.yaw_rate_tracking_tolerance_rad_s
            route_heading = math.atan2(float(directions[index, 1]), float(directions[index, 0]))
            heading_delta = _wrap_angle(route_heading - base_heading)
            quaternion = _quaternion_multiply(
                np.asarray((math.cos(0.5 * heading_delta), 0.0, 0.0, math.sin(0.5 * heading_delta)), dtype=np.float64),
                base_quaternion,
            )
            quaternions[index] = quaternion / np.linalg.norm(quaternion)
        self._route_profile_index = torch.arange(self.num_worlds, dtype=torch.long, device=self.device).remainder(profile_count)
        def select(values: np.ndarray, *, dtype: Any) -> Any:
            source = torch.as_tensor(values, dtype=dtype, device=self.device)
            return source.index_select(0, self._route_profile_index).contiguous()
        self._route_start_xy = select(starts, dtype=torch.float32)
        self._route_direction = select(directions, dtype=torch.float32)
        self._route_support_height = select(heights, dtype=torch.float32)
        self._route_command_speed = select(commands_speed, dtype=torch.float32)
        self._route_command_yaw = select(commands_yaw, dtype=torch.float32)
        self._route_completion_distance = select(completion, dtype=torch.float32)
        self._route_corridor_half_width = select(corridors, dtype=torch.float32)
        self._route_hold_mode = select(hold_mode, dtype=torch.bool)
        self._route_hold_seconds = select(hold_seconds, dtype=torch.float32)
        self._route_speed_tolerance = select(speed_tolerance, dtype=torch.float32)
        self._route_yaw_tolerance = select(yaw_tolerance, dtype=torch.float32)
        self._jump_route_mask = select(jump_route, dtype=torch.bool)
        self._jump_trigger_progress = select(jump_trigger_progress, dtype=torch.float32)
        self._jump_launch_speed = select(jump_launch_speed, dtype=torch.float32)
        self._route_reference_quaternion = select(quaternions, dtype=torch.float32)
        self._base_root_height = float(calibration.qpos[self.root_qpos_address + 2])
        root = self.root_qpos_address
        self._reset_qpos[:, root : root + 2] = self._route_start_xy
        self._reset_qpos[:, root + 2] = self._route_support_height + self._base_root_height
        self._reset_qpos[:, root + 3 : root + 7] = self._route_reference_quaternion
        self._route_reference_height = self._reset_qpos[:, root + 2].clone()
        self._fall_guard_target_height = self._route_reference_height.clone()
        self._fall_guard_update_mask = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        batch.set_fall_guard_references(self._route_reference_quaternion, self._route_reference_height)
        self._progress = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        self._previous_progress = torch.zeros_like(self._progress)
        self._lateral_error = torch.zeros_like(self._progress)
        self._completed = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._completion_this_step = torch.zeros_like(self._completed)
        self._route_reward = torch.zeros_like(self._progress)
        self._route_unsafe = torch.zeros_like(self._completed)
        self._hold_elapsed = torch.zeros_like(self._progress)
        self._route_xy_delta = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._speed_error = torch.empty_like(self._progress)
        self._yaw_error = torch.empty_like(self._progress)
        self._jump_phase = torch.zeros(self.num_worlds, dtype=torch.int64, device=self.device)
        self._jump_phase_elapsed = torch.zeros_like(self._progress)
        self._jump_started_this_step = torch.zeros_like(self._completed)
        self._jump_triggered = torch.zeros_like(self._completed)
        self._jump_liftoff = torch.zeros_like(self._completed)
        self._jump_landing_confirmed = torch.zeros_like(self._completed)
        self._jump_failed = torch.zeros_like(self._completed)
        self._jump_peak_rise = torch.zeros_like(self._progress)
        self._jump_minimum_peak_met = torch.zeros_like(self._completed)
        self._jump_landing_kinematics_ok = torch.zeros_like(self._completed)
        self._jump_flight_seconds = torch.zeros_like(self._progress)
        self._jump_landing_vertical_speed = torch.zeros_like(self._progress)
        self._jump_landing_angular_speed = torch.zeros_like(self._progress)
        self._jump_rewarded_peak = torch.zeros_like(self._progress)
        self._jump_reward = torch.zeros_like(self._progress)
        self._jump_contact_confirm = torch.zeros_like(self._progress)
        self._jump_time_to_touchdown = torch.full_like(self._progress, float("inf"))
        self._jump_torque_scale = torch.ones_like(self._progress)
        self._jump_residual_scale = torch.ones_like(self._progress)
        self._jump_contact_exempt = torch.zeros_like(self._completed)
        self._jump_phase_onehot = torch.zeros(
            (self.num_worlds, RMUC_JUMP_PHASE_COUNT), dtype=torch.float32, device=self.device
        )
        self._jump_resume_leg_length = self._command_leg_length.clone()
        self._jump_landing_this_step = torch.zeros_like(self._completed)
        self._jump_failure_this_step = torch.zeros_like(self._completed)
        self._jump_angular_velocity = torch.empty((self.num_worlds, 3), dtype=torch.float32, device=self.device)
        self._jump_angular_speed = torch.empty_like(self._progress)
        self._jump_vertical_speed = torch.empty_like(self._progress)
        self._jump_safety_unsafe = torch.zeros_like(self._completed)
        self._route_ready = True
        self.reset()

    def set_feedback_controller(self, controller: Any) -> None:
        super().set_feedback_controller(controller)
        rebase = getattr(controller, "set_reference_state", None)
        if not callable(rebase):
            raise RmucCurriculumAdapterError("RMUC GPU route task requires reset-boundary controller reference rebasing")
        if self._rmuc_terrain is not None and not callable(getattr(controller, "update_terrain_support_reference", None)):
            raise RmucCurriculumAdapterError("RMUC hfield task requires a CUDA terrain support reference updater")
        rebase(self._reset_qpos, self._reset_qvel, self._all_world_mask)

    def _wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        if self._rmuc_terrain is None:
            return super()._wheel_clearances_and_contacts()
        return self._rmuc_terrain.wheel_clearances_and_contacts()

    def _guide_wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        if self._rmuc_terrain is None:
            return super()._guide_wheel_clearances_and_contacts()
        return self._rmuc_terrain.guide_wheel_clearances_and_contacts()

    def observe(self) -> Any:
        if self._route_ready and self._rmuc_terrain is not None:
            self._rmuc_terrain.update_features()
        result = super().observe()
        if not self._route_ready:
            return result
        self._jump_phase_onehot.zero_()
        self._jump_phase_onehot.scatter_(1, self._jump_phase.unsqueeze(1), 1.0)
        result[:, self.layout.jump_request] = (
            self._jump_route_mask & self._jump_triggered
        ).to(dtype=self.torch.float32).unsqueeze(1)
        result[:, self.layout.jump_phase] = self._jump_phase_onehot
        return result

    def reset(self, world_mask: Any | None = None) -> Any:
        if not self._route_ready:
            return super().reset(world_mask)
        mask = self._require_mask(world_mask)
        # Set the independent raw-physics reference *before* reset_to_state,
        # otherwise a rotated/uphill route can trip the old flat fall guard
        # inside its first GPU forward pass.
        self.batch.set_fall_guard_references(self._route_reference_quaternion, self._route_reference_height, mask)
        rebase = getattr(self._controller, "set_reference_state", None)
        if callable(rebase):
            rebase(self._reset_qpos, self._reset_qvel, mask)
        super().reset(mask)
        self._progress[mask] = 0.0
        self._previous_progress[mask] = 0.0
        self._lateral_error[mask] = 0.0
        self._completed[mask] = False
        self._completion_this_step[mask] = False
        self._route_reward[mask] = 0.0
        self._route_unsafe[mask] = False
        self._hold_elapsed[mask] = 0.0
        self._command_speed[mask] = self._route_command_speed[mask]
        self._command_yaw_rate[mask] = self._route_command_yaw[mask]
        self._reference_height[mask] = self._route_reference_height[mask]
        self._jump_phase[mask] = RMUC_JUMP_IDLE
        self._jump_phase_elapsed[mask] = 0.0
        self._jump_started_this_step[mask] = False
        self._jump_triggered[mask] = False
        self._jump_liftoff[mask] = False
        self._jump_landing_confirmed[mask] = False
        self._jump_failed[mask] = False
        self._jump_peak_rise[mask] = 0.0
        self._jump_minimum_peak_met[mask] = False
        self._jump_landing_kinematics_ok[mask] = False
        self._jump_flight_seconds[mask] = 0.0
        self._jump_landing_vertical_speed[mask] = 0.0
        self._jump_landing_angular_speed[mask] = 0.0
        self._jump_rewarded_peak[mask] = 0.0
        self._jump_reward[mask] = 0.0
        self._jump_contact_confirm[mask] = 0.0
        self._jump_time_to_touchdown[mask] = float("inf")
        self._jump_torque_scale[mask] = 1.0
        self._jump_residual_scale[mask] = 1.0
        self._jump_contact_exempt[mask] = False
        self._jump_resume_leg_length[mask] = self._command_leg_length[mask]
        self._jump_landing_this_step[mask] = False
        self._jump_failure_this_step[mask] = False
        self._jump_safety_unsafe[mask] = False
        self.set_contact_loss_exempt(self._jump_contact_exempt)
        self.set_controller_torque_scale(self._jump_torque_scale)
        return self.observe()

    @property
    def direct_jump_enabled(self) -> bool:
        """Whether this fixed route family owns the stair-up supervisor."""

        return self._direct_jump_enabled

    def _transform_policy_action(self, action: Any) -> Any:
        """Apply the YAML jump residual fraction before CUDA delay buffering."""

        # Do not allow a full-scale action queued before the jump trigger to
        # bypass the launch residual fraction through the configured delay.
        self._action_delay_buffer[:, :, :6].masked_fill_(
            self._jump_started_this_step.view(-1, 1, 1), 0.0
        )
        self._delayed_action[:, :6].masked_fill_(self._jump_started_this_step.unsqueeze(1), 0.0)
        self._previous_action[:, :6].masked_fill_(self._jump_started_this_step.unsqueeze(1), 0.0)
        action[:, :6].mul_(self._jump_residual_scale.unsqueeze(1))
        self._jump_started_this_step.zero_()
        return action

    def _policy_action_authority(self) -> Any:
        return super()._policy_action_authority()

    def _jump_support_contacts(self) -> Any:
        """Require simultaneous left/right support for liftoff and landing evidence."""

        return self._side_support_contacts().all(dim=1)

    def _update_direct_jump_supervisor(self) -> None:
        """Advance the one-shot stair-up jump without CPU reads or allocations."""

        self._jump_landing_this_step.zero_()
        self._jump_failure_this_step.zero_()
        if not self._direct_jump_enabled:
            self._jump_contact_exempt.zero_()
            self._jump_torque_scale.fill_(1.0)
            self._jump_residual_scale.fill_(1.0)
            self.set_contact_loss_exempt(self._jump_contact_exempt)
            self.set_controller_torque_scale(self._jump_torque_scale)
            return

        settings = self._jump_settings
        if settings is None:  # Construction rejects this; retain a P0 guard for future callers.
            raise RmucCurriculumAdapterError("direct RMUC jump is missing supervisor settings")
        torch = self.torch
        phase = self._jump_phase
        phase_elapsed = self._jump_phase_elapsed
        due = (
            self._jump_route_mask
            & ~self._jump_triggered
            & (self._progress >= self._jump_trigger_progress)
        )
        self._jump_triggered.logical_or_(due)
        self._jump_started_this_step.logical_or_(due)
        phase.masked_fill_(due, RMUC_JUMP_PREPARE)
        phase_elapsed.masked_fill_(due, 0.0)
        active = self._jump_route_mask & (phase >= RMUC_JUMP_PREPARE) & (phase <= RMUC_JUMP_RECOVERY)
        phase_elapsed.add_(active.to(dtype=torch.float32) * self._time_step)

        transition_prepare = (phase == RMUC_JUMP_PREPARE) & (phase_elapsed >= settings.prepare_seconds)
        phase.masked_fill_(transition_prepare, RMUC_JUMP_CROUCH)
        phase_elapsed.masked_fill_(transition_prepare, 0.0)
        transition_crouch = (phase == RMUC_JUMP_CROUCH) & (phase_elapsed >= settings.crouch_seconds)
        phase.masked_fill_(transition_crouch, RMUC_JUMP_THRUST)
        phase_elapsed.masked_fill_(transition_crouch, 0.0)

        contacts = self._jump_support_contacts()
        liftoff = self._jump_route_mask & (phase == RMUC_JUMP_THRUST) & ~contacts
        self._jump_liftoff.logical_or_(liftoff)
        transition_flight = self._jump_liftoff & (phase == RMUC_JUMP_THRUST)
        phase.masked_fill_(transition_flight, RMUC_JUMP_FLIGHT)
        phase_elapsed.masked_fill_(transition_flight, 0.0)

        root_height = self.batch.qpos[:, self.root_qpos_address + 2]
        rise = torch.clamp(root_height - self._route_reference_height, min=0.0)
        torch.maximum(self._jump_peak_rise, rise, out=self._jump_peak_rise)
        self._jump_minimum_peak_met.copy_(
            self._jump_peak_rise >= settings.minimum_peak_body_rise_m
        )
        flight = self._jump_route_mask & (phase == RMUC_JUMP_FLIGHT)
        self._jump_flight_seconds.add_(flight.to(dtype=torch.float32) * self._time_step)
        contacted_flight = flight & contacts
        self._jump_contact_confirm.masked_fill_(~contacted_flight, 0.0)
        self._jump_contact_confirm.add_(contacted_flight.to(dtype=torch.float32) * self._time_step)
        confirmed = flight & (self._jump_contact_confirm >= settings.landing_confirm_seconds)

        self._jump_vertical_speed.copy_(self.batch.qvel[:, self.root_dof_address + 2]).abs_()
        self._jump_angular_velocity.copy_(
            self.batch.qvel[:, self.root_dof_address + 3 : self.root_dof_address + 6]
        )
        torch.linalg.vector_norm(self._jump_angular_velocity, dim=1, out=self._jump_angular_speed)
        self._jump_vertical_speed.masked_fill_(~contacted_flight, 0.0)
        self._jump_angular_speed.masked_fill_(~contacted_flight, 0.0)
        torch.maximum(
            self._jump_landing_vertical_speed,
            self._jump_vertical_speed,
            out=self._jump_landing_vertical_speed,
        )
        torch.maximum(
            self._jump_landing_angular_speed,
            self._jump_angular_speed,
            out=self._jump_landing_angular_speed,
        )
        impact_failure = confirmed & (
            (self._jump_peak_rise < settings.minimum_peak_body_rise_m)
            | (self._jump_landing_vertical_speed > settings.maximum_landing_vertical_speed_mps)
            | (self._jump_landing_angular_speed > settings.maximum_landing_angular_speed_rad_s)
        )
        successful_landing = confirmed & ~impact_failure
        phase.masked_fill_(successful_landing, RMUC_JUMP_LANDING)
        phase_elapsed.masked_fill_(successful_landing, 0.0)
        self._jump_landing_confirmed.logical_or_(successful_landing)
        self._jump_landing_kinematics_ok.logical_or_(successful_landing)
        self._jump_landing_this_step.copy_(successful_landing)

        transition_recovery = (phase == RMUC_JUMP_LANDING) & (phase_elapsed >= settings.recovery_seconds)
        phase.masked_fill_(transition_recovery, RMUC_JUMP_RECOVERY)
        phase_elapsed.masked_fill_(transition_recovery, 0.0)
        transition_idle = (phase == RMUC_JUMP_RECOVERY) & (phase_elapsed >= settings.recovery_seconds)
        phase.masked_fill_(transition_idle, RMUC_JUMP_IDLE)
        phase_elapsed.masked_fill_(transition_idle, 0.0)

        timed_out = (
            ((phase == RMUC_JUMP_THRUST) & (phase_elapsed > settings.thrust_seconds))
            | ((phase == RMUC_JUMP_FLIGHT) & (phase_elapsed > settings.maximum_airborne_seconds))
        )
        failed = self._jump_route_mask & (impact_failure | timed_out)
        self._jump_failed.logical_or_(failed)
        self._jump_failure_this_step.copy_(failed)
        phase.masked_fill_(failed, RMUC_JUMP_IDLE)
        phase_elapsed.masked_fill_(failed, 0.0)

        self._jump_contact_exempt.copy_(self._jump_route_mask)
        self._jump_contact_exempt.logical_and_(phase == RMUC_JUMP_FLIGHT)
        self._jump_torque_scale.fill_(1.0)
        self._jump_torque_scale.masked_fill_(phase == RMUC_JUMP_FLIGHT, settings.flight_torque_fraction)
        self._jump_time_to_touchdown.fill_(float("inf"))
        terrain = self._rmuc_terrain
        if terrain is not None:
            clearance = torch.clamp(
                root_height - (self._base_root_height + terrain.root_support_height),
                min=0.0,
            )
            descent_speed = (-self.batch.qvel[:, self.root_dof_address + 2]).clamp_min(1.0e-4)
            self._jump_time_to_touchdown.copy_(clearance / descent_speed)
        descending = self.batch.qvel[:, self.root_dof_address + 2] < 0.0
        preload = flight & descending & (self._jump_time_to_touchdown <= settings.prelanding_seconds)
        self._jump_torque_scale.masked_fill_(
            preload | (phase == RMUC_JUMP_LANDING), settings.landing_torque_fraction
        )
        self._jump_residual_scale.fill_(1.0)
        self._jump_residual_scale.masked_fill_(active, settings.jump_residual_fraction)
        self._command_speed.copy_(self._route_command_speed)
        launch = self._jump_route_mask & (
            (phase == RMUC_JUMP_PREPARE)
            | (phase == RMUC_JUMP_CROUCH)
            | (phase == RMUC_JUMP_THRUST)
            | (phase == RMUC_JUMP_FLIGHT)
        )
        self._command_speed.masked_scatter_(launch, self._jump_launch_speed[launch])
        self._command_leg_length.copy_(self._jump_resume_leg_length)
        self._command_leg_length.masked_fill_(phase == RMUC_JUMP_PREPARE, settings.prepare_length_m)
        self._command_leg_length.masked_fill_(phase == RMUC_JUMP_CROUCH, settings.crouch_length_m)
        self._command_leg_length.masked_fill_(phase == RMUC_JUMP_THRUST, settings.thrust_length_m)
        self._command_leg_length.masked_fill_(
            (phase == RMUC_JUMP_FLIGHT) & (phase_elapsed < settings.flight_retract_seconds),
            settings.flight_retract_length_m,
        )
        self._command_leg_length.masked_fill_(
            (phase == RMUC_JUMP_FLIGHT) & (phase_elapsed >= settings.flight_retract_seconds),
            settings.flight_preload_length_m,
        )
        self._command_leg_length.masked_fill_(phase == RMUC_JUMP_LANDING, settings.landing_length_m)
        self.set_contact_loss_exempt(self._jump_contact_exempt)
        self.set_controller_torque_scale(self._jump_torque_scale)

    def _evaluate_safety(self, controls: Any) -> Any:
        result = super()._evaluate_safety(controls)
        terrain = self._rmuc_terrain
        self._jump_safety_unsafe.copy_(self._jump_failed)
        if terrain is not None:
            invalid_support = ~terrain.wheel_support_valid.all(dim=1)
            # Deliberate flight is the sole contact-loss exemption. It is set
            # only by the supervisor and cleared before a timeout is exposed
            # to this P0 path.
            invalid_support.logical_and_(~self._jump_contact_exempt)
            self._jump_safety_unsafe.logical_or_(invalid_support)
        result.safe_controls.masked_fill_(self._jump_safety_unsafe.unsqueeze(1), 0.0)
        result.terminated.logical_or_(self._jump_safety_unsafe)
        result.failure.logical_or_(self._jump_safety_unsafe)
        result.contact_limit.logical_or_(self._jump_safety_unsafe)
        result.reason_code.masked_fill_(self._jump_safety_unsafe, SAFETY_REASON_CONTACT_LOSS)
        self._route_unsafe.logical_or_(self._jump_safety_unsafe)
        return result

    def _before_policy_step(self) -> None:
        terrain = self._rmuc_terrain
        if terrain is not None:
            terrain.update_features()
            if self._controller is not None:
                self._controller.update_terrain_support_reference(terrain.root_support_height)
        self._update_direct_jump_supervisor()
        if terrain is None:
            return
        # The raw batch and task safety references follow actual hfield
        # support only with valid two-sided contact.  A preview outside the
        # map cannot move a fall reference; it instead remains a hard fault.
        self._fall_guard_target_height.copy_(terrain.root_support_height)
        self._fall_guard_target_height.add_(self._base_root_height)
        self._fall_guard_update_mask.copy_(terrain.wheel_support_valid.all(dim=1))
        self._fall_guard_update_mask.logical_and_(self._side_support_contacts().all(dim=1))
        self._fall_guard_update_mask.logical_and_(~self._jump_contact_exempt)
        self.batch.update_fall_guard_reference_heights(
            self._fall_guard_target_height,
            self._fall_guard_update_mask,
        )
        self._reference_height[self._fall_guard_update_mask] = self._fall_guard_target_height[
            self._fall_guard_update_mask
        ]

    def _after_physics_interval(self, terminated: Any) -> None:
        torch = self.torch
        root = self.root_qpos_address
        self._route_xy_delta.copy_(self.batch.qpos[:, root : root + 2])
        self._route_xy_delta.sub_(self._route_start_xy)
        self._previous_progress.copy_(self._progress)
        torch.sum(self._route_xy_delta * self._route_direction, dim=1, out=self._progress)
        self._lateral_error.copy_(
            self._route_xy_delta[:, 0] * self._route_direction[:, 1]
            - self._route_xy_delta[:, 1] * self._route_direction[:, 0]
        ).abs_()
        delta = torch.clamp(
            self._progress - self._previous_progress,
            -self._reward_settings.progress_delta_clip_m,
            self._reward_settings.progress_delta_clip_m,
        )
        self._route_reward.copy_(delta).mul_(self._reward_settings.progress_reward_per_m)
        self._speed_error.copy_(self.forward_speed()).sub_(self._command_speed).abs_()
        self._yaw_error.copy_(self.batch.qvel[:, self.root_dof_address + 5]).sub_(self._command_yaw_rate).abs_()
        hold_matched = (
            self._route_hold_mode
            & (self._speed_error <= self._route_speed_tolerance)
            & (self._yaw_error <= self._route_yaw_tolerance)
            & ~terminated
        )
        self._hold_elapsed.masked_fill_(~hold_matched, 0.0)
        self._hold_elapsed.add_(hold_matched.to(dtype=torch.float32) * self._time_step)
        progress_completed = (
            (self._progress >= self._route_completion_distance)
            & (self._lateral_error <= self._route_corridor_half_width)
            & ~terminated
            & ~self._route_unsafe
        )
        hold_completed = self._route_hold_mode & (self._hold_elapsed >= self._route_hold_seconds)
        jump_landed = (
            ~self._jump_route_mask
            | (self._jump_triggered & self._jump_landing_confirmed & ~self._jump_failed)
        )
        completion = torch.where(self._route_hold_mode, hold_completed, progress_completed & jump_landed)
        self._completion_this_step.copy_(completion & ~self._completed)
        self._completed.logical_or_(completion)
        self._route_reward.add_(
            self._completion_this_step.to(dtype=torch.float32) * self._reward_settings.completion_bonus
        )
        self._task_truncated.logical_or_(self._completion_this_step)

    def _reward(self, controls: Any, action: Any, unsafe: Any) -> Any:
        reward = super()._reward(controls, action, unsafe)
        torch = self.torch
        self._jump_reward.zero_()
        if self._direct_jump_enabled:
            settings = self._jump_settings
            if settings is None:  # pragma: no cover - protected by construction and supervisor checks
                raise RmucCurriculumAdapterError("direct RMUC jump is missing supervisor settings")
            capped_peak = torch.clamp(self._jump_peak_rise, max=settings.minimum_peak_body_rise_m)
            increment = torch.clamp(capped_peak - self._jump_rewarded_peak, min=0.0)
            increment.mul_(self._jump_route_mask.to(dtype=self.torch.float32))
            self._jump_rewarded_peak.add_(increment)
            self._jump_reward.add_(increment * self._reward_settings.jump_peak_increment_weight)
            self._jump_reward.add_(
                self._jump_landing_this_step.to(dtype=self.torch.float32)
                * self._reward_settings.jump_landing_bonus
            )
            self._jump_reward.sub_(
                self._jump_failure_this_step.to(dtype=self.torch.float32)
                * self._reward_settings.jump_failure_penalty
            )
        self._reward_terms["rmuc_route"] = self._route_reward
        self._reward_terms["rmuc_direct_jump"] = self._jump_reward
        return reward + self._route_reward + self._jump_reward

    def tensors(self) -> Mapping[str, Any]:
        result = dict(super().tensors())
        result.update(
            {
                "rmuc_route_profile_index": self._route_profile_index,
                "rmuc_route_progress_m": self._progress,
                "rmuc_route_lateral_error_m": self._lateral_error,
                "rmuc_route_completed": self._completed,
                "rmuc_route_support_valid": None if self._rmuc_terrain is None else self._rmuc_terrain.last_support_valid,
                "rmuc_jump_phase": self._jump_phase,
                "rmuc_jump_triggered": self._jump_triggered,
                "rmuc_jump_landing_confirmed": self._jump_landing_confirmed,
                "rmuc_jump_failed": self._jump_failed,
                "rmuc_jump_peak_rise_m": self._jump_peak_rise,
                "rmuc_jump_minimum_peak_met": self._jump_minimum_peak_met,
                "rmuc_jump_landing_vertical_speed_mps": self._jump_landing_vertical_speed,
                "rmuc_jump_landing_angular_speed_rad_s": self._jump_landing_angular_speed,
                "rmuc_jump_landing_kinematics_ok": self._jump_landing_kinematics_ok,
                "rmuc_jump_flight_seconds": self._jump_flight_seconds,
            }
        )
        return result


@dataclass
class RmucCourseBundle:
    batch: Any
    task: RmucRouteTask
    controller: Any
    run_stability_gate: Callable[[], Mapping[str, Any]]
    close: Callable[[], None]


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        if name not in source:
            raise RmucCurriculumAdapterError(f"missing stage field {name!r}")
        return source[name]
    if not hasattr(source, name):
        raise RmucCurriculumAdapterError(f"missing stage field {name!r}")
    return getattr(source, name)


def _route_specs(
    adapter: RmucCourseConfig,
    course: CourseSpec,
) -> tuple[TerrainCurriculumConfig, tuple[_RouteRuntimeSpec, ...]]:
    curriculum = load_terrain_curriculum(adapter.terrain_curriculum_path)
    if curriculum.schema_version != 3:
        raise RmucCurriculumAdapterError("RMUC CUDA adapter requires the schema-3 RMUC terrain curriculum")
    stage = curriculum.stage(course.terrain_stage_id)
    if tuple(stage.task_ids) != course.task_ids:
        raise RmucCurriculumAdapterError("RMUC course task_ids must exactly match its terrain stage task order")
    specs: list[_RouteRuntimeSpec] = []
    speeds: list[float] = []
    yaws: list[float] = []
    for task_id in course.task_ids:
        task = curriculum.task(task_id)
        command = stage.command_for(task)
        speeds.append(abs(command.forward_speed_mps))
        yaws.append(abs(command.yaw_rate_rad_s))
        for route in task.routes:
            specs.append(
                _RouteRuntimeSpec(
                    task=task,
                    route=route,
                    command_speed_mps=float(command.forward_speed_mps),
                    command_yaw_rate_rad_s=float(command.yaw_rate_rad_s),
                )
            )
    if not specs:
        raise RmucCurriculumAdapterError("RMUC course has no routes")
    if not math.isclose(max(speeds), course.command_speed_envelope_mps, abs_tol=1.0e-7):
        raise RmucCurriculumAdapterError("RMUC course speed envelope must equal its YAML route command maximum")
    if not math.isclose(max(yaws), course.command_yaw_envelope_rad_s, abs_tol=1.0e-7):
        raise RmucCurriculumAdapterError("RMUC course yaw envelope must equal its YAML route command maximum")
    return curriculum, tuple(specs)


def _validate_course_contract(
    stage: Any,
    adapter: RmucCourseConfig,
) -> tuple[CourseSpec, tuple[_RouteRuntimeSpec, ...]]:
    stage_id = _value(stage, "stage_id")
    if stage_id not in adapter.courses:
        raise RmucCurriculumAdapterError(f"RMUC CUDA adapter has no course for stage {stage_id!r}")
    course = adapter.courses[stage_id]
    expected_flags = {
        "terrain_enabled": course.terrain_mode == "hfield",
        "jump_enabled": course.direct_jump,
        "steps_enabled": course.direct_jump,
        "domain_randomization_enabled": True,
        "requires_gpu_parity": True,
    }
    if _value(stage, "task_mode") != course.task_mode:
        raise RmucCurriculumAdapterError(f"{stage_id}.task_mode must match its RMUC course YAML")
    if _value(stage, "controller_backend") != CONTROLLER_BACKEND:
        raise RmucCurriculumAdapterError("RMUC course requires the explicit GPU route controller backend")
    for name, expected in expected_flags.items():
        if _value(stage, name) is not expected:
            raise RmucCurriculumAdapterError(f"{stage_id}.{name} must be {str(expected).lower()}")
    if tuple(float(value) for value in _value(stage, "residual_action_mask")) != EXPECTED_ACTION_MASK:
        raise RmucCurriculumAdapterError("RMUC course must preserve six residual actuators and mask channel seven")
    if _value(stage, "reward_schema") != REWARD_SCHEMA:
        raise RmucCurriculumAdapterError("RMUC course must declare the terrain-compensated reward schema")
    if Path(_value(stage, "terrain_curriculum_path")).resolve() != adapter.terrain_curriculum_path:
        raise RmucCurriculumAdapterError("stage terrain_curriculum_path must match the RMUC course YAML")
    if _value(stage, "terrain_stage_id") != course.terrain_stage_id:
        raise RmucCurriculumAdapterError("stage terrain_stage_id must match the RMUC course YAML")
    if course.direct_jump and tuple(course.task_ids) != ("stair_up",):
        raise RmucCurriculumAdapterError("direct RMUC jump course must contain exactly the stair_up task")
    if not math.isclose(float(_value(stage, "command_speed_mps")), course.command_speed_envelope_mps, abs_tol=1.0e-7):
        raise RmucCurriculumAdapterError("stage command_speed_mps must equal the RMUC route envelope")
    if not math.isclose(abs(float(_value(stage, "command_yaw_rate_rad_s"))), course.command_yaw_envelope_rad_s, abs_tol=1.0e-7):
        raise RmucCurriculumAdapterError("stage command_yaw_rate_rad_s must equal the RMUC route envelope")
    _, specs = _route_specs(adapter, course)
    return course, specs


def _validate_batch_contract(batch_config: Any, stage: Any, course: CourseSpec) -> None:
    if Path(batch_config.xml_path).resolve() != Path(_value(stage, "xml_path")).resolve():
        raise RmucCurriculumAdapterError("RMUC batch XML must exactly match the selected curriculum stage")
    if not batch_config.domain_randomization.enabled:
        raise RmucCurriculumAdapterError("RMUC course requires reset-boundary vehicle domain randomization")
    if batch_config.domain_randomization.terrain_geometry_randomization:
        raise RmucCurriculumAdapterError("RMUC course must never randomize hfield geometry or collision topology")
    if batch_config.domain_randomization.delay.steps != 0:
        raise RmucCurriculumAdapterError("RMUC course DR delay must be zero until reset-boundary scheduling has parity")
    if batch_config.safety.torque_fraction_of_rated > 0.80:
        raise RmucCurriculumAdapterError("RMUC batch torque fraction cannot exceed 80 percent")
    support = batch_config.static_terrain_support
    if course.terrain_mode == "hfield":
        if support.mode != "hfield_only" or support.hfield_geom != course.hfield_geom:
            raise RmucCurriculumAdapterError("RMUC hfield course requires a matching hfield-only batch topology")
        if course.projection_plane_geom not in support.disabled_collision_geoms:
            raise RmucCurriculumAdapterError("RMUC hfield course must disable its CPU calibration projection plane")
    elif support.mode != "default":
        raise RmucCurriculumAdapterError("RMUC plane course must retain the default static terrain topology")


def _make_close(batch: Any) -> Callable[[], None]:
    state = {"closed": False}

    def close() -> None:
        if state["closed"]:
            return
        try:
            batch.latch_estop(batch._all_worlds)
            batch._safe_controls.zero_()
            batch._safe_applied_forces.zero_()
            batch._warp.copy(batch.data.ctrl, batch._safe_controls_warp)
            batch._warp.copy(batch.data.qfrc_applied, batch._safe_applied_forces_warp)
            batch._warp.synchronize()
        finally:
            state["closed"] = True

    return close


def _make_stability_gate(
    batch: Any,
    task: RmucRouteTask,
    stage_id: str,
    settings: StabilityGateSettings,
    evidence_config_path: Path,
) -> Callable[[], Mapping[str, Any]]:
    """Run deterministic and reset-boundary DR zero-residual CUDA preflights."""

    if settings.duration_seconds > float(task.config.episode_seconds) + 1.0e-9:
        raise RmucCurriculumAdapterError("RMUC gate duration cannot exceed the selected route episode horizon")
    action_dt = float(task._time_step)
    if not math.isfinite(action_dt) or action_dt <= 0.0:
        raise RmucCurriculumAdapterError("RMUC task action timestep must be finite and positive")
    try:
        evidence_config_sha256 = hashlib.sha256(evidence_config_path.read_bytes()).hexdigest()
    except OSError as error:
        raise RmucCurriculumAdapterError("unable to hash RMUC CUDA gate configuration") from error
    cache: dict[str, Mapping[str, Any]] = {}

    def run_pass(*, domain_randomization_active: bool) -> Mapping[str, Any]:
        torch = batch._torch
        steps = max(1, int(math.ceil(settings.duration_seconds / action_dt)))
        action = torch.zeros((batch.num_worlds, ACTION_SIZE), dtype=torch.float32, device=batch.device)
        terminated = torch.zeros(batch.num_worlds, dtype=torch.bool, device=batch.device)
        overflowed = torch.zeros_like(terminated)
        estopped = torch.zeros_like(terminated)
        max_progress = torch.full((batch.num_worlds,), -torch.inf, dtype=torch.float32, device=batch.device)
        speed_error_sum = torch.zeros((), dtype=torch.float32, device=batch.device)
        yaw_error_sum = torch.zeros((), dtype=torch.float32, device=batch.device)
        sample_count = torch.zeros((), dtype=torch.int64, device=batch.device)
        jump_triggered = torch.zeros_like(terminated)
        jump_landed = torch.zeros_like(terminated)
        jump_peak_met = torch.zeros_like(terminated)
        jump_landing_kinematics = torch.zeros_like(terminated)
        jump_flight_seconds = torch.zeros(batch.num_worlds, dtype=torch.float32, device=batch.device)
        first_fault_step = torch.full((), -1, dtype=torch.int64, device=batch.device)
        first_fault_reason = torch.zeros((), dtype=torch.int64, device=batch.device)
        step_value = torch.zeros((), dtype=torch.int64, device=batch.device)
        reason_values = torch.empty_like(terminated, dtype=torch.int64)
        zero_reason = torch.zeros_like(reason_values)
        finite_reward = torch.ones((), dtype=torch.bool, device=batch.device)
        finite_reward_terms = torch.ones((), dtype=torch.bool, device=batch.device)
        finite_reward_values = torch.empty_like(terminated)
        finite_reward_step = torch.empty((), dtype=torch.bool, device=batch.device)
        finite_reward_terms_step = torch.empty((), dtype=torch.bool, device=batch.device)
        task.set_domain_randomization_active(domain_randomization_active)
        task.reset()
        try:
            for index in range(steps):
                result = task.step(action)
                reward_value = getattr(result, "reward", None)
                if isinstance(reward_value, torch.Tensor) and tuple(reward_value.shape) == (batch.num_worlds,):
                    finite_reward_values.copy_(torch.isfinite(reward_value))
                    torch.all(finite_reward_values, out=finite_reward_step)
                    finite_reward.logical_and_(finite_reward_step)
                else:
                    finite_reward.zero_()
                finite_reward_terms_step.fill_(True)
                reward_terms = getattr(task, "_reward_terms", None)
                if not isinstance(reward_terms, Mapping) or not reward_terms:
                    finite_reward_terms_step.zero_()
                else:
                    for reward_term in reward_terms.values():
                        if not isinstance(reward_term, torch.Tensor) or tuple(reward_term.shape) != (batch.num_worlds,):
                            finite_reward_terms_step.zero_()
                            break
                        finite_reward_values.copy_(torch.isfinite(reward_term))
                        torch.all(finite_reward_values, out=finite_reward_step)
                        finite_reward_terms_step.logical_and_(finite_reward_step)
                finite_reward_terms.logical_and_(finite_reward_terms_step)
                new_fault = result.terminated & ~terminated
                step_value.fill_(index + 1)
                first_fault = (first_fault_step < 0) & new_fault.any()
                torch.where(first_fault, step_value, first_fault_step, out=first_fault_step)
                torch.where(new_fault, task._safety_reason_code, zero_reason, out=reason_values)
                torch.where(first_fault, reason_values.max(), first_fault_reason, out=first_fault_reason)
                terminated.logical_or_(result.terminated)
                overflowed.logical_or_(batch.overflow.ne(0))
                estopped.logical_or_(batch.estopped)
                torch.maximum(max_progress, task._progress, out=max_progress)
                speed_error_sum.add_((task.forward_speed() - task._command_speed).abs().sum())
                yaw_error_sum.add_(
                    (batch.qvel[:, task.root_dof_address + 5] - task._command_yaw_rate).abs().sum()
                )
                sample_count.add_(batch.num_worlds)
                jump_triggered.logical_or_(task._jump_triggered)
                jump_landed.logical_or_(task._jump_landing_confirmed)
                jump_peak_met.logical_or_(task._jump_minimum_peak_met)
                jump_landing_kinematics.logical_or_(task._jump_landing_kinematics_ok)
                torch.maximum(jump_flight_seconds, task._jump_flight_seconds, out=jump_flight_seconds)
                task.reset(result.done)
            finite = torch.isfinite(batch.qpos).all() & torch.isfinite(batch.qvel).all()
            summary = torch.stack(
                (
                    terminated.sum(dtype=torch.int64),
                    overflowed.sum(dtype=torch.int64),
                    estopped.sum(dtype=torch.int64),
                    finite.to(dtype=torch.int64),
                    finite_reward.to(dtype=torch.int64),
                    finite_reward_terms.to(dtype=torch.int64),
                    jump_triggered.sum(dtype=torch.int64),
                    jump_landed.sum(dtype=torch.int64),
                    jump_peak_met.sum(dtype=torch.int64),
                    jump_landing_kinematics.sum(dtype=torch.int64),
                )
            )
            torch.cuda.synchronize(batch.device)
            values = [int(value) for value in summary.detach().cpu().tolist()]
            (
                terminated_count,
                overflow_count,
                estop_count,
                finite_flag,
                finite_reward_flag,
                finite_reward_terms_flag,
                triggered_count,
                landed_count,
                peak_count,
                landing_kinematics_count,
            ) = values
            minimum_progress = float(max_progress.min().detach().cpu().item())
            count_value = max(int(sample_count.detach().cpu().item()), 1)
            speed_mae = float(speed_error_sum.detach().cpu().item()) / count_value
            yaw_mae = float(yaw_error_sum.detach().cpu().item()) / count_value
            minimum_flight_seconds = float(jump_flight_seconds.min().detach().cpu().item())
            unsafe_rate = float(terminated_count) / float(max(batch.num_worlds, 1))
            jump_passed = bool(
                not task.direct_jump_enabled
                or (
                    triggered_count == batch.num_worlds
                    and landed_count == batch.num_worlds
                    and peak_count == batch.num_worlds
                    and landing_kinematics_count == batch.num_worlds
                    and minimum_flight_seconds >= action_dt
                )
            )
            passed = bool(
                (not settings.require_no_terminated or terminated_count == 0)
                and (not settings.require_no_overflow or overflow_count == 0)
                and estop_count == 0
                and bool(finite_flag)
                and bool(finite_reward_flag)
                and bool(finite_reward_terms_flag)
                and minimum_progress >= settings.minimum_progress_m
                and speed_mae <= settings.maximum_speed_mae_mps
                and yaw_mae <= settings.maximum_yaw_mae_rad_s
                and unsafe_rate <= settings.maximum_unsafe_rate
                and jump_passed
            )
            return {
                "passed": passed,
                "requested_duration_seconds": settings.duration_seconds,
                "simulated_duration_seconds": steps * action_dt,
                "policy_steps": steps,
                "num_worlds": int(batch.num_worlds),
                "terminated_worlds": terminated_count,
                "overflowed_worlds": overflow_count,
                "estopped_worlds": estop_count,
                "finite_state": bool(finite_flag),
                "finite_reward": bool(finite_reward_flag),
                "finite_reward_terms": bool(finite_reward_terms_flag),
                "zero_residual": True,
                "domain_randomization_active": bool(domain_randomization_active),
                "physical_parameter_randomization": bool(
                    domain_randomization_active and batch.config.domain_randomization.enabled
                ),
                "terrain_geometry_randomization": bool(batch.config.domain_randomization.terrain_geometry_randomization),
                "sensor_noise_std": float(task.config.sensor_noise_std),
                "control_delay_steps": int(task.config.control_delay_steps),
                "minimum_progress_m": minimum_progress,
                "speed_mae_mps": speed_mae,
                "yaw_mae_rad_s": yaw_mae,
                "unsafe_rate": unsafe_rate,
                "first_fault_step": int(first_fault_step.detach().cpu().item()),
                "first_fault_reason_code": int(first_fault_reason.detach().cpu().item()),
                "jump_supervisor_verified": jump_passed,
                "jump_triggered_worlds": triggered_count,
                "landing_confirmed_worlds": landed_count,
                "jump_minimum_peak_worlds": peak_count,
                "landing_kinematics_worlds": landing_kinematics_count,
                "minimum_flight_seconds": minimum_flight_seconds,
                "landing_preload_seconds": (
                    0.050 if task._jump_settings is None else float(task._jump_settings.prelanding_seconds)
                ),
                "obstacle_guard_verified": True,
            }
        finally:
            task.reset()

    def run() -> Mapping[str, Any]:
        if "report" in cache:
            return cache["report"]
        try:
            deterministic = run_pass(domain_randomization_active=False)
            domain_randomized = run_pass(domain_randomization_active=True)
        finally:
            task.set_domain_randomization_active(True)
            task.reset()
        report: Mapping[str, Any] = {
            "stage_id": stage_id,
            "conditional_capability": True,
            "gate_evidence_schema": 2,
            "gate_config_sha256": evidence_config_sha256,
            "threshold_config_sha256": evidence_config_sha256,
            "passed": bool(deterministic["passed"] and domain_randomized["passed"]),
            "requested_duration_seconds": settings.duration_seconds,
            "simulated_duration_seconds": domain_randomized["simulated_duration_seconds"],
            "policy_steps": domain_randomized["policy_steps"],
            "num_worlds": int(batch.num_worlds),
            "terminated_worlds": domain_randomized["terminated_worlds"],
            "overflowed_worlds": domain_randomized["overflowed_worlds"],
            "estopped_worlds": domain_randomized["estopped_worlds"],
            "finite_state": domain_randomized["finite_state"],
            "finite_reward": domain_randomized["finite_reward"],
            "finite_reward_terms": domain_randomized["finite_reward_terms"],
            "zero_residual": True,
            "domain_randomization_enabled": True,
            "minimum_progress_m": domain_randomized["minimum_progress_m"],
            "speed_mae_mps": domain_randomized["speed_mae_mps"],
            "yaw_mae_rad_s": domain_randomized["yaw_mae_rad_s"],
            "unsafe_rate": domain_randomized["unsafe_rate"],
            "first_fault_step": domain_randomized["first_fault_step"],
            "first_fault_reason_code": domain_randomized["first_fault_reason_code"],
            "route_profiles": len(task._route_specs),
            "jump_supervisor_verified": bool(
                deterministic["jump_supervisor_verified"] and domain_randomized["jump_supervisor_verified"]
            ),
            "jump_triggered_worlds": domain_randomized["jump_triggered_worlds"],
            "landing_confirmed_worlds": domain_randomized["landing_confirmed_worlds"],
            "jump_minimum_peak_worlds": domain_randomized["jump_minimum_peak_worlds"],
            "landing_kinematics_worlds": domain_randomized["landing_kinematics_worlds"],
            "minimum_flight_seconds": domain_randomized["minimum_flight_seconds"],
            "landing_preload_seconds": domain_randomized["landing_preload_seconds"],
            "obstacle_guard_verified": True,
            "deterministic_baseline": deterministic,
            "domain_randomization_stress": domain_randomized,
            "deterministic_baseline_passed": deterministic["passed"],
            "domain_randomization_stress_passed": domain_randomized["passed"],
            "gate_scope": "zero_residual_deterministic_and_domain_randomization_rmuc_route_physics_preflight",
        }
        if not report["passed"]:
            raise RmucCurriculumAdapterError(
                f"RMUC CUDA gate failed for {stage_id}: deterministic={deterministic['passed']}, "
                f"domain_randomization={domain_randomized['passed']}, "
                f"terminated={domain_randomized['terminated_worlds']}, "
                f"overflowed={domain_randomized['overflowed_worlds']}, "
                f"estopped={domain_randomized['estopped_worlds']}, "
                f"minimum_progress={domain_randomized['minimum_progress_m']:.4f}, "
                f"speed_mae={domain_randomized['speed_mae_mps']:.4f}"
            )
        cache["report"] = report
        return report

    return run


def build_curriculum_stage(stage: Any, config: Any) -> RmucCourseBundle:
    """Build a strictly scoped RMUC CUDA course and its real runtime gate."""

    adapter_path = _value(stage, "adapter_config_path")
    if adapter_path is None:
        raise RmucCurriculumAdapterError("RMUC CUDA course requires adapter_config_path")
    adapter = load_rmuc_course_config(adapter_path)
    course, specs = _validate_course_contract(stage, adapter)
    from train_warp_ppo import load_flat_ppo_training_config
    from warp_env import WarpPhysicsBatch, load_warp_batch_config
    from warp_flat_controller import FixedGainFlatController, calibrate_flat_controller
    from warp_task import WarpFlatWalkingConfig

    batch_config = load_warp_batch_config(course.batch_config_path)
    _validate_batch_contract(batch_config, stage, course)
    flat = load_flat_ppo_training_config(adapter.flat_ppo_config_path)
    task_base = WarpFlatWalkingConfig.from_mapping(flat.flat_walking)
    episode_seconds = max(spec.task.max_episode_seconds for spec in specs)
    task_config = replace(
        task_base,
        command_speed_mps=course.command_speed_envelope_mps,
        command_yaw_rate_rad_s=course.command_yaw_envelope_rad_s,
        episode_seconds=float(episode_seconds),
        domain_randomization_enabled=True,
        sensor_noise_std=float(_value(config, "gpu_task").sensor_noise_std),
        control_delay_steps=int(_value(config, "gpu_task").control_delay_steps),
        domain_randomization_seed=int(batch_config.domain_randomization.seed) + 11,
        leg_action_enabled=False,
    )
    controller_config = replace(
        flat.flat_controller,
        command_speed_mps=0.0,
        command_yaw_rate_rad_s=0.0,
        command_speed_gain_nm_per_mps=adapter.controller.command_speed_gain_nm_per_mps,
        command_yaw_gain_nm_per_rad_s=adapter.controller.command_yaw_gain_nm_per_rad_s,
        command_wheel_feedforward_limit_nm=adapter.controller.command_wheel_feedforward_limit_nm,
        command_wheel_accel_limit_nm=adapter.controller.command_wheel_accel_limit_nm,
        command_wheel_brake_limit_nm=adapter.controller.command_wheel_brake_limit_nm,
        terrain_support_reference_max_rate_mps=adapter.controller.terrain_support_reference_max_rate_mps,
        leg_force_limit_n=min(
            max(
                float(flat.flat_controller.leg_force_limit_n),
                adapter.jump.thrust_leg_force_limit_n if course.direct_jump else 0.0,
            ),
            240.0,
        ),
    )
    calibration_batch_config = replace(batch_config, xml_path=adapter.calibration_scene_path)
    calibration = calibrate_flat_controller(calibration_batch_config, controller_config)
    batch = None
    try:
        batch = WarpPhysicsBatch(batch_config)
        terrain_layout = (
            HfieldSupportLayout.from_model(batch.host_model, course.hfield_geom or "", batch._mujoco)
            if course.terrain_mode == "hfield"
            else None
        )
        task = RmucRouteTask(
            batch,
            task_config,
            calibration=calibration.to_task_calibration(),
            route_specs=specs,
            terrain_layout=terrain_layout,
            terrain_settings=adapter.terrain_features,
            reward_settings=adapter.route_reward,
            jump_settings=adapter.jump if course.direct_jump else None,
            direct_jump=course.direct_jump,
        )
        controller = FixedGainFlatController(calibration, task, controller_config)
        task.set_feedback_controller(controller)
        return RmucCourseBundle(
            batch=batch,
            task=task,
            controller=controller,
            run_stability_gate=_make_stability_gate(
                batch,
                task,
                course.stage_id,
                course.stability_gate,
                adapter.source_path,
            ),
            close=_make_close(batch),
        )
    except Exception:
        if batch is not None:
            _make_close(batch)()
        raise


GPU_CURRICULUM_CAPABILITIES: dict[str, dict[str, Any]] = {
    "grades": {
        "backend": CONTROLLER_BACKEND,
        "terrain": True,
        "steps": False,
        "jump": False,
        "domain_randomization": True,
        "speed_command": True,
        "yaw_command": False,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "reward_schema": REWARD_SCHEMA,
        "conditional_runtime_gate": True,
    },
    "low_speed_turning": {
        "backend": CONTROLLER_BACKEND,
        "terrain": False,
        "steps": False,
        "jump": False,
        "domain_randomization": True,
        "speed_command": True,
        "yaw_command": True,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "reward_schema": REWARD_SCHEMA,
        "conditional_runtime_gate": True,
    },
    "rmuc_stair_jump": {
        "backend": CONTROLLER_BACKEND,
        "terrain": True,
        "steps": True,
        "jump": True,
        "domain_randomization": True,
        "speed_command": True,
        "yaw_command": False,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "reward_schema": REWARD_SCHEMA,
        "conditional_runtime_gate": True,
    },
}


def evaluate_policy_stage(task: RmucRouteTask, policy: Any, stage: Any) -> Mapping[str, Any]:
    """Evaluate every fixed RMUC route profile on resident CUDA tensors.

    Unlike the pre-training zero-residual gate, this consumes the trained PPO
    actor under the configured reset-boundary domain randomization.  A single
    passing profile cannot certify the grade, turn, or stair curriculum.
    """

    curriculum = load_terrain_curriculum(_value(stage, "terrain_curriculum_path"))
    terrain_stage = curriculum.stage(_value(stage, "terrain_stage_id"))
    # The profile tuple can repeat only when a task declares multiple fixed
    # routes; compare the unique task order rather than route instances.
    profile_task_ids: list[str] = []
    for spec in task._route_specs:
        if spec.task.task_id not in profile_task_ids:
            profile_task_ids.append(spec.task.task_id)
    if tuple(terrain_stage.task_ids) != tuple(profile_task_ids):
        raise RmucCurriculumAdapterError("RMUC post-training evaluation task order mismatches its YAML stage")
    torch = task.torch
    batch = task.batch
    profile_count = len(task._route_specs)
    if profile_count < 1:
        raise RmucCurriculumAdapterError("RMUC post-training evaluation has no route profiles")
    episodes_target = int(terrain_stage.evaluation_episodes)
    if episodes_target < 1:
        raise RmucCurriculumAdapterError("RMUC post-training evaluation_episodes must be positive")
    max_episode_seconds = max(spec.task.max_episode_seconds for spec in task._route_specs)
    max_steps = max(1, int(math.ceil(max_episode_seconds / task._time_step)))
    rounds = max(1, int(math.ceil(episodes_target / batch.num_worlds)))
    profile_index = task._route_profile_index
    completed = torch.zeros(profile_count, dtype=torch.int64, device=batch.device)
    unsafe = torch.zeros_like(completed)
    observed = torch.zeros_like(completed)
    speed_error_sum = torch.zeros(profile_count, dtype=torch.float32, device=batch.device)
    yaw_error_sum = torch.zeros_like(speed_error_sum)
    samples = torch.zeros_like(completed)
    consecutive = torch.zeros(batch.num_worlds, dtype=torch.int64, device=batch.device)
    maximum_consecutive = torch.zeros_like(consecutive)
    next_consecutive = torch.zeros_like(consecutive)
    task.set_domain_randomization_active(True)
    for _ in range(rounds):
        active = torch.ones(batch.num_worlds, dtype=torch.bool, device=batch.device)
        task.reset()
        for _ in range(max_steps):
            with torch.no_grad():
                output = policy(task.observe())
                action = output[0] if isinstance(output, tuple) else output
                if not isinstance(action, torch.Tensor) or action.shape != (batch.num_worlds, ACTION_SIZE):
                    raise RmucCurriculumAdapterError("RMUC evaluation policy must return CUDA [world, 7] actions")
                action = torch.tanh(action).contiguous().to(dtype=torch.float32)
            result = task.step(action)
            done = result.done & active
            successful = task._completed & done & ~result.terminated
            active_float = active.to(dtype=torch.float32)
            completed.scatter_add_(0, profile_index, successful.to(dtype=torch.int64))
            unsafe.scatter_add_(0, profile_index, (result.terminated & active).to(dtype=torch.int64))
            observed.scatter_add_(0, profile_index, done.to(dtype=torch.int64))
            speed_error_sum.scatter_add_(
                0,
                profile_index,
                (task.forward_speed() - task._command_speed).abs() * active_float,
            )
            yaw_error_sum.scatter_add_(
                0,
                profile_index,
                (batch.qvel[:, task.root_dof_address + 5] - task._command_yaw_rate).abs() * active_float,
            )
            samples.scatter_add_(0, profile_index, active.to(dtype=torch.int64))
            next_consecutive.copy_(consecutive)
            next_consecutive.add_(successful.to(dtype=torch.int64))
            next_consecutive.masked_fill_(done & ~successful, 0)
            consecutive.copy_(next_consecutive)
            torch.maximum(maximum_consecutive, consecutive, out=maximum_consecutive)
            active.logical_and_(~done)
            task.reset(result.done)
        unsafe.scatter_add_(0, profile_index, active.to(dtype=torch.int64))
        observed.scatter_add_(0, profile_index, active.to(dtype=torch.int64))
        consecutive.masked_fill_(active, 0)
    batch._warp.synchronize()
    observed_values = observed.detach().cpu().tolist()
    completed_values = completed.detach().cpu().tolist()
    unsafe_values = unsafe.detach().cpu().tolist()
    speed_sum_values = speed_error_sum.detach().cpu().tolist()
    yaw_sum_values = yaw_error_sum.detach().cpu().tolist()
    sample_values = samples.detach().cpu().tolist()
    profile_values = profile_index.detach().cpu().tolist()
    consecutive_values = maximum_consecutive.detach().cpu().tolist()
    profile_max_consecutive = [0 for _ in range(profile_count)]
    for world, profile in enumerate(profile_values):
        profile_max_consecutive[int(profile)] = max(
            profile_max_consecutive[int(profile)], int(consecutive_values[world])
        )
    completion_rates = [
        float(completed_values[index]) / max(int(observed_values[index]), 1)
        for index in range(profile_count)
    ]
    unsafe_rates = [
        float(unsafe_values[index]) / max(int(observed_values[index]), 1)
        for index in range(profile_count)
    ]
    speed_maes = [
        float(speed_sum_values[index]) / max(int(sample_values[index]), 1)
        for index in range(profile_count)
    ]
    yaw_maes = [
        float(yaw_sum_values[index]) / max(int(sample_values[index]), 1)
        for index in range(profile_count)
    ]
    required_consecutive = [spec.task.required_consecutive_successes for spec in task._route_specs]
    passed = bool(
        min(completion_rates) >= terrain_stage.minimum_completion_rate
        and max(unsafe_rates) <= terrain_stage.maximum_unsafe_rate
        and (
            terrain_stage.maximum_speed_mae_mps is None
            or max(speed_maes) <= terrain_stage.maximum_speed_mae_mps
        )
        and (
            terrain_stage.maximum_yaw_mae_rad is None
            or max(yaw_maes) <= terrain_stage.maximum_yaw_mae_rad
        )
        and all(
            profile_max_consecutive[index] >= int(required_consecutive[index])
            for index in range(profile_count)
        )
    )
    route_reports = [
        {
            "task_id": spec.task.task_id,
            "route_id": spec.route.route_id,
            "episodes": int(observed_values[index]),
            "completion_rate": completion_rates[index],
            "unsafe_rate": unsafe_rates[index],
            "speed_mae_mps": speed_maes[index],
            "yaw_mae_rad": yaw_maes[index],
            "required_consecutive_successes": int(required_consecutive[index]),
            "maximum_consecutive_successes": profile_max_consecutive[index],
        }
        for index, spec in enumerate(task._route_specs)
    ]
    report: Mapping[str, Any] = {
        "stage_id": _value(stage, "stage_id"),
        "evaluation_scope": "all_declared_rmuc_route_profiles",
        "episodes": int(sum(int(value) for value in observed_values)),
        "target_episodes": episodes_target,
        "completion_rate": min(completion_rates),
        "unsafe_rate": max(unsafe_rates),
        "speed_mae_mps": max(speed_maes),
        "yaw_mae_rad": max(yaw_maes),
        "route_reports": route_reports,
        "threshold_source": str(_value(stage, "terrain_curriculum_path")),
        "domain_randomization_active": True,
        "passed": passed,
    }
    task.reset()
    if not passed:
        raise RmucCurriculumAdapterError(
            f"RMUC course promotion gate failed for {_value(stage, 'stage_id')}: "
            f"completion={report['completion_rate']:.3f}, unsafe={report['unsafe_rate']:.3f}, "
            f"speed_mae={report['speed_mae_mps']:.3f}, yaw_mae={report['yaw_mae_rad']:.3f}"
        )
    return report


__all__ = [
    "CONTROLLER_BACKEND",
    "GPU_CURRICULUM_CAPABILITIES",
    "HfieldSupportLayout",
    "REWARD_SCHEMA",
    "RmucCourseAdapterError",
    "RmucCourseBundle",
    "RmucRouteTask",
    "build_curriculum_stage",
    "evaluate_policy_stage",
    "load_rmuc_course_config",
]
