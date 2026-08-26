"""Fail-closed CUDA adapter for the minimal official 15-degree uphill route.

Only ``grade15_up`` is exposed here.  The robot starts on the horizontal lead
surface with the calibrated yaw and climbs the 15-degree static box ramp.  It
does not imply GPU parity for grade15-down, 20-degree ramps, steps, jumps, or
the doghole.  Those task ids are rejected before CUDA allocation.

The adapter supplies three pieces missing from the flat task: static-box
support-height queries, the existing 16-D terrain-observation contract, and
per-world route progress/completion metadata.  MuJoCo-Warp still owns all
physics, control clipping, estop, and torque-safe force staging.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from terrain_curriculum import (
    TerrainCurriculumConfig,
    TerrainCurriculumError,
    TerrainRoute,
    TerrainTask,
    load_terrain_curriculum,
    validate_scene_contract,
)
from warp_safety import SAFETY_REASON_CONTACT_LOSS
from warp_task import (
    ACTION_SIZE,
    GUIDE_WHEEL_CONTACT_GEOM_NAMES,
    OBSERVATION_SIZE,
    WarpFlatWalkingTask,
    WarpTaskStep,
    combine_side_support_contacts,
)


STAGE_ID = "official_grade15_up"
TASK_MODE = "official_grade15_up"
CONTROLLER_BACKEND = "fixed_gain_flat_controller_v2"
REWARD_SCHEMA = "warp_official_grade15_route_reward_v1"
SUPPORTED_TERRAIN_STAGE = "grade15"
SUPPORTED_TASK_ID = "grade15_up"
SUPPORTED_ROUTE_INDEX = 0
EXPECTED_ACTION_MASK = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
EXPECTED_SUPPORT_GEOMS = (
    "support_grade15_lead",
    "support_grade15_ramp",
    "support_grade15_platform",
)


class OfficialGrade15AdapterError(ValueError):
    """Raised before an unsupported official task reaches the CUDA backend."""


@dataclass(frozen=True)
class TerrainFeatureSettings:
    """Observation normalization defined by the adapter YAML."""

    lookahead_distances_m: tuple[float, ...]
    lateral_offsets_m: tuple[float, ...]
    height_normalization_m: float
    slope_normalization: float

    def __post_init__(self) -> None:
        if not self.lookahead_distances_m or not self.lateral_offsets_m:
            raise OfficialGrade15AdapterError("terrain feature samples must not be empty")
        if len(self.lookahead_distances_m) * len(self.lateral_offsets_m) != 12:
            raise OfficialGrade15AdapterError("terrain features require exactly 12 height-preview samples")
        for name, values in (
            ("lookahead_distances_m", self.lookahead_distances_m),
            ("lateral_offsets_m", self.lateral_offsets_m),
        ):
            if any(not math.isfinite(float(value)) for value in values):
                raise OfficialGrade15AdapterError(f"{name} must contain finite values")
        if any(value <= 0.0 for value in self.lookahead_distances_m):
            raise OfficialGrade15AdapterError("lookahead_distances_m must be positive")
        if not math.isfinite(self.height_normalization_m) or self.height_normalization_m <= 0.0:
            raise OfficialGrade15AdapterError("height_normalization_m must be positive")
        if not math.isfinite(self.slope_normalization) or self.slope_normalization <= 0.0:
            raise OfficialGrade15AdapterError("slope_normalization must be positive")


@dataclass(frozen=True)
class RouteRewardSettings:
    """Bounded route reward terms; physical unsafe penalties remain task-owned."""

    progress_reward_per_m: float
    progress_delta_clip_m: float
    completion_bonus: float

    def __post_init__(self) -> None:
        values = (
            ("progress_reward_per_m", self.progress_reward_per_m),
            ("progress_delta_clip_m", self.progress_delta_clip_m),
            ("completion_bonus", self.completion_bonus),
        )
        for name, value in values:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise OfficialGrade15AdapterError(f"{name} must be finite and non-negative")
        if self.progress_delta_clip_m <= 0.0:
            raise OfficialGrade15AdapterError("progress_delta_clip_m must be positive")


@dataclass(frozen=True)
class ControllerSettings:
    """Bounded command feedforward settings for the grade15 YAML stage."""

    command_speed_gain_nm_per_mps: float
    command_yaw_gain_nm_per_rad_s: float
    command_wheel_feedforward_limit_nm: float
    command_wheel_accel_limit_nm: float | None
    command_wheel_brake_limit_nm: float | None
    terrain_support_reference_max_rate_mps: float

    def __post_init__(self) -> None:
        for name in (
            "command_speed_gain_nm_per_mps",
            "command_yaw_gain_nm_per_rad_s",
            "command_wheel_feedforward_limit_nm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise OfficialGrade15AdapterError(f"controller.{name} must be finite and non-negative")
        for name in ("command_wheel_accel_limit_nm", "command_wheel_brake_limit_nm"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise OfficialGrade15AdapterError(f"controller.{name} must be finite and non-negative when supplied")


@dataclass(frozen=True)
class StabilityGateSettings:
    """Runtime evidence required before this conditional capability trains."""

    duration_seconds: float
    require_no_terminated: bool
    require_no_overflow: bool
    require_finite_state: bool
    minimum_progress_m: float
    maximum_speed_mae_mps: float
    maximum_unsafe_rate: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.duration_seconds)) or self.duration_seconds <= 0.0:
            raise OfficialGrade15AdapterError("stability_gate.duration_seconds must be positive")
        if not (
            self.require_no_terminated
            and self.require_no_overflow
            and self.require_finite_state
        ):
            raise OfficialGrade15AdapterError(
                "the official grade15 gate must require no terminations/overflow and finite state"
            )
        if not math.isfinite(float(self.minimum_progress_m)) or self.minimum_progress_m <= 0.0:
            raise OfficialGrade15AdapterError("stability_gate.minimum_progress_m must be positive")
        if not math.isfinite(float(self.maximum_speed_mae_mps)) or self.maximum_speed_mae_mps <= 0.0:
            raise OfficialGrade15AdapterError("stability_gate.maximum_speed_mae_mps must be positive")
        if not math.isfinite(float(self.maximum_unsafe_rate)) or not 0.0 <= self.maximum_unsafe_rate <= 1.0:
            raise OfficialGrade15AdapterError("stability_gate.maximum_unsafe_rate must be within [0, 1]")


@dataclass(frozen=True)
class OfficialGrade15AdapterConfig:
    """Strict YAML contract for the only currently admissible official route."""

    source_path: Path
    stage_id: str
    task_mode: str
    batch_config_path: Path
    flat_ppo_config_path: Path
    terrain_curriculum_path: Path
    canonical_scene_path: Path
    terrain_stage_id: str
    task_id: str
    route_index: int
    support_geoms: tuple[str, ...]
    terrain_features: TerrainFeatureSettings
    route_reward: RouteRewardSettings
    controller: ControllerSettings
    stability_gate: StabilityGateSettings


@dataclass(frozen=True)
class StaticBoxSupportLayout:
    """Validated, immutable top-surface planes for declared static box geoms."""

    names: tuple[str, ...]
    center: np.ndarray
    rotation: np.ndarray
    half_size: np.ndarray
    inverse_top_xy: np.ndarray

    @classmethod
    def from_model(cls, model: Any, support_geoms: Sequence[str], mujoco: Any) -> "StaticBoxSupportLayout":
        names = tuple(str(name) for name in support_geoms)
        if names != EXPECTED_SUPPORT_GEOMS:
            raise OfficialGrade15AdapterError(
                "minimal official adapter requires exactly support_grade15_lead/ramp/platform"
            )
        geom_ids: list[int] = []
        for name in names:
            try:
                geom_id = int(model.geom(name).id)
            except KeyError as error:
                raise OfficialGrade15AdapterError(f"required static support geom is missing: {name}") from error
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise OfficialGrade15AdapterError(f"support geom {name!r} must be a box")
            if int(model.geom_bodyid[geom_id]) != 0:
                raise OfficialGrade15AdapterError(f"support geom {name!r} must be fixed to the world body")
            geom_ids.append(geom_id)
        center = np.asarray(model.geom_pos[geom_ids], dtype=np.float64).copy()
        half_size = np.asarray(model.geom_size[geom_ids, :3], dtype=np.float64).copy()
        quaternion = np.asarray(model.geom_quat[geom_ids], dtype=np.float64).copy()
        rotation = np.stack(tuple(_rotation_matrix_from_quaternion(value) for value in quaternion))
        inverse_top_xy = np.empty((len(geom_ids), 2, 2), dtype=np.float64)
        for index, (name, matrix, size) in enumerate(zip(names, rotation, half_size)):
            if not np.isfinite(matrix).all() or not np.isfinite(size).all() or np.any(size <= 0.0):
                raise OfficialGrade15AdapterError(f"support geom {name!r} has invalid static geometry")
            xy = matrix[:2, :2]
            determinant = float(np.linalg.det(xy))
            if not math.isfinite(determinant) or abs(determinant) <= 1.0e-7:
                raise OfficialGrade15AdapterError(f"support geom {name!r} top face is not a world XY graph")
            inverse_top_xy[index] = np.linalg.inv(xy)
        return cls(
            names=names,
            center=center,
            rotation=rotation,
            half_size=half_size,
            inverse_top_xy=inverse_top_xy,
        )

    def surface_height_cpu(self, world_xy: Sequence[float]) -> tuple[float, bool]:
        """Return the highest declared support top under one XY query."""

        xy = np.asarray(world_xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("world_xy must be a finite [x, y] vector")
        heights: list[float] = []
        for center, rotation, size, inverse in zip(
            self.center, self.rotation, self.half_size, self.inverse_top_xy
        ):
            top_xy_offset = rotation[:2, 2] * size[2]
            local_xy = inverse @ (xy - center[:2] - top_xy_offset)
            if np.all(np.abs(local_xy) <= size[:2] + 1.0e-8):
                height = float(
                    center[2]
                    + np.dot(rotation[2, :2], local_xy)
                    + rotation[2, 2] * size[2]
                )
                heights.append(height)
        return (max(heights), True) if heights else (0.0, False)


def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    if quaternion.shape != (4,):
        raise OfficialGrade15AdapterError("static geom quaternion must have four elements")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-8:
        raise OfficialGrade15AdapterError("static geom quaternion must be finite and non-zero")
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
    """Compose scalar-first quaternions without importing a runtime backend."""

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
    """Mirror ``WarpFlatWalkingTask.forward_direction`` on the host at setup."""

    matrix = _rotation_matrix_from_quaternion(np.asarray(quaternion, dtype=np.float64))
    axle_x, axle_y = float(matrix[0, 0]), float(matrix[1, 0])
    return math.atan2(axle_x, -axle_y)


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialGrade15AdapterError(f"{name} must be a YAML mapping")
    return value


def _exact_keys(mapping: Mapping[str, Any], name: str, expected: set[str]) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        pieces: list[str] = []
        if missing:
            pieces.append(f"missing={missing}")
        if unknown:
            pieces.append(f"unknown={unknown}")
        raise OfficialGrade15AdapterError(f"{name} keys are invalid: {', '.join(pieces)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise OfficialGrade15AdapterError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialGrade15AdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        raise OfficialGrade15AdapterError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _path(source: Path, value: Any, name: str) -> Path:
    candidate = Path(_string(value, name))
    result = candidate.resolve() if candidate.is_absolute() else (source.parent / candidate).resolve()
    if not result.is_file():
        raise OfficialGrade15AdapterError(f"{name} does not exist: {result}")
    return result


def _tuple_of_floats(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise OfficialGrade15AdapterError(f"{name} must be a non-empty sequence")
    return tuple(_finite(entry, f"{name}[{index}]") for index, entry in enumerate(value))


def load_official_grade15_adapter_config(path: str | Path) -> OfficialGrade15AdapterConfig:
    """Load the explicit YAML settings for the grade15-up probe/training gate."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise OfficialGrade15AdapterError(f"unable to read official grade15 adapter config {source}: {error}") from error
    root = _mapping(raw, "official grade15 adapter config")
    _exact_keys(
        root,
        "official grade15 adapter config",
        {
            "schema_version",
            "stage_id",
            "task_mode",
            "batch_config",
            "flat_ppo_config",
            "terrain_curriculum",
            "canonical_scene",
            "terrain_stage_id",
            "task_id",
            "route_index",
            "support_geoms",
            "terrain_features",
            "route_reward",
            "controller",
            "stability_gate",
        },
    )
    if root["schema_version"] != 1:
        raise OfficialGrade15AdapterError("official grade15 adapter schema_version must be 1")
    route_index = root["route_index"]
    if isinstance(route_index, bool) or not isinstance(route_index, int) or route_index < 0:
        raise OfficialGrade15AdapterError("route_index must be a non-negative integer")
    support_geoms = root["support_geoms"]
    if not isinstance(support_geoms, list):
        raise OfficialGrade15AdapterError("support_geoms must be a sequence")
    support_names = tuple(_string(value, f"support_geoms[{index}]") for index, value in enumerate(support_geoms))
    terrain = _mapping(root["terrain_features"], "terrain_features")
    _exact_keys(
        terrain,
        "terrain_features",
        {"lookahead_distances_m", "lateral_offsets_m", "height_normalization_m", "slope_normalization"},
    )
    reward = _mapping(root["route_reward"], "route_reward")
    _exact_keys(reward, "route_reward", {"progress_reward_per_m", "progress_delta_clip_m", "completion_bonus"})
    controller = _mapping(root["controller"], "controller")
    _exact_keys(
        controller,
        "controller",
        {
            "command_speed_gain_nm_per_mps",
            "command_yaw_gain_nm_per_rad_s",
            "command_wheel_feedforward_limit_nm",
            "command_wheel_accel_limit_nm",
            "command_wheel_brake_limit_nm",
            "terrain_support_reference_max_rate_mps",
        },
    )
    gate = _mapping(root["stability_gate"], "stability_gate")
    _exact_keys(
        gate,
        "stability_gate",
        {
            "duration_seconds", "require_no_terminated", "require_no_overflow", "require_finite_state",
            "minimum_progress_m", "maximum_speed_mae_mps", "maximum_unsafe_rate",
        },
    )
    return OfficialGrade15AdapterConfig(
        source_path=source,
        stage_id=_string(root["stage_id"], "stage_id"),
        task_mode=_string(root["task_mode"], "task_mode"),
        batch_config_path=_path(source, root["batch_config"], "batch_config"),
        flat_ppo_config_path=_path(source, root["flat_ppo_config"], "flat_ppo_config"),
        terrain_curriculum_path=_path(source, root["terrain_curriculum"], "terrain_curriculum"),
        canonical_scene_path=_path(source, root["canonical_scene"], "canonical_scene"),
        terrain_stage_id=_string(root["terrain_stage_id"], "terrain_stage_id"),
        task_id=_string(root["task_id"], "task_id"),
        route_index=int(route_index),
        support_geoms=support_names,
        terrain_features=TerrainFeatureSettings(
            lookahead_distances_m=_tuple_of_floats(terrain["lookahead_distances_m"], "terrain_features.lookahead_distances_m"),
            lateral_offsets_m=_tuple_of_floats(terrain["lateral_offsets_m"], "terrain_features.lateral_offsets_m"),
            height_normalization_m=_finite(terrain["height_normalization_m"], "terrain_features.height_normalization_m", positive=True),
            slope_normalization=_finite(terrain["slope_normalization"], "terrain_features.slope_normalization", positive=True),
        ),
        route_reward=RouteRewardSettings(
            progress_reward_per_m=_finite(reward["progress_reward_per_m"], "route_reward.progress_reward_per_m", nonnegative=True),
            progress_delta_clip_m=_finite(reward["progress_delta_clip_m"], "route_reward.progress_delta_clip_m", positive=True),
            completion_bonus=_finite(reward["completion_bonus"], "route_reward.completion_bonus", nonnegative=True),
        ),
        controller=ControllerSettings(
            command_speed_gain_nm_per_mps=_finite(
                controller["command_speed_gain_nm_per_mps"],
                "controller.command_speed_gain_nm_per_mps",
                nonnegative=True,
            ),
            command_yaw_gain_nm_per_rad_s=_finite(
                controller["command_yaw_gain_nm_per_rad_s"],
                "controller.command_yaw_gain_nm_per_rad_s",
                nonnegative=True,
            ),
            command_wheel_feedforward_limit_nm=_finite(
                controller["command_wheel_feedforward_limit_nm"],
                "controller.command_wheel_feedforward_limit_nm",
                nonnegative=True,
            ),
            command_wheel_accel_limit_nm=(
                None
                if controller["command_wheel_accel_limit_nm"] is None
                else _finite(
                    controller["command_wheel_accel_limit_nm"],
                    "controller.command_wheel_accel_limit_nm",
                    nonnegative=True,
                )
            ),
            command_wheel_brake_limit_nm=(
                None
                if controller["command_wheel_brake_limit_nm"] is None
                else _finite(
                    controller["command_wheel_brake_limit_nm"],
                    "controller.command_wheel_brake_limit_nm",
                    nonnegative=True,
                )
            ),
            terrain_support_reference_max_rate_mps=_finite(
                controller["terrain_support_reference_max_rate_mps"],
                "controller.terrain_support_reference_max_rate_mps",
                nonnegative=True,
            ),
        ),
        stability_gate=StabilityGateSettings(
            duration_seconds=_finite(gate["duration_seconds"], "stability_gate.duration_seconds", positive=True),
            require_no_terminated=_boolean(gate["require_no_terminated"], "stability_gate.require_no_terminated"),
            require_no_overflow=_boolean(gate["require_no_overflow"], "stability_gate.require_no_overflow"),
            require_finite_state=_boolean(gate["require_finite_state"], "stability_gate.require_finite_state"),
            minimum_progress_m=_finite(gate["minimum_progress_m"], "stability_gate.minimum_progress_m", positive=True),
            maximum_speed_mae_mps=_finite(
                gate["maximum_speed_mae_mps"], "stability_gate.maximum_speed_mae_mps", positive=True
            ),
            maximum_unsafe_rate=_finite(
                gate["maximum_unsafe_rate"], "stability_gate.maximum_unsafe_rate", nonnegative=True
            ),
        ),
    )


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OfficialGrade15AdapterError(f"{name} must be boolean")
    return value


def validate_official_warp_scene_variant(
    curriculum: TerrainCurriculumConfig,
    *,
    canonical_scene_path: Path,
    variant_scene_path: Path,
    curriculum_path: Path,
) -> None:
    """Validate the collision-safe Warp scene against the canonical contract.

    The variant is admitted only when it retains the official model identity
    and every declared support/obstacle geom.  Its sole allowed support-plane
    change is disabling the inherited global projection plane, which prevents
    it from masking a missing route support in MuJoCo-Warp.
    """

    validate_scene_contract(curriculum, canonical_scene_path, curriculum_path=curriculum_path)
    if variant_scene_path.resolve() == canonical_scene_path.resolve():
        return
    if variant_scene_path.name != "official_standard_warp_ground.xml":
        raise OfficialGrade15AdapterError(
            "official GPU curriculum accepts only the explicitly validated official_standard_warp_ground.xml variant"
        )
    try:
        canonical_root = ET.parse(canonical_scene_path).getroot()
        variant_root = ET.parse(variant_scene_path).getroot()
    except (OSError, ET.ParseError) as error:
        raise OfficialGrade15AdapterError(f"cannot parse official Warp scene variant: {error}") from error
    contract = curriculum.scene_contract
    if contract is None or variant_root.tag != "mujoco" or variant_root.attrib.get("model") != contract.mjcf_model:
        raise OfficialGrade15AdapterError("official Warp scene variant model identity does not match the canonical contract")

    def geoms(root: Any) -> dict[str, Mapping[str, str]]:
        return {
            str(geom.attrib["name"]): dict(geom.attrib)
            for geom in root.iter("geom")
            if "name" in geom.attrib
        }

    canonical_geoms = geoms(canonical_root)
    variant_geoms = geoms(variant_root)
    expected = set(contract.support_geoms) | set(contract.obstacle_geoms)
    missing = sorted(expected - set(variant_geoms))
    if missing:
        raise OfficialGrade15AdapterError("official Warp scene variant omits contract geoms: " + ", ".join(missing))
    # Support shape/pose/collision attributes define the actual official
    # terrain.  They must be byte-identical across the canonical and Warp
    # scenes; cosmetic rendering attributes are intentionally irrelevant.
    support_fields = ("type", "pos", "size", "quat", "fromto", "friction", "solref", "contype", "conaffinity")
    for name in contract.support_geoms:
        canonical_geom = canonical_geoms.get(name)
        variant_geom = variant_geoms.get(name)
        if canonical_geom is None or variant_geom is None:
            raise OfficialGrade15AdapterError(f"official support geom {name!r} is absent from canonical or Warp scene")
        if any(canonical_geom.get(field) != variant_geom.get(field) for field in support_fields):
            raise OfficialGrade15AdapterError(f"official Warp scene alters support geometry or collision contract: {name}")
    ground = variant_geoms.get("ground")
    if ground is None or ground.get("type") != "plane" or ground.get("contype") != "0" or ground.get("conaffinity") != "0":
        raise OfficialGrade15AdapterError(
            "official Warp scene must disable the global projection plane collision to fail closed outside route support"
        )


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        if name not in source:
            raise OfficialGrade15AdapterError(f"missing stage field {name!r}")
        return source[name]
    if not hasattr(source, name):
        raise OfficialGrade15AdapterError(f"missing stage field {name!r}")
    return getattr(source, name)


def _adapter_path(stage: Any) -> Path:
    value = _value(stage, "adapter_config_path")
    if value is None:
        raise OfficialGrade15AdapterError("official grade15 stage requires adapter_config_path")
    path = Path(value).resolve()
    if not path.is_file():
        raise OfficialGrade15AdapterError(f"adapter_config_path does not exist: {path}")
    return path


def validate_official_grade15_contract(
    stage: Any,
    adapter: OfficialGrade15AdapterConfig,
) -> tuple[TerrainCurriculumConfig, TerrainTask, TerrainRoute]:
    """Validate scene, stage and route identities before a CUDA allocation."""

    expected_flags = {
        "terrain_enabled": True,
        "steps_enabled": False,
        "jump_enabled": False,
        "domain_randomization_enabled": False,
        "requires_gpu_parity": True,
    }
    if _value(stage, "stage_id") != adapter.stage_id or adapter.stage_id != STAGE_ID:
        raise OfficialGrade15AdapterError("adapter supports only the official_grade15_up stage")
    if _value(stage, "task_mode") != adapter.task_mode or adapter.task_mode != TASK_MODE:
        raise OfficialGrade15AdapterError("adapter task_mode must be official_grade15_up")
    if _value(stage, "controller_backend") != CONTROLLER_BACKEND:
        raise OfficialGrade15AdapterError("grade15-up requires the fixed-gain CUDA controller backend")
    if _value(stage, "scene_variant") != "official_warp_compat":
        raise OfficialGrade15AdapterError(
            "official grade15-up requires the strict official_warp_compat scene variant"
        )
    for name, expected in expected_flags.items():
        if _value(stage, name) is not expected:
            raise OfficialGrade15AdapterError(f"official_grade15_up.{name} must be {str(expected).lower()}")
    action_mask = tuple(float(value) for value in _value(stage, "residual_action_mask"))
    if action_mask != EXPECTED_ACTION_MASK:
        raise OfficialGrade15AdapterError("official_grade15_up must retain six residual controls and mask channel seven")
    if adapter.terrain_stage_id != SUPPORTED_TERRAIN_STAGE or adapter.task_id != SUPPORTED_TASK_ID:
        raise OfficialGrade15AdapterError("only official grade15_up in stage grade15 is supported")
    if adapter.route_index != SUPPORTED_ROUTE_INDEX:
        raise OfficialGrade15AdapterError("only official_grade15_up route_index 0 is supported")
    if adapter.support_geoms != EXPECTED_SUPPORT_GEOMS:
        raise OfficialGrade15AdapterError("official grade15-up support geometry contract is invalid")
    if Path(_value(stage, "terrain_curriculum_path")).resolve() != adapter.terrain_curriculum_path:
        raise OfficialGrade15AdapterError("stage terrain_curriculum_path must match adapter YAML")
    curriculum = load_terrain_curriculum(adapter.terrain_curriculum_path)
    if curriculum.schema_version != 4:
        raise OfficialGrade15AdapterError("official grade15-up requires terrain curriculum schema 4")
    if _value(stage, "terrain_stage_id") != adapter.terrain_stage_id:
        raise OfficialGrade15AdapterError("stage terrain_stage_id must match adapter YAML")
    terrain_stage = curriculum.stage(adapter.terrain_stage_id)
    if tuple(terrain_stage.task_ids) != (SUPPORTED_TASK_ID, "grade15_down"):
        raise OfficialGrade15AdapterError("official grade15 stage changed; refusing partial route substitution")
    task = curriculum.task(adapter.task_id)
    if task.command.jump_request or task.has_progress_jump_trigger or task.jump_launch_speed_mps is not None:
        raise OfficialGrade15AdapterError("jump-capable official tasks are not admitted by the grade15 adapter")
    route = task.route_at(adapter.route_index)
    if abs(route.spawn.yaw_rad) > 1.0e-7:
        raise OfficialGrade15AdapterError("grade15-up must retain the calibrated zero-yaw spawn")
    expected_command = terrain_stage.command_for(task)
    if not math.isclose(float(_value(stage, "command_speed_mps")), expected_command.forward_speed_mps, abs_tol=1.0e-7):
        raise OfficialGrade15AdapterError("stage speed command must equal the YAML grade15 scaled command")
    if not math.isclose(float(_value(stage, "command_yaw_rate_rad_s")), expected_command.yaw_rate_rad_s, abs_tol=1.0e-7):
        raise OfficialGrade15AdapterError("stage yaw command must equal the YAML grade15 scaled command")
    xml_path = Path(_value(stage, "xml_path")).resolve()
    try:
        from build_official_standard_ground import validate_official_warp_scene

        validate_official_warp_scene(xml_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise OfficialGrade15AdapterError(
            f"official grade15-up Warp scene variant validation failed: {error}"
        ) from error
    validate_official_warp_scene_variant(
        curriculum,
        canonical_scene_path=adapter.canonical_scene_path,
        variant_scene_path=xml_path,
        curriculum_path=adapter.terrain_curriculum_path,
    )
    binding = next(
        (
            item
            for item in (curriculum.scene_contract.route_bindings if curriculum.scene_contract else ())
            if item.task_id == task.task_id and item.route_id == route.route_id
        ),
        None,
    )
    if binding is None or binding.support_geoms != adapter.support_geoms or binding.obstacle_geoms:
        raise OfficialGrade15AdapterError("grade15-up route must bind exactly the declared non-obstacle support boxes")
    return curriculum, task, route


class StaticBoxTerrain16D:
    """GPU provider for the declared grade15 static-box support surfaces."""

    def __init__(self, task: WarpFlatWalkingTask, layout: StaticBoxSupportLayout, settings: TerrainFeatureSettings) -> None:
        self.task = task
        self.layout = layout
        self.settings = settings
        torch = task.torch
        self.torch = torch
        self.device = task.device
        self.num_worlds = task.num_worlds
        count = len(layout.names)
        self._center_xy = torch.as_tensor(layout.center[:, :2], dtype=torch.float32, device=self.device)
        self._center_z = torch.as_tensor(layout.center[:, 2], dtype=torch.float32, device=self.device)
        self._half_xy = torch.as_tensor(layout.half_size[:, :2], dtype=torch.float32, device=self.device)
        self._half_z = torch.as_tensor(layout.half_size[:, 2], dtype=torch.float32, device=self.device)
        self._inverse_xy = torch.as_tensor(layout.inverse_top_xy, dtype=torch.float32, device=self.device)
        self._rotation_xy_z = torch.as_tensor(layout.rotation[:, :2, 2], dtype=torch.float32, device=self.device)
        self._rotation_z_xy = torch.as_tensor(layout.rotation[:, 2, :2], dtype=torch.float32, device=self.device)
        self._rotation_zz = torch.as_tensor(layout.rotation[:, 2, 2], dtype=torch.float32, device=self.device)
        offsets = tuple(
            (distance, lateral)
            for distance in settings.lookahead_distances_m
            for lateral in settings.lateral_offsets_m
        )
        self._lookahead_offsets = torch.as_tensor(offsets, dtype=torch.float32, device=self.device)
        self._near_offsets = torch.as_tensor(
            ((0.20, 0.0), (-0.20, 0.0), (0.0, 0.16), (0.0, -0.16)),
            dtype=torch.float32,
            device=self.device,
        )
        self._feature_buffer = torch.zeros((self.num_worlds, 16), dtype=torch.float32, device=self.device)
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
        self._guide_wheel_xy = torch.empty(
            (self.num_worlds, len(GUIDE_WHEEL_CONTACT_GEOM_NAMES), 2),
            dtype=torch.float32,
            device=self.device,
        )
        self._guide_wheel_height = torch.empty(
            (self.num_worlds, len(GUIDE_WHEEL_CONTACT_GEOM_NAMES)),
            dtype=torch.float32,
            device=self.device,
        )
        self._guide_wheel_valid = torch.empty(
            (self.num_worlds, len(GUIDE_WHEEL_CONTACT_GEOM_NAMES)),
            dtype=torch.bool,
            device=self.device,
        )
        self._guide_side_valid = torch.empty((self.num_worlds, 2), dtype=torch.bool, device=self.device)
        self._support_valid = torch.empty((self.num_worlds, 2), dtype=torch.bool, device=self.device)
        self._forward = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._lateral = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._last_support_valid = torch.empty(self.num_worlds, dtype=torch.bool, device=self.device)
        self._box_count = count

    @property
    def wheel_support_valid(self) -> Any:
        """Per-side active-or-lower-guide terrain coverage on CUDA."""
        return self._support_valid

    @property
    def last_support_valid(self) -> Any:
        return self._last_support_valid

    def _sample_surface(self, xy: Any, height_out: Any, valid_out: Any) -> None:
        """Evaluate the top of every declared box and retain the highest one."""

        torch = self.torch
        shifted = (
            xy.unsqueeze(2)
            - self._center_xy.view(1, 1, self._box_count, 2)
            - (self._rotation_xy_z * self._half_z.unsqueeze(1)).view(1, 1, self._box_count, 2)
        )
        local_xy = torch.einsum("bij,nkbj->nkbi", self._inverse_xy, shifted)
        inside = (local_xy.abs() <= self._half_xy.view(1, 1, self._box_count, 2)).all(dim=-1)
        height = (
            self._center_z.view(1, 1, self._box_count)
            + (local_xy * self._rotation_z_xy.view(1, 1, self._box_count, 2)).sum(dim=-1)
            + (self._rotation_zz * self._half_z).view(1, 1, self._box_count)
        )
        valid_out.copy_(inside.any(dim=2))
        height_out.copy_(torch.where(inside, height, torch.full_like(height, -torch.inf)).amax(dim=2))
        height_out.masked_fill_(~valid_out, 0.0)

    def _update_heading(self) -> None:
        # Reuse the same wheel-axis-derived rolling direction used by the
        # controller and reward. Root yaw alone is not this MJCF's forward
        # convention, particularly after a route reset rotates the robot.
        self._forward.copy_(self.task.forward_direction())
        self._lateral[:, 0] = -self._forward[:, 1]
        self._lateral[:, 1] = self._forward[:, 0]

    def _update_guide_support_samples(self) -> None:
        """Refresh private lower-guide support data without changing observations."""

        torch = self.torch
        task = self.task
        torch.index_select(
            task._geom_xpos,
            1,
            task._guide_wheel_geom_gpu,
            out=task._guide_wheel_positions,
        )
        self._guide_wheel_xy.copy_(task._guide_wheel_positions[..., :2])
        self._sample_surface(
            self._guide_wheel_xy,
            self._guide_wheel_height,
            self._guide_wheel_valid,
        )
        torch.index_select(
            self._guide_wheel_valid,
            1,
            task._guide_left_indices,
            out=task._guide_left_contact_values,
        )
        torch.index_select(
            self._guide_wheel_valid,
            1,
            task._guide_right_indices,
            out=task._guide_right_contact_values,
        )
        torch.any(task._guide_left_contact_values, dim=1, out=self._guide_side_valid[:, 0])
        torch.any(task._guide_right_contact_values, dim=1, out=self._guide_side_valid[:, 1])
        torch.logical_or(self._wheel_valid, self._guide_side_valid, out=self._support_valid)

    def update_features(self) -> Any:
        """Write the 16-D terrain preview for the current CUDA task state."""

        torch = self.torch
        qpos = self.task.batch.qpos
        start = self.task.root_qpos_address
        self._root_xy[:, 0].copy_(qpos[:, start : start + 2])
        self._update_heading()
        self._sample_xy.copy_(self._root_xy)
        self._sample_xy.add_(
            self._forward.unsqueeze(1) * self._lookahead_offsets[:, 0].view(1, 12, 1)
        )
        self._sample_xy.add_(
            self._lateral.unsqueeze(1) * self._lookahead_offsets[:, 1].view(1, 12, 1)
        )
        self._near_xy.copy_(self._root_xy)
        self._near_xy.add_(
            self._forward.unsqueeze(1) * self._near_offsets[:, 0].view(1, 4, 1)
        )
        self._near_xy.add_(
            self._lateral.unsqueeze(1) * self._near_offsets[:, 1].view(1, 4, 1)
        )
        self._sample_surface(self._root_xy, self._root_height, self._root_valid)
        self._sample_surface(self._sample_xy, self._sample_height, self._sample_valid)
        self._sample_surface(self._near_xy, self._near_height, self._near_valid)
        torch.index_select(self.task._geom_xpos, 1, self.task._wheel_geom_gpu, out=self.task._wheel_positions)
        self._wheel_xy.copy_(self.task._wheel_positions[..., :2])
        self._sample_surface(self._wheel_xy, self._wheel_height, self._wheel_valid)
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
        torch.logical_and(self._root_valid[:, 0], self._sample_valid.all(dim=1), out=self._last_support_valid)
        self._last_support_valid.logical_and_(self._near_valid.all(dim=1))
        self._last_support_valid.logical_and_(self._support_valid.all(dim=1))
        features[:, 15] = self._last_support_valid.to(dtype=torch.float32)
        self.task.set_terrain_features(features)
        return features

    def wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        """Use verified static support heights for contact safety and rewards."""

        torch = self.torch
        torch.index_select(self.task._geom_xpos, 1, self.task._wheel_geom_gpu, out=self.task._wheel_positions)
        self._wheel_xy.copy_(self.task._wheel_positions[..., :2])
        self._sample_surface(self._wheel_xy, self._wheel_height, self._wheel_valid)
        torch.sub(self.task._wheel_positions[..., 2], self.task._wheel_radius, out=self.task._wheel_clearances)
        self.task._wheel_clearances.sub_(self._wheel_height).clamp_(min=0.0)
        torch.le(
            self.task._wheel_clearances,
            self.task.config.contact_clearance_m,
            out=self.task._wheel_contacts,
        )
        self.task._wheel_contacts.logical_and_(self._wheel_valid)
        return self.task._wheel_clearances, self.task._wheel_contacts

    def guide_wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        """Return current private lower-guide contacts against static supports."""

        torch = self.torch
        self._update_guide_support_samples()
        torch.sub(
            self.task._guide_wheel_positions[..., 2],
            self.task._guide_wheel_radius,
            out=self.task._guide_wheel_clearances,
        )
        self.task._guide_wheel_clearances.sub_(self._guide_wheel_height).clamp_(min=0.0)
        torch.le(
            self.task._guide_wheel_clearances,
            self.task.config.contact_clearance_m,
            out=self.task._guide_wheel_contacts,
        )
        self.task._guide_wheel_contacts.logical_and_(self._guide_wheel_valid)
        return self.task._guide_wheel_clearances, self.task._guide_wheel_contacts


class OfficialGrade15UpTask(WarpFlatWalkingTask):
    """Flat-task safety/control path with a verified grade15-up route layer."""

    def __init__(
        self,
        batch: Any,
        config: Any,
        *,
        calibration: Any,
        terrain_layout: StaticBoxSupportLayout,
        terrain_settings: TerrainFeatureSettings,
        terrain_task: TerrainTask,
        route: TerrainRoute,
        route_reward: RouteRewardSettings,
        command_speed_mps: float,
        controller: Any | None = None,
    ) -> None:
        self._official_terrain: StaticBoxTerrain16D | None = None
        self._official_route_ready = False
        self._official_task = terrain_task
        self._official_route = route
        self._official_route_reward_settings = route_reward
        super().__init__(batch, config, calibration=calibration, controller=controller)
        self._official_terrain = StaticBoxTerrain16D(self, terrain_layout, terrain_settings)
        torch = self.torch
        direction = np.asarray(
            (route.goal.x_m - route.spawn.x_m, route.goal.y_m - route.spawn.y_m), dtype=np.float64
        )
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length <= 1.0e-6:
            raise OfficialGrade15AdapterError("official grade15 route must have non-zero XY length")
        support_height, valid = terrain_layout.surface_height_cpu(route.spawn.xy())
        if not valid or abs(support_height) > 1.0e-6:
            raise OfficialGrade15AdapterError(
                "grade15-up spawn must remain on the zero-height horizontal lead support"
            )
        self._official_start_xy = torch.tensor(route.spawn.xy(), dtype=torch.float32, device=self.device)
        self._official_direction = torch.tensor(direction / length, dtype=torch.float32, device=self.device)
        self._official_completion_distance = float(
            terrain_task.required_distance_m - terrain_task.completion_tolerance_m
        )
        self._official_corridor_half_width = float(route.corridor_half_width_m)
        self._official_command_speed_mps = float(command_speed_mps)
        self._official_progress = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        self._official_previous_progress = torch.zeros_like(self._official_progress)
        self._official_lateral_error = torch.zeros_like(self._official_progress)
        self._official_completed = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._official_completion_this_step = torch.zeros_like(self._official_completed)
        self._official_route_reward = torch.zeros_like(self._official_progress)
        self._official_route_xy_delta = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._official_route_unsafe = torch.zeros_like(self._official_completed)
        # The calibrated root's wheel axis produces world +Y rolling. The
        # official uphill route runs along world +X, so rotate the root by the
        # exact delta required to align *physical* rolling direction with the
        # declared corridor. This also becomes the P0 fall-guard reference.
        root = self.root_qpos_address
        self._reset_qpos[:, root : root + 2] = self._official_start_xy
        base_quaternion = np.asarray(calibration.qpos[root + 3 : root + 7], dtype=np.float64)
        base_heading = _rolling_heading_from_quaternion(base_quaternion)
        route_heading = math.atan2(float(direction[1]), float(direction[0]))
        heading_delta = _wrap_angle(route_heading - base_heading)
        route_quaternion = _quaternion_multiply(
            np.asarray((math.cos(0.5 * heading_delta), 0.0, 0.0, math.sin(0.5 * heading_delta)), dtype=np.float64),
            base_quaternion,
        )
        route_quaternion /= np.linalg.norm(route_quaternion)
        self._reset_qpos[:, root + 3 : root + 7] = torch.as_tensor(
            route_quaternion, dtype=torch.float32, device=self.device
        )
        batch.set_fall_guard_reference(
            self._reset_qpos[0, root + 3 : root + 7].contiguous(),
            float(calibration.qpos[root + 2]),
        )
        self._official_route_ready = True
        self._command_speed.fill_(self._official_command_speed_mps)
        self._command_yaw_rate.zero_()
        self.reset()

    def set_feedback_controller(self, controller: Any) -> None:
        super().set_feedback_controller(controller)
        rebase = getattr(controller, "set_reference_state", None)
        if not callable(rebase):
            raise OfficialGrade15AdapterError(
                "official grade15-up requires a controller with reset-boundary reference rebasing"
            )
        if not callable(getattr(controller, "update_terrain_support_reference", None)):
            raise OfficialGrade15AdapterError(
                "official grade15-up requires a GPU terrain support reference updater"
            )
        rebase(self._reset_qpos, self._reset_qvel, self._all_world_mask)

    def _wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        if self._official_terrain is None:
            return super()._wheel_clearances_and_contacts()
        return self._official_terrain.wheel_clearances_and_contacts()

    def _guide_wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        if self._official_terrain is None:
            return super()._guide_wheel_clearances_and_contacts()
        return self._official_terrain.guide_wheel_clearances_and_contacts()

    def observe(self) -> Any:
        if self._official_terrain is not None:
            self._official_terrain.update_features()
        return super().observe()

    def reset(self, world_mask: Any | None = None) -> Any:
        result = super().reset(world_mask)
        if not self._official_route_ready:
            return result
        mask = self._require_mask(world_mask)
        self._official_progress[mask] = 0.0
        self._official_previous_progress[mask] = 0.0
        self._official_lateral_error[mask] = 0.0
        self._official_completed[mask] = False
        self._official_completion_this_step[mask] = False
        self._official_route_reward[mask] = 0.0
        self._official_route_unsafe[mask] = False
        self._command_speed[mask] = self._official_command_speed_mps
        self._command_yaw_rate[mask] = 0.0
        return self.observe()

    def _evaluate_safety(self, controls: Any) -> Any:
        result = super()._evaluate_safety(controls)
        terrain = self._official_terrain
        if terrain is None:
            return result
        invalid_support = ~terrain.wheel_support_valid.all(dim=1)
        result.safe_controls.masked_fill_(invalid_support.unsqueeze(1), 0.0)
        result.terminated.logical_or_(invalid_support)
        result.failure.logical_or_(invalid_support)
        result.contact_limit.logical_or_(invalid_support)
        result.reason_code.masked_fill_(invalid_support, SAFETY_REASON_CONTACT_LOSS)
        self._official_route_unsafe.copy_(invalid_support)
        return result

    def _update_route_progress(self, physically_terminated: Any) -> None:
        torch = self.torch
        root = self.root_qpos_address
        self._official_route_xy_delta.copy_(self.batch.qpos[:, root : root + 2])
        self._official_route_xy_delta.sub_(self._official_start_xy)
        self._official_previous_progress.copy_(self._official_progress)
        torch.sum(
            self._official_route_xy_delta * self._official_direction.unsqueeze(0),
            dim=1,
            out=self._official_progress,
        )
        self._official_lateral_error.copy_(
            self._official_route_xy_delta[:, 0] * self._official_direction[1]
            - self._official_route_xy_delta[:, 1] * self._official_direction[0]
        ).abs_()
        delta = torch.clamp(
            self._official_progress - self._official_previous_progress,
            -self._official_route_reward_settings.progress_delta_clip_m,
            self._official_route_reward_settings.progress_delta_clip_m,
        )
        self._official_route_reward.copy_(delta).mul_(self._official_route_reward_settings.progress_reward_per_m)
        completion = (
            (self._official_progress >= self._official_completion_distance)
            & (self._official_lateral_error <= self._official_corridor_half_width)
            & ~physically_terminated
            & ~self._official_route_unsafe
        )
        self._official_completion_this_step.copy_(completion & ~self._official_completed)
        self._official_completed.logical_or_(completion)
        self._official_route_reward.add_(
            self._official_completion_this_step.to(dtype=torch.float32)
            * self._official_route_reward_settings.completion_bonus
        )

    def step(self, action: Any, controls: Any | None = None) -> WarpTaskStep:
        if self._official_terrain is not None and self._controller is not None:
            # ``observe()`` at the previous policy boundary already refreshed
            # this resident support-height buffer; consume it before the next
            # controller call without duplicating terrain sampling in the hot
            # path.
            self._controller.update_terrain_support_reference(
                self._official_terrain._root_height[:, 0]
            )
        result = super().step(action, controls)
        if not self._official_route_ready:
            return result
        self._update_route_progress(result.terminated)
        completed = self._official_completion_this_step & ~result.terminated
        truncated = result.truncated | completed
        done = result.terminated | truncated
        self._episode_done.logical_or_(done)
        return WarpTaskStep(
            observation=result.observation,
            reward=result.reward + self._official_route_reward,
            terminated=result.terminated,
            truncated=truncated,
            done=done,
            physics=result.physics,
        )

    def tensors(self) -> Mapping[str, Any]:
        result = dict(super().tensors())
        result.update(
            {
                "official_route_progress_m": self._official_progress,
                "official_route_lateral_error_m": self._official_lateral_error,
                "official_route_completed": self._official_completed,
                "official_route_completion_this_step": self._official_completion_this_step,
                "official_route_reward": self._official_route_reward,
                "official_route_support_valid": self._official_terrain.last_support_valid if self._official_terrain else None,
            }
        )
        return result


@dataclass
class OfficialGrade15Bundle:
    """Factory return shape consumed by the generic CUDA curriculum runner."""

    batch: Any
    task: OfficialGrade15UpTask
    controller: Any
    run_stability_gate: Callable[[], Mapping[str, Any]]
    close: Callable[[], None]


def _make_stability_gate(batch: Any, task: OfficialGrade15UpTask, settings: StabilityGateSettings) -> Callable[[], Mapping[str, Any]]:
    if settings.duration_seconds > float(task.config.episode_seconds) + 1.0e-9:
        raise OfficialGrade15AdapterError("grade15 stability gate cannot exceed the configured route episode horizon")
    action_dt = float(task._time_step)
    if not math.isfinite(action_dt) or action_dt <= 0.0:
        raise OfficialGrade15AdapterError("grade15 task action timestep must be finite and positive")
    cache: dict[str, Mapping[str, Any]] = {}

    def run() -> Mapping[str, Any]:
        if "report" in cache:
            return cache["report"]
        torch = batch._torch
        steps = max(1, int(math.ceil(settings.duration_seconds / action_dt)))
        action = torch.zeros((batch.num_worlds, ACTION_SIZE), dtype=torch.float32, device=batch.device)
        terminated = torch.zeros(batch.num_worlds, dtype=torch.bool, device=batch.device)
        overflowed = torch.zeros_like(terminated)
        estopped = torch.zeros_like(terminated)
        max_progress = torch.full((batch.num_worlds,), -torch.inf, dtype=torch.float32, device=batch.device)
        max_lateral_error = torch.zeros_like(max_progress)
        speed_error = torch.empty_like(max_progress)
        speed_error_sum = torch.zeros((), dtype=torch.float32, device=batch.device)
        speed_sample_count = torch.zeros((), dtype=torch.int64, device=batch.device)
        unsafe_episode_count = torch.zeros((), dtype=torch.int64, device=batch.device)
        completed_episode_count = torch.zeros((), dtype=torch.int64, device=batch.device)
        observed_episode_count = torch.zeros((), dtype=torch.int64, device=batch.device)
        first_fault_step = torch.full((), -1, dtype=torch.int64, device=batch.device)
        first_fault_reason = torch.zeros((), dtype=torch.int64, device=batch.device)
        current_step_tensor = torch.zeros((), dtype=torch.int64, device=batch.device)
        fault_reason_buffer = torch.empty_like(terminated, dtype=torch.int64)
        zero_reason = torch.zeros_like(fault_reason_buffer)
        task.reset()
        for step_index in range(steps):
            result = task.step(action)
            new_fault = result.terminated & ~terminated
            current_step_tensor.fill_(step_index + 1)
            first_fault_mask = (~first_fault_step.ge(0)) & new_fault.any()
            torch.where(first_fault_mask, current_step_tensor, first_fault_step, out=first_fault_step)
            torch.where(new_fault, task._safety_reason_code, zero_reason, out=fault_reason_buffer)
            current_reason = fault_reason_buffer.max()
            torch.where(first_fault_mask, current_reason, first_fault_reason, out=first_fault_reason)
            terminated.logical_or_(result.terminated)
            overflowed.logical_or_(batch.overflow.ne(0))
            estopped.logical_or_(batch.estopped)
            torch.maximum(max_progress, task._official_progress, out=max_progress)
            torch.maximum(max_lateral_error, task._official_lateral_error, out=max_lateral_error)
            speed_error.copy_(task.forward_speed())
            speed_error.sub_(task._official_command_speed_mps).abs_()
            speed_error_sum.add_(speed_error.sum())
            speed_sample_count.add_(batch.num_worlds)
            unsafe_episode_count.add_(result.terminated.to(dtype=torch.int64).sum())
            completed_episode_count.add_(task._official_completion_this_step.to(dtype=torch.int64).sum())
            observed_episode_count.add_(result.done.to(dtype=torch.int64).sum())
            # Mirror the CUDA collector: a successful route completion is a
            # truncation, not a reason to drive past the declared support.
            # Resetting only completed/unsafe worlds keeps the gate focused on
            # repeated physical safety, while its masks stay on the device.
            task.reset(result.done)
        finite = torch.isfinite(batch.qpos).all() & torch.isfinite(batch.qvel).all()
        summary = torch.stack(
            (
                terminated.sum(dtype=torch.int64),
                overflowed.sum(dtype=torch.int64),
                estopped.sum(dtype=torch.int64),
                finite.to(dtype=torch.int64),
                speed_sample_count,
                unsafe_episode_count,
                completed_episode_count,
                observed_episode_count,
            )
        )
        torch.cuda.synchronize(batch.device)
        values = summary.detach().cpu().tolist()
        terminated_count, overflow_count, estop_count, finite_flag = (int(value) for value in values[:4])
        minimum_progress_m = float(max_progress.min().detach().cpu().item())
        maximum_lateral_error_m = float(max_lateral_error.max().detach().cpu().item())
        speed_mae_mps = float(speed_error_sum.detach().cpu().item()) / max(int(speed_sample_count.item()), 1)
        unsafe_episodes = int(unsafe_episode_count.detach().cpu().item())
        completed_episodes = int(completed_episode_count.detach().cpu().item())
        observed_episodes = int(observed_episode_count.detach().cpu().item())
        unsafe_rate = unsafe_episodes / max(observed_episodes, 1)
        first_fault_step_value = int(first_fault_step.detach().cpu().item())
        first_fault_reason_value = int(first_fault_reason.detach().cpu().item())
        passed = bool(
            (not settings.require_no_terminated or terminated_count == 0)
            and (not settings.require_no_overflow or overflow_count == 0)
            and estop_count == 0
            and finite_flag
            and minimum_progress_m >= settings.minimum_progress_m
            and speed_mae_mps <= settings.maximum_speed_mae_mps
            and unsafe_rate <= settings.maximum_unsafe_rate
        )
        report: Mapping[str, Any] = {
            "stage_id": STAGE_ID,
            "conditional_capability": True,
            "passed": passed,
            "requested_duration_seconds": settings.duration_seconds,
            "simulated_duration_seconds": steps * action_dt,
            "policy_steps": steps,
            "num_worlds": int(batch.num_worlds),
            "terminated_worlds": terminated_count,
            "overflowed_worlds": overflow_count,
            "estopped_worlds": estop_count,
            "finite_state": bool(finite_flag),
            "zero_residual": True,
            "minimum_progress_m": minimum_progress_m,
            "maximum_lateral_error_m": maximum_lateral_error_m,
            "speed_mae_mps": speed_mae_mps,
            "unsafe_episodes": unsafe_episodes,
            "completed_episodes": completed_episodes,
            "observed_episodes": observed_episodes,
            "unsafe_rate": unsafe_rate,
            "first_fault_step": first_fault_step_value,
            "first_fault_reason_code": first_fault_reason_value,
            "gate_scope": "motion_stability_preflight_not_course_evaluation",
        }
        task.reset()
        if not passed:
            raise OfficialGrade15AdapterError(
                "official grade15-up conditional stability gate failed: "
                f"terminated={terminated_count}, overflowed={overflow_count}, estopped={estop_count}, "
                f"finite_state={bool(finite_flag)}, minimum_progress_m={minimum_progress_m:.4f}, "
                f"speed_mae_mps={speed_mae_mps:.4f}, unsafe_rate={unsafe_rate:.4f}, "
                f"first_fault_step={first_fault_step_value}, first_fault_reason_code={first_fault_reason_value}, "
                f"duration={steps * action_dt:.6f}s"
            )
        cache["report"] = report
        return report

    return run


def _make_close(batch: Any) -> Callable[[], None]:
    closed = {"value": False}

    def close() -> None:
        if closed["value"]:
            return
        try:
            batch.latch_estop(batch._all_worlds)
            batch._safe_controls.zero_()
            batch._safe_applied_forces.zero_()
            batch._warp.copy(batch.data.ctrl, batch._safe_controls_warp)
            batch._warp.copy(batch.data.qfrc_applied, batch._safe_applied_forces_warp)
            batch._warp.synchronize()
        finally:
            closed["value"] = True

    return close


def build_curriculum_stage(stage: Any, config: Any) -> OfficialGrade15Bundle:
    """Build grade15-up only; the returned runner must pass its real GPU gate."""

    adapter = load_official_grade15_adapter_config(_adapter_path(stage))
    _, terrain_task, route = validate_official_grade15_contract(stage, adapter)
    from train_warp_ppo import load_flat_ppo_training_config
    from warp_env import WarpPhysicsBatch, load_warp_batch_config
    from warp_flat_controller import FixedGainFlatController, calibrate_flat_controller
    from warp_task import WarpFlatWalkingConfig

    batch_config = load_warp_batch_config(adapter.batch_config_path)
    if batch_config.xml_path != Path(_value(stage, "xml_path")).resolve():
        raise OfficialGrade15AdapterError("official grade15 batch XML must exactly match the selected stage scene")
    if batch_config.domain_randomization.enabled:
        raise OfficialGrade15AdapterError("official grade15-up remains no-DR until terrain controller parity is proven")
    if batch_config.safety.torque_fraction_of_rated > 0.80:
        raise OfficialGrade15AdapterError("official grade15 batch torque fraction cannot exceed 80 percent")
    flat = load_flat_ppo_training_config(adapter.flat_ppo_config_path)
    base_task_config = WarpFlatWalkingConfig.from_mapping(flat.flat_walking)
    task_config = replace(
        base_task_config,
        command_speed_mps=float(_value(stage, "command_speed_mps")),
        command_yaw_rate_rad_s=0.0,
        episode_seconds=float(terrain_task.max_episode_seconds),
        domain_randomization_enabled=False,
        sensor_noise_std=0.0,
        control_delay_steps=0,
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
    )
    # The Warp scene disables only the inherited global projection-plane
    # collision.  Calibrate once against the strictly validated canonical
    # scene so the CPU standing-contact projection remains meaningful, then
    # upload that immutable state/control equilibrium into the identical Warp
    # model topology. Runtime physics never switches back to the canonical
    # scene.
    calibration_batch_config = replace(batch_config, xml_path=adapter.canonical_scene_path)
    calibration = calibrate_flat_controller(calibration_batch_config, controller_config)
    batch = None
    try:
        batch = WarpPhysicsBatch(batch_config)
        layout = StaticBoxSupportLayout.from_model(batch.host_model, adapter.support_geoms, batch._mujoco)
        task = OfficialGrade15UpTask(
            batch,
            task_config,
            calibration=calibration.to_task_calibration(),
            terrain_layout=layout,
            terrain_settings=adapter.terrain_features,
            terrain_task=terrain_task,
            route=route,
            route_reward=adapter.route_reward,
            command_speed_mps=float(_value(stage, "command_speed_mps")),
        )
        controller = FixedGainFlatController(calibration, task, controller_config)
        task.set_feedback_controller(controller)
        return OfficialGrade15Bundle(
            batch=batch,
            task=task,
            controller=controller,
            run_stability_gate=_make_stability_gate(batch, task, adapter.stability_gate),
            close=_make_close(batch),
        )
    except Exception:
        if batch is not None:
            _make_close(batch)()
        raise


# This capability is conditional: ``run_stability_gate`` must pass on the
# selected GPU before the generic runner allocates a policy or PPO storage.
GPU_CURRICULUM_CAPABILITIES: dict[str, dict[str, Any]] = {
    STAGE_ID: {
        "backend": CONTROLLER_BACKEND,
        "terrain": True,
        "steps": False,
        "jump": False,
        "domain_randomization": False,
        "speed_command": True,
        "yaw_command": False,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "reward_schema": REWARD_SCHEMA,
        "conditional_runtime_gate": True,
        "supported_task_ids": (SUPPORTED_TASK_ID,),
    }
}


__all__ = [
    "CONTROLLER_BACKEND",
    "EXPECTED_SUPPORT_GEOMS",
    "GPU_CURRICULUM_CAPABILITIES",
    "OfficialGrade15AdapterConfig",
    "OfficialGrade15AdapterError",
    "OfficialGrade15Bundle",
    "OfficialGrade15UpTask",
    "REWARD_SCHEMA",
    "STAGE_ID",
    "StaticBoxSupportLayout",
    "StaticBoxTerrain16D",
    "TASK_MODE",
    "build_curriculum_stage",
    "load_official_grade15_adapter_config",
    "validate_official_grade15_contract",
]
