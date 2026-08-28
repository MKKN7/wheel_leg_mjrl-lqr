"""Fail-closed GPU adapters for every individual official-field exercise.

The adapter deliberately runs one declared route family per CUDA batch.  This
keeps fall-guard references, static support geometry and landing evidence
unambiguous while the curriculum launcher advances stages explicitly.  It is
not a shortcut around the official full-course evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from official_curriculum_warp import (
    ControllerSettings,
    StaticBoxTerrain16D,
    TerrainFeatureSettings,
    _quaternion_multiply,
    _rolling_heading_from_quaternion,
    _rotation_matrix_from_quaternion,
    _wrap_angle,
    validate_official_warp_scene_variant,
)
from terrain_curriculum import TerrainCurriculumConfig, TerrainRoute, TerrainTask, load_terrain_curriculum
from warp_safety import SAFETY_REASON_CONTACT_LOSS
from warp_task import ACTION_SIZE, OBSERVATION_SIZE, WarpFlatWalkingTask, WarpTaskStep


CONTROLLER_BACKEND = "official_course_controller_v1"
REWARD_SCHEMA = "warp_official_terrain_compensated_reward_v2"
EXPECTED_ACTION_MASK = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)

PHASE_IDLE = 0
PHASE_PREPARE = 1
PHASE_CROUCH = 2
PHASE_THRUST = 3
PHASE_FLIGHT = 4
PHASE_LANDING = 5
PHASE_RECOVERY = 6
PHASE_COUNT = 7


class OfficialCourseAdapterError(ValueError):
    """Raised before an unsupported official route can reach CUDA physics."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialCourseAdapterError(f"{name} must be a YAML mapping")
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
        raise OfficialCourseAdapterError(f"{name} keys are invalid: {', '.join(detail)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise OfficialCourseAdapterError(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OfficialCourseAdapterError(f"{name} must be boolean")
    return value


def _finite(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialCourseAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        qualifier = "finite and positive" if positive else "finite and non-negative" if nonnegative else "finite"
        raise OfficialCourseAdapterError(f"{name} must be {qualifier}")
    return result


def _path(source: Path, value: Any, name: str) -> Path:
    candidate = Path(_string(value, name))
    result = candidate.resolve() if candidate.is_absolute() else (source.parent / candidate).resolve()
    if not result.is_file():
        raise OfficialCourseAdapterError(f"{name} does not exist: {result}")
    return result


def _names(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OfficialCourseAdapterError(f"{name} must be a sequence")
    result = tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise OfficialCourseAdapterError(f"{name} must not contain duplicate geometry names")
    return result


def _float_list(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise OfficialCourseAdapterError(f"{name} must be a non-empty sequence")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class CourseRewardSettings:
    progress_reward_per_m: float
    progress_delta_clip_m: float
    completion_bonus: float
    jump_peak_increment_weight: float
    jump_landing_bonus: float
    jump_failure_penalty: float
    obstacle_penalty: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise OfficialCourseAdapterError(f"route_reward.{name} must be finite and non-negative")
        if self.progress_delta_clip_m <= 0.0:
            raise OfficialCourseAdapterError("route_reward.progress_delta_clip_m must be positive")


@dataclass(frozen=True)
class JumpSupervisorSettings:
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
    collision_guard_margin_m: float

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.items())
        for name, value in values:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise OfficialCourseAdapterError(f"jump_supervisor.{name} must be finite and non-negative")
        for name in (
            "prepare_seconds", "crouch_seconds", "thrust_seconds", "maximum_airborne_seconds",
            "landing_confirm_seconds", "recovery_seconds", "prelanding_seconds", "thrust_leg_force_limit_n",
            "maximum_landing_vertical_speed_mps", "maximum_landing_angular_speed_rad_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise OfficialCourseAdapterError(f"jump_supervisor.{name} must be positive")
        if self.prelanding_seconds < 0.050:
            raise OfficialCourseAdapterError("jump_supervisor.prelanding_seconds must be at least 50 ms")
        for name in ("landing_torque_fraction", "flight_torque_fraction", "jump_residual_fraction"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise OfficialCourseAdapterError(f"jump_supervisor.{name} must be within [0, 1]")


@dataclass(frozen=True)
class CourseGateSettings:
    duration_seconds: float
    require_no_terminated: bool
    require_no_overflow: bool
    require_finite_state: bool
    minimum_progress_m: float
    maximum_speed_mae_mps: float
    maximum_unsafe_rate: float

    def __post_init__(self) -> None:
        for name in ("duration_seconds", "minimum_progress_m", "maximum_speed_mae_mps", "maximum_unsafe_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise OfficialCourseAdapterError(f"stability_gate.{name} must be finite and non-negative")
        if self.duration_seconds <= 0.0 or self.maximum_speed_mae_mps <= 0.0:
            raise OfficialCourseAdapterError("stability_gate duration and speed MAE limit must be positive")
        if not 0.0 <= self.maximum_unsafe_rate <= 1.0:
            raise OfficialCourseAdapterError("stability_gate.maximum_unsafe_rate must be within [0, 1]")
        if not (self.require_no_terminated and self.require_no_overflow and self.require_finite_state):
            raise OfficialCourseAdapterError("stability_gate must require finite, non-terminated, non-overflow physics")


@dataclass(frozen=True)
class CourseSpec:
    stage_id: str
    terrain_stage_id: str
    task_id: str
    route_index: int
    support_geoms: tuple[str, ...]
    obstacle_geoms: tuple[str, ...]
    direct_jump: bool


@dataclass(frozen=True)
class OfficialCourseConfig:
    source_path: Path
    batch_config_path: Path
    flat_ppo_config_path: Path
    terrain_curriculum_path: Path
    canonical_scene_path: Path
    terrain_features: TerrainFeatureSettings
    route_reward: CourseRewardSettings
    controller: ControllerSettings
    jump: JumpSupervisorSettings
    stability_gate: CourseGateSettings
    courses: Mapping[str, CourseSpec]


def load_official_course_config(path: str | Path) -> OfficialCourseConfig:
    """Load the strict, YAML-owned configuration for all official GPU routes."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise OfficialCourseAdapterError(f"unable to read official course adapter config {source}: {error}") from error
    root = _mapping(raw, "official course adapter config")
    _exact_keys(
        root,
        "official course adapter config",
        {
            "schema_version", "batch_config", "flat_ppo_config", "terrain_curriculum", "canonical_scene",
            "terrain_features", "route_reward", "controller", "jump_supervisor", "stability_gate", "courses",
        },
    )
    if root["schema_version"] != 1:
        raise OfficialCourseAdapterError("official course adapter schema_version must be 1")
    terrain_raw = _mapping(root["terrain_features"], "terrain_features")
    _exact_keys(
        terrain_raw,
        "terrain_features",
        {"lookahead_distances_m", "lateral_offsets_m", "height_normalization_m", "slope_normalization"},
    )
    terrain_features = TerrainFeatureSettings(
        lookahead_distances_m=_float_list(terrain_raw["lookahead_distances_m"], "terrain_features.lookahead_distances_m"),
        lateral_offsets_m=_float_list(terrain_raw["lateral_offsets_m"], "terrain_features.lateral_offsets_m"),
        height_normalization_m=_finite(terrain_raw["height_normalization_m"], "terrain_features.height_normalization_m", positive=True),
        slope_normalization=_finite(terrain_raw["slope_normalization"], "terrain_features.slope_normalization", positive=True),
    )
    reward_raw = _mapping(root["route_reward"], "route_reward")
    _exact_keys(
        reward_raw,
        "route_reward",
        {
            "progress_reward_per_m", "progress_delta_clip_m", "completion_bonus", "jump_peak_increment_weight",
            "jump_landing_bonus", "jump_failure_penalty", "obstacle_penalty",
        },
    )
    route_reward = CourseRewardSettings(**{
        name: _finite(value, f"route_reward.{name}", nonnegative=True)
        for name, value in reward_raw.items()
    })
    controller_raw = _mapping(root["controller"], "controller")
    _exact_keys(
        controller_raw,
        "controller",
        {
            "command_speed_gain_nm_per_mps", "command_yaw_gain_nm_per_rad_s",
            "command_wheel_feedforward_limit_nm", "command_wheel_accel_limit_nm",
            "command_wheel_brake_limit_nm", "terrain_support_reference_max_rate_mps",
        },
    )
    controller = ControllerSettings(**{
        name: _finite(value, f"controller.{name}", nonnegative=True)
        for name, value in controller_raw.items()
    })
    jump_raw = _mapping(root["jump_supervisor"], "jump_supervisor")
    jump_expected = set(JumpSupervisorSettings.__dataclass_fields__)
    _exact_keys(jump_raw, "jump_supervisor", jump_expected)
    jump = JumpSupervisorSettings(**{
        name: _finite(value, f"jump_supervisor.{name}", nonnegative=True)
        for name, value in jump_raw.items()
    })
    gate_raw = _mapping(root["stability_gate"], "stability_gate")
    _exact_keys(gate_raw, "stability_gate", set(CourseGateSettings.__dataclass_fields__))
    stability_gate = CourseGateSettings(
        duration_seconds=_finite(gate_raw["duration_seconds"], "stability_gate.duration_seconds", positive=True),
        require_no_terminated=_boolean(gate_raw["require_no_terminated"], "stability_gate.require_no_terminated"),
        require_no_overflow=_boolean(gate_raw["require_no_overflow"], "stability_gate.require_no_overflow"),
        require_finite_state=_boolean(gate_raw["require_finite_state"], "stability_gate.require_finite_state"),
        minimum_progress_m=_finite(gate_raw["minimum_progress_m"], "stability_gate.minimum_progress_m", nonnegative=True),
        maximum_speed_mae_mps=_finite(gate_raw["maximum_speed_mae_mps"], "stability_gate.maximum_speed_mae_mps", positive=True),
        maximum_unsafe_rate=_finite(gate_raw["maximum_unsafe_rate"], "stability_gate.maximum_unsafe_rate", nonnegative=True),
    )
    courses_raw = _mapping(root["courses"], "courses")
    courses: dict[str, CourseSpec] = {}
    for stage_id, raw_course in courses_raw.items():
        normalized_id = _string(stage_id, "courses key")
        course = _mapping(raw_course, f"courses.{normalized_id}")
        _exact_keys(
            course,
            f"courses.{normalized_id}",
            {"terrain_stage_id", "task_id", "route_index", "support_geoms", "obstacle_geoms", "direct_jump"},
        )
        route_index = course["route_index"]
        if isinstance(route_index, bool) or not isinstance(route_index, int) or route_index < 0:
            raise OfficialCourseAdapterError(f"courses.{normalized_id}.route_index must be a non-negative integer")
        courses[normalized_id] = CourseSpec(
            stage_id=normalized_id,
            terrain_stage_id=_string(course["terrain_stage_id"], f"courses.{normalized_id}.terrain_stage_id"),
            task_id=_string(course["task_id"], f"courses.{normalized_id}.task_id"),
            route_index=int(route_index),
            support_geoms=_names(course["support_geoms"], f"courses.{normalized_id}.support_geoms"),
            obstacle_geoms=_names(course["obstacle_geoms"], f"courses.{normalized_id}.obstacle_geoms"),
            direct_jump=_boolean(course["direct_jump"], f"courses.{normalized_id}.direct_jump"),
        )
    if not courses:
        raise OfficialCourseAdapterError("courses must not be empty")
    return OfficialCourseConfig(
        source_path=source,
        batch_config_path=_path(source, root["batch_config"], "batch_config"),
        flat_ppo_config_path=_path(source, root["flat_ppo_config"], "flat_ppo_config"),
        terrain_curriculum_path=_path(source, root["terrain_curriculum"], "terrain_curriculum"),
        canonical_scene_path=_path(source, root["canonical_scene"], "canonical_scene"),
        terrain_features=terrain_features,
        route_reward=route_reward,
        controller=controller,
        jump=jump,
        stability_gate=stability_gate,
        courses=courses,
    )


@dataclass(frozen=True)
class StaticBoxLayout:
    """Immutable static-box top-surface layout for an admitted route."""

    names: tuple[str, ...]
    center: np.ndarray
    rotation: np.ndarray
    half_size: np.ndarray
    inverse_top_xy: np.ndarray

    @classmethod
    def from_model(cls, model: Any, names: Sequence[str], mujoco: Any) -> "StaticBoxLayout":
        result_names = tuple(str(name) for name in names)
        if not result_names:
            raise OfficialCourseAdapterError("a route must declare at least one static support box")
        ids: list[int] = []
        for name in result_names:
            try:
                geom_id = int(model.geom(name).id)
            except KeyError as error:
                raise OfficialCourseAdapterError(f"required static geometry is missing: {name}") from error
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise OfficialCourseAdapterError(f"static geometry {name!r} must be a box")
            if int(model.geom_bodyid[geom_id]) != 0:
                raise OfficialCourseAdapterError(f"static geometry {name!r} must be fixed to the world body")
            ids.append(geom_id)
        center = np.asarray(model.geom_pos[ids], dtype=np.float64).copy()
        half_size = np.asarray(model.geom_size[ids, :3], dtype=np.float64).copy()
        quaternion = np.asarray(model.geom_quat[ids], dtype=np.float64).copy()
        rotation = np.stack(tuple(_rotation_matrix_from_quaternion(value) for value in quaternion))
        inverse_top_xy = np.empty((len(ids), 2, 2), dtype=np.float64)
        for index, (name, matrix, size) in enumerate(zip(result_names, rotation, half_size)):
            if not np.isfinite(matrix).all() or not np.isfinite(size).all() or np.any(size <= 0.0):
                raise OfficialCourseAdapterError(f"static geometry {name!r} has invalid geometry")
            xy = matrix[:2, :2]
            determinant = float(np.linalg.det(xy))
            if not math.isfinite(determinant) or abs(determinant) <= 1.0e-7:
                raise OfficialCourseAdapterError(f"static geometry {name!r} top face is not a world XY graph")
            inverse_top_xy[index] = np.linalg.inv(xy)
        return cls(result_names, center, rotation, half_size, inverse_top_xy)

    def surface_height_cpu(self, world_xy: Sequence[float]) -> tuple[float, bool]:
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
                heights.append(float(center[2] + np.dot(rotation[2, :2], local_xy) + rotation[2, 2] * size[2]))
        return (max(heights), True) if heights else (0.0, False)


@dataclass(frozen=True)
class ObstacleBoxLayout:
    """Validated immutable obstacle boxes for analytic doghole guarding."""

    names: tuple[str, ...]
    center: np.ndarray
    rotation: np.ndarray
    half_size: np.ndarray

    @classmethod
    def from_model(cls, model: Any, names: Sequence[str], mujoco: Any) -> "ObstacleBoxLayout | None":
        result_names = tuple(str(name) for name in names)
        if not result_names:
            return None
        ids: list[int] = []
        for name in result_names:
            try:
                geom_id = int(model.geom(name).id)
            except KeyError as error:
                raise OfficialCourseAdapterError(f"required obstacle geometry is missing: {name}") from error
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise OfficialCourseAdapterError(f"obstacle geometry {name!r} must be a box")
            if int(model.geom_bodyid[geom_id]) != 0:
                raise OfficialCourseAdapterError(f"obstacle geometry {name!r} must be fixed to the world body")
            ids.append(geom_id)
        quaternion = np.asarray(model.geom_quat[ids], dtype=np.float64).copy()
        return cls(
            names=result_names,
            center=np.asarray(model.geom_pos[ids], dtype=np.float64).copy(),
            rotation=np.stack(tuple(_rotation_matrix_from_quaternion(value) for value in quaternion)),
            half_size=np.asarray(model.geom_size[ids, :3], dtype=np.float64).copy(),
        )


class AnalyticObstacleGuard:
    """Conservative GPU clearance guard where Warp contact pairs are limited."""

    def __init__(self, task: WarpFlatWalkingTask, layout: ObstacleBoxLayout | None, margin_m: float) -> None:
        self.task = task
        self.layout = layout
        self.margin_m = float(margin_m)
        self.torch = task.torch
        self.device = task.device
        self.num_worlds = task.num_worlds
        self._unsafe = self.torch.zeros(self.num_worlds, dtype=self.torch.bool, device=self.device)
        self._body_center = self.torch.empty((self.num_worlds, 3), dtype=self.torch.float32, device=self.device)
        try:
            base_geom_id = int(task.batch.host_model.geom("base_collision").id)
        except KeyError as error:
            raise OfficialCourseAdapterError("analytic obstacle guard requires base_collision geometry") from error
        if int(task.batch.host_model.geom_type[base_geom_id]) != int(task.batch._mujoco.mjtGeom.mjGEOM_BOX):
            raise OfficialCourseAdapterError("base_collision must be a box for the analytic obstacle guard")
        base_half_size = np.asarray(task.batch.host_model.geom_size[base_geom_id, :3], dtype=np.float32)
        if not np.isfinite(base_half_size).all() or np.any(base_half_size <= 0.0):
            raise OfficialCourseAdapterError("base_collision size must be finite and positive")
        # A sphere enclosing the collision box is conservative under arbitrary
        # chassis yaw/pitch, and does not rely on the incomplete cylinder-box
        # contact manifold exposed by the current Warp release.
        self._body_radius = float(np.linalg.norm(base_half_size))
        self._base_geom_id = self.torch.as_tensor([base_geom_id], dtype=self.torch.long, device=self.device)
        self._base_geom_position = self.torch.empty((self.num_worlds, 1, 3), dtype=self.torch.float32, device=self.device)
        if layout is None:
            self._center = self._rotation = self._half_size = None
            self._local = self._distance = None
            return
        self._center = self.torch.as_tensor(layout.center, dtype=self.torch.float32, device=self.device)
        self._rotation = self.torch.as_tensor(layout.rotation, dtype=self.torch.float32, device=self.device)
        self._half_size = self.torch.as_tensor(layout.half_size, dtype=self.torch.float32, device=self.device)
        count = len(layout.names)
        self._local = self.torch.empty((self.num_worlds, count, 3), dtype=self.torch.float32, device=self.device)
        self._distance = self.torch.empty((self.num_worlds, count, 3), dtype=self.torch.float32, device=self.device)

    def unsafe(self) -> Any:
        if self.layout is None:
            self._unsafe.zero_()
            return self._unsafe
        torch = self.torch
        torch.index_select(self.task._geom_xpos, 1, self._base_geom_id, out=self._base_geom_position)
        self._body_center.copy_(self._base_geom_position[:, 0])
        # Transform a conservative chassis-sphere center into every obstacle
        # frame.  Inflating each obstacle by the bounding-sphere radius makes
        # roof/wall intrusion a P0 stop before relying on contact reporting.
        delta = self._body_center.unsqueeze(1) - self._center.unsqueeze(0)
        self._local.copy_(torch.einsum("bij,nkj->nki", self._rotation.transpose(1, 2), delta))
        self._distance.copy_(self._local.abs())
        self._distance.sub_(self._half_size.unsqueeze(0)).sub_(self._body_radius + self.margin_m)
        self._unsafe.copy_((self._distance <= 0.0).all(dim=2).any(dim=1))
        return self._unsafe


class OfficialCourseTask(WarpFlatWalkingTask):
    """One official route with GPU terrain, direct-jump and obstacle guards."""

    def __init__(
        self,
        batch: Any,
        config: Any,
        *,
        calibration: Any,
        terrain_layout: StaticBoxLayout,
        obstacle_layout: ObstacleBoxLayout | None,
        terrain_settings: TerrainFeatureSettings,
        terrain_task: TerrainTask,
        route: TerrainRoute,
        course: CourseSpec,
        reward_settings: CourseRewardSettings,
        jump_settings: JumpSupervisorSettings,
        command_speed_mps: float,
    ) -> None:
        self._course_terrain: StaticBoxTerrain16D | None = None
        self._route_ready = False
        self._terrain_task = terrain_task
        self._route = route
        self._course = course
        self._reward_settings = reward_settings
        self._jump_settings = jump_settings
        super().__init__(batch, config, calibration=calibration)
        # StaticBoxTerrain16D only depends on the geometry data shape; the
        # generic layout deliberately supplies the exact stage support list.
        self._course_terrain = StaticBoxTerrain16D(self, terrain_layout, terrain_settings)
        self._obstacle_guard = AnalyticObstacleGuard(self, obstacle_layout, jump_settings.collision_guard_margin_m)
        torch = self.torch
        direction = np.asarray(
            (route.goal.x_m - route.spawn.x_m, route.goal.y_m - route.spawn.y_m), dtype=np.float64
        )
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length <= 1.0e-6:
            raise OfficialCourseAdapterError("official route must have non-zero XY length")
        support_height, valid = terrain_layout.surface_height_cpu(route.spawn.xy())
        if not valid:
            raise OfficialCourseAdapterError("official route spawn is outside its declared support geometry")
        self._route_start_xy = torch.tensor(route.spawn.xy(), dtype=torch.float32, device=self.device)
        self._route_direction = torch.tensor(direction / length, dtype=torch.float32, device=self.device)
        self._completion_distance = float(terrain_task.required_distance_m - terrain_task.completion_tolerance_m)
        self._corridor_half_width = float(route.corridor_half_width_m)
        self._command_speed_value = float(command_speed_mps)
        launch_speed = terrain_task.jump_launch_speed_mps
        if course.direct_jump:
            if launch_speed is None:
                raise OfficialCourseAdapterError("direct-jump route must declare jump_launch_speed_mps")
            self._jump_launch_speed_mps = float(launch_speed)
            if self._jump_launch_speed_mps > float(config.command_speed_limit_mps):
                raise OfficialCourseAdapterError("direct-jump launch speed exceeds the configured CUDA command limit")
        else:
            self._jump_launch_speed_mps = self._command_speed_value
        root = self.root_qpos_address
        self._base_root_height = float(calibration.qpos[root + 2])
        self._reset_qpos[:, root : root + 2] = self._route_start_xy
        self._reset_qpos[:, root + 2].add_(float(support_height))
        base_quaternion = np.asarray(calibration.qpos[root + 3 : root + 7], dtype=np.float64)
        base_heading = _rolling_heading_from_quaternion(base_quaternion)
        route_heading = math.atan2(float(direction[1]), float(direction[0]))
        heading_delta = _wrap_angle(route_heading - base_heading)
        route_quaternion = _quaternion_multiply(
            np.asarray((math.cos(0.5 * heading_delta), 0.0, 0.0, math.sin(0.5 * heading_delta)), dtype=np.float64),
            base_quaternion,
        )
        route_quaternion /= np.linalg.norm(route_quaternion)
        self._reset_qpos[:, root + 3 : root + 7] = torch.as_tensor(route_quaternion, dtype=torch.float32, device=self.device)
        self._route_reference_quaternion = self._reset_qpos[:, root + 3 : root + 7].clone()
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
        self._route_xy_delta = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._route_unsafe = torch.zeros_like(self._completed)
        self._jump_phase = torch.zeros(self.num_worlds, dtype=torch.int64, device=self.device)
        self._jump_phase_elapsed = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        self._jump_started_this_step = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._jump_triggered = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._jump_liftoff = torch.zeros_like(self._jump_triggered)
        self._jump_landing_confirmed = torch.zeros_like(self._jump_triggered)
        self._jump_failed = torch.zeros_like(self._jump_triggered)
        self._jump_peak_rise = torch.zeros_like(self._progress)
        self._jump_minimum_peak_met = torch.zeros_like(self._jump_triggered)
        self._jump_landing_kinematics_ok = torch.zeros_like(self._jump_triggered)
        self._jump_flight_seconds = torch.zeros_like(self._progress)
        self._jump_landing_vertical_speed = torch.zeros_like(self._progress)
        self._jump_landing_angular_speed = torch.zeros_like(self._progress)
        self._jump_current_vertical_speed = torch.empty_like(self._progress)
        self._jump_current_angular_speed = torch.empty_like(self._progress)
        self._jump_angular_velocity = torch.empty((self.num_worlds, 3), dtype=torch.float32, device=self.device)
        self._jump_rewarded_peak = torch.zeros_like(self._progress)
        self._jump_reward = torch.zeros_like(self._progress)
        self._jump_contact_confirm = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        self._jump_time_to_touchdown = torch.full_like(self._progress, float("inf"))
        self._jump_torque_scale = torch.ones(self.num_worlds, dtype=torch.float32, device=self.device)
        self._jump_contact_exempt = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._jump_phase_onehot = torch.zeros((self.num_worlds, PHASE_COUNT), dtype=torch.float32, device=self.device)
        self._jump_policy_scale = torch.ones(self.num_worlds, dtype=torch.float32, device=self.device)
        self._jump_resume_leg_length = self._command_leg_length.clone()
        self._jump_landing_this_step = torch.zeros_like(self._jump_triggered)
        self._jump_failure_this_step = torch.zeros_like(self._jump_triggered)
        self._command_speed.fill_(self._command_speed_value)
        self._command_yaw_rate.zero_()
        self._route_ready = True
        self.reset()

    def set_feedback_controller(self, controller: Any) -> None:
        super().set_feedback_controller(controller)
        rebase = getattr(controller, "set_reference_state", None)
        update_support = getattr(controller, "update_terrain_support_reference", None)
        if not callable(rebase) or not callable(update_support):
            raise OfficialCourseAdapterError("official GPU course requires terrain-reference capable CUDA controller")
        rebase(self._reset_qpos, self._reset_qvel, self._all_world_mask)

    def _wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        if self._course_terrain is None:
            return super()._wheel_clearances_and_contacts()
        return self._course_terrain.wheel_clearances_and_contacts()

    def _guide_wheel_clearances_and_contacts(self) -> tuple[Any, Any]:
        if self._course_terrain is None:
            return super()._guide_wheel_clearances_and_contacts()
        return self._course_terrain.guide_wheel_clearances_and_contacts()

    def observe(self) -> Any:
        if not self._route_ready:
            return super().observe()
        if self._course_terrain is not None:
            self._course_terrain.update_features()
        result = super().observe()
        self._jump_phase_onehot.zero_()
        self._jump_phase_onehot.scatter_(1, self._jump_phase.unsqueeze(1), 1.0)
        result[:, self.layout.jump_request] = self._jump_triggered.unsqueeze(1).to(dtype=self.torch.float32)
        result[:, self.layout.jump_phase] = self._jump_phase_onehot
        return result

    def reset(self, world_mask: Any | None = None) -> Any:
        if self._route_ready:
            pre_mask = self._require_mask(world_mask)
            # Raw batch reset performs its own independent fall-health check;
            # install the route reference before that check can observe a
            # different-height previous episode.
            self.batch.set_fall_guard_references(
                self._route_reference_quaternion,
                self._route_reference_height,
                pre_mask,
            )
        result = super().reset(world_mask)
        if not self._route_ready:
            return result
        mask = self._require_mask(world_mask)
        self.batch.set_fall_guard_references(self._route_reference_quaternion, self._route_reference_height, mask)
        rebase = getattr(self._controller, "set_reference_state", None)
        if callable(rebase):
            rebase(self._reset_qpos, self._reset_qvel, mask)
        self._progress[mask] = 0.0
        self._previous_progress[mask] = 0.0
        self._lateral_error[mask] = 0.0
        self._completed[mask] = False
        self._completion_this_step[mask] = False
        self._route_reward[mask] = 0.0
        self._route_unsafe[mask] = False
        self._jump_phase[mask] = PHASE_IDLE
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
        self._jump_contact_exempt[mask] = False
        self._jump_policy_scale[mask] = 1.0
        self._jump_resume_leg_length[mask] = self._command_leg_length[mask]
        self._jump_landing_this_step[mask] = False
        self._jump_failure_this_step[mask] = False
        self.set_contact_loss_exempt(self._jump_contact_exempt)
        self.set_controller_torque_scale(self._jump_torque_scale)
        self._command_speed[mask] = self._command_speed_value
        self._command_yaw_rate[mask] = 0.0
        return self.observe()

    def _update_progress(self, terminated: Any) -> None:
        torch = self.torch
        root = self.root_qpos_address
        self._route_xy_delta.copy_(self.batch.qpos[:, root : root + 2])
        self._route_xy_delta.sub_(self._route_start_xy)
        self._previous_progress.copy_(self._progress)
        torch.sum(self._route_xy_delta * self._route_direction.unsqueeze(0), dim=1, out=self._progress)
        self._lateral_error.copy_(
            self._route_xy_delta[:, 0] * self._route_direction[1]
            - self._route_xy_delta[:, 1] * self._route_direction[0]
        ).abs_()
        delta = torch.clamp(
            self._progress - self._previous_progress,
            -self._reward_settings.progress_delta_clip_m,
            self._reward_settings.progress_delta_clip_m,
        )
        self._route_reward.copy_(delta).mul_(self._reward_settings.progress_reward_per_m)
        completion = (
            (self._progress >= self._completion_distance)
            & (self._lateral_error <= self._corridor_half_width)
            & ~terminated
            & ~self._route_unsafe
        )
        self._completion_this_step.copy_(completion & ~self._completed)
        self._completed.logical_or_(completion)
        self._route_reward.add_(
            self._completion_this_step.to(dtype=torch.float32) * self._reward_settings.completion_bonus
        )

    def _jump_support_contacts(self) -> Any:
        return self._side_support_contacts().all(dim=1)

    def _update_jump_supervisor(self) -> None:
        """Advance one direct jump using only persistent CUDA buffers."""

        torch = self.torch
        self._jump_landing_this_step.zero_()
        self._jump_failure_this_step.zero_()
        if not self._course.direct_jump:
            self._jump_contact_exempt.zero_()
            self._jump_torque_scale.fill_(1.0)
            self._jump_policy_scale.fill_(1.0)
            self.set_contact_loss_exempt(self._jump_contact_exempt)
            self.set_controller_torque_scale(self._jump_torque_scale)
            return
        settings = self._jump_settings
        trigger_progress = self._terrain_task.jump_trigger_progress_m
        if trigger_progress is None:
            raise OfficialCourseAdapterError("direct-jump route must declare jump_trigger_progress_m")
        due = (
            ~self._jump_triggered
            & (self._progress >= float(trigger_progress))
        )
        self._jump_triggered.logical_or_(due)
        self._jump_started_this_step.logical_or_(due)
        self._jump_phase.masked_fill_(due, PHASE_PREPARE)
        self._jump_phase_elapsed.masked_fill_(due, 0.0)
        active = (self._jump_phase >= PHASE_PREPARE) & (self._jump_phase <= PHASE_RECOVERY)
        self._jump_phase_elapsed.add_(active.to(dtype=torch.float32) * self._time_step)
        phase = self._jump_phase
        phase_elapsed = self._jump_phase_elapsed
        transition_prepare = (phase == PHASE_PREPARE) & (phase_elapsed >= settings.prepare_seconds)
        phase.masked_fill_(transition_prepare, PHASE_CROUCH)
        phase_elapsed.masked_fill_(transition_prepare, 0.0)
        transition_crouch = (phase == PHASE_CROUCH) & (phase_elapsed >= settings.crouch_seconds)
        phase.masked_fill_(transition_crouch, PHASE_THRUST)
        phase_elapsed.masked_fill_(transition_crouch, 0.0)
        contacts = self._jump_support_contacts()
        liftoff = (phase == PHASE_THRUST) & ~contacts
        self._jump_liftoff.logical_or_(liftoff)
        transition_flight = self._jump_liftoff & (phase == PHASE_THRUST)
        phase.masked_fill_(transition_flight, PHASE_FLIGHT)
        phase_elapsed.masked_fill_(transition_flight, 0.0)
        root_height = self.batch.qpos[:, self.root_qpos_address + 2]
        rise = torch.clamp(root_height - self._route_reference_height, min=0.0)
        torch.maximum(self._jump_peak_rise, rise, out=self._jump_peak_rise)
        self._jump_minimum_peak_met.copy_(
            self._jump_peak_rise >= settings.minimum_peak_body_rise_m
        )
        descending = self.batch.qvel[:, self.root_dof_address + 2] < 0.0
        flight = phase == PHASE_FLIGHT
        self._jump_flight_seconds.add_(flight.to(dtype=torch.float32) * self._time_step)
        contacted_flight = flight & contacts
        self._jump_contact_confirm.masked_fill_(~contacted_flight, 0.0)
        self._jump_contact_confirm.add_(contacted_flight.to(dtype=torch.float32) * self._time_step)
        landing_candidate = flight & (self._jump_contact_confirm >= settings.landing_confirm_seconds)
        self._jump_current_vertical_speed.copy_(self.batch.qvel[:, self.root_dof_address + 2]).abs_()
        self._jump_angular_velocity.copy_(
            self.batch.qvel[:, self.root_dof_address + 3 : self.root_dof_address + 6]
        )
        self._jump_angular_velocity.square_()
        torch.sum(self._jump_angular_velocity, dim=1, out=self._jump_current_angular_speed)
        self._jump_current_angular_speed.sqrt_()
        # Latch the peak contact speed during the whole confirmation window.
        # Evaluating only the final 80 ms sample could hide the impact pulse.
        self._jump_current_vertical_speed.masked_fill_(~contacted_flight, 0.0)
        self._jump_current_angular_speed.masked_fill_(~contacted_flight, 0.0)
        torch.maximum(
            self._jump_landing_vertical_speed,
            self._jump_current_vertical_speed,
            out=self._jump_landing_vertical_speed,
        )
        torch.maximum(
            self._jump_landing_angular_speed,
            self._jump_current_angular_speed,
            out=self._jump_landing_angular_speed,
        )
        landing_kinematics_ok = (
            (self._jump_landing_vertical_speed <= settings.maximum_landing_vertical_speed_mps)
            & (self._jump_landing_angular_speed <= settings.maximum_landing_angular_speed_rad_s)
        )
        confirmed = landing_candidate & self._jump_minimum_peak_met & landing_kinematics_ok
        rejected_landing = landing_candidate & ~confirmed
        phase.masked_fill_(confirmed, PHASE_LANDING)
        phase_elapsed.masked_fill_(confirmed, 0.0)
        self._jump_landing_confirmed.logical_or_(confirmed)
        self._jump_landing_kinematics_ok.logical_or_(confirmed)
        self._jump_landing_this_step.copy_(confirmed)
        transition_recovery = (phase == PHASE_LANDING) & (phase_elapsed >= settings.recovery_seconds)
        phase.masked_fill_(transition_recovery, PHASE_RECOVERY)
        phase_elapsed.masked_fill_(transition_recovery, 0.0)
        transition_idle = (phase == PHASE_RECOVERY) & (phase_elapsed >= settings.recovery_seconds)
        phase.masked_fill_(transition_idle, PHASE_IDLE)
        phase_elapsed.masked_fill_(transition_idle, 0.0)
        failed = (
            ((phase == PHASE_THRUST) & (phase_elapsed > settings.thrust_seconds))
            | ((phase == PHASE_FLIGHT) & (phase_elapsed > settings.maximum_airborne_seconds))
            | rejected_landing
        )
        self._jump_failed.logical_or_(failed)
        self._jump_failure_this_step.copy_(failed)
        phase.masked_fill_(failed, PHASE_IDLE)
        phase_elapsed.masked_fill_(failed, 0.0)
        self._jump_contact_exempt.copy_(phase == PHASE_FLIGHT)
        self._jump_torque_scale.fill_(1.0)
        self._jump_torque_scale.masked_fill_(phase == PHASE_FLIGHT, settings.flight_torque_fraction)
        self._jump_time_to_touchdown.fill_(float("inf"))
        if self._course_terrain is not None:
            root_support = self._course_terrain._root_height[:, 0]
            clearance = torch.clamp(root_height - (self._base_root_height + root_support), min=0.0)
            descent_speed = (-self.batch.qvel[:, self.root_dof_address + 2]).clamp_min(1.0e-4)
            self._jump_time_to_touchdown.copy_(clearance / descent_speed)
        preload = flight & descending & (self._jump_time_to_touchdown <= settings.prelanding_seconds)
        self._jump_torque_scale.masked_fill_(preload | (phase == PHASE_LANDING), settings.landing_torque_fraction)
        self._jump_policy_scale.fill_(1.0)
        limited_residual = (phase >= PHASE_PREPARE) & (phase <= PHASE_RECOVERY)
        self._jump_policy_scale.masked_fill_(limited_residual, settings.jump_residual_fraction)
        self._jump_policy_scale.masked_fill_(
            phase == PHASE_LANDING,
            min(settings.jump_residual_fraction, settings.landing_torque_fraction),
        )
        launch_phase = (phase == PHASE_PREPARE) | (phase == PHASE_CROUCH) | (phase == PHASE_THRUST)
        self._command_speed.fill_(self._command_speed_value)
        self._command_speed.masked_fill_(launch_phase, self._jump_launch_speed_mps)
        target = self._command_leg_length
        target.copy_(self._jump_resume_leg_length)
        target.masked_fill_(phase == PHASE_PREPARE, settings.prepare_length_m)
        target.masked_fill_(phase == PHASE_CROUCH, settings.crouch_length_m)
        target.masked_fill_(phase == PHASE_THRUST, settings.thrust_length_m)
        target.masked_fill_(
            (phase == PHASE_FLIGHT) & (phase_elapsed < settings.flight_retract_seconds),
            settings.flight_retract_length_m,
        )
        target.masked_fill_(
            (phase == PHASE_FLIGHT) & (phase_elapsed >= settings.flight_retract_seconds),
            settings.flight_preload_length_m,
        )
        target.masked_fill_(phase == PHASE_LANDING, settings.landing_length_m)
        self.set_contact_loss_exempt(self._jump_contact_exempt)
        self.set_controller_torque_scale(self._jump_torque_scale)

    def _transform_policy_action(self, action: Any) -> Any:
        """Keep direct-jump residual torque authority below its YAML fraction."""

        # A delayed pre-jump residual may otherwise reach the controller one
        # policy interval after the supervisor starts.  Flush it per world so
        # launch authority is bounded from its first physical substep.
        self._action_delay_buffer[:, :, :6].masked_fill_(
            self._jump_started_this_step.view(-1, 1, 1), 0.0
        )
        self._delayed_action[:, :6].masked_fill_(self._jump_started_this_step.unsqueeze(1), 0.0)
        self._previous_action[:, :6].masked_fill_(self._jump_started_this_step.unsqueeze(1), 0.0)
        action[:, :6].mul_(self._jump_policy_scale.unsqueeze(1))
        # Consume the trigger only after all delayed action buffers have seen
        # it. This keeps a direct supervisor invocation and the normal step
        # path semantically identical without a CPU branch.
        self._jump_started_this_step.zero_()
        return action

    def _policy_action_authority(self) -> Any:
        # PPO masks are categorical channel availability, not a physical
        # torque scale. The six residual channels remain learnable while
        # ``_transform_policy_action`` applies their bounded jump fraction.
        return super()._policy_action_authority()

    def _evaluate_safety(self, controls: Any) -> Any:
        result = super()._evaluate_safety(controls)
        terrain = self._course_terrain
        if terrain is None:
            return result
        invalid_support = ~terrain.wheel_support_valid.all(dim=1)
        invalid_support.logical_and_(~self._jump_contact_exempt)
        obstacle = self._obstacle_guard.unsafe()
        unsafe = invalid_support | obstacle | self._jump_failed
        result.safe_controls.masked_fill_(unsafe.unsqueeze(1), 0.0)
        result.terminated.logical_or_(unsafe)
        result.failure.logical_or_(unsafe)
        result.contact_limit.logical_or_(unsafe)
        result.reason_code.masked_fill_(unsafe, SAFETY_REASON_CONTACT_LOSS)
        self._route_unsafe.logical_or_(unsafe)
        return result

    def _before_policy_step(self) -> None:
        if self._course_terrain is None:
            return
        self._course_terrain.update_features()
        if self._controller is not None:
            self._controller.update_terrain_support_reference(self._course_terrain._root_height[:, 0])
        # Rebase the independent raw-physics height guard only when both
        # sides are physically supported and no intentional flight is active.
        self._fall_guard_target_height.copy_(self._course_terrain._root_height[:, 0])
        self._fall_guard_target_height.add_(self._base_root_height)
        self._fall_guard_update_mask.copy_(self._course_terrain.wheel_support_valid.all(dim=1))
        self._fall_guard_update_mask.logical_and_(self._side_support_contacts().all(dim=1))
        self._fall_guard_update_mask.logical_and_(~self._jump_contact_exempt)
        self.batch.update_fall_guard_reference_heights(
            self._fall_guard_target_height,
            self._fall_guard_update_mask,
        )
        self._reference_height[self._fall_guard_update_mask] = self._fall_guard_target_height[
            self._fall_guard_update_mask
        ]
        self._post_physics_terminated.copy_(self._safety_terminated)
        self._post_physics_terminated.logical_or_(self.batch.estopped)
        self._update_progress(self._post_physics_terminated)
        self._update_jump_supervisor()

    def _after_physics_interval(self, terminated: Any) -> None:
        self._update_progress(terminated)
        completed = self._completion_this_step & ~terminated
        self._task_truncated.logical_or_(completed)

    def _reward(self, controls: Any, action: Any, unsafe: Any) -> Any:
        reward = super()._reward(controls, action, unsafe)
        torch = self.torch
        self._jump_reward.zero_()
        if self._course.direct_jump:
            capped_peak = torch.clamp(self._jump_peak_rise, max=self._jump_settings.minimum_peak_body_rise_m)
            increment = torch.clamp(capped_peak - self._jump_rewarded_peak, min=0.0)
            self._jump_rewarded_peak.add_(increment)
            self._jump_reward.add_(increment * self._reward_settings.jump_peak_increment_weight)
            self._jump_reward.add_(
                self._jump_landing_this_step.to(dtype=torch.float32) * self._reward_settings.jump_landing_bonus
            )
            self._jump_reward.sub_(
                self._jump_failure_this_step.to(dtype=torch.float32) * self._reward_settings.jump_failure_penalty
            )
        self._jump_reward.sub_(self._route_unsafe.to(dtype=torch.float32) * self._reward_settings.obstacle_penalty)
        self._reward_terms.update({"official_route": self._route_reward, "direct_jump": self._jump_reward})
        return reward + self._route_reward + self._jump_reward

    def tensors(self) -> Mapping[str, Any]:
        result = dict(super().tensors())
        result.update({
            "official_route_progress_m": self._progress,
            "official_route_lateral_error_m": self._lateral_error,
            "official_route_completed": self._completed,
            "official_route_support_valid": self._course_terrain.last_support_valid if self._course_terrain else None,
            "jump_phase": self._jump_phase,
            "jump_triggered": self._jump_triggered,
            "jump_landing_confirmed": self._jump_landing_confirmed,
            "jump_failed": self._jump_failed,
            "jump_peak_rise_m": self._jump_peak_rise,
            "jump_minimum_peak_met": self._jump_minimum_peak_met,
            "jump_landing_vertical_speed_mps": self._jump_landing_vertical_speed,
            "jump_landing_angular_speed_rad_s": self._jump_landing_angular_speed,
            "jump_landing_kinematics_ok": self._jump_landing_kinematics_ok,
            "jump_flight_seconds": self._jump_flight_seconds,
        })
        return result


@dataclass
class OfficialCourseBundle:
    batch: Any
    task: OfficialCourseTask
    controller: Any
    run_stability_gate: Callable[[], Mapping[str, Any]]
    close: Callable[[], None]


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        if name not in source:
            raise OfficialCourseAdapterError(f"missing stage field {name!r}")
        return source[name]
    if not hasattr(source, name):
        raise OfficialCourseAdapterError(f"missing stage field {name!r}")
    return getattr(source, name)


def _validate_course_contract(
    stage: Any,
    adapter: OfficialCourseConfig,
) -> tuple[CourseSpec, TerrainCurriculumConfig, TerrainTask, TerrainRoute]:
    stage_id = _value(stage, "stage_id")
    if stage_id not in adapter.courses:
        raise OfficialCourseAdapterError(f"official course adapter has no configuration for stage {stage_id!r}")
    course = adapter.courses[stage_id]
    expected_task_mode = "official_course_doghole" if course.obstacle_geoms else "official_course"
    if _value(stage, "task_mode") != expected_task_mode:
        raise OfficialCourseAdapterError(f"{stage_id}.task_mode must be {expected_task_mode}")
    if _value(stage, "controller_backend") != CONTROLLER_BACKEND:
        raise OfficialCourseAdapterError("official course requires the explicit official CUDA controller backend")
    if _value(stage, "scene_variant") != "official_warp_compat":
        raise OfficialCourseAdapterError("official course requires the strict official_warp_compat scene variant")
    expected_flags = {
        "terrain_enabled": True,
        "jump_enabled": course.direct_jump,
        "steps_enabled": course.task_id.startswith(("step", "stair")),
        "domain_randomization_enabled": True,
        "requires_gpu_parity": True,
    }
    for name, expected in expected_flags.items():
        if _value(stage, name) is not expected:
            raise OfficialCourseAdapterError(f"{stage_id}.{name} must be {str(expected).lower()}")
    if tuple(float(value) for value in _value(stage, "residual_action_mask")) != EXPECTED_ACTION_MASK:
        raise OfficialCourseAdapterError("official course must retain six residual actuators and mask channel seven")
    if _value(stage, "reward_schema") != REWARD_SCHEMA:
        raise OfficialCourseAdapterError("official course must declare the terrain-compensated reward schema")
    curriculum = load_terrain_curriculum(adapter.terrain_curriculum_path)
    if curriculum.schema_version != 4 or curriculum.scene_contract is None:
        raise OfficialCourseAdapterError("official course requires the v4 official terrain curriculum scene contract")
    if _value(stage, "terrain_curriculum_path") != adapter.terrain_curriculum_path:
        raise OfficialCourseAdapterError("stage terrain_curriculum_path must match official course YAML")
    if _value(stage, "terrain_stage_id") != course.terrain_stage_id:
        raise OfficialCourseAdapterError("stage terrain_stage_id must match its official course YAML")
    terrain_stage = curriculum.stage(course.terrain_stage_id)
    if course.task_id not in terrain_stage.task_ids:
        raise OfficialCourseAdapterError("official course task is absent from its declared terrain stage")
    task = curriculum.task(course.task_id)
    route = task.route_at(course.route_index)
    command = terrain_stage.command_for(task)
    if not math.isclose(float(_value(stage, "command_speed_mps")), command.forward_speed_mps, abs_tol=1.0e-7):
        raise OfficialCourseAdapterError("stage speed command must equal the YAML stage-scaled command")
    if not math.isclose(float(_value(stage, "command_yaw_rate_rad_s")), command.yaw_rate_rad_s, abs_tol=1.0e-7):
        raise OfficialCourseAdapterError("stage yaw command must equal the YAML stage-scaled command")
    if bool(task.has_progress_jump_trigger) is not course.direct_jump:
        raise OfficialCourseAdapterError("direct_jump must exactly match the terrain task jump trigger contract")
    binding = next(
        (
            item for item in curriculum.scene_contract.route_bindings
            if item.task_id == task.task_id and item.route_id == route.route_id
        ),
        None,
    )
    if binding is None:
        raise OfficialCourseAdapterError("official route has no immutable scene binding")
    if binding.support_geoms != course.support_geoms or binding.obstacle_geoms != course.obstacle_geoms:
        raise OfficialCourseAdapterError("official course support/obstacle YAML does not match the scene contract")
    xml_path = Path(_value(stage, "xml_path")).resolve()
    try:
        from build_official_standard_ground import validate_official_warp_scene

        validate_official_warp_scene(xml_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise OfficialCourseAdapterError(f"official Warp scene validation failed: {error}") from error
    validate_official_warp_scene_variant(
        curriculum,
        canonical_scene_path=adapter.canonical_scene_path,
        variant_scene_path=xml_path,
        curriculum_path=adapter.terrain_curriculum_path,
    )
    return course, curriculum, task, route


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


def _make_stability_gate(
    batch: Any,
    task: OfficialCourseTask,
    stage_id: str,
    settings: CourseGateSettings,
    evidence_config_path: Path,
) -> Callable[[], Mapping[str, Any]]:
    """Run deterministic and reset-boundary DR CUDA safety preflights."""

    if settings.duration_seconds > float(task.config.episode_seconds) + 1.0e-9:
        raise OfficialCourseAdapterError("official course stability gate cannot exceed its route episode horizon")
    action_dt = float(task._time_step)
    if not math.isfinite(action_dt) or action_dt <= 0.0:
        raise OfficialCourseAdapterError("official course action timestep must be finite and positive")
    try:
        evidence_config_sha256 = hashlib.sha256(evidence_config_path.read_bytes()).hexdigest()
    except OSError as error:
        raise OfficialCourseAdapterError("unable to hash the official CUDA gate configuration") from error
    cache: dict[str, Mapping[str, Any]] = {}

    def run_pass(*, domain_randomization_active: bool) -> Mapping[str, Any]:
        """Execute one zero-residual pass and retain only aggregate CUDA evidence."""

        torch = batch._torch
        steps = max(1, int(math.ceil(settings.duration_seconds / action_dt)))
        action = torch.zeros((batch.num_worlds, ACTION_SIZE), dtype=torch.float32, device=batch.device)
        terminated = torch.zeros(batch.num_worlds, dtype=torch.bool, device=batch.device)
        overflowed = torch.zeros_like(terminated)
        estopped = torch.zeros_like(terminated)
        max_progress = torch.full((batch.num_worlds,), -torch.inf, dtype=torch.float32, device=batch.device)
        speed_error_sum = torch.zeros((), dtype=torch.float32, device=batch.device)
        speed_samples = torch.zeros((), dtype=torch.int64, device=batch.device)
        jump_triggered = torch.zeros_like(terminated)
        jump_landed = torch.zeros_like(terminated)
        jump_peak_met = torch.zeros_like(terminated)
        jump_landing_kinematics = torch.zeros_like(terminated)
        jump_flight_seconds = torch.zeros(batch.num_worlds, dtype=torch.float32, device=batch.device)
        obstacle_seen = torch.zeros_like(terminated)
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
                speed_samples.add_(batch.num_worlds)
                jump_triggered.logical_or_(task._jump_triggered)
                jump_landed.logical_or_(task._jump_landing_confirmed)
                jump_peak_met.logical_or_(task._jump_minimum_peak_met)
                jump_landing_kinematics.logical_or_(task._jump_landing_kinematics_ok)
                torch.maximum(jump_flight_seconds, task._jump_flight_seconds, out=jump_flight_seconds)
                obstacle_seen.logical_or_(task._route_unsafe)
                task.reset(result.done)
            finite = torch.isfinite(batch.qpos).all() & torch.isfinite(batch.qvel).all()
            summary = torch.stack((
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
                obstacle_seen.sum(dtype=torch.int64),
            ))
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
                obstacle_count,
            ) = values
            minimum_progress = float(max_progress.min().detach().cpu().item())
            speed_mae = float(speed_error_sum.detach().cpu().item()) / max(int(speed_samples.detach().cpu().item()), 1)
            minimum_flight_seconds = float(jump_flight_seconds.min().detach().cpu().item())
            unsafe_rate = float(terminated_count) / float(max(batch.num_worlds, 1))
            jump_passed = bool(
                not task._course.direct_jump
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
                and unsafe_rate <= settings.maximum_unsafe_rate
                and jump_passed
                and obstacle_count == 0
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
                "unsafe_rate": unsafe_rate,
                "first_fault_step": int(first_fault_step.detach().cpu().item()),
                "first_fault_reason_code": int(first_fault_reason.detach().cpu().item()),
                "jump_supervisor_verified": jump_passed,
                "jump_triggered_worlds": triggered_count,
                "landing_confirmed_worlds": landed_count,
                "jump_minimum_peak_worlds": peak_count,
                "landing_kinematics_worlds": landing_kinematics_count,
                "minimum_flight_seconds": minimum_flight_seconds,
                "landing_preload_seconds": float(task._jump_settings.prelanding_seconds),
                "obstacle_guard_verified": obstacle_count == 0,
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
            "unsafe_rate": domain_randomized["unsafe_rate"],
            "first_fault_step": domain_randomized["first_fault_step"],
            "first_fault_reason_code": domain_randomized["first_fault_reason_code"],
            "jump_supervisor_verified": bool(
                deterministic["jump_supervisor_verified"] and domain_randomized["jump_supervisor_verified"]
            ),
            "jump_triggered_worlds": domain_randomized["jump_triggered_worlds"],
            "landing_confirmed_worlds": domain_randomized["landing_confirmed_worlds"],
            "jump_minimum_peak_worlds": domain_randomized["jump_minimum_peak_worlds"],
            "landing_kinematics_worlds": domain_randomized["landing_kinematics_worlds"],
            "minimum_flight_seconds": domain_randomized["minimum_flight_seconds"],
            "landing_preload_seconds": float(task._jump_settings.prelanding_seconds),
            "obstacle_guard_verified": bool(
                deterministic["obstacle_guard_verified"] and domain_randomized["obstacle_guard_verified"]
            ),
            "deterministic_baseline": deterministic,
            "domain_randomization_stress": domain_randomized,
            "deterministic_baseline_passed": deterministic["passed"],
            "domain_randomization_stress_passed": domain_randomized["passed"],
            "gate_scope": "zero_residual_deterministic_and_domain_randomization_course_physics_preflight",
        }
        if not report["passed"]:
            raise OfficialCourseAdapterError(
                f"official course GPU gate failed for {stage_id}: "
                f"deterministic={deterministic['passed']}, domain_randomization={domain_randomized['passed']}, "
                f"terminated={domain_randomized['terminated_worlds']}, "
                f"overflowed={domain_randomized['overflowed_worlds']}, "
                f"estopped={domain_randomized['estopped_worlds']}, "
                f"minimum_progress={domain_randomized['minimum_progress_m']:.4f}, "
                f"speed_mae={domain_randomized['speed_mae_mps']:.4f}"
            )
        cache["report"] = report
        return report

    return run


def build_curriculum_stage(stage: Any, config: Any) -> OfficialCourseBundle:
    """Build one strictly verified official route task on CUDA."""

    adapter_path = _value(stage, "adapter_config_path")
    if adapter_path is None:
        raise OfficialCourseAdapterError("official course stage requires adapter_config_path")
    adapter = load_official_course_config(adapter_path)
    course, _, terrain_task, route = _validate_course_contract(stage, adapter)
    from train_warp_ppo import load_flat_ppo_training_config
    from warp_env import WarpPhysicsBatch, load_warp_batch_config
    from warp_flat_controller import FixedGainFlatController, calibrate_flat_controller
    from warp_task import WarpFlatWalkingConfig

    batch_config = load_warp_batch_config(adapter.batch_config_path)
    if batch_config.xml_path != Path(_value(stage, "xml_path")).resolve():
        raise OfficialCourseAdapterError("official course batch XML must exactly match the selected Warp scene")
    if not batch_config.domain_randomization.enabled:
        raise OfficialCourseAdapterError("official course requires reset-boundary vehicle domain randomization")
    if batch_config.domain_randomization.terrain_geometry_randomization:
        raise OfficialCourseAdapterError("official course must never randomize terrain geometry")
    if batch_config.safety.torque_fraction_of_rated > 0.80:
        raise OfficialCourseAdapterError("official course batch torque fraction cannot exceed 80 percent")
    flat = load_flat_ppo_training_config(adapter.flat_ppo_config_path)
    task_base = WarpFlatWalkingConfig.from_mapping(flat.flat_walking)
    task_config = replace(
        task_base,
        command_speed_mps=float(_value(stage, "command_speed_mps")),
        command_yaw_rate_rad_s=float(_value(stage, "command_yaw_rate_rad_s")),
        episode_seconds=float(terrain_task.max_episode_seconds),
        domain_randomization_enabled=True,
        sensor_noise_std=float(_value(config, "gpu_task").sensor_noise_std),
        control_delay_steps=int(_value(config, "gpu_task").control_delay_steps),
        domain_randomization_seed=int(batch_config.domain_randomization.seed) + 3,
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
            max(float(flat.flat_controller.leg_force_limit_n), adapter.jump.thrust_leg_force_limit_n if course.direct_jump else 0.0),
            240.0,
        ),
    )
    calibration_batch_config = replace(batch_config, xml_path=adapter.canonical_scene_path)
    calibration = calibrate_flat_controller(calibration_batch_config, controller_config)
    batch = None
    try:
        batch = WarpPhysicsBatch(batch_config)
        layout = StaticBoxLayout.from_model(batch.host_model, course.support_geoms, batch._mujoco)
        obstacles = ObstacleBoxLayout.from_model(batch.host_model, course.obstacle_geoms, batch._mujoco)
        task = OfficialCourseTask(
            batch,
            task_config,
            calibration=calibration.to_task_calibration(),
            terrain_layout=layout,
            obstacle_layout=obstacles,
            terrain_settings=adapter.terrain_features,
            terrain_task=terrain_task,
            route=route,
            course=course,
            reward_settings=adapter.route_reward,
            jump_settings=adapter.jump,
            command_speed_mps=float(_value(stage, "command_speed_mps")),
        )
        controller = FixedGainFlatController(calibration, task, controller_config)
        task.set_feedback_controller(controller)
        return OfficialCourseBundle(
            batch=batch,
            task=task,
            controller=controller,
            run_stability_gate=_make_stability_gate(
                batch,
                task,
                course.stage_id,
                adapter.stability_gate,
                adapter.source_path,
            ),
            close=_make_close(batch),
        )
    except Exception:
        if batch is not None:
            _make_close(batch)()
        raise


_COURSE_CAPABILITY_IDS = (
    "official_grade15_up", "official_grade15_down", "official_grade20_up", "official_grade20_down",
    "official_step150_up", "official_step150_down", "official_stair2x100_up", "official_stair2x100_down",
    "official_step200_lab", "official_fly17_jump", "official_doghole450",
)
_DIRECT_JUMP_IDS = frozenset((
    "official_step150_up", "official_stair2x100_up", "official_step200_lab", "official_fly17_jump",
))
_STEP_IDS = frozenset((
    "official_step150_up", "official_step150_down", "official_stair2x100_up", "official_stair2x100_down",
    "official_step200_lab",
))

GPU_CURRICULUM_CAPABILITIES: dict[str, dict[str, Any]] = {
    stage_id: {
        "backend": CONTROLLER_BACKEND,
        "terrain": True,
        "steps": stage_id in _STEP_IDS,
        "jump": stage_id in _DIRECT_JUMP_IDS,
        "domain_randomization": True,
        "speed_command": True,
        "yaw_command": False,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "reward_schema": REWARD_SCHEMA,
        "conditional_runtime_gate": True,
    }
    for stage_id in _COURSE_CAPABILITY_IDS
}


def evaluate_course_stage(task: OfficialCourseTask, stage: Any) -> Mapping[str, Any]:
    """Evaluate one already-trained CUDA route against its YAML stage gate.

    The evaluator intentionally does not drive the policy itself.  The generic
    PPO runner passes an actor callback so all action selection remains on
    CUDA and the only host transfer is this final aggregated result.
    """

    raise OfficialCourseAdapterError(
        "evaluate_course_stage requires the PPO runner's CUDA policy callback; use evaluate_policy_stage"
    )


def evaluate_policy_stage(
    task: OfficialCourseTask,
    policy: Any,
    stage: Any,
    *,
    threshold_terrain_stage_id: str | None = None,
) -> Mapping[str, Any]:
    """Evaluate exactly one declared route and its task-owned promotion rules.

    An individual CUDA training stage owns one immutable route.  It must never
    claim that the sibling routes listed by a broader terrain curriculum stage
    were evaluated.  The terrain-stage thresholds remain the YAML authority,
    while the task's own consecutive-success requirement is enforced here.
    """

    curriculum = load_terrain_curriculum(_value(stage, "terrain_curriculum_path"))
    threshold_stage_id = (
        _value(stage, "terrain_stage_id")
        if threshold_terrain_stage_id is None
        else _string(threshold_terrain_stage_id, "threshold_terrain_stage_id")
    )
    terrain_stage = curriculum.stage(threshold_stage_id)
    task_id = task._terrain_task.task_id
    if task_id not in terrain_stage.task_ids:
        raise OfficialCourseAdapterError("evaluation task is absent from its declared terrain stage")
    torch = task.torch
    batch = task.batch
    episodes_target = int(terrain_stage.evaluation_episodes)
    if episodes_target < 1:
        raise OfficialCourseAdapterError("terrain stage evaluation_episodes must be positive")
    # Every world executes the same immutable route; count completions,
    # failures, and consecutive successes on device until the declared sample
    # count and the task-owned consecutive-success requirement are both met.
    completed = torch.zeros((), dtype=torch.int64, device=batch.device)
    unsafe = torch.zeros((), dtype=torch.int64, device=batch.device)
    observed = torch.zeros((), dtype=torch.int64, device=batch.device)
    speed_error_sum = torch.zeros((), dtype=torch.float32, device=batch.device)
    yaw_error_sum = torch.zeros((), dtype=torch.float32, device=batch.device)
    samples = torch.zeros((), dtype=torch.int64, device=batch.device)
    consecutive = torch.zeros(batch.num_worlds, dtype=torch.int64, device=batch.device)
    maximum_consecutive = torch.zeros_like(consecutive)
    next_consecutive = torch.zeros_like(consecutive)
    required_consecutive = int(task._terrain_task.required_consecutive_successes)
    if required_consecutive < 1:
        raise OfficialCourseAdapterError("route evaluation requires at least one consecutive success")
    rounds = max(required_consecutive, int(math.ceil(episodes_target / batch.num_worlds)))
    max_steps = max(1, int(math.ceil(task._terrain_task.max_episode_seconds / task._time_step)))
    task.set_domain_randomization_active(True)
    for _ in range(rounds):
        active = torch.ones(batch.num_worlds, dtype=torch.bool, device=batch.device)
        task.reset()
        for _ in range(max_steps):
            with torch.no_grad():
                output = policy(task.observe())
                action = output[0] if isinstance(output, tuple) else output
                if not isinstance(action, torch.Tensor) or action.shape != (batch.num_worlds, ACTION_SIZE):
                    raise OfficialCourseAdapterError("evaluation policy must return CUDA [world, 7] actions")
                action = torch.tanh(action).contiguous().to(dtype=torch.float32)
            result = task.step(action)
            speed_error_sum.add_((task.forward_speed() - task._command_speed).abs().sum())
            yaw_error_sum.add_((batch.qvel[:, task.root_dof_address + 5] - task._command_yaw_rate).abs().sum())
            samples.add_(batch.num_worlds)
            done = result.done & active
            successful = task._completed & done & ~result.terminated
            completed.add_(successful.sum(dtype=torch.int64))
            unsafe.add_((result.terminated & active).sum(dtype=torch.int64))
            observed.add_(done.sum(dtype=torch.int64))
            next_consecutive.copy_(consecutive)
            next_consecutive.add_(successful.to(dtype=torch.int64))
            next_consecutive.masked_fill_(done & ~successful, 0)
            consecutive.copy_(next_consecutive)
            torch.maximum(maximum_consecutive, consecutive, out=maximum_consecutive)
            active.logical_and_(~done)
            # Keep completed/unsafe worlds torque-safe for the remainder of
            # the fixed vector evaluation horizon.  They no longer contribute
            # to metrics because ``active`` is already cleared, but resetting
            # them prevents a terminal state from being integrated again.
            task.reset(result.done)
        unresolved = active
        unsafe.add_(unresolved.sum(dtype=torch.int64))
        observed.add_(unresolved.sum(dtype=torch.int64))
        consecutive.masked_fill_(unresolved, 0)
    batch._warp.synchronize()
    observed_value = max(int(observed.detach().cpu().item()), 1)
    completion_rate = float(completed.detach().cpu().item()) / observed_value
    unsafe_rate = float(unsafe.detach().cpu().item()) / observed_value
    speed_mae = float(speed_error_sum.detach().cpu().item()) / max(int(samples.detach().cpu().item()), 1)
    yaw_mae = float(yaw_error_sum.detach().cpu().item()) / max(int(samples.detach().cpu().item()), 1)
    maximum_consecutive_value = int(maximum_consecutive.max().detach().cpu().item())
    passed = bool(
        completion_rate >= terrain_stage.minimum_completion_rate
        and unsafe_rate <= terrain_stage.maximum_unsafe_rate
        and (
            terrain_stage.maximum_speed_mae_mps is None
            or speed_mae <= terrain_stage.maximum_speed_mae_mps
        )
        and (
            terrain_stage.maximum_yaw_mae_rad is None
            or yaw_mae <= terrain_stage.maximum_yaw_mae_rad
        )
        and maximum_consecutive_value >= required_consecutive
    )
    report: Mapping[str, Any] = {
        "stage_id": _value(stage, "stage_id"),
        "evaluation_scope": "single_declared_route",
        "task_id": task_id,
        "route_id": task._route.route_id,
        "episodes": observed_value,
        "target_episodes": episodes_target,
        "completion_rate": completion_rate,
        "unsafe_rate": unsafe_rate,
        "speed_mae_mps": speed_mae,
        "yaw_mae_rad": yaw_mae,
        "required_consecutive_successes": required_consecutive,
        "maximum_consecutive_successes": maximum_consecutive_value,
        "threshold_source": str(_value(stage, "terrain_curriculum_path")),
        "threshold_terrain_stage_id": threshold_stage_id,
        "domain_randomization_active": True,
        "passed": passed,
    }
    if not passed:
        raise OfficialCourseAdapterError(
            f"course promotion gate failed for {_value(stage, 'stage_id')}: "
            f"completion={completion_rate:.3f}, unsafe={unsafe_rate:.3f}, "
            f"speed_mae={speed_mae:.3f}, yaw_mae={yaw_mae:.3f}, "
            f"consecutive={maximum_consecutive_value}/{required_consecutive}"
        )
    return report


__all__ = [
    "CONTROLLER_BACKEND", "CourseSpec", "GPU_CURRICULUM_CAPABILITIES", "OfficialCourseAdapterError",
    "OfficialCourseBundle", "OfficialCourseConfig", "OfficialCourseTask", "REWARD_SCHEMA",
    "build_curriculum_stage", "evaluate_policy_stage", "load_official_course_config",
]
