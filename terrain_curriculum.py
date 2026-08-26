"""Strict, non-navigating task definitions for locomotion training.

The route endpoints in this module are reset and evaluation metadata only.
They never become policy observations and never produce actuator commands.  A
locomotion policy receives only :class:`LocomotionCommandSpec` values:
forward speed, yaw rate, and a jump edge.  A task can optionally declare one
fixed *route-progress* threshold for that jump edge.  This is a deterministic
high-level task schedule, not navigation or a generated trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


RMUC_SCHEMA_VERSION = 3
OFFICIAL_TERRAIN_SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = frozenset((RMUC_SCHEMA_VERSION, OFFICIAL_TERRAIN_SCHEMA_VERSION))
# Keep the historic public constant stable for callers that identify the
# original RMUC curriculum format directly.
SCHEMA_VERSION = RMUC_SCHEMA_VERSION
DEFAULT_CONFIG_PATH = Path(__file__).with_name("rmuc_terrain_curriculum.yaml")
REQUIRED_TASK_IDS = frozenset(
    (
        "flat",
        "uphill",
        "downhill",
        "stair_up",
        "stair_down",
        "turn_left",
        "turn_right",
        "accel_turn",
        "accel_turn_right",
    )
)


class TerrainCurriculumError(ValueError):
    """Raised when a terrain curriculum file is malformed or inconsistent."""


@dataclass(frozen=True)
class TerrainSpawn:
    """Planar reset or evaluation pose in the RMUC world frame."""

    x_m: float
    y_m: float
    yaw_rad: float

    def xy(self) -> tuple[float, float]:
        return (self.x_m, self.y_m)


@dataclass(frozen=True)
class LocomotionCommandSpec:
    """The only controller-facing input supplied by a terrain task."""

    forward_speed_mps: float
    yaw_rate_rad_s: float
    jump_request: bool

    def as_reset_option(self) -> dict[str, float | bool]:
        """Return the mapping accepted by ``WheelLegResidualEnv.reset``."""
        return {
            "forward_speed_mps": self.forward_speed_mps,
            "yaw_rate_rad_s": self.yaw_rate_rad_s,
            "jump_request": self.jump_request,
        }


@dataclass(frozen=True)
class TerrainRoute:
    """One fixed test corridor for a task; not a navigation path."""

    route_id: str
    spawn: TerrainSpawn
    goal: TerrainSpawn
    corridor_half_width_m: float

    @property
    def nominal_distance_m(self) -> float:
        return math.hypot(self.goal.x_m - self.spawn.x_m, self.goal.y_m - self.spawn.y_m)


@dataclass(frozen=True)
class TerrainRouteBinding:
    """One v4 route's required static collision assets.

    This is scene-validation metadata only. It is deliberately not passed to
    the controller, observation, reward, or actuator layers.
    """

    task_id: str
    route_id: str
    support_geoms: tuple[str, ...]
    obstacle_geoms: tuple[str, ...]


@dataclass(frozen=True)
class TerrainSceneContract:
    """Immutable v4 contract between a curriculum and an MJCF scene."""

    mjcf_filename: str
    mjcf_model: str
    terrain_spec_filename: str
    support_geoms: tuple[str, ...]
    obstacle_geoms: tuple[str, ...]
    route_bindings: tuple[TerrainRouteBinding, ...]


@dataclass(frozen=True)
class TerrainTask:
    """One command-conditioned locomotion task with one or more fixed routes."""

    task_id: str
    sampling_weight: float
    command: LocomotionCommandSpec
    required_distance_m: float
    completion_tolerance_m: float
    max_episode_seconds: float
    routes: tuple[TerrainRoute, ...]
    jump_trigger_progress_m: float | None = None
    completion_mode: str = "route_progress"
    command_tracking_hold_seconds: float | None = None
    speed_tracking_tolerance_mps: float | None = None
    yaw_rate_tracking_tolerance_rad_s: float | None = None
    jump_launch_speed_mps: float | None = None
    lqr_speed_reference_scale: float = 1.0
    required_consecutive_successes: int = 1

    def route_at(self, index: int) -> TerrainRoute:
        """Select a declared route cyclically without generating a path."""
        if not self.routes:
            raise TerrainCurriculumError(f"task {self.task_id!r} has no routes")
        return self.routes[index % len(self.routes)]

    @property
    def has_progress_jump_trigger(self) -> bool:
        """Whether this task has a fixed one-shot jump trigger."""
        return self.jump_trigger_progress_m is not None

    @property
    def uses_command_tracking_completion(self) -> bool:
        """Whether evaluation ends after sustained command tracking.

        This condition is evaluation metadata only. It never supplies a route
        target, trajectory, or navigation signal to the policy.
        """
        return self.completion_mode == "command_tracking_hold"

    def jump_edge_due(self, progress_m: float, *, already_triggered: bool) -> bool:
        """Return whether the task's single scheduled jump edge is due.

        ``progress_m`` is measured along the declared reset/evaluation corridor
        by the environment.  The caller owns ``already_triggered`` so the
        method is stateless and can never turn a jump edge into a repeated
        command.  No route point is exposed to the policy or used to steer it.
        """
        progress = float(progress_m)
        if not math.isfinite(progress):
            raise ValueError("progress_m must be finite")
        return bool(
            not already_triggered
            and self.jump_trigger_progress_m is not None
            and progress >= self.jump_trigger_progress_m
        )


@dataclass(frozen=True)
class TerrainCurriculumStage:
    """A measurable curriculum gate with bounded high-level command scaling."""

    stage_id: str
    task_ids: tuple[str, ...]
    evaluation_episodes: int
    maximum_unsafe_rate: float
    minimum_completion_rate: float
    command_speed_scale: float
    command_yaw_rate_scale: float
    maximum_speed_mae_mps: float | None = None
    maximum_yaw_mae_rad: float | None = None

    def __post_init__(self) -> None:
        for name in ("command_speed_scale", "command_yaw_rate_scale"):
            scale = float(getattr(self, name))
            if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
                raise TerrainCurriculumError(f"{name} must be finite and within (0, 1]")
        for name in ("maximum_speed_mae_mps", "maximum_yaw_mae_rad"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
                raise TerrainCurriculumError(f"{name} must be finite and > 0 when supplied")

    def command_for(self, task: TerrainTask) -> LocomotionCommandSpec:
        """Return this stage's scaled high-level command for one declared task.

        The route remains reset/evaluation metadata.  This method only scales
        the command seen by the locomotion controller and preserves a jump
        edge exactly as declared by the task.
        """
        if task.task_id not in self.task_ids:
            raise TerrainCurriculumError(
                f"task {task.task_id!r} is not enabled by stage {self.stage_id!r}"
            )
        return LocomotionCommandSpec(
            forward_speed_mps=task.command.forward_speed_mps * self.command_speed_scale,
            yaw_rate_rad_s=task.command.yaw_rate_rad_s * self.command_yaw_rate_scale,
            jump_request=task.command.jump_request,
        )


@dataclass(frozen=True)
class TerrainCommandLimits:
    """Global command limits used to validate all task commands."""

    max_forward_speed_mps: float
    max_yaw_rate_rad_s: float


@dataclass(frozen=True)
class TerrainCurriculumConfig:
    """Immutable validated locomotion training curriculum."""

    schema_version: int
    name: str
    limits: TerrainCommandLimits
    tasks: tuple[TerrainTask, ...]
    stages: tuple[TerrainCurriculumStage, ...]
    scene_id: str | None = None
    scene_contract: TerrainSceneContract | None = None
    _task_index: Mapping[str, TerrainTask] = field(init=False, repr=False, compare=False)
    _stage_index: Mapping[str, TerrainCurriculumStage] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        task_index = {task.task_id: task for task in self.tasks}
        stage_index = {stage.stage_id: stage for stage in self.stages}
        object.__setattr__(self, "_task_index", MappingProxyType(task_index))
        object.__setattr__(self, "_stage_index", MappingProxyType(stage_index))

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def task(self, task_id: str) -> TerrainTask:
        try:
            return self._task_index[task_id]
        except KeyError as error:
            raise KeyError(f"unknown terrain task {task_id!r}") from error

    def stage(self, stage_id: str) -> TerrainCurriculumStage:
        try:
            return self._stage_index[stage_id]
        except KeyError as error:
            raise KeyError(f"unknown terrain curriculum stage {stage_id!r}") from error

    def stage_max_episode_seconds(self, stage_id: str) -> float:
        """Return the longest declared task horizon for one curriculum stage."""
        stage = self.stage(stage_id)
        return max(self.task(task_id).max_episode_seconds for task_id in stage.task_ids)


def validate_scene_contract(
    curriculum: TerrainCurriculumConfig,
    xml_path: str | Path,
    *,
    curriculum_path: str | Path | None = None,
) -> None:
    """Fail closed when a v4 curriculum is paired with the wrong MJCF scene.

    RMUC v3 has no scene contract and intentionally keeps its historical
    caller behavior.  Official v4 scenes bind the XML model and every static
    support/obstacle geom by name before MuJoCo state is created for training.
    """
    contract = curriculum.scene_contract
    if curriculum.schema_version < OFFICIAL_TERRAIN_SCHEMA_VERSION or contract is None:
        return
    path = Path(xml_path).expanduser().resolve()
    if not path.is_file():
        raise TerrainCurriculumError(f"scene contract MJCF does not exist: {path}")
    if path.name != contract.mjcf_filename:
        raise TerrainCurriculumError(
            f"scene contract expects {contract.mjcf_filename!r}, got {path.name!r}"
        )
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise TerrainCurriculumError(f"cannot parse scene contract MJCF {path}: {error}") from error
    if root.tag != "mujoco" or root.attrib.get("model") != contract.mjcf_model:
        raise TerrainCurriculumError(
            f"scene contract expects model {contract.mjcf_model!r} in {path.name!r}"
        )
    geom_names = {
        str(geom.attrib["name"])
        for geom in root.iter("geom")
        if "name" in geom.attrib
    }
    expected = set(contract.support_geoms) | set(contract.obstacle_geoms)
    missing = sorted(expected - geom_names)
    if missing:
        raise TerrainCurriculumError(
            "scene contract geoms missing from MJCF: " + ", ".join(missing)
        )
    if curriculum_path is not None:
        spec_path = Path(curriculum_path).expanduser().resolve().parent / contract.terrain_spec_filename
        if not spec_path.is_file():
            raise TerrainCurriculumError(
                f"scene contract terrain specification does not exist: {spec_path}"
            )


def _yaml_loader() -> type[Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required to load RMUC terrain curriculum files; install PyYAML>=6.0."
        ) from error

    class StrictSafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise TerrainCurriculumError("YAML mapping keys must be hashable") from error
            if duplicate:
                raise TerrainCurriculumError(f"duplicate YAML key {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    StrictSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return StrictSafeLoader


def _path(path: str) -> str:
    return path if path else "root"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TerrainCurriculumError(f"{_path(path)} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TerrainCurriculumError(f"{_path(path)} keys must be strings")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TerrainCurriculumError(f"{_path(path)} must be a YAML sequence")
    return value


def _exact_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    path: str,
    *,
    optional: set[str] | None = None,
) -> None:
    actual = set(mapping)
    missing = sorted(required - actual)
    allowed = required if optional is None else required | optional
    unexpected = sorted(actual - allowed)
    if missing or unexpected:
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ", ".join(missing))
        if unexpected:
            detail.append("unexpected=" + ", ".join(unexpected))
        raise TerrainCurriculumError(f"{_path(path)} has invalid keys: {'; '.join(detail)}")


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TerrainCurriculumError(f"{_path(path)} must be a non-empty string")
    if value.strip() != value or any(character.isspace() for character in value):
        raise TerrainCurriculumError(f"{_path(path)} must not contain whitespace")
    return value


def _filename(value: Any, path: str, *, suffix: str) -> str:
    filename = _identifier(value, path)
    if "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise TerrainCurriculumError(f"{_path(path)} must be a filename, not a path")
    if not filename.endswith(suffix):
        raise TerrainCurriculumError(f"{_path(path)} must end with {suffix!r}")
    return filename


def _identifier_tuple(value: Any, path: str, *, allow_empty: bool) -> tuple[str, ...]:
    identifiers = tuple(
        _identifier(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    )
    if not identifiers and not allow_empty:
        raise TerrainCurriculumError(f"{_path(path)} must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise TerrainCurriculumError(f"{_path(path)} contains duplicates")
    return identifiers


def _route_binding(
    value: Any,
    path: str,
    *,
    known_support_geoms: set[str],
    known_obstacle_geoms: set[str],
) -> TerrainRouteBinding:
    mapping = _mapping(value, path)
    _exact_keys(mapping, {"task_id", "route_id", "support_geoms", "obstacle_geoms"}, path)
    support_geoms = _identifier_tuple(
        mapping["support_geoms"], f"{path}.support_geoms", allow_empty=False
    )
    obstacle_geoms = _identifier_tuple(
        mapping["obstacle_geoms"], f"{path}.obstacle_geoms", allow_empty=True
    )
    unknown_support = sorted(set(support_geoms) - known_support_geoms)
    if unknown_support:
        raise TerrainCurriculumError(
            f"{path}.support_geoms references assets absent from scene_contract.support_geoms: "
            + ", ".join(unknown_support)
        )
    unknown_obstacles = sorted(set(obstacle_geoms) - known_obstacle_geoms)
    if unknown_obstacles:
        raise TerrainCurriculumError(
            f"{path}.obstacle_geoms references assets absent from scene_contract.obstacle_geoms: "
            + ", ".join(unknown_obstacles)
        )
    return TerrainRouteBinding(
        task_id=_identifier(mapping["task_id"], f"{path}.task_id"),
        route_id=_identifier(mapping["route_id"], f"{path}.route_id"),
        support_geoms=support_geoms,
        obstacle_geoms=obstacle_geoms,
    )


def _scene_contract(value: Any, path: str) -> TerrainSceneContract:
    mapping = _mapping(value, path)
    _exact_keys(
        mapping,
        {
            "mjcf_filename",
            "mjcf_model",
            "terrain_spec_filename",
            "support_geoms",
            "obstacle_geoms",
            "route_bindings",
        },
        path,
    )
    support_geoms = _identifier_tuple(
        mapping["support_geoms"], f"{path}.support_geoms", allow_empty=False
    )
    obstacle_geoms = _identifier_tuple(
        mapping["obstacle_geoms"], f"{path}.obstacle_geoms", allow_empty=True
    )
    shared_geoms = sorted(set(support_geoms) & set(obstacle_geoms))
    if shared_geoms:
        raise TerrainCurriculumError(
            f"{path}.support_geoms and {path}.obstacle_geoms overlap: "
            + ", ".join(shared_geoms)
        )
    route_bindings = tuple(
        _route_binding(
            item,
            f"{path}.route_bindings[{index}]",
            known_support_geoms=set(support_geoms),
            known_obstacle_geoms=set(obstacle_geoms),
        )
        for index, item in enumerate(_sequence(mapping["route_bindings"], f"{path}.route_bindings"))
    )
    if not route_bindings:
        raise TerrainCurriculumError(f"{path}.route_bindings must not be empty")
    binding_keys = tuple((binding.task_id, binding.route_id) for binding in route_bindings)
    if len(set(binding_keys)) != len(binding_keys):
        raise TerrainCurriculumError(f"{path}.route_bindings contains duplicate task_id/route_id pairs")
    return TerrainSceneContract(
        mjcf_filename=_filename(mapping["mjcf_filename"], f"{path}.mjcf_filename", suffix=".xml"),
        mjcf_model=_identifier(mapping["mjcf_model"], f"{path}.mjcf_model"),
        terrain_spec_filename=_filename(
            mapping["terrain_spec_filename"], f"{path}.terrain_spec_filename", suffix=".yaml"
        ),
        support_geoms=support_geoms,
        obstacle_geoms=obstacle_geoms,
        route_bindings=route_bindings,
    )


def _validate_v4_route_bindings(
    tasks: Sequence[TerrainTask],
    scene_contract: TerrainSceneContract,
) -> None:
    declared = {(binding.task_id, binding.route_id) for binding in scene_contract.route_bindings}
    actual = {(task.task_id, route.route_id) for task in tasks for route in task.routes}
    missing = sorted(actual - declared)
    unexpected = sorted(declared - actual)
    if missing or unexpected:
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ", ".join(f"{task_id}/{route_id}" for task_id, route_id in missing))
        if unexpected:
            detail.append(
                "unexpected=" + ", ".join(f"{task_id}/{route_id}" for task_id, route_id in unexpected)
            )
        raise TerrainCurriculumError(
            "root.scene_contract.route_bindings must match root.tasks routes exactly: "
            + "; ".join(detail)
        )


def _finite_float(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TerrainCurriculumError(f"{_path(path)} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TerrainCurriculumError(f"{_path(path)} must be finite")
    if minimum is not None and result < minimum:
        raise TerrainCurriculumError(f"{_path(path)} must be >= {minimum:g}")
    if maximum is not None and result > maximum:
        raise TerrainCurriculumError(f"{_path(path)} must be <= {maximum:g}")
    return result


def _positive_float(value: Any, path: str) -> float:
    return _finite_float(value, path, minimum=math.ulp(1.0))


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TerrainCurriculumError(f"{_path(path)} must be an integer")
    if value < minimum:
        raise TerrainCurriculumError(f"{_path(path)} must be >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TerrainCurriculumError(f"{_path(path)} must be boolean")
    return value


def _spawn(value: Any, path: str) -> TerrainSpawn:
    mapping = _mapping(value, path)
    _exact_keys(mapping, {"x_m", "y_m", "yaw_rad"}, path)
    yaw_rad = _finite_float(mapping["yaw_rad"], f"{path}.yaw_rad")
    if abs(yaw_rad) > math.pi + 1e-9:
        raise TerrainCurriculumError(f"{path}.yaw_rad must be within -pi..pi")
    return TerrainSpawn(
        x_m=_finite_float(mapping["x_m"], f"{path}.x_m"),
        y_m=_finite_float(mapping["y_m"], f"{path}.y_m"),
        yaw_rad=yaw_rad,
    )


def _route(value: Any, path: str) -> TerrainRoute:
    mapping = _mapping(value, path)
    _exact_keys(mapping, {"id", "spawn", "goal", "corridor_half_width_m"}, path)
    route = TerrainRoute(
        route_id=_identifier(mapping["id"], f"{path}.id"),
        spawn=_spawn(mapping["spawn"], f"{path}.spawn"),
        goal=_spawn(mapping["goal"], f"{path}.goal"),
        corridor_half_width_m=_positive_float(
            mapping["corridor_half_width_m"], f"{path}.corridor_half_width_m"
        ),
    )
    if route.nominal_distance_m <= 1e-6:
        raise TerrainCurriculumError(f"{path} must have distinct spawn and goal XY positions")
    return route


def _command(value: Any, path: str, limits: TerrainCommandLimits) -> LocomotionCommandSpec:
    mapping = _mapping(value, path)
    _exact_keys(mapping, {"forward_speed_mps", "yaw_rate_rad_s", "jump_request"}, path)
    speed = _finite_float(mapping["forward_speed_mps"], f"{path}.forward_speed_mps")
    yaw_rate = _finite_float(mapping["yaw_rate_rad_s"], f"{path}.yaw_rate_rad_s")
    if abs(speed) > limits.max_forward_speed_mps:
        raise TerrainCurriculumError(
            f"{path}.forward_speed_mps exceeds limits.max_forward_speed_mps"
        )
    if abs(yaw_rate) > limits.max_yaw_rate_rad_s:
        raise TerrainCurriculumError(
            f"{path}.yaw_rate_rad_s exceeds limits.max_yaw_rate_rad_s"
        )
    if abs(speed) <= 1e-6:
        raise TerrainCurriculumError(f"{path}.forward_speed_mps must be non-zero for a locomotion task")
    return LocomotionCommandSpec(
        forward_speed_mps=speed,
        yaw_rate_rad_s=yaw_rate,
        jump_request=_boolean(mapping["jump_request"], f"{path}.jump_request"),
    )


def _task(
    value: Any,
    path: str,
    limits: TerrainCommandLimits,
    *,
    schema_version: int,
) -> TerrainTask:
    mapping = _mapping(value, path)
    required_keys = {
        "id",
        "sampling_weight",
        "command",
        "required_distance_m",
        "completion_tolerance_m",
        "max_episode_seconds",
        "routes",
    }
    if schema_version == OFFICIAL_TERRAIN_SCHEMA_VERSION:
        required_keys.add("required_consecutive_successes")
    _exact_keys(
        mapping,
        required_keys,
        path,
        optional={
            "jump_trigger_progress_m",
            "completion_mode",
            "command_tracking_hold_seconds",
            "speed_tracking_tolerance_mps",
            "yaw_rate_tracking_tolerance_rad_s",
            "jump_launch_speed_mps",
            "lqr_speed_reference_scale",
        },
    )
    routes = tuple(_route(item, f"{path}.routes[{index}]") for index, item in enumerate(
        _sequence(mapping["routes"], f"{path}.routes")
    ))
    if not routes:
        raise TerrainCurriculumError(f"{path}.routes must not be empty")
    route_ids = tuple(route.route_id for route in routes)
    if len(set(route_ids)) != len(route_ids):
        raise TerrainCurriculumError(f"{path}.routes contains duplicate ids")
    command = _command(mapping["command"], f"{path}.command", limits)
    jump_trigger_progress_m = (
        None
        if "jump_trigger_progress_m" not in mapping
        else _finite_float(
            mapping["jump_trigger_progress_m"],
            f"{path}.jump_trigger_progress_m",
            minimum=0.0,
        )
    )
    completion_mode = mapping.get("completion_mode", "route_progress")
    if completion_mode not in ("route_progress", "command_tracking_hold"):
        raise TerrainCurriculumError(
            f"{path}.completion_mode must be route_progress or command_tracking_hold"
        )
    tracking_keys = (
        "command_tracking_hold_seconds",
        "speed_tracking_tolerance_mps",
        "yaw_rate_tracking_tolerance_rad_s",
    )
    jump_launch_speed_mps = (
        None
        if "jump_launch_speed_mps" not in mapping
        else _finite_float(
            mapping["jump_launch_speed_mps"],
            f"{path}.jump_launch_speed_mps",
            minimum=0.0,
            maximum=0.55,
        )
    )
    if jump_launch_speed_mps is not None and not (
        jump_trigger_progress_m is not None or command.jump_request
    ):
        raise TerrainCurriculumError(
            f"{path}.jump_launch_speed_mps requires a jump command or progress jump trigger"
        )
    lqr_speed_reference_scale = _finite_float(
        mapping.get("lqr_speed_reference_scale", 1.0),
        f"{path}.lqr_speed_reference_scale",
        minimum=math.ulp(1.0),
        maximum=1.0,
    )
    required_consecutive_successes = (
        _integer(
            mapping["required_consecutive_successes"],
            f"{path}.required_consecutive_successes",
            minimum=1,
        )
        if schema_version == OFFICIAL_TERRAIN_SCHEMA_VERSION
        else 1
    )
    supplied_tracking_keys = [key for key in tracking_keys if key in mapping]
    if completion_mode == "command_tracking_hold":
        missing_tracking_keys = [key for key in tracking_keys if key not in mapping]
        if missing_tracking_keys:
            raise TerrainCurriculumError(
                f"{path}.completion_mode=command_tracking_hold requires "
                + ", ".join(missing_tracking_keys)
            )
        command_tracking_hold_seconds = _positive_float(
            mapping["command_tracking_hold_seconds"],
            f"{path}.command_tracking_hold_seconds",
        )
        speed_tracking_tolerance_mps = _positive_float(
            mapping["speed_tracking_tolerance_mps"],
            f"{path}.speed_tracking_tolerance_mps",
        )
        yaw_rate_tracking_tolerance_rad_s = _positive_float(
            mapping["yaw_rate_tracking_tolerance_rad_s"],
            f"{path}.yaw_rate_tracking_tolerance_rad_s",
        )
    else:
        if supplied_tracking_keys:
            raise TerrainCurriculumError(
                f"{path} supplies command-tracking fields but completion_mode is route_progress"
            )
        command_tracking_hold_seconds = None
        speed_tracking_tolerance_mps = None
        yaw_rate_tracking_tolerance_rad_s = None
    task = TerrainTask(
        task_id=_identifier(mapping["id"], f"{path}.id"),
        sampling_weight=_positive_float(mapping["sampling_weight"], f"{path}.sampling_weight"),
        command=command,
        required_distance_m=_positive_float(
            mapping["required_distance_m"], f"{path}.required_distance_m"
        ),
        completion_tolerance_m=_positive_float(
            mapping["completion_tolerance_m"], f"{path}.completion_tolerance_m"
        ),
        max_episode_seconds=_positive_float(
            mapping["max_episode_seconds"], f"{path}.max_episode_seconds"
        ),
        routes=routes,
        jump_trigger_progress_m=jump_trigger_progress_m,
        completion_mode=completion_mode,
        command_tracking_hold_seconds=command_tracking_hold_seconds,
        speed_tracking_tolerance_mps=speed_tracking_tolerance_mps,
        yaw_rate_tracking_tolerance_rad_s=yaw_rate_tracking_tolerance_rad_s,
        jump_launch_speed_mps=jump_launch_speed_mps,
        lqr_speed_reference_scale=lqr_speed_reference_scale,
        required_consecutive_successes=required_consecutive_successes,
    )
    shortest_route = min(route.nominal_distance_m for route in task.routes)
    if task.required_distance_m > shortest_route + 1e-6:
        raise TerrainCurriculumError(
            f"{path}.required_distance_m exceeds the shortest declared route ({shortest_route:.3f} m)"
        )
    completion_progress_m = task.required_distance_m - task.completion_tolerance_m
    if completion_progress_m <= 0.0:
        raise TerrainCurriculumError(
            f"{path}.completion_tolerance_m must be smaller than required_distance_m"
        )
    if task.command.jump_request and task.jump_trigger_progress_m is not None:
        raise TerrainCurriculumError(
            f"{path} cannot combine command.jump_request with jump_trigger_progress_m"
        )
    if (
        task.jump_trigger_progress_m is not None
        and task.jump_trigger_progress_m >= completion_progress_m
    ):
        raise TerrainCurriculumError(
            f"{path}.jump_trigger_progress_m must precede the task completion threshold"
        )
    return task


def _stage(
    value: Any,
    path: str,
    known_task_ids: set[str],
    *,
    schema_version: int,
) -> TerrainCurriculumStage:
    mapping = _mapping(value, path)
    required_keys = {
        "id",
        "task_ids",
        "evaluation_episodes",
        "maximum_unsafe_rate",
        "minimum_completion_rate",
        "command_speed_scale",
        "command_yaw_rate_scale",
    }
    tracking_keys = {"maximum_speed_mae_mps", "maximum_yaw_mae_rad"}
    if schema_version == OFFICIAL_TERRAIN_SCHEMA_VERSION:
        required_keys.update(tracking_keys)
        optional_keys: set[str] | None = None
    else:
        # Keep schema-3 files backward compatible while allowing a stage to
        # opt into the same tracking gates used by official schema 4.  The
        # pair is intentionally all-or-nothing so a typo cannot create a
        # one-sided acceptance gate.
        optional_keys = tracking_keys
        supplied_tracking_keys = set(mapping) & tracking_keys
        if supplied_tracking_keys and supplied_tracking_keys != tracking_keys:
            missing = sorted(tracking_keys - supplied_tracking_keys)
            raise TerrainCurriculumError(
                f"{path} must provide both tracking gate keys; missing={', '.join(missing)}"
            )
    _exact_keys(mapping, required_keys, path, optional=optional_keys)
    task_ids = tuple(
        _identifier(item, f"{path}.task_ids[{index}]")
        for index, item in enumerate(_sequence(mapping["task_ids"], f"{path}.task_ids"))
    )
    if not task_ids:
        raise TerrainCurriculumError(f"{path}.task_ids must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise TerrainCurriculumError(f"{path}.task_ids contains duplicates")
    unknown = sorted(set(task_ids) - known_task_ids)
    if unknown:
        raise TerrainCurriculumError(f"{path}.task_ids references unknown tasks: {', '.join(unknown)}")
    return TerrainCurriculumStage(
        stage_id=_identifier(mapping["id"], f"{path}.id"),
        task_ids=task_ids,
        evaluation_episodes=_integer(mapping["evaluation_episodes"], f"{path}.evaluation_episodes", minimum=1),
        maximum_unsafe_rate=_finite_float(
            mapping["maximum_unsafe_rate"], f"{path}.maximum_unsafe_rate", minimum=0.0, maximum=1.0
        ),
        minimum_completion_rate=_finite_float(
            mapping["minimum_completion_rate"], f"{path}.minimum_completion_rate", minimum=0.0, maximum=1.0
        ),
        command_speed_scale=_finite_float(
            mapping["command_speed_scale"],
            f"{path}.command_speed_scale",
            minimum=math.ulp(1.0),
            maximum=1.0,
        ),
        command_yaw_rate_scale=_finite_float(
            mapping["command_yaw_rate_scale"],
            f"{path}.command_yaw_rate_scale",
            minimum=math.ulp(1.0),
            maximum=1.0,
        ),
        maximum_speed_mae_mps=(
            _positive_float(
                mapping["maximum_speed_mae_mps"],
                f"{path}.maximum_speed_mae_mps",
            )
            if "maximum_speed_mae_mps" in mapping
            else None
        ),
        maximum_yaw_mae_rad=(
            _positive_float(
                mapping["maximum_yaw_mae_rad"],
                f"{path}.maximum_yaw_mae_rad",
            )
            if "maximum_yaw_mae_rad" in mapping
            else None
        ),
    )


def load_terrain_curriculum(path: str | Path = DEFAULT_CONFIG_PATH) -> TerrainCurriculumConfig:
    """Load a complete, strict locomotion curriculum from YAML.

    Unknown and duplicate keys are rejected so a typo cannot silently alter a
    training run.  The returned object is immutable and contains no navigation
    logic; callers choose a fixed route at reset and pass only ``task.command``
    to the locomotion controller.  For a task with
    ``jump_trigger_progress_m``, callers use ``task.jump_edge_due(...)`` once
    while progressing along that fixed corridor.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"terrain curriculum YAML does not exist: {config_path}")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required to load RMUC terrain curriculum files; install PyYAML>=6.0."
        ) from error
    try:
        loaded = yaml.load(config_path.read_text(encoding="ascii"), Loader=_yaml_loader())
    except TerrainCurriculumError:
        raise
    except yaml.YAMLError as error:
        raise TerrainCurriculumError(f"invalid YAML in {config_path}: {error}") from error
    root = _mapping(loaded, "root")
    schema_version = _integer(root.get("schema_version"), "root.schema_version", minimum=1)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise TerrainCurriculumError(
            "root.schema_version must be one of "
            + ", ".join(str(version) for version in sorted(SUPPORTED_SCHEMA_VERSIONS))
            + f", got {schema_version}"
        )
    root_keys = {"schema_version", "name", "limits", "tasks", "stages"}
    if schema_version == RMUC_SCHEMA_VERSION:
        _exact_keys(root, root_keys, "root")
        scene_id = None
        scene_contract = None
    else:
        _exact_keys(root, root_keys | {"scene_id", "scene_contract"}, "root")
        scene_id = _identifier(root["scene_id"], "root.scene_id")
        scene_contract = _scene_contract(root["scene_contract"], "root.scene_contract")
    limits_mapping = _mapping(root["limits"], "root.limits")
    _exact_keys(limits_mapping, {"max_forward_speed_mps", "max_yaw_rate_rad_s"}, "root.limits")
    limits = TerrainCommandLimits(
        max_forward_speed_mps=_positive_float(
            limits_mapping["max_forward_speed_mps"], "root.limits.max_forward_speed_mps"
        ),
        max_yaw_rate_rad_s=_positive_float(
            limits_mapping["max_yaw_rate_rad_s"], "root.limits.max_yaw_rate_rad_s"
        ),
    )
    tasks = tuple(
        _task(item, f"root.tasks[{index}]", limits, schema_version=schema_version)
        for index, item in enumerate(_sequence(root["tasks"], "root.tasks"))
    )
    if not tasks:
        raise TerrainCurriculumError("root.tasks must not be empty")
    task_ids = tuple(task.task_id for task in tasks)
    if len(set(task_ids)) != len(task_ids):
        raise TerrainCurriculumError("root.tasks contains duplicate ids")
    if schema_version == RMUC_SCHEMA_VERSION:
        actual_task_ids = frozenset(task_ids)
        if actual_task_ids != REQUIRED_TASK_IDS:
            missing = sorted(REQUIRED_TASK_IDS - actual_task_ids)
            unexpected = sorted(actual_task_ids - REQUIRED_TASK_IDS)
            detail: list[str] = []
            if missing:
                detail.append("missing=" + ", ".join(missing))
            if unexpected:
                detail.append("unexpected=" + ", ".join(unexpected))
            raise TerrainCurriculumError(
                "root.tasks must contain the RMUC task ids exactly: " + "; ".join(detail)
            )
    elif scene_contract is not None:
        _validate_v4_route_bindings(tasks, scene_contract)
    stages = tuple(
        _stage(
            item,
            f"root.stages[{index}]",
            set(task_ids),
            schema_version=schema_version,
        )
        for index, item in enumerate(_sequence(root["stages"], "root.stages"))
    )
    if not stages:
        raise TerrainCurriculumError("root.stages must not be empty")
    stage_ids = tuple(stage.stage_id for stage in stages)
    if len(set(stage_ids)) != len(stage_ids):
        raise TerrainCurriculumError("root.stages contains duplicate ids")
    staged_task_ids = {task_id for stage in stages for task_id in stage.task_ids}
    unstaged = sorted(set(task_ids) - staged_task_ids)
    if unstaged:
        raise TerrainCurriculumError("tasks absent from every stage: " + ", ".join(unstaged))
    return TerrainCurriculumConfig(
        schema_version=schema_version,
        name=_identifier(root["name"], "root.name"),
        limits=limits,
        tasks=tasks,
        stages=stages,
        scene_id=scene_id,
        scene_contract=scene_contract,
    )


def main() -> None:
    curriculum = load_terrain_curriculum()
    print(
        f"loaded {curriculum.name}: tasks={','.join(curriculum.task_ids)} "
        f"stages={','.join(stage.stage_id for stage in curriculum.stages)}"
    )


if __name__ == "__main__":
    main()
