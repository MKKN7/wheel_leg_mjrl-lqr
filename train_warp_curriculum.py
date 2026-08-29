"""Capability-gated MuJoCo-Warp curriculum PPO training.

    The manifest and stage capability checks run before GPU model allocation.
    External backends publish a ``GPU_CURRICULUM_CAPABILITIES`` mapping and a
    ``build_curriculum_stage`` factory.  Every non-flat curriculum stage remains
    conditional until its own real CUDA runtime gate passes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from entrypoint_paths import project_path, resolve_cli_input

CURRICULUM_CONFIG_SCHEMA = 3
CURRICULUM_CHECKPOINT_FORMAT = 2
CURRICULUM_BACKEND = "mujoco_warp_curriculum_ppo_v2"
FLAT_BACKEND = "mujoco_warp_flat_ppo_fixed_gain_v2"
FLAT_CONTROLLER_BACKEND = "fixed_gain_flat_controller_v2"
GATE_EVIDENCE_SCHEMA = 2
OBSERVATION_SIZE = 67
ACTION_SIZE = 7
REWARD_SCHEMA = "warp_flat_terrain_compensated_reward_v2"
LEGACY_FLAT_REWARD_SCHEMA = "warp_flat_walking_reward_v1"
ACTION_SEMANTICS = "seven_dimensional_fixed_gain_residual"


class WarpCurriculumConfigError(ValueError):
    """Raised when a curriculum cannot be proven safe and reproducible."""


@dataclass(frozen=True)
class PpoSettings:
    total_timesteps: int
    rollout_steps: int
    epochs: int
    minibatch_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_coefficient: float
    entropy_coefficient: float
    max_gradient_norm: float
    hidden_size: int
    initial_action_std: float
    seed: int


@dataclass(frozen=True)
class ArtifactSettings:
    checkpoint_path: Path
    metrics_path: Path
    checkpoint_interval_updates: int


@dataclass(frozen=True)
class SmokeSettings:
    rollout_steps: int
    epochs: int
    minibatch_size: int
    checkpoint_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class StageConfig:
    stage_id: str
    task_mode: str
    xml_path: Path
    terrain_curriculum_path: Path
    terrain_stage_id: str
    controller_backend: str
    terrain_enabled: bool
    jump_enabled: bool
    steps_enabled: bool
    domain_randomization_enabled: bool
    command_speed_mps: float
    command_yaw_rate_rad_s: float
    residual_action_mask: tuple[float, ...]
    requires_gpu_parity: bool
    prerequisite_stage_ids: tuple[str, ...] = ()
    adapter_config_path: Path | None = None
    scene_variant: str = "canonical"
    reward_schema: str | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        result: list[str] = []
        if self.terrain_enabled:
            result.append("terrain")
        if self.steps_enabled:
            result.append("steps")
        if self.jump_enabled:
            result.append("jump")
        if self.domain_randomization_enabled:
            result.append("domain_randomization")
        if abs(self.command_speed_mps) > 1.0e-12:
            result.append("speed_command")
        if abs(self.command_yaw_rate_rad_s) > 1.0e-12:
            result.append("yaw_command")
        return tuple(result)


@dataclass(frozen=True)
class GpuTaskSettings:
    """Explicit CUDA task settings shared by curriculum-stage factories."""

    sensor_noise_std: float
    control_delay_steps: int
    stability_gate_seconds: float
    command_speed_gain_nm_per_mps: float
    command_yaw_gain_nm_per_rad_s: float
    command_wheel_feedforward_limit_nm: float
    allow_cpu_residual_warm_start: bool


@dataclass(frozen=True)
class GpuStageCapability:
    backend: str
    terrain: bool
    steps: bool
    jump: bool
    domain_randomization: bool
    speed_command: bool
    yaw_command: bool
    observation_size: int = OBSERVATION_SIZE
    action_size: int = ACTION_SIZE
    reward_schema: str = REWARD_SCHEMA
    runtime_gate_required: bool = False


@dataclass(frozen=True)
class CourseEvaluationResult:
    """Host summary of a post-training, CUDA-resident course evaluation."""

    stage_id: str
    episodes: int
    completion_rate: float
    unsafe_rate: float
    speed_mae_mps: float
    yaw_mae_rad: float
    passed: bool


@dataclass(frozen=True)
class CurriculumConfig:
    source_path: Path
    name: str
    batch_config_path: Path
    flat_ppo_config_path: Path
    observation_size: int
    action_size: int
    reward_schema: str
    action_semantics: str
    stages: tuple[StageConfig, ...]
    gpu_task: GpuTaskSettings
    ppo: PpoSettings
    output: ArtifactSettings
    smoke: SmokeSettings
    require_gpu_parity: bool
    allow_external_task_factory: bool
    source_digest: str

    def stage(self, stage_id: str) -> StageConfig:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        available = ", ".join(item.stage_id for item in self.stages)
        raise WarpCurriculumConfigError(
            f"unknown curriculum stage {stage_id!r}; available stages: {available}"
        )


@dataclass(frozen=True)
class ResidualCheckpoint:
    path: Path
    state_dict: dict[str, Any]
    metadata: Mapping[str, Any]
    sha256: str
    transfer_kind: str


GPU_CURRICULUM_CAPABILITIES: dict[str, GpuStageCapability] = {
    # This is the only capability proven by the current repository.
    "rmuc_flat": GpuStageCapability(
        backend=FLAT_CONTROLLER_BACKEND,
        terrain=False,
        steps=False,
        jump=False,
        domain_randomization=False,
        speed_command=False,
        yaw_command=False,
    ),
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WarpCurriculumConfigError(f"{name} must be a YAML mapping")
    return value


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise WarpCurriculumConfigError(f"missing required configuration key: {name}")
    return mapping[name]


def _exact_keys(mapping: Mapping[str, Any], name: str, expected: set[str]) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing}")
        if unknown:
            detail.append(f"unknown={unknown}")
        raise WarpCurriculumConfigError(f"{name} keys are invalid: {', '.join(detail)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise WarpCurriculumConfigError(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise WarpCurriculumConfigError(f"{name} must be boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WarpCurriculumConfigError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WarpCurriculumConfigError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarpCurriculumConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise WarpCurriculumConfigError(f"{name} must be {qualifier}")
    return result


def _resolve_path(source: Path, value: Any, name: str, *, must_exist: bool = False) -> Path:
    candidate = Path(_string(value, name))
    result = candidate.resolve() if candidate.is_absolute() else (source.parent / candidate).resolve()
    if must_exist and not result.is_file():
        raise WarpCurriculumConfigError(f"{name} does not exist: {result}")
    return result


def _load_ppo(raw: Mapping[str, Any]) -> PpoSettings:
    expected = {
        "total_timesteps", "rollout_steps", "epochs", "minibatch_size", "learning_rate",
        "gamma", "gae_lambda", "clip_ratio", "value_coefficient", "entropy_coefficient",
        "max_gradient_norm", "hidden_size", "initial_action_std", "seed",
    }
    _exact_keys(raw, "ppo", expected)
    result = PpoSettings(
        total_timesteps=_positive_int(_required(raw, "total_timesteps"), "ppo.total_timesteps"),
        rollout_steps=_positive_int(_required(raw, "rollout_steps"), "ppo.rollout_steps"),
        epochs=_positive_int(_required(raw, "epochs"), "ppo.epochs"),
        minibatch_size=_positive_int(_required(raw, "minibatch_size"), "ppo.minibatch_size"),
        learning_rate=_finite_float(_required(raw, "learning_rate"), "ppo.learning_rate", positive=True),
        gamma=_finite_float(_required(raw, "gamma"), "ppo.gamma"),
        gae_lambda=_finite_float(_required(raw, "gae_lambda"), "ppo.gae_lambda"),
        clip_ratio=_finite_float(_required(raw, "clip_ratio"), "ppo.clip_ratio", positive=True),
        value_coefficient=_finite_float(_required(raw, "value_coefficient"), "ppo.value_coefficient", positive=True),
        entropy_coefficient=_finite_float(_required(raw, "entropy_coefficient"), "ppo.entropy_coefficient"),
        max_gradient_norm=_finite_float(_required(raw, "max_gradient_norm"), "ppo.max_gradient_norm", positive=True),
        hidden_size=_positive_int(_required(raw, "hidden_size"), "ppo.hidden_size"),
        initial_action_std=_finite_float(_required(raw, "initial_action_std"), "ppo.initial_action_std", positive=True),
        seed=_positive_int(_required(raw, "seed"), "ppo.seed"),
    )
    if not 0.0 <= result.gamma <= 1.0 or not 0.0 <= result.gae_lambda <= 1.0:
        raise WarpCurriculumConfigError("ppo.gamma and ppo.gae_lambda must be within [0, 1]")
    if not 0.0 < result.clip_ratio < 1.0:
        raise WarpCurriculumConfigError("ppo.clip_ratio must be within (0, 1)")
    if not 0.0 < result.initial_action_std <= 1.0:
        raise WarpCurriculumConfigError("ppo.initial_action_std must be within (0, 1]")
    if result.entropy_coefficient < 0.0:
        raise WarpCurriculumConfigError("ppo.entropy_coefficient must be non-negative")
    return result


def _load_artifacts(raw: Mapping[str, Any], source: Path, name: str) -> ArtifactSettings:
    _exact_keys(raw, name, {"checkpoint_path", "metrics_path", "checkpoint_interval_updates"})
    checkpoint = _resolve_path(source, _required(raw, "checkpoint_path"), f"{name}.checkpoint_path")
    metrics = _resolve_path(source, _required(raw, "metrics_path"), f"{name}.metrics_path")
    if checkpoint == metrics:
        raise WarpCurriculumConfigError(f"{name} checkpoint_path and metrics_path must differ")
    if "warp_curriculum" not in checkpoint.name or "warp_curriculum" not in metrics.name:
        raise WarpCurriculumConfigError(f"{name} paths must use the warp_curriculum artifact namespace")
    return ArtifactSettings(
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        checkpoint_interval_updates=_positive_int(
            _required(raw, "checkpoint_interval_updates"), f"{name}.checkpoint_interval_updates"
        ),
    )


def _load_smoke(raw: Mapping[str, Any], source: Path) -> SmokeSettings:
    _exact_keys(raw, "smoke", {"rollout_steps", "epochs", "minibatch_size", "checkpoint_path", "metrics_path"})
    artifact = _load_artifacts(
        {
            "checkpoint_path": _required(raw, "checkpoint_path"),
            "metrics_path": _required(raw, "metrics_path"),
            "checkpoint_interval_updates": 1,
        },
        source,
        "smoke",
    )
    return SmokeSettings(
        rollout_steps=_positive_int(_required(raw, "rollout_steps"), "smoke.rollout_steps"),
        epochs=_positive_int(_required(raw, "epochs"), "smoke.epochs"),
        minibatch_size=_positive_int(_required(raw, "minibatch_size"), "smoke.minibatch_size"),
        checkpoint_path=artifact.checkpoint_path,
        metrics_path=artifact.metrics_path,
    )


def _load_stage(raw: Mapping[str, Any], source: Path, index: int) -> StageConfig:
    name = f"curriculum.stages[{index}]"
    expected = {
        "id", "task_mode", "xml_path", "controller_backend", "terrain_enabled", "jump_enabled",
        "steps_enabled", "domain_randomization_enabled", "command_speed_mps", "command_yaw_rate_rad_s",
        "residual_action_mask", "requires_gpu_parity", "terrain_curriculum_path", "terrain_stage_id",
    }
    optional = {"adapter_config_path", "scene_variant", "reward_schema", "prerequisite_stage_ids"}
    actual = set(raw)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected - optional)
    if missing or unknown:
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing}")
        if unknown:
            detail.append(f"unknown={unknown}")
        raise WarpCurriculumConfigError(f"{name} keys are invalid: {', '.join(detail)}")
    mask_raw = _required(raw, "residual_action_mask")
    if not isinstance(mask_raw, (list, tuple)) or len(mask_raw) != ACTION_SIZE:
        raise WarpCurriculumConfigError(f"{name}.residual_action_mask must contain seven values")
    mask = tuple(_finite_float(value, f"{name}.residual_action_mask[{i}]") for i, value in enumerate(mask_raw))
    if any(value < 0.0 or value > 1.0 for value in mask):
        raise WarpCurriculumConfigError(f"{name}.residual_action_mask values must be within [0, 1]")
    if mask[:6] != (1.0,) * 6 or mask[6] != 0.0:
        raise WarpCurriculumConfigError(
            f"{name}.residual_action_mask must authorize six actuators and mask channel seven"
        )
    speed = _finite_float(_required(raw, "command_speed_mps"), f"{name}.command_speed_mps")
    yaw = _finite_float(_required(raw, "command_yaw_rate_rad_s"), f"{name}.command_yaw_rate_rad_s")
    if abs(speed) > 3.0 or abs(yaw) > 0.45:
        raise WarpCurriculumConfigError(f"{name} command exceeds GPU task limits")
    prerequisites_raw = raw.get("prerequisite_stage_ids", [])
    if not isinstance(prerequisites_raw, list):
        raise WarpCurriculumConfigError(f"{name}.prerequisite_stage_ids must be a sequence")
    prerequisites = tuple(
        _string(value, f"{name}.prerequisite_stage_ids[{index}]")
        for index, value in enumerate(prerequisites_raw)
    )
    if len(set(prerequisites)) != len(prerequisites):
        raise WarpCurriculumConfigError(f"{name}.prerequisite_stage_ids must not contain duplicates")
    stage = StageConfig(
        stage_id=_string(_required(raw, "id"), f"{name}.id"),
        task_mode=_string(_required(raw, "task_mode"), f"{name}.task_mode"),
        xml_path=_resolve_path(source, _required(raw, "xml_path"), f"{name}.xml_path", must_exist=True),
        terrain_curriculum_path=_resolve_path(
            source,
            _required(raw, "terrain_curriculum_path"),
            f"{name}.terrain_curriculum_path",
            must_exist=True,
        ),
        terrain_stage_id=_string(_required(raw, "terrain_stage_id"), f"{name}.terrain_stage_id"),
        controller_backend=_string(_required(raw, "controller_backend"), f"{name}.controller_backend"),
        terrain_enabled=_boolean(_required(raw, "terrain_enabled"), f"{name}.terrain_enabled"),
        jump_enabled=_boolean(_required(raw, "jump_enabled"), f"{name}.jump_enabled"),
        steps_enabled=_boolean(_required(raw, "steps_enabled"), f"{name}.steps_enabled"),
        domain_randomization_enabled=_boolean(
            _required(raw, "domain_randomization_enabled"), f"{name}.domain_randomization_enabled"
        ),
        command_speed_mps=speed,
        command_yaw_rate_rad_s=yaw,
        residual_action_mask=mask,
        requires_gpu_parity=_boolean(_required(raw, "requires_gpu_parity"), f"{name}.requires_gpu_parity"),
        prerequisite_stage_ids=prerequisites,
        adapter_config_path=(
            None
            if "adapter_config_path" not in raw
            else _resolve_path(source, raw["adapter_config_path"], f"{name}.adapter_config_path", must_exist=True)
        ),
        scene_variant=_string(raw.get("scene_variant", "canonical"), f"{name}.scene_variant"),
        reward_schema=(
            None
            if "reward_schema" not in raw
            else _string(raw["reward_schema"], f"{name}.reward_schema")
        ),
    )
    if not stage.stage_id.replace("_", "").isalnum():
        raise WarpCurriculumConfigError(f"{name}.id must contain only letters, digits, and underscores")
    if stage.scene_variant not in {"canonical", "official_warp_compat"}:
        raise WarpCurriculumConfigError(
            f"{name}.scene_variant must be canonical or official_warp_compat"
        )
    if stage.scene_variant == "official_warp_compat" and stage.adapter_config_path is None:
        raise WarpCurriculumConfigError(
            f"{name}.official_warp_compat requires adapter_config_path"
        )
    return stage


def _load_gpu_task_settings(raw: Mapping[str, Any]) -> GpuTaskSettings:
    _exact_keys(
        raw,
        "gpu_task",
        {
            "sensor_noise_std",
            "control_delay_steps",
            "stability_gate_seconds",
            "command_speed_gain_nm_per_mps",
            "command_yaw_gain_nm_per_rad_s",
            "command_wheel_feedforward_limit_nm",
            "allow_cpu_residual_warm_start",
        },
    )
    sensor_noise_std = _finite_float(
        _required(raw, "sensor_noise_std"), "gpu_task.sensor_noise_std"
    )
    if sensor_noise_std < 0.0 or sensor_noise_std > 0.10:
        raise WarpCurriculumConfigError("gpu_task.sensor_noise_std must be within [0, 0.10]")
    delay = _nonnegative_int(_required(raw, "control_delay_steps"), "gpu_task.control_delay_steps")
    if delay > 2:
        raise WarpCurriculumConfigError("gpu_task.control_delay_steps must be within [0, 2]")
    values = {
        name: _finite_float(_required(raw, name), f"gpu_task.{name}")
        for name in (
            "command_speed_gain_nm_per_mps",
            "command_yaw_gain_nm_per_rad_s",
            "command_wheel_feedforward_limit_nm",
        )
    }
    if any(value < 0.0 for value in values.values()):
        raise WarpCurriculumConfigError("gpu_task command gains and limit must be non-negative")
    return GpuTaskSettings(
        sensor_noise_std=sensor_noise_std,
        control_delay_steps=delay,
        stability_gate_seconds=_finite_float(
            _required(raw, "stability_gate_seconds"), "gpu_task.stability_gate_seconds", positive=True
        ),
        command_speed_gain_nm_per_mps=values["command_speed_gain_nm_per_mps"],
        command_yaw_gain_nm_per_rad_s=values["command_yaw_gain_nm_per_rad_s"],
        command_wheel_feedforward_limit_nm=values["command_wheel_feedforward_limit_nm"],
        allow_cpu_residual_warm_start=_boolean(
            _required(raw, "allow_cpu_residual_warm_start"),
            "gpu_task.allow_cpu_residual_warm_start",
        ),
    )


def load_curriculum_config(path: str | Path) -> CurriculumConfig:
    """Load and strictly validate a curriculum manifest."""

    source = Path(path).resolve()
    try:
        source_text = source.read_text(encoding="utf-8")
        raw = yaml.safe_load(source_text)
    except OSError as error:
        raise WarpCurriculumConfigError(f"unable to read curriculum config {source}: {error}") from error
    root = _mapping(raw, "curriculum configuration")
    expected = {
        "schema_version", "backend", "observation_size", "action_size", "reward_schema",
        "action_semantics", "batch_config", "flat_ppo_config", "curriculum", "ppo", "output",
        "smoke", "capabilities", "gpu_task",
    }
    _exact_keys(root, "curriculum configuration", expected)
    if _required(root, "schema_version") != CURRICULUM_CONFIG_SCHEMA:
        raise WarpCurriculumConfigError(f"unsupported curriculum schema; expected {CURRICULUM_CONFIG_SCHEMA}")
    if _required(root, "backend") != CURRICULUM_BACKEND:
        raise WarpCurriculumConfigError(f"backend must be {CURRICULUM_BACKEND!r}")
    observation_size = _positive_int(_required(root, "observation_size"), "observation_size")
    action_size = _positive_int(_required(root, "action_size"), "action_size")
    if observation_size != OBSERVATION_SIZE or action_size != ACTION_SIZE:
        raise WarpCurriculumConfigError(f"curriculum interface must be ({OBSERVATION_SIZE}, {ACTION_SIZE})")
    reward_schema = _string(_required(root, "reward_schema"), "reward_schema")
    action_semantics = _string(_required(root, "action_semantics"), "action_semantics")
    if reward_schema != REWARD_SCHEMA or action_semantics != ACTION_SEMANTICS:
        raise WarpCurriculumConfigError("curriculum reward/action schema is incompatible with GPU task")

    curriculum_raw = _mapping(_required(root, "curriculum"), "curriculum")
    _exact_keys(curriculum_raw, "curriculum", {"name", "stages"})
    stage_values = curriculum_raw["stages"]
    if not isinstance(stage_values, list) or not stage_values:
        raise WarpCurriculumConfigError("curriculum.stages must be a non-empty sequence")
    stages = tuple(
        _load_stage(_mapping(item, f"curriculum.stages[{i}]"), source, i)
        for i, item in enumerate(stage_values)
    )
    stage_ids = [stage.stage_id for stage in stages]
    if len(set(stage_ids)) != len(stage_ids):
        raise WarpCurriculumConfigError("curriculum.stages contains duplicate ids")
    stage_positions = {stage_id: index for index, stage_id in enumerate(stage_ids)}
    for index, stage in enumerate(stages):
        for prerequisite in stage.prerequisite_stage_ids:
            if prerequisite not in stage_positions:
                raise WarpCurriculumConfigError(
                    f"stage {stage.stage_id!r} references an unknown prerequisite {prerequisite!r}"
                )
            if prerequisite == stage.stage_id or stage_positions[prerequisite] >= index:
                raise WarpCurriculumConfigError(
                    f"stage {stage.stage_id!r} prerequisite {prerequisite!r} must occur earlier in the YAML order"
                )
    capabilities_raw = _mapping(_required(root, "capabilities"), "capabilities")
    _exact_keys(capabilities_raw, "capabilities", {"require_gpu_parity", "allow_external_task_factory"})
    require_gpu_parity = _boolean(_required(capabilities_raw, "require_gpu_parity"), "capabilities.require_gpu_parity")
    allow_external = _boolean(
        _required(capabilities_raw, "allow_external_task_factory"),
        "capabilities.allow_external_task_factory",
    )
    return CurriculumConfig(
        source_path=source,
        name=_string(_required(curriculum_raw, "name"), "curriculum.name"),
        batch_config_path=_resolve_path(source, _required(root, "batch_config"), "batch_config", must_exist=True),
        flat_ppo_config_path=_resolve_path(source, _required(root, "flat_ppo_config"), "flat_ppo_config", must_exist=True),
        observation_size=observation_size,
        action_size=action_size,
        reward_schema=reward_schema,
        action_semantics=action_semantics,
        stages=stages,
        gpu_task=_load_gpu_task_settings(_mapping(_required(root, "gpu_task"), "gpu_task")),
        ppo=_load_ppo(_mapping(_required(root, "ppo"), "ppo")),
        output=_load_artifacts(_mapping(_required(root, "output"), "output"), source, "output"),
        smoke=_load_smoke(_mapping(_required(root, "smoke"), "smoke"), source),
        require_gpu_parity=require_gpu_parity,
        allow_external_task_factory=allow_external,
        source_digest=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
    )


def _coerce_capability(value: Any) -> GpuStageCapability:
    if isinstance(value, GpuStageCapability):
        return value
    raw = _mapping(value, "GPU capability")
    required = {"backend", "terrain", "steps", "jump", "domain_randomization", "speed_command", "yaw_command"}
    missing = sorted(required - set(raw))
    if missing:
        raise WarpCurriculumConfigError(f"GPU capability entry missing keys: {missing}")
    return GpuStageCapability(
        backend=_string(raw["backend"], "capability.backend"),
        terrain=_boolean(raw["terrain"], "capability.terrain"),
        steps=_boolean(raw["steps"], "capability.steps"),
        jump=_boolean(raw["jump"], "capability.jump"),
        domain_randomization=_boolean(raw["domain_randomization"], "capability.domain_randomization"),
        speed_command=_boolean(raw["speed_command"], "capability.speed_command"),
        yaw_command=_boolean(raw["yaw_command"], "capability.yaw_command"),
        observation_size=int(raw.get("observation_size", OBSERVATION_SIZE)),
        action_size=int(raw.get("action_size", ACTION_SIZE)),
        reward_schema=str(raw.get("reward_schema", REWARD_SCHEMA)),
        runtime_gate_required=_boolean(
            raw.get("conditional_runtime_gate", False),
            "capability.conditional_runtime_gate",
        ),
    )


def gpu_stage_capability(stage_id: str, *, allow_external: bool = True) -> GpuStageCapability | None:
    """Discover a declared capability without allocating a simulator."""

    if allow_external:
        for module_name in (
            "warp_curriculum_task",
            "rmuc_curriculum_warp",
            "official_course_warp",
            "official_curriculum_warp",
            "curriculum_warp",
            "warp_task",
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            published = getattr(module, "GPU_CURRICULUM_CAPABILITIES", None)
            if isinstance(published, Mapping) and stage_id in published:
                return _coerce_capability(published[stage_id])
    return GPU_CURRICULUM_CAPABILITIES.get(stage_id)


def validate_stage_capability(config: CurriculumConfig, stage: StageConfig) -> GpuStageCapability:
    """Fail closed when a stage lacks complete GPU parity evidence."""

    from terrain_curriculum import load_terrain_curriculum, validate_scene_contract

    terrain_curriculum = load_terrain_curriculum(stage.terrain_curriculum_path)
    terrain_curriculum.stage(stage.terrain_stage_id)
    if stage.scene_variant == "official_warp_compat":
        from build_official_standard_ground import validate_official_warp_scene

        validate_official_warp_scene(stage.xml_path)
    else:
        validate_scene_contract(
            terrain_curriculum,
            stage.xml_path,
            curriculum_path=stage.terrain_curriculum_path,
        )
    capability = gpu_stage_capability(stage.stage_id, allow_external=config.allow_external_task_factory)
    if capability is None:
        requested = ", ".join(stage.feature_names) or "none"
        raise WarpCurriculumConfigError(
            f"stage {stage.stage_id!r} requests [{requested}] but no GPU parity capability is declared; "
            "official grades/steps/jumps remain blocked until a GPU task backend publishes it"
        )
    if config.require_gpu_parity and not stage.requires_gpu_parity:
        raise WarpCurriculumConfigError(f"stage {stage.stage_id!r} must set requires_gpu_parity=true")
    if capability.observation_size != config.observation_size or capability.action_size != config.action_size:
        raise WarpCurriculumConfigError(f"stage {stage.stage_id!r} capability has incompatible observation/action shape")
    expected_reward_schema = stage.reward_schema or config.reward_schema
    if capability.reward_schema != expected_reward_schema:
        raise WarpCurriculumConfigError(f"stage {stage.stage_id!r} capability has incompatible reward schema")
    checks = (
        (stage.terrain_enabled, capability.terrain, "terrain"),
        (stage.steps_enabled, capability.steps, "steps"),
        (stage.jump_enabled, capability.jump, "jump"),
        (stage.domain_randomization_enabled, capability.domain_randomization, "domain randomization"),
        (abs(stage.command_speed_mps) > 1.0e-12, capability.speed_command, "speed command"),
        (abs(stage.command_yaw_rate_rad_s) > 1.0e-12, capability.yaw_command, "yaw command"),
    )
    unsupported = [label for requested, enabled, label in checks if requested and not enabled]
    if unsupported:
        raise WarpCurriculumConfigError(
            f"stage {stage.stage_id!r} has no GPU parity for: {', '.join(unsupported)}"
        )
    if stage.controller_backend != capability.backend:
        raise WarpCurriculumConfigError(
            f"stage {stage.stage_id!r} controller backend {stage.controller_backend!r} does not match "
            f"declared GPU backend {capability.backend!r}"
        )
    return capability


def _validate_course_certificate(
    metadata: Mapping[str, Any], *, config: CurriculumConfig, source_stage_id: str
) -> None:
    """Reject candidate, smoke, or unsigned CUDA-stage checkpoints as warm starts."""

    if metadata.get("artifact_status") != "certified":
        raise WarpCurriculumConfigError("GPU curriculum warm start is not a certified checkpoint")
    certificate = metadata.get("course_certificate")
    evaluation = metadata.get("course_evaluation")
    if not isinstance(certificate, Mapping) or not isinstance(evaluation, Mapping):
        raise WarpCurriculumConfigError("GPU curriculum warm start is missing stage-promotion evidence")
    if (
        certificate.get("certificate_schema") != 1
        or certificate.get("passed") is not True
        or certificate.get("stage_id") != source_stage_id
        or evaluation.get("passed") is not True
        or evaluation.get("stage_id") != source_stage_id
    ):
        raise WarpCurriculumConfigError("GPU curriculum warm start has an invalid stage-promotion certificate")
    try:
        source_stage = config.stage(source_stage_id)
    except WarpCurriculumConfigError as error:
        raise WarpCurriculumConfigError("GPU curriculum warm start names an unknown certificate stage") from error
    if certificate.get("curriculum_config_sha256") != config.source_digest:
        raise WarpCurriculumConfigError("GPU curriculum warm start certificate uses a different curriculum YAML")
    expected_adapter_digest = (
        None
        if source_stage.adapter_config_path is None
        else hashlib.sha256(source_stage.adapter_config_path.read_bytes()).hexdigest()
    )
    if certificate.get("adapter_config_sha256") != expected_adapter_digest:
        raise WarpCurriculumConfigError("GPU curriculum warm start certificate adapter provenance mismatches YAML")


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any], *, config: CurriculumConfig, stage: StageConfig
) -> None:
    """Validate a residual warm-start against the current interface contract."""

    if not isinstance(metadata, Mapping) or metadata.get("algorithm") != "ppo":
        raise WarpCurriculumConfigError("residual checkpoint must be a PPO metadata mapping")
    if metadata.get("observation_size") != config.observation_size or metadata.get("action_size") != config.action_size:
        raise WarpCurriculumConfigError("residual checkpoint observation/action shape is incompatible")
    backend = metadata.get("checkpoint_backend")
    cpu_transfer = backend is None
    if cpu_transfer:
        if not config.gpu_task.allow_cpu_residual_warm_start:
            raise WarpCurriculumConfigError("CPU residual warm starts are disabled by gpu_task configuration")
        from env import LOCOMOTION_COMMAND_SCHEMA, RESIDUAL_AUTHORITY_SCHEMA
        from train_ppo import CHECKPOINT_FORMAT_VERSION

        if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise WarpCurriculumConfigError("CPU residual checkpoint format version is incompatible")
        if metadata.get("action_semantics") != "residual":
            raise WarpCurriculumConfigError("CPU residual checkpoint must use residual actions")
        task_config = metadata.get("task_config")
        if not isinstance(task_config, Mapping):
            raise WarpCurriculumConfigError("CPU residual checkpoint is missing task_config")
        if task_config.get("locomotion_command_schema") != LOCOMOTION_COMMAND_SCHEMA:
            raise WarpCurriculumConfigError("CPU residual checkpoint locomotion-command schema is incompatible")
        if task_config.get("residual_authority_schema") != RESIDUAL_AUTHORITY_SCHEMA:
            raise WarpCurriculumConfigError("CPU residual checkpoint authority schema is incompatible")
    else:
        if backend not in {CURRICULUM_BACKEND, FLAT_BACKEND}:
            raise WarpCurriculumConfigError("residual checkpoint backend is not a supported GPU backend")
        expected_format = CURRICULUM_CHECKPOINT_FORMAT if backend == CURRICULUM_BACKEND else 2
        if metadata.get("format_version") != expected_format:
            raise WarpCurriculumConfigError("residual checkpoint format version is incompatible with its backend")
        if metadata.get("action_semantics") not in {ACTION_SEMANTICS, "seven_dimensional_fixed_gain_residual"}:
            raise WarpCurriculumConfigError("residual checkpoint action semantics are incompatible")
    source_stage_id = metadata.get("stage_id")
    explicit_predecessor = bool(
        not cpu_transfer
        and backend == CURRICULUM_BACKEND
        and isinstance(source_stage_id, str)
        and source_stage_id in stage.prerequisite_stage_ids
    )
    exact_gpu_stage = bool(
        not cpu_transfer and backend == CURRICULUM_BACKEND and source_stage_id == stage.stage_id
    )
    if not cpu_transfer and backend == CURRICULUM_BACKEND:
        if not (explicit_predecessor or exact_gpu_stage):
            raise WarpCurriculumConfigError(
                "GPU curriculum warm start must be the selected stage or an explicitly declared prerequisite"
            )
        _validate_course_certificate(metadata, config=config, source_stage_id=str(source_stage_id))
    expected_reward_schema = stage.reward_schema or config.reward_schema
    saved_reward = metadata.get("reward_schema")
    if cpu_transfer:
        # CPU checkpoints use the established command-conditioned reward
        # schema. This is a deliberate cross-backend initialization only;
        # the resulting GPU checkpoint records the transfer rather than
        # claiming reward-equivalent replay.
        if not isinstance(saved_reward, str) or not saved_reward:
            task_config = metadata.get("task_config")
            saved_reward = task_config.get("reward_schema") if isinstance(task_config, Mapping) else None
        if not isinstance(saved_reward, str) or not saved_reward:
            raise WarpCurriculumConfigError("CPU residual checkpoint is missing reward provenance")
    elif saved_reward is None:
        scope = metadata.get("task_scope")
        # The v2 flat checkpoint predates the explicit reward_schema field.
        # It is accepted only when its immutable scope proves it used this
        # exact flat task; arbitrary legacy checkpoints remain rejected.
        if (
            backend == FLAT_BACKEND
            and expected_reward_schema == LEGACY_FLAT_REWARD_SCHEMA
            and isinstance(scope, Mapping)
            and scope.get("task_mode") == "flat_walking_only"
            and scope.get("terrain_enabled") is False
            and scope.get("jump_enabled") is False
            and scope.get("domain_randomization_enabled") is False
            and scope.get("flat_terrain_features_zeroed") is True
            and scope.get("jump_features_zeroed") is True
        ):
            saved_reward = LEGACY_FLAT_REWARD_SCHEMA
        else:
            raise WarpCurriculumConfigError("residual checkpoint is missing an auditable reward schema")
    if not cpu_transfer and saved_reward != expected_reward_schema and not explicit_predecessor:
        raise WarpCurriculumConfigError("residual checkpoint reward schema is incompatible")
    if not cpu_transfer and backend == CURRICULUM_BACKEND:
        scope = metadata.get("stage_scope")
        if not isinstance(scope, Mapping):
            raise WarpCurriculumConfigError("GPU curriculum checkpoint is missing auditable stage_scope provenance")
        if not explicit_predecessor:
            for name, expected in (
                ("task_mode", stage.task_mode),
                ("xml_path", str(stage.xml_path)),
                ("scene_variant", stage.scene_variant),
                ("terrain_curriculum_path", str(stage.terrain_curriculum_path)),
                ("terrain_stage_id", stage.terrain_stage_id),
                ("controller_backend", stage.controller_backend),
            ):
                if scope.get(name) != expected:
                    raise WarpCurriculumConfigError(f"checkpoint stage_scope.{name} does not match the selected curriculum stage")
        experiment = metadata.get("experiment_config")
        if not isinstance(experiment, Mapping):
            raise WarpCurriculumConfigError("GPU curriculum checkpoint is missing experiment_config provenance")
        if explicit_predecessor:
            for field in (
                "xml_sha256",
                "terrain_curriculum_sha256",
                "flat_ppo_config_sha256",
            ):
                value = experiment.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise WarpCurriculumConfigError("prerequisite checkpoint provenance is incomplete")
            source_stage = config.stage(str(source_stage_id))
            adapter_digest = experiment.get("adapter_config_sha256")
            if source_stage.adapter_config_path is None:
                if adapter_digest is not None:
                    raise WarpCurriculumConfigError("flat prerequisite checkpoint must not claim an adapter digest")
            elif not isinstance(adapter_digest, str) or len(adapter_digest) != 64:
                raise WarpCurriculumConfigError("prerequisite checkpoint adapter provenance is incomplete")
        else:
            digest_fields: tuple[tuple[str, Path], ...] = (
                ("xml_sha256", stage.xml_path),
                ("terrain_curriculum_sha256", stage.terrain_curriculum_path),
                ("flat_ppo_config_sha256", config.flat_ppo_config_path),
            )
            if stage.adapter_config_path is not None:
                digest_fields += (("adapter_config_sha256", stage.adapter_config_path),)
            for field, path in digest_fields:
                try:
                    expected_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                except OSError as error:
                    raise WarpCurriculumConfigError(f"unable to hash curriculum provenance file {path}") from error
                if experiment.get(field) != expected_digest:
                    raise WarpCurriculumConfigError(f"GPU curriculum checkpoint provenance digest mismatch for {field}")
    mask = metadata.get("policy_action_mask")
    if not cpu_transfer:
        if not isinstance(mask, (list, tuple)) or len(mask) != config.action_size:
            raise WarpCurriculumConfigError("residual checkpoint policy_action_mask must contain seven values")
        try:
            mask_values = tuple(float(value) for value in mask)
        except (TypeError, ValueError) as error:
            raise WarpCurriculumConfigError("residual checkpoint policy_action_mask is not numeric") from error
        if (
            any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in mask_values)
            or mask_values != stage.residual_action_mask
        ):
            raise WarpCurriculumConfigError("residual checkpoint action authority does not match the selected stage")
    if not isinstance(metadata.get("model_state_dict"), Mapping) or not metadata["model_state_dict"]:
        raise WarpCurriculumConfigError("residual checkpoint is missing model_state_dict")


def load_residual_checkpoint(
    checkpoint_path: str | Path, *, config: CurriculumConfig, stage: StageConfig
) -> ResidualCheckpoint:
    """Load and validate an optional residual policy warm start."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise WarpCurriculumConfigError(f"residual checkpoint does not exist: {path}")
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - torch backend errors vary
        raise WarpCurriculumConfigError(f"unable to load residual checkpoint {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise WarpCurriculumConfigError("residual checkpoint root must be a mapping")
    validate_checkpoint_metadata(payload, config=config, stage=stage)
    return ResidualCheckpoint(
        path=path,
        state_dict=dict(payload["model_state_dict"]),
        metadata=payload,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        transfer_kind="cpu_residual_transfer" if payload.get("checkpoint_backend") is None else "gpu_checkpoint",
    )


def build_checkpoint_metadata(
    *, config: CurriculumConfig, stage: StageConfig, timesteps: int, update_index: int,
    source_checkpoint: ResidualCheckpoint | None, batch: Any, task: Any, smoke: bool
) -> dict[str, Any]:
    """Create provenance metadata for a curriculum checkpoint."""

    reward_schema = stage.reward_schema or config.reward_schema
    terrain_reward = bool(
        getattr(getattr(task, "config", None), "terrain_compensated_leg_reward", None)
        and getattr(task.config.terrain_compensated_leg_reward, "enabled", False)
    )
    if terrain_reward:
        if not reward_schema.endswith("_v2"):
            raise WarpCurriculumConfigError(
                "terrain-compensated task cannot be checkpointed under a pre-v2 reward schema"
            )
        # The task configuration, rather than whether a stage overrides the
        # global schema, is the source of truth.  This covers the flat and
        # flat-DR stages as well as terrain route adapters.
        reward_terms = (
            "speed_tracking", "leg_tracking", "terrain_attitude_tracking", "contact", "yaw_tracking",
            "energy_cost", "residual_cost", "yaw_rate_cost",
            "terrain_compensated_leg_difference_cost", "unsafe_penalty",
        )
        if stage.adapter_config_path is not None and stage.stage_id.startswith("official_"):
            reward_terms += ("official_route", "direct_jump")
        elif stage.adapter_config_path is not None:
            reward_terms += ("rmuc_route", "rmuc_direct_jump")
    else:
        reward_terms = (
            "speed_tracking", "leg_tracking", "attitude_tracking", "contact", "yaw_tracking",
            "energy_cost", "residual_cost", "yaw_rate_cost", "leg_symmetry_cost", "unsafe_penalty",
        )
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "format_version": CURRICULUM_CHECKPOINT_FORMAT,
        "checkpoint_backend": CURRICULUM_BACKEND,
        "algorithm": "ppo",
        "action_semantics": config.action_semantics,
        "observation_size": config.observation_size,
        "action_size": config.action_size,
        "reward_schema": reward_schema,
        "stage_id": stage.stage_id,
        "timesteps": int(timesteps),
        "update_index": int(update_index),
        "smoke": bool(smoke),
        "policy_action_mask": list(stage.residual_action_mask),
        "stage_scope": {
            "task_mode": stage.task_mode,
            "xml_path": str(stage.xml_path),
            "scene_variant": stage.scene_variant,
            "terrain_curriculum_path": str(stage.terrain_curriculum_path),
            "terrain_stage_id": stage.terrain_stage_id,
            "adapter_config_path": (
                None if stage.adapter_config_path is None else str(stage.adapter_config_path)
            ),
            "reward_schema": reward_schema,
            "terrain_compensated_leg_reward_enabled": terrain_reward,
            "controller_backend": stage.controller_backend,
            "terrain_enabled": stage.terrain_enabled,
            "steps_enabled": stage.steps_enabled,
            "jump_enabled": stage.jump_enabled,
            "domain_randomization_enabled": stage.domain_randomization_enabled,
            "command_speed_mps": stage.command_speed_mps,
            "command_yaw_rate_rad_s": stage.command_yaw_rate_rad_s,
            "prerequisite_stage_ids": list(stage.prerequisite_stage_ids),
        },
        "experiment_config": {
            "source_path": str(config.source_path),
            "source_digest": config.source_digest,
            "batch_config_path": str(config.batch_config_path),
            "num_worlds": int(batch.num_worlds),
            "device": str(batch.device),
            "xml_sha256": digest(stage.xml_path),
            "terrain_curriculum_sha256": digest(stage.terrain_curriculum_path),
            "flat_ppo_config_sha256": digest(config.flat_ppo_config_path),
            "adapter_config_sha256": (
                None if stage.adapter_config_path is None else digest(stage.adapter_config_path)
            ),
        },
        "gpu_task_config": asdict(config.gpu_task),
        "batch_domain_randomization": asdict(batch.config.domain_randomization),
        "reward_terms": reward_terms,
        "residual_warm_start": (
            None if source_checkpoint is None else {
                "path": str(source_checkpoint.path),
                "sha256": source_checkpoint.sha256,
                "transfer_kind": source_checkpoint.transfer_kind,
                "source_reward_schema": (
                    source_checkpoint.metadata.get("reward_schema")
                    if source_checkpoint.transfer_kind == "gpu_checkpoint"
                    else (
                        source_checkpoint.metadata.get("task_config", {}).get("reward_schema")
                        if isinstance(source_checkpoint.metadata.get("task_config"), Mapping)
                        else None
                    )
                ),
            }
        ),
    }


def _load_flat_manifest(config: CurriculumConfig, stage: StageConfig) -> tuple[Any, Any]:
    """Load the proven flat manifests and verify their scene before GPU setup."""

    from train_warp_ppo import load_flat_ppo_training_config
    from warp_env import load_warp_batch_config

    if stage.stage_id != "rmuc_flat":
        raise WarpCurriculumConfigError(
            f"stage {stage.stage_id!r} has no built-in runner; an external GPU task factory is required"
        )
    flat = load_flat_ppo_training_config(config.flat_ppo_config_path)
    batch_config = load_warp_batch_config(flat.batch_config_path)
    if batch_config.xml_path != stage.xml_path:
        raise WarpCurriculumConfigError(
            f"stage {stage.stage_id!r} XML does not match its flat batch config"
        )
    if stage.controller_backend != FLAT_CONTROLLER_BACKEND or stage.feature_names:
        raise WarpCurriculumConfigError(
            "built-in rmuc_flat has only fixed-gain, zero-command, no-DR parity"
        )
    return flat, batch_config


def _external_stage_bundle(config: CurriculumConfig, stage: StageConfig) -> Any:
    """Resolve a parity-tested external GPU task bundle without fallback."""

    if not config.allow_external_task_factory:
        raise WarpCurriculumConfigError(
            f"stage {stage.stage_id!r} requires an external GPU task factory, but it is disabled"
        )
    for module_name in (
        "warp_curriculum_task",
        "rmuc_curriculum_warp",
        "official_course_warp",
        "official_curriculum_warp",
        "curriculum_warp",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        published = getattr(module, "GPU_CURRICULUM_CAPABILITIES", None)
        if isinstance(published, Mapping) and stage.stage_id not in published:
            continue
        factory = getattr(module, "build_curriculum_stage", None)
        if callable(factory):
            try:
                signature = inspect.signature(factory)
            except (TypeError, ValueError) as error:
                raise WarpCurriculumConfigError(
                    f"unable to inspect GPU task factory {module_name}.build_curriculum_stage"
                ) from error
            parameters = tuple(signature.parameters.values())
            keyword_safe = all(
                parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY,)
                for parameter in parameters
            )
            accepts_keywords = keyword_safe and (
                any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
                or {"stage", "config"}.issubset(signature.parameters)
            )
            if accepts_keywords:
                bundle = factory(stage=stage, config=config)
            else:
                bundle = factory(stage, config)
            if bundle is None:
                raise WarpCurriculumConfigError(
                    f"GPU task factory {module_name}.build_curriculum_stage returned None"
                )
            return bundle
    raise WarpCurriculumConfigError(
        f"stage {stage.stage_id!r} has no build_curriculum_stage factory; "
        "publish a parity-tested warp_curriculum_task.py backend first"
    )


def _bundle_value(bundle: Any, name: str, *, required: bool = True) -> Any:
    value = bundle.get(name) if isinstance(bundle, Mapping) else getattr(bundle, name, None)
    if required and value is None:
        raise WarpCurriculumConfigError(f"GPU curriculum bundle must provide {name!r}")
    return value


def _gate_float(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarpCurriculumConfigError(f"conditional GPU gate field {name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise WarpCurriculumConfigError(f"conditional GPU gate field {name!r} must be finite")
    return result


def _gate_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WarpCurriculumConfigError(f"conditional GPU gate field {name!r} must be an integer")
    if minimum is not None and value < minimum:
        raise WarpCurriculumConfigError(f"conditional GPU gate field {name!r} is below its allowed range")
    return int(value)


def _validate_gate_pass(
    value: Any,
    *,
    name: str,
    stage: StageConfig,
    config: CurriculumConfig,
    expected_worlds: int,
    expected_domain_randomization_active: bool,
) -> Mapping[str, Any]:
    """Validate one immutable, zero-residual CUDA gate pass."""

    if not isinstance(value, Mapping):
        raise WarpCurriculumConfigError(f"conditional GPU gate is missing mapping evidence for {name!r}")
    required = (
        "passed", "requested_duration_seconds", "simulated_duration_seconds", "policy_steps",
        "num_worlds", "terminated_worlds", "overflowed_worlds", "estopped_worlds", "finite_state",
        "finite_reward", "finite_reward_terms",
        "zero_residual", "domain_randomization_active", "physical_parameter_randomization",
        "terrain_geometry_randomization", "sensor_noise_std", "control_delay_steps",
        "minimum_progress_m", "speed_mae_mps", "unsafe_rate", "first_fault_step",
        "first_fault_reason_code", "obstacle_guard_verified",
    )
    for field in required:
        if field not in value:
            raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} is missing {field!r}")
    if value["passed"] is not True or value["zero_residual"] is not True:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} did not pass its zero-residual physical test")
    worlds = _gate_int(value["num_worlds"], f"{name}.num_worlds", minimum=1)
    if worlds != expected_worlds:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} world count changed between stress passes")
    counts = tuple(
        _gate_int(value[field], f"{name}.{field}", minimum=0)
        for field in ("terminated_worlds", "overflowed_worlds", "estopped_worlds")
    )
    if any(count > worlds for count in counts) or any(counts):
        raise WarpCurriculumConfigError(
            f"conditional GPU gate {name!r} must have zero terminated, overflowed, and estopped worlds"
        )
    if value["finite_state"] is not True:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} must prove finite_state=true")
    if value["finite_reward"] is not True:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} must prove finite_reward=true")
    if value["finite_reward_terms"] is not True:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} must prove finite_reward_terms=true")
    requested = _gate_float(value["requested_duration_seconds"], f"{name}.requested_duration_seconds", nonnegative=True)
    simulated = _gate_float(value["simulated_duration_seconds"], f"{name}.simulated_duration_seconds", nonnegative=True)
    steps = _gate_int(value["policy_steps"], f"{name}.policy_steps", minimum=1)
    if requested < config.gpu_task.stability_gate_seconds - 1.0e-9:
        raise WarpCurriculumConfigError("conditional GPU gate did not run for the YAML-required safety duration")
    if simulated + 1.0e-9 < requested:
        raise WarpCurriculumConfigError("conditional GPU gate simulated less than its declared safety duration")
    if simulated <= 0.0 or simulated / float(steps) <= 0.0:
        raise WarpCurriculumConfigError("conditional GPU gate has an invalid physics action timestep")
    if value["domain_randomization_active"] is not expected_domain_randomization_active:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} has the wrong DR activation state")
    if value["physical_parameter_randomization"] is not expected_domain_randomization_active:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} did not exercise the required physical DR")
    if value["terrain_geometry_randomization"] is not False:
        raise WarpCurriculumConfigError("conditional GPU gate may not randomize terrain geometry or collision topology")
    sensor_noise = _gate_float(value["sensor_noise_std"], f"{name}.sensor_noise_std", nonnegative=True)
    if not math.isclose(sensor_noise, config.gpu_task.sensor_noise_std, rel_tol=0.0, abs_tol=1.0e-8):
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} sensor-noise evidence mismatches YAML")
    if _gate_int(value["control_delay_steps"], f"{name}.control_delay_steps", minimum=0) != config.gpu_task.control_delay_steps:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} action-delay evidence mismatches YAML")
    for field in ("minimum_progress_m", "speed_mae_mps", "unsafe_rate"):
        _gate_float(value[field], f"{name}.{field}", nonnegative=True)
    _gate_int(value["first_fault_step"], f"{name}.first_fault_step", minimum=-1)
    _gate_int(value["first_fault_reason_code"], f"{name}.first_fault_reason_code", minimum=0)
    if value["obstacle_guard_verified"] is not True:
        raise WarpCurriculumConfigError(f"conditional GPU gate {name!r} did not verify its analytic obstacle guard")
    if stage.jump_enabled:
        jump_fields = (
            "jump_supervisor_verified", "jump_triggered_worlds", "landing_confirmed_worlds",
            "jump_minimum_peak_worlds", "landing_kinematics_worlds", "minimum_flight_seconds",
            "landing_preload_seconds",
        )
        for field in jump_fields:
            if field not in value:
                raise WarpCurriculumConfigError(f"jump GPU gate {name!r} is missing {field!r}")
        if value["jump_supervisor_verified"] is not True:
            raise WarpCurriculumConfigError(f"jump GPU gate {name!r} did not verify the direct-jump supervisor")
        for field in (
            "jump_triggered_worlds", "landing_confirmed_worlds", "jump_minimum_peak_worlds",
            "landing_kinematics_worlds",
        ):
            if _gate_int(value[field], f"{name}.{field}", minimum=0) != worlds:
                raise WarpCurriculumConfigError(f"jump GPU gate {name!r} must prove {field} for every world")
        minimum_flight = _gate_float(value["minimum_flight_seconds"], f"{name}.minimum_flight_seconds", nonnegative=True)
        if minimum_flight + 1.0e-9 < simulated / float(steps):
            raise WarpCurriculumConfigError(f"jump GPU gate {name!r} did not prove a physical flight interval")
        preload = _gate_float(value["landing_preload_seconds"], f"{name}.landing_preload_seconds", nonnegative=True)
        if preload < 0.050:
            raise WarpCurriculumConfigError("jump GPU gate must reduce landing torque at least 50 ms before touchdown")
    return value


def _validate_runtime_gate_report(
    report: Any, *, config: CurriculumConfig, stage: StageConfig, capability: GpuStageCapability
) -> Mapping[str, Any]:
    """Reject any GPU gate without versioned, YAML-bound dual-pass evidence."""

    del capability
    if not isinstance(report, Mapping):
        raise WarpCurriculumConfigError("conditional GPU gate must return a mapping report")
    if report.get("stage_id") != stage.stage_id:
        raise WarpCurriculumConfigError("conditional GPU gate report stage_id does not match the selected stage")
    if report.get("conditional_capability") is not True:
        raise WarpCurriculumConfigError("conditional GPU gate report must set conditional_capability=true")
    if report.get("gate_evidence_schema") != GATE_EVIDENCE_SCHEMA:
        raise WarpCurriculumConfigError("conditional GPU gate report has an incompatible evidence schema")
    if stage.adapter_config_path is None:
        raise WarpCurriculumConfigError("conditional GPU stage lacks an auditable adapter configuration")
    try:
        expected_digest = hashlib.sha256(stage.adapter_config_path.read_bytes()).hexdigest()
    except OSError as error:
        raise WarpCurriculumConfigError("unable to hash conditional GPU gate configuration") from error
    for field in ("gate_config_sha256", "threshold_config_sha256"):
        if report.get(field) != expected_digest:
            raise WarpCurriculumConfigError(f"conditional GPU gate {field} does not match the selected YAML")
    if report.get("passed") is not True:
        raise WarpCurriculumConfigError("conditional GPU gate report must set passed=true")
    if report.get("zero_residual") is not True:
        raise WarpCurriculumConfigError("conditional GPU gate must be a zero-residual baseline test")
    if report.get("domain_randomization_enabled") is not True or not stage.domain_randomization_enabled:
        raise WarpCurriculumConfigError("conditional GPU stage must require reset-boundary domain randomization")
    worlds = _gate_int(report.get("num_worlds"), "num_worlds", minimum=1)
    deterministic = _validate_gate_pass(
        report.get("deterministic_baseline"),
        name="deterministic_baseline",
        stage=stage,
        config=config,
        expected_worlds=worlds,
        expected_domain_randomization_active=False,
    )
    domain_randomized = _validate_gate_pass(
        report.get("domain_randomization_stress"),
        name="domain_randomization_stress",
        stage=stage,
        config=config,
        expected_worlds=worlds,
        expected_domain_randomization_active=True,
    )
    if report.get("deterministic_baseline_passed") is not True or report.get("domain_randomization_stress_passed") is not True:
        raise WarpCurriculumConfigError("conditional GPU gate must pass both deterministic and DR physical stress runs")
    for field in (
        "requested_duration_seconds", "simulated_duration_seconds", "policy_steps", "terminated_worlds",
        "overflowed_worlds", "estopped_worlds", "finite_state", "finite_reward", "finite_reward_terms",
        "minimum_progress_m", "speed_mae_mps",
        "unsafe_rate", "first_fault_step", "first_fault_reason_code", "obstacle_guard_verified",
    ):
        if field not in report:
            raise WarpCurriculumConfigError(f"conditional GPU gate report is missing {field!r}")
    # The top-level summary must be the DR stress result, never a stale
    # deterministic summary that masks a randomized-world failure.
    for field in (
        "requested_duration_seconds", "simulated_duration_seconds", "policy_steps", "terminated_worlds",
        "overflowed_worlds", "estopped_worlds", "finite_state", "finite_reward", "finite_reward_terms",
        "minimum_progress_m", "speed_mae_mps",
        "unsafe_rate", "first_fault_step", "first_fault_reason_code", "obstacle_guard_verified",
    ):
        if report[field] != domain_randomized[field]:
            raise WarpCurriculumConfigError("conditional GPU gate top-level summary must match its DR stress evidence")
    if "doghole" in stage.task_mode and (
        deterministic["obstacle_guard_verified"] is not True or domain_randomized["obstacle_guard_verified"] is not True
    ):
        raise WarpCurriculumConfigError("doghole GPU gate must verify the obstacle collision guard in both passes")
    return report


def _atomic_save(payload: Mapping[str, Any], path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_metrics(path: Path, row: Mapping[str, Any]) -> None:
    fields = (
        "update", "timesteps", "rollout_world_steps", "mean_reward", "completed_episodes",
        "mean_completed_return", "partial_mean_return", "policy_loss", "value_loss", "entropy",
        "aggregate_world_steps_per_second",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="ascii") as handle:
            if tuple(next(csv.reader(handle), [])) != fields:
                raise WarpCurriculumConfigError(f"metrics schema mismatch at {path}")
    with path.open("a", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(dict(row))


def _stage_artifact_path(path: Path, stage: StageConfig) -> Path:
    """Keep external curriculum stages from sharing/mixing flat artifacts."""

    if stage.stage_id == "rmuc_flat":
        return path
    return path.with_name(f"{path.stem}.{stage.stage_id}{path.suffix}")


def _stage_candidate_path(path: Path) -> Path:
    """Keep unverified policy snapshots outside the warm-start namespace."""

    return path.with_name(f"{path.stem}.candidate{path.suffix}")


def _stage_requires_promotion(stage: StageConfig, *, smoke: bool) -> bool:
    """Every formal CUDA stage must be certified before it becomes a baseline."""

    del stage
    return not smoke


def _certify_course_evaluation(
    evaluation: Mapping[str, Any] | None, *, config: CurriculumConfig, stage: StageConfig
) -> Mapping[str, Any]:
    """Turn a passed CUDA route evaluation into checkpoint-bound evidence."""

    if not isinstance(evaluation, Mapping):
        raise WarpCurriculumConfigError("CUDA curriculum stage did not return promotion evidence")
    if evaluation.get("stage_id") != stage.stage_id or evaluation.get("passed") is not True:
        raise WarpCurriculumConfigError("CUDA curriculum stage did not pass its post-training evaluation")
    return {
        "certificate_schema": 1,
        "passed": True,
        "stage_id": stage.stage_id,
        "curriculum_config_sha256": config.source_digest,
        "adapter_config_sha256": (
            None
            if stage.adapter_config_path is None
            else hashlib.sha256(stage.adapter_config_path.read_bytes()).hexdigest()
        ),
        "evaluation_scope": evaluation.get("evaluation_scope", "declared_route"),
    }


def _run_vector_training(
    *, config: CurriculumConfig, stage: StageConfig, batch: Any, task: Any,
    gate_report: Any, source_checkpoint: ResidualCheckpoint | None, smoke: bool,
    close_bundle: Any = None,
) -> Path:
    """Run the common CUDA collector/update loop after a backend gate."""

    from train_ppo import ActorCritic
    from warp_ppo import WarpPPOCollector, update_policy_cuda

    if int(getattr(task, "observation_size", OBSERVATION_SIZE)) != config.observation_size:
        raise WarpCurriculumConfigError("GPU task observation_size does not match curriculum schema")
    if int(getattr(task, "action_size", ACTION_SIZE)) != config.action_size:
        raise WarpCurriculumConfigError("GPU task action_size does not match curriculum schema")
    torch = getattr(batch, "_torch", None)
    if torch is None or not str(getattr(batch, "device", "")).startswith("cuda"):
        raise WarpCurriculumConfigError("curriculum task bundle must be CUDA-backed")
    torch.manual_seed(config.ppo.seed)
    torch.cuda.manual_seed_all(config.ppo.seed)
    policy = ActorCritic(
        config.observation_size, config.action_size,
        hidden_size=config.ppo.hidden_size,
        initial_action_std=config.ppo.initial_action_std,
    ).to(batch.device)
    if source_checkpoint is not None:
        try:
            policy.load_state_dict(source_checkpoint.state_dict, strict=True)
        except RuntimeError as error:
            raise WarpCurriculumConfigError(
                "residual checkpoint model shape does not match the configured PPO network"
            ) from error
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo.learning_rate)
    collector = WarpPPOCollector(task, policy, gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda)
    if smoke:
        rollout_steps, epochs, minibatch_size = config.smoke.rollout_steps, config.smoke.epochs, config.smoke.minibatch_size
        output_path = _stage_artifact_path(config.smoke.checkpoint_path, stage)
        metrics_path = _stage_artifact_path(config.smoke.metrics_path, stage)
        total_timesteps, max_updates = rollout_steps * int(batch.num_worlds), 1
    else:
        rollout_steps, epochs, minibatch_size = config.ppo.rollout_steps, config.ppo.epochs, config.ppo.minibatch_size
        output_path = _stage_artifact_path(config.output.checkpoint_path, stage)
        metrics_path = _stage_artifact_path(config.output.metrics_path, stage)
        total_timesteps, max_updates = config.ppo.total_timesteps, None
    requires_promotion = _stage_requires_promotion(stage, smoke=smoke)
    write_path = _stage_candidate_path(output_path) if requires_promotion else output_path
    if total_timesteps % int(batch.num_worlds):
        raise WarpCurriculumConfigError("total_timesteps must be divisible by the GPU world count")
    observations = task.reset()
    episode_returns = torch.zeros(int(batch.num_worlds), dtype=torch.float32, device=batch.device)
    timesteps = 0
    update_index = 0
    while timesteps < total_timesteps:
        current_steps = min(rollout_steps, (total_timesteps - timesteps) // int(batch.num_worlds))
        if current_steps < 1:
            raise WarpCurriculumConfigError("curriculum PPO reached a fractional vector update")
        started = perf_counter()
        rollout, observations, episode_returns = collector.collect(
            current_steps, observations=observations, episode_returns=episode_returns
        )
        metrics = update_policy_cuda(
            policy, optimizer, rollout, epochs=epochs, minibatch_size=minibatch_size,
            clip_ratio=config.ppo.clip_ratio, value_coefficient=config.ppo.value_coefficient,
            entropy_coefficient=config.ppo.entropy_coefficient,
            max_gradient_norm=config.ppo.max_gradient_norm,
        )
        torch.cuda.synchronize(batch.device)
        elapsed = max(perf_counter() - started, 1.0e-9)
        timesteps += current_steps * int(batch.num_worlds)
        update_index += 1
        done = rollout.continuation_masks.eq(0.0)
        completed = int(done.sum().detach().cpu().item())
        mean_completed = (
            float(rollout.completed_episode_returns[done].mean().detach().cpu().item())
            if completed else float("nan")
        )
        payload = {
            **build_checkpoint_metadata(
                config=config, stage=stage, timesteps=timesteps, update_index=update_index,
                source_checkpoint=source_checkpoint, batch=batch, task=task, smoke=smoke,
            ),
            "model_state_dict": {key: value.detach().clone() for key, value in policy.state_dict().items()},
            "actor_state_dict": {key: value.detach().clone() for key, value in policy.actor.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "gate_report": gate_report,
            "artifact_status": "candidate" if requires_promotion else ("smoke" if smoke else "uncertified"),
            "ppo_config": asdict(config.ppo),
        }
        _append_metrics(metrics_path, {
            "update": update_index, "timesteps": timesteps,
            "rollout_world_steps": current_steps * int(batch.num_worlds),
            "mean_reward": float(rollout.rewards.mean().detach().cpu().item()),
            "completed_episodes": completed,
            "mean_completed_return": mean_completed,
            "partial_mean_return": float(episode_returns.mean().detach().cpu().item()),
            "policy_loss": metrics["policy_loss"], "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "aggregate_world_steps_per_second": current_steps * int(batch.num_worlds) / elapsed,
        })
        if smoke or timesteps == total_timesteps or update_index % config.output.checkpoint_interval_updates == 0:
            _atomic_save(payload, write_path, torch)
        if max_updates is not None and update_index >= max_updates:
            break
    if smoke and update_index != 1:
        raise RuntimeError("curriculum PPO smoke must execute exactly one update")
    evaluation = _run_post_training_evaluation(
        task=task,
        batch=batch,
        policy=policy,
        stage=stage,
        config=config,
        smoke=smoke,
    )
    if requires_promotion:
        certificate = _certify_course_evaluation(evaluation, config=config, stage=stage)
        payload = {
            **build_checkpoint_metadata(
                config=config, stage=stage, timesteps=timesteps, update_index=update_index,
                source_checkpoint=source_checkpoint, batch=batch, task=task, smoke=smoke,
            ),
            "model_state_dict": {key: value.detach().clone() for key, value in policy.state_dict().items()},
            "actor_state_dict": {key: value.detach().clone() for key, value in policy.actor.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "gate_report": gate_report,
            "course_evaluation": evaluation,
            "course_certificate": certificate,
            "artifact_status": "certified",
            "ppo_config": asdict(config.ppo),
        }
        _atomic_save(payload, output_path, torch)
    if callable(close_bundle):
        close_bundle()
    return output_path


def _evaluate_flat_policy_stage(
    *, task: Any, batch: Any, policy: Any, stage: StageConfig, config: CurriculumConfig
) -> Mapping[str, Any]:
    """Run a deterministic post-PPO CUDA safety evaluation for flat stages."""

    torch = getattr(batch, "_torch", None)
    if torch is None or not str(getattr(batch, "device", "")).startswith("cuda"):
        raise WarpCurriculumConfigError("flat post-policy evaluation requires a CUDA task bundle")
    actor = getattr(policy, "actor", None)
    if actor is None:
        raise WarpCurriculumConfigError("PPO policy does not expose an actor for flat CUDA evaluation")
    action_dt = float(getattr(task, "_time_step", 0.0))
    if not math.isfinite(action_dt) or action_dt <= 0.0:
        raise WarpCurriculumConfigError("flat post-policy evaluation has an invalid action timestep")
    duration = float(config.gpu_task.stability_gate_seconds)
    if duration > float(task.config.episode_seconds) + 1.0e-9:
        raise WarpCurriculumConfigError("flat post-policy evaluation exceeds the task episode horizon")
    steps = max(1, int(math.ceil(duration / action_dt)))
    terminated = torch.zeros(batch.num_worlds, dtype=torch.bool, device=batch.device)
    overflowed = torch.zeros_like(terminated)
    estopped = torch.zeros_like(terminated)
    finite_reward = torch.ones((), dtype=torch.bool, device=batch.device)
    finite_reward_terms = torch.ones((), dtype=torch.bool, device=batch.device)
    finite_values = torch.empty_like(terminated)
    finite_step = torch.empty((), dtype=torch.bool, device=batch.device)
    finite_terms_step = torch.empty((), dtype=torch.bool, device=batch.device)
    task.reset()
    policy.eval()
    try:
        for _ in range(steps):
            with torch.no_grad():
                output = actor(task.observe())
                action = output[0] if isinstance(output, tuple) else output
                if not isinstance(action, torch.Tensor) or action.shape != (batch.num_worlds, ACTION_SIZE):
                    raise WarpCurriculumConfigError("flat evaluation policy must return CUDA [world, 7] actions")
                action = torch.tanh(action).contiguous().to(dtype=torch.float32)
            result = task.step(action)
            reward_value = getattr(result, "reward", None)
            if isinstance(reward_value, torch.Tensor) and tuple(reward_value.shape) == (batch.num_worlds,):
                finite_values.copy_(torch.isfinite(reward_value))
                torch.all(finite_values, out=finite_step)
                finite_reward.logical_and_(finite_step)
            else:
                finite_reward.zero_()
            finite_terms_step.fill_(True)
            reward_terms = getattr(task, "_reward_terms", None)
            if not isinstance(reward_terms, Mapping) or not reward_terms:
                finite_terms_step.zero_()
            else:
                for reward_term in reward_terms.values():
                    if not isinstance(reward_term, torch.Tensor) or tuple(reward_term.shape) != (batch.num_worlds,):
                        finite_terms_step.zero_()
                        break
                    finite_values.copy_(torch.isfinite(reward_term))
                    torch.all(finite_values, out=finite_step)
                    finite_terms_step.logical_and_(finite_step)
            finite_reward_terms.logical_and_(finite_terms_step)
            terminated.logical_or_(result.terminated)
            overflowed.logical_or_(batch.overflow.ne(0))
            estopped.logical_or_(batch.estopped)
            task.reset(result.done)
        finite_state = torch.isfinite(batch.qpos).all() & torch.isfinite(batch.qvel).all()
        summary = torch.stack((
            terminated.sum(dtype=torch.int64),
            overflowed.sum(dtype=torch.int64),
            estopped.sum(dtype=torch.int64),
            finite_state.to(dtype=torch.int64),
            finite_reward.to(dtype=torch.int64),
            finite_reward_terms.to(dtype=torch.int64),
        ))
        torch.cuda.synchronize(batch.device)
        (
            terminated_count,
            overflow_count,
            estop_count,
            finite_state_flag,
            finite_reward_flag,
            finite_reward_terms_flag,
        ) = (int(value) for value in summary.detach().cpu().tolist())
    finally:
        task.reset()
    passed = bool(
        terminated_count == 0
        and overflow_count == 0
        and estop_count == 0
        and bool(finite_state_flag)
        and bool(finite_reward_flag)
        and bool(finite_reward_terms_flag)
    )
    report: Mapping[str, Any] = {
        "stage_id": stage.stage_id,
        "evaluation_scope": "post_policy_flat_cuda_safety",
        "requested_duration_seconds": duration,
        "simulated_duration_seconds": steps * action_dt,
        "policy_steps": steps,
        "num_worlds": int(batch.num_worlds),
        "terminated_worlds": terminated_count,
        "overflowed_worlds": overflow_count,
        "estopped_worlds": estop_count,
        "finite_state": bool(finite_state_flag),
        "finite_reward": bool(finite_reward_flag),
        "finite_reward_terms": bool(finite_reward_terms_flag),
        "domain_randomization_active": bool(getattr(task.config, "domain_randomization_enabled", False)),
        "passed": passed,
    }
    if not passed:
        raise WarpCurriculumConfigError(
            f"flat post-policy CUDA safety evaluation failed for {stage.stage_id}: "
            f"terminated={terminated_count}, overflowed={overflow_count}, estopped={estop_count}, "
            f"finite_state={bool(finite_state_flag)}, finite_reward={bool(finite_reward_flag)}, "
            f"finite_reward_terms={bool(finite_reward_terms_flag)}"
        )
    return report


def _run_post_training_evaluation(
    *, task: Any, batch: Any, policy: Any, stage: StageConfig, config: CurriculumConfig, smoke: bool
) -> Mapping[str, Any] | None:
    """Dispatch a CUDA-only policy evaluation for every formal curriculum stage."""

    if smoke:
        return None
    if stage.adapter_config_path is None:
        return _evaluate_flat_policy_stage(task=task, batch=batch, policy=policy, stage=stage, config=config)

    def deterministic_action(observation: Any) -> Any:
        actor = getattr(policy, "actor", None)
        if actor is None:
            raise WarpCurriculumConfigError("PPO policy does not expose an actor for CUDA course evaluation")
        return actor(observation)
    try:
        from official_course_warp import OfficialCourseTask, evaluate_policy_stage as evaluate_official

        if isinstance(task, OfficialCourseTask):
            return evaluate_official(task, deterministic_action, stage)
    except ImportError:
        pass
    try:
        from rmuc_curriculum_warp import RmucRouteTask, evaluate_policy_stage as evaluate_rmuc

        if isinstance(task, RmucRouteTask):
            return evaluate_rmuc(task, deterministic_action, stage)
    except ImportError:
        pass
    raise WarpCurriculumConfigError(
        "CUDA curriculum stage did not provide a verified post-training evaluator"
    )


def _load_required_prerequisite_checkpoint(
    config: CurriculumConfig, stage: StageConfig, *, smoke: bool
) -> ResidualCheckpoint | None:
    """Require all YAML predecessors and select the final certified baseline.

    This runs before a simulator is constructed.  A user may still supply a
    separately audited CPU residual warm start, but cannot skip the completed
    predecessor certificates that define the curriculum order.
    """

    if smoke or not stage.prerequisite_stage_ids:
        return None
    selected: ResidualCheckpoint | None = None
    for predecessor_id in stage.prerequisite_stage_ids:
        predecessor = config.stage(predecessor_id)
        artifact = _stage_artifact_path(config.output.checkpoint_path, predecessor)
        if not artifact.is_file():
            raise WarpCurriculumConfigError(
                f"stage {stage.stage_id!r} is blocked until prerequisite {predecessor_id!r} has a certified checkpoint"
            )
        # Verify the source artifact against the stage that produced it before
        # validating the explicit target-stage transfer.  This binds both the
        # certificate and the source stage scope to the same checked-in YAML.
        selected = load_residual_checkpoint(artifact, config=config, stage=predecessor)
        validate_checkpoint_metadata(selected.metadata, config=config, stage=stage)
    return selected


def run_curriculum_training(
    config: CurriculumConfig, *, stage_id: str, smoke: bool = False,
    init_residual_checkpoint: str | Path | None = None,
) -> Path:
    """Validate and train one curriculum stage."""

    stage = config.stage(stage_id)
    capability = validate_stage_capability(config, stage)
    certified_predecessor = _load_required_prerequisite_checkpoint(config, stage, smoke=smoke)
    source_checkpoint = certified_predecessor
    if init_residual_checkpoint is not None:
        source_checkpoint = load_residual_checkpoint(init_residual_checkpoint, config=config, stage=stage)
    if stage.stage_id != "rmuc_flat":
        bundle = _external_stage_bundle(config, stage)
        batch = _bundle_value(bundle, "batch")
        task = _bundle_value(bundle, "task")
        close_bundle = _bundle_value(bundle, "close", required=False)
        try:
            gate = _bundle_value(bundle, "gate_report", required=False)
            gate_runner = _bundle_value(bundle, "run_stability_gate", required=False)
            if capability.runtime_gate_required and not callable(gate_runner):
                raise WarpCurriculumConfigError(
                    f"stage {stage.stage_id!r} requires a callable runtime GPU gate"
                )
            if callable(gate_runner):
                gate = gate_runner()
            if gate is None:
                raise WarpCurriculumConfigError("external GPU bundle must provide gate_report or run_stability_gate")
            if capability.runtime_gate_required:
                gate = _validate_runtime_gate_report(gate, config=config, stage=stage, capability=capability)
            return _run_vector_training(
                config=config, stage=stage, batch=batch, task=task, gate_report=gate,
                source_checkpoint=source_checkpoint, smoke=smoke,
                close_bundle=close_bundle,
            )
        except BaseException:
            # A failed external CUDA gate must leave physical controls and
            # generalized forces latched at zero before propagating the error.
            # The official adapter's close function is idempotent, and third
            # party factories can opt out by omitting it.
            if callable(close_bundle):
                close_bundle()
            raise

    from train_warp_ppo import _run_zero_residual_stability_gate
    from warp_env import WarpPhysicsBatch
    from warp_flat_controller import FixedGainFlatController, calibrate_flat_controller
    from warp_task import WarpFlatWalkingConfig, WarpFlatWalkingTask

    flat, batch_config = _load_flat_manifest(config, stage)
    calibration = calibrate_flat_controller(batch_config, flat.flat_controller)
    batch = WarpPhysicsBatch(batch_config)
    try:
        task_config = WarpFlatWalkingConfig.from_mapping(flat.flat_walking)
        task = WarpFlatWalkingTask(batch, task_config, calibration=calibration.to_task_calibration())
        controller = FixedGainFlatController(calibration, task, flat.flat_controller)
        task.set_feedback_controller(controller)
        gate_report = _run_zero_residual_stability_gate(batch, task, flat)
        return _run_vector_training(
            config=config, stage=stage, batch=batch, task=task, gate_report=asdict(gate_report),
            source_checkpoint=source_checkpoint, smoke=smoke,
        )
    finally:
        batch._safe_controls.zero_()
        batch._warp.copy(batch.data.ctrl, batch._safe_controls_warp)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a capability-gated MuJoCo-Warp curriculum PPO stage.")
    parser.add_argument(
        "--curriculum",
        type=resolve_cli_input,
        default=project_path("configs", "warp_curriculum_ppo.yaml"),
    )
    parser.add_argument("--stage", required=True, help="Curriculum stage id, for example rmuc_flat or grades.")
    parser.add_argument("--init-residual-checkpoint", type=resolve_cli_input, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one short GPU PPO update.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = load_curriculum_config(resolve_cli_input(args.curriculum))
        output = run_curriculum_training(
            config,
            stage_id=args.stage,
            smoke=args.smoke,
            init_residual_checkpoint=args.init_residual_checkpoint,
        )
    except (WarpCurriculumConfigError, RuntimeError, ValueError, OSError) as error:
        raise SystemExit(f"MuJoCo-Warp curriculum PPO blocked/failed: {error}") from error
    print(f"MuJoCo-Warp curriculum PPO complete: stage={args.stage} checkpoint={output}")


__all__ = [
    "ACTION_SIZE", "ACTION_SEMANTICS", "CURRICULUM_BACKEND", "CURRICULUM_CHECKPOINT_FORMAT",
    "CURRICULUM_CONFIG_SCHEMA", "CurriculumConfig", "FLAT_BACKEND", "FLAT_CONTROLLER_BACKEND",
    "GPU_CURRICULUM_CAPABILITIES", "GpuStageCapability", "OBSERVATION_SIZE", "PpoSettings",
    "REWARD_SCHEMA", "ResidualCheckpoint", "StageConfig", "WarpCurriculumConfigError",
    "build_checkpoint_metadata", "gpu_stage_capability", "load_curriculum_config",
    "load_residual_checkpoint", "main", "parse_args", "run_curriculum_training",
    "validate_checkpoint_metadata", "validate_stage_capability",
]


if __name__ == "__main__":
    main()
