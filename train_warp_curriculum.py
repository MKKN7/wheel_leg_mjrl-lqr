"""Capability-gated MuJoCo-Warp curriculum PPO training.

The manifest and stage capability checks run before GPU model allocation.  The
repository currently proves only the fixed-gain flat task; RMUC grades,
steps/jumps, and domain-randomized turning remain blocked until a GPU task
backend publishes matching parity evidence.  External backends may register a
``GPU_CURRICULUM_CAPABILITIES`` mapping and a ``build_curriculum_stage``
factory in ``warp_curriculum_task``.
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


CURRICULUM_CONFIG_SCHEMA = 2
CURRICULUM_CHECKPOINT_FORMAT = 2
CURRICULUM_BACKEND = "mujoco_warp_curriculum_ppo_v2"
FLAT_BACKEND = "mujoco_warp_flat_ppo_fixed_gain_v2"
FLAT_CONTROLLER_BACKEND = "fixed_gain_flat_controller_v2"
OBSERVATION_SIZE = 67
ACTION_SIZE = 7
REWARD_SCHEMA = "warp_flat_walking_reward_v1"
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
    optional = {"adapter_config_path", "scene_variant", "reward_schema"}
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
            and isinstance(scope, Mapping)
            and scope.get("task_mode") == "flat_walking_only"
            and scope.get("terrain_enabled") is False
            and scope.get("jump_enabled") is False
            and scope.get("domain_randomization_enabled") is False
            and scope.get("flat_terrain_features_zeroed") is True
            and scope.get("jump_features_zeroed") is True
        ):
            saved_reward = REWARD_SCHEMA
        else:
            raise WarpCurriculumConfigError("residual checkpoint is missing an auditable reward schema")
    if not cpu_transfer and saved_reward != expected_reward_schema:
        raise WarpCurriculumConfigError("residual checkpoint reward schema is incompatible")
    if stage.reward_schema is not None and not cpu_transfer:
        scope = metadata.get("stage_scope")
        if not isinstance(scope, Mapping):
            raise WarpCurriculumConfigError("official checkpoint is missing auditable stage_scope provenance")
        for name, expected in (
            ("task_mode", stage.task_mode),
            ("xml_path", str(stage.xml_path)),
            ("scene_variant", stage.scene_variant),
            ("terrain_curriculum_path", str(stage.terrain_curriculum_path)),
            ("terrain_stage_id", stage.terrain_stage_id),
            ("controller_backend", stage.controller_backend),
        ):
            if scope.get(name) != expected:
                raise WarpCurriculumConfigError(f"checkpoint stage_scope.{name} does not match the selected official stage")
        experiment = metadata.get("experiment_config")
        if not isinstance(experiment, Mapping):
            raise WarpCurriculumConfigError("official checkpoint is missing experiment_config provenance")
        digest_fields: tuple[tuple[str, Path], ...] = (
            ("xml_sha256", stage.xml_path),
            ("terrain_curriculum_sha256", stage.terrain_curriculum_path),
        )
        if stage.adapter_config_path is not None:
            digest_fields += (("adapter_config_sha256", stage.adapter_config_path),)
        for field, path in digest_fields:
            try:
                expected_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError as error:
                raise WarpCurriculumConfigError(f"unable to hash official provenance file {path}") from error
            if experiment.get(field) != expected_digest:
                raise WarpCurriculumConfigError(f"official checkpoint provenance digest mismatch for {field}")
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
    source_checkpoint: ResidualCheckpoint | None, batch: Any, smoke: bool
) -> dict[str, Any]:
    """Create provenance metadata for a curriculum checkpoint."""

    reward_schema = stage.reward_schema or config.reward_schema
    reward_terms = (
        (
            "speed_tracking", "leg_tracking", "attitude_tracking", "contact", "yaw_tracking",
            "energy_cost", "residual_cost", "yaw_rate_cost", "leg_symmetry_cost", "unsafe_penalty",
            "official_route_progress", "official_route_completion",
        )
        if stage.reward_schema is not None
        else (
            "speed_tracking", "leg_tracking", "attitude_tracking", "contact", "yaw_tracking",
            "energy_cost", "residual_cost", "yaw_rate_cost", "leg_symmetry_cost", "unsafe_penalty",
        )
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
            "controller_backend": stage.controller_backend,
            "terrain_enabled": stage.terrain_enabled,
            "steps_enabled": stage.steps_enabled,
            "jump_enabled": stage.jump_enabled,
            "domain_randomization_enabled": stage.domain_randomization_enabled,
            "command_speed_mps": stage.command_speed_mps,
            "command_yaw_rate_rad_s": stage.command_yaw_rate_rad_s,
        },
        "experiment_config": {
            "source_path": str(config.source_path),
            "source_digest": config.source_digest,
            "batch_config_path": str(config.batch_config_path),
            "num_worlds": int(batch.num_worlds),
            "device": str(batch.device),
            "xml_sha256": digest(stage.xml_path),
            "terrain_curriculum_sha256": digest(stage.terrain_curriculum_path),
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


def _validate_runtime_gate_report(
    report: Any, *, stage: StageConfig, capability: GpuStageCapability
) -> Mapping[str, Any]:
    """Validate the minimum evidence required by a conditional GPU stage."""

    if not isinstance(report, Mapping):
        raise WarpCurriculumConfigError("conditional GPU gate must return a mapping report")
    if report.get("stage_id") != stage.stage_id:
        raise WarpCurriculumConfigError("conditional GPU gate report stage_id does not match the selected stage")
    if report.get("conditional_capability") is not True:
        raise WarpCurriculumConfigError("conditional GPU gate report must set conditional_capability=true")
    if report.get("passed") is not True:
        raise WarpCurriculumConfigError("conditional GPU gate report must set passed=true")
    required = ("num_worlds", "terminated_worlds", "overflowed_worlds", "estopped_worlds", "finite_state")
    for name in required:
        if name not in report:
            raise WarpCurriculumConfigError(f"conditional GPU gate report is missing {name!r}")
    try:
        worlds = int(report["num_worlds"])
        counts = tuple(int(report[name]) for name in required[1:4])
    except (TypeError, ValueError) as error:
        raise WarpCurriculumConfigError("conditional GPU gate counts must be integers") from error
    if worlds < 1 or any(value < 0 or value > worlds for value in counts):
        raise WarpCurriculumConfigError("conditional GPU gate counts are outside [0, num_worlds]")
    if report["finite_state"] is not True:
        raise WarpCurriculumConfigError("conditional GPU gate must prove finite_state=true")
    if stage.stage_id == "official_grade15_up":
        for name in (
            "zero_residual", "minimum_progress_m", "speed_mae_mps", "unsafe_rate",
            "first_fault_step", "first_fault_reason_code",
        ):
            if name not in report:
                raise WarpCurriculumConfigError(f"official GPU gate report is missing {name!r}")
        if report["zero_residual"] is not True:
            raise WarpCurriculumConfigError("official GPU gate must be a zero-residual baseline test")
        for name in ("minimum_progress_m", "speed_mae_mps", "unsafe_rate"):
            try:
                value = float(report[name])
            except (TypeError, ValueError) as error:
                raise WarpCurriculumConfigError(f"official GPU gate field {name!r} must be numeric") from error
            if not math.isfinite(value):
                raise WarpCurriculumConfigError(f"official GPU gate field {name!r} must be finite")
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
                source_checkpoint=source_checkpoint, batch=batch, smoke=smoke,
            ),
            "model_state_dict": {key: value.detach().clone() for key, value in policy.state_dict().items()},
            "actor_state_dict": {key: value.detach().clone() for key, value in policy.actor.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "gate_report": gate_report,
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
            _atomic_save(payload, output_path, torch)
        if max_updates is not None and update_index >= max_updates:
            break
    if smoke and update_index != 1:
        raise RuntimeError("curriculum PPO smoke must execute exactly one update")
    if callable(close_bundle):
        close_bundle()
    return output_path


def run_curriculum_training(
    config: CurriculumConfig, *, stage_id: str, smoke: bool = False,
    init_residual_checkpoint: str | Path | None = None,
) -> Path:
    """Validate and train one curriculum stage."""

    stage = config.stage(stage_id)
    capability = validate_stage_capability(config, stage)
    source_checkpoint = (
        None if init_residual_checkpoint is None
        else load_residual_checkpoint(init_residual_checkpoint, config=config, stage=stage)
    )
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
                gate = _validate_runtime_gate_report(gate, stage=stage, capability=capability)
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
    parser.add_argument("--curriculum", type=Path, default=Path("configs/warp_curriculum_ppo.yaml"))
    parser.add_argument("--stage", required=True, help="Curriculum stage id, for example rmuc_flat or grades.")
    parser.add_argument("--init-residual-checkpoint", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one short GPU PPO update.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = load_curriculum_config(args.curriculum)
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
