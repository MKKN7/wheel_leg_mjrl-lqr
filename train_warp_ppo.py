"""MuJoCo-Warp flat fixed-gain PPO entry point and physics preflight.

The training path is deliberately narrow: one CPU calibration creates an
immutable fixed-gain controller, the calibrated state is uploaded once, and
all rollout/reward/control tensors remain on CUDA. A full 128-world,
zero-residual stability gate must survive before PPO allocates a policy or
writes a checkpoint. Terrain, jumping, domain randomization, nonzero command
training, and the seventh leg-length action are rejected by the manifest
loader.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import yaml

from warp_env import (
    WarpBatchConfig,
    WarpBatchError,
    WarpPhysicsBatch,
    load_warp_batch_config,
    run_warp_preflight,
)
from warp_flat_controller import WarpFlatControllerConfig


FLAT_PPO_CONFIG_SCHEMA = 2
FLAT_PPO_CHECKPOINT_FORMAT = 2
FLAT_PPO_BACKEND = "mujoco_warp_flat_ppo_fixed_gain_v2"
FIXED_GAIN_CONTROLLER_BACKEND = "fixed_gain_flat_controller_v2"
OBSERVATION_SIZE = 67
ACTION_SIZE = 7

_FLAT_WALKING_KEYS = {
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
    "direct_control_mode",
    "residual_limits",
    "leg_action_enabled",
}
_FLAT_CONTROLLER_KEYS = {
    "command_speed_mps",
    "command_yaw_rate_rad_s",
    "command_leg_length_m",
    "calibration_seed",
    "gas_spring_enabled",
    "gas_spring_torque_nm",
    "gas_spring_max_abs_generalized_force_nm",
    "stance_guard_kp_nm_per_rad",
    "stance_guard_kd_nm_per_rad_per_s",
    "leg_force_kp_n_per_m",
    "leg_force_kd_ns_per_m",
    "leg_force_limit_n",
    "max_forward_feedback_mps",
    "lqr_reference_speed_limit_mps",
    "command_speed_gain_nm_per_mps",
    "command_yaw_gain_nm_per_rad_s",
    "command_wheel_feedforward_limit_nm",
    "max_torque_fraction",
    "yaw_alignment_enabled",
}


class WarpFlatPpoConfigError(ValueError):
    """Raised before GPU allocation when a flat PPO experiment is invalid."""


@dataclass(frozen=True)
class PpoHyperparameters:
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
class OutputConfig:
    checkpoint_path: Path
    metrics_path: Path
    checkpoint_interval_updates: int


@dataclass(frozen=True)
class SmokeConfig:
    rollout_steps: int
    epochs: int
    minibatch_size: int
    checkpoint_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class ScopeConfig:
    task_mode: str
    terrain_enabled: bool
    jump_enabled: bool
    domain_randomization_enabled: bool
    controller_backend: str
    dynamic_lqr_enabled: bool
    zero_command_only: bool
    leg_action_enabled: bool


@dataclass(frozen=True)
class StabilityGateConfig:
    duration_seconds: float
    required_num_worlds: int
    zero_residual: bool
    require_no_terminated: bool
    require_no_overflow: bool
    require_finite_state: bool


@dataclass(frozen=True)
class FlatPpoTrainingConfig:
    source_path: Path
    backend: str
    batch_config_path: Path
    flat_walking: dict[str, Any]
    flat_controller: WarpFlatControllerConfig
    ppo: PpoHyperparameters
    output: OutputConfig
    smoke: SmokeConfig
    scope: ScopeConfig
    stability_gate: StabilityGateConfig
    source_digest: str


@dataclass(frozen=True)
class StabilityGateReport:
    requested_duration_seconds: float
    simulated_duration_seconds: float
    policy_steps: int
    num_worlds: int
    terminated_worlds: int
    overflowed_worlds: int
    estopped_worlds: int
    finite_state: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WarpFlatPpoConfigError(f"{name} must be a YAML mapping")
    return value


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise WarpFlatPpoConfigError(f"missing required configuration key: {name}")
    return mapping[name]


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise WarpFlatPpoConfigError(f"{name} must be boolean")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WarpFlatPpoConfigError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WarpFlatPpoConfigError(f"{name} must be a positive integer")
    return int(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WarpFlatPpoConfigError(f"{name} must be an integer")
    return int(value)


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarpFlatPpoConfigError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise WarpFlatPpoConfigError(f"{name} must be {qualifier}")
    return result


def _optional_finite_float(value: object, name: str) -> float | None:
    return None if value is None else _finite_float(value, name)


def _resolve_path(source_path: Path, value: object, name: str) -> Path:
    candidate = Path(_string(value, name))
    return candidate.resolve() if candidate.is_absolute() else (source_path.parent / candidate).resolve()


def _expect_exact_keys(mapping: Mapping[str, Any], name: str, keys: set[str]) -> None:
    actual = set(mapping)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing or unknown:
        pieces: list[str] = []
        if missing:
            pieces.append(f"missing={missing}")
        if unknown:
            pieces.append(f"unknown={unknown}")
        raise WarpFlatPpoConfigError(f"{name} keys are invalid: {', '.join(pieces)}")


def _load_ppo_hyperparameters(raw: Mapping[str, Any]) -> PpoHyperparameters:
    expected = {
        "total_timesteps", "rollout_steps", "epochs", "minibatch_size", "learning_rate", "gamma",
        "gae_lambda", "clip_ratio", "value_coefficient", "entropy_coefficient", "max_gradient_norm",
        "hidden_size", "initial_action_std", "seed",
    }
    _expect_exact_keys(raw, "ppo", expected)
    result = PpoHyperparameters(
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
        raise WarpFlatPpoConfigError("ppo.gamma and ppo.gae_lambda must be within [0, 1]")
    if not 0.0 < result.clip_ratio < 1.0:
        raise WarpFlatPpoConfigError("ppo.clip_ratio must be within (0, 1)")
    if not 0.0 < result.initial_action_std <= 1.0:
        raise WarpFlatPpoConfigError("ppo.initial_action_std must be within (0, 1]")
    if result.entropy_coefficient < 0.0:
        raise WarpFlatPpoConfigError("ppo.entropy_coefficient must be non-negative")
    return result


def _load_output_config(raw: Mapping[str, Any], source_path: Path, *, name: str) -> OutputConfig:
    _expect_exact_keys(raw, name, {"checkpoint_path", "metrics_path", "checkpoint_interval_updates"})
    checkpoint_path = _resolve_path(source_path, _required(raw, "checkpoint_path"), f"{name}.checkpoint_path")
    metrics_path = _resolve_path(source_path, _required(raw, "metrics_path"), f"{name}.metrics_path")
    legacy_names = {"warp_flat_ppo.pt", "warp_flat_ppo.metrics.csv", "warp_flat_ppo_smoke.pt", "warp_flat_ppo_smoke.metrics.csv"}
    if checkpoint_path.name in legacy_names or metrics_path.name in legacy_names:
        raise WarpFlatPpoConfigError(f"{name} paths may not overwrite legacy flat PPO artifacts")
    if checkpoint_path == metrics_path:
        raise WarpFlatPpoConfigError(f"{name}.checkpoint_path and metrics_path must differ")
    if "fixed_gain_v2" not in checkpoint_path.name or "fixed_gain_v2" not in metrics_path.name:
        raise WarpFlatPpoConfigError(f"{name} paths must use the fixed_gain_v2 artifact namespace")
    return OutputConfig(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        checkpoint_interval_updates=_positive_int(
            _required(raw, "checkpoint_interval_updates"), f"{name}.checkpoint_interval_updates"
        ),
    )


def _load_smoke_config(raw: Mapping[str, Any], source_path: Path) -> SmokeConfig:
    _expect_exact_keys(raw, "smoke", {"rollout_steps", "epochs", "minibatch_size", "checkpoint_path", "metrics_path"})
    output = _load_output_config(
        {
            "checkpoint_path": _required(raw, "checkpoint_path"),
            "metrics_path": _required(raw, "metrics_path"),
            "checkpoint_interval_updates": 1,
        },
        source_path,
        name="smoke",
    )
    return SmokeConfig(
        rollout_steps=_positive_int(_required(raw, "rollout_steps"), "smoke.rollout_steps"),
        epochs=_positive_int(_required(raw, "epochs"), "smoke.epochs"),
        minibatch_size=_positive_int(_required(raw, "minibatch_size"), "smoke.minibatch_size"),
        checkpoint_path=output.checkpoint_path,
        metrics_path=output.metrics_path,
    )


def _load_flat_walking_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    _expect_exact_keys(raw, "flat_walking", _FLAT_WALKING_KEYS)
    normalized = dict(raw)
    limits = _required(raw, "residual_limits")
    if not isinstance(limits, (list, tuple)) or len(limits) != ACTION_SIZE:
        raise WarpFlatPpoConfigError("flat_walking.residual_limits must contain seven values")
    normalized["residual_limits"] = tuple(
        _finite_float(value, f"flat_walking.residual_limits[{index}]")
        for index, value in enumerate(limits)
    )
    try:
        from warp_task import WarpFlatWalkingConfig

        task_config = WarpFlatWalkingConfig.from_mapping(normalized)
    except (TypeError, ValueError) as error:
        raise WarpFlatPpoConfigError(f"invalid flat_walking section: {error}") from error
    if task_config.direct_control_mode:
        raise WarpFlatPpoConfigError("flat_walking.direct_control_mode must be false for fixed-gain PPO")
    if task_config.leg_action_enabled:
        raise WarpFlatPpoConfigError("flat_walking.leg_action_enabled must be false; channel seven is masked")
    if task_config.residual_limits[-1] != 0.0:
        raise WarpFlatPpoConfigError("flat_walking.residual_limits[6] must be zero with the leg channel masked")
    if task_config.command_speed_mps != 0.0 or task_config.command_yaw_rate_rad_s != 0.0:
        raise WarpFlatPpoConfigError("flat_walking training commands must be exactly zero speed and zero yaw")
    return normalized


def _load_flat_controller_config(raw: Mapping[str, Any]) -> WarpFlatControllerConfig:
    _expect_exact_keys(raw, "flat_controller", _FLAT_CONTROLLER_KEYS)
    for name in ("gas_spring_enabled", "yaw_alignment_enabled"):
        _boolean(_required(raw, name), f"flat_controller.{name}")
    _integer(_required(raw, "calibration_seed"), "flat_controller.calibration_seed")
    for name in (
        "command_speed_mps", "command_yaw_rate_rad_s", "gas_spring_torque_nm",
        "gas_spring_max_abs_generalized_force_nm",
        "stance_guard_kp_nm_per_rad", "stance_guard_kd_nm_per_rad_per_s",
        "leg_force_kp_n_per_m", "leg_force_kd_ns_per_m", "leg_force_limit_n",
        "max_forward_feedback_mps", "lqr_reference_speed_limit_mps", "max_torque_fraction",
    ):
        _finite_float(_required(raw, name), f"flat_controller.{name}")
    _optional_finite_float(_required(raw, "command_leg_length_m"), "flat_controller.command_leg_length_m")
    try:
        config = WarpFlatControllerConfig.from_mapping(dict(raw))
    except (TypeError, ValueError) as error:
        raise WarpFlatPpoConfigError(f"invalid flat_controller section: {error}") from error
    if config.command_speed_mps != 0.0 or config.command_yaw_rate_rad_s != 0.0:
        raise WarpFlatPpoConfigError("flat_controller commands must be exactly zero speed and zero yaw")
    if config.max_torque_fraction > 0.80:
        raise WarpFlatPpoConfigError("flat_controller.max_torque_fraction cannot exceed 0.80")
    return config


def _load_scope_config(raw: Mapping[str, Any]) -> ScopeConfig:
    expected = {
        "task_mode", "terrain_enabled", "jump_enabled", "domain_randomization_enabled",
        "controller_backend", "dynamic_lqr_enabled", "zero_command_only", "leg_action_enabled",
    }
    _expect_exact_keys(raw, "scope", expected)
    result = ScopeConfig(
        task_mode=_string(_required(raw, "task_mode"), "scope.task_mode"),
        terrain_enabled=_boolean(_required(raw, "terrain_enabled"), "scope.terrain_enabled"),
        jump_enabled=_boolean(_required(raw, "jump_enabled"), "scope.jump_enabled"),
        domain_randomization_enabled=_boolean(_required(raw, "domain_randomization_enabled"), "scope.domain_randomization_enabled"),
        controller_backend=_string(_required(raw, "controller_backend"), "scope.controller_backend"),
        dynamic_lqr_enabled=_boolean(_required(raw, "dynamic_lqr_enabled"), "scope.dynamic_lqr_enabled"),
        zero_command_only=_boolean(_required(raw, "zero_command_only"), "scope.zero_command_only"),
        leg_action_enabled=_boolean(_required(raw, "leg_action_enabled"), "scope.leg_action_enabled"),
    )
    if result.task_mode != "flat_walking_only":
        raise WarpFlatPpoConfigError("scope.task_mode must be 'flat_walking_only'")
    if result.terrain_enabled or result.jump_enabled or result.domain_randomization_enabled:
        raise WarpFlatPpoConfigError("terrain, jump, and domain randomization are disabled for flat PPO")
    if result.controller_backend != FIXED_GAIN_CONTROLLER_BACKEND:
        raise WarpFlatPpoConfigError(f"scope.controller_backend must be {FIXED_GAIN_CONTROLLER_BACKEND!r}")
    if result.dynamic_lqr_enabled or not result.zero_command_only or result.leg_action_enabled:
        raise WarpFlatPpoConfigError("scope must enforce fixed-gain, zero-command, masked-leg training")
    return result


def _load_stability_gate_config(raw: Mapping[str, Any]) -> StabilityGateConfig:
    expected = {
        "duration_seconds", "required_num_worlds", "zero_residual", "require_no_terminated",
        "require_no_overflow", "require_finite_state",
    }
    _expect_exact_keys(raw, "stability_gate", expected)
    result = StabilityGateConfig(
        duration_seconds=_finite_float(_required(raw, "duration_seconds"), "stability_gate.duration_seconds", positive=True),
        required_num_worlds=_positive_int(_required(raw, "required_num_worlds"), "stability_gate.required_num_worlds"),
        zero_residual=_boolean(_required(raw, "zero_residual"), "stability_gate.zero_residual"),
        require_no_terminated=_boolean(_required(raw, "require_no_terminated"), "stability_gate.require_no_terminated"),
        require_no_overflow=_boolean(_required(raw, "require_no_overflow"), "stability_gate.require_no_overflow"),
        require_finite_state=_boolean(_required(raw, "require_finite_state"), "stability_gate.require_finite_state"),
    )
    if not result.zero_residual or not result.require_no_terminated or not result.require_no_overflow or not result.require_finite_state:
        raise WarpFlatPpoConfigError("stability_gate must require zero residual, no termination/overflow, and finite state")
    return result


def load_flat_ppo_training_config(path: str | Path) -> FlatPpoTrainingConfig:
    """Load a strict v2 fixed-gain flat PPO experiment manifest."""

    source_path = Path(path).resolve()
    try:
        source_text = source_path.read_text(encoding="utf-8")
        raw = yaml.safe_load(source_text)
    except OSError as error:
        raise WarpFlatPpoConfigError(f"unable to read flat PPO configuration {source_path}: {error}") from error
    root = _mapping(raw, "flat PPO configuration")
    expected = {
        "schema_version", "backend", "batch_config", "flat_walking", "flat_controller", "ppo",
        "output", "smoke", "scope", "stability_gate",
    }
    _expect_exact_keys(root, "flat PPO configuration", expected)
    if _required(root, "schema_version") != FLAT_PPO_CONFIG_SCHEMA:
        raise WarpFlatPpoConfigError(f"unsupported flat PPO schema; expected {FLAT_PPO_CONFIG_SCHEMA}")
    backend = _string(_required(root, "backend"), "backend")
    if backend != FLAT_PPO_BACKEND:
        raise WarpFlatPpoConfigError(f"backend must be {FLAT_PPO_BACKEND!r}")
    flat_walking = _load_flat_walking_config(_mapping(_required(root, "flat_walking"), "flat_walking"))
    flat_controller = _load_flat_controller_config(
        _mapping(_required(root, "flat_controller"), "flat_controller")
    )
    ppo = _load_ppo_hyperparameters(_mapping(_required(root, "ppo"), "ppo"))
    output = _load_output_config(_mapping(_required(root, "output"), "output"), source_path, name="output")
    smoke = _load_smoke_config(_mapping(_required(root, "smoke"), "smoke"), source_path)
    scope = _load_scope_config(_mapping(_required(root, "scope"), "scope"))
    stability_gate = _load_stability_gate_config(
        _mapping(_required(root, "stability_gate"), "stability_gate")
    )
    if flat_controller.command_leg_length_m is not None and flat_walking["command_leg_length_m"] is not None:
        if not np.isclose(flat_controller.command_leg_length_m, flat_walking["command_leg_length_m"]):
            raise WarpFlatPpoConfigError("flat_controller and flat_walking command_leg_length_m must match")
    if ppo.total_timesteps % stability_gate.required_num_worlds:
        raise WarpFlatPpoConfigError("ppo.total_timesteps must be divisible by stability_gate.required_num_worlds")
    return FlatPpoTrainingConfig(
        source_path=source_path,
        backend=backend,
        batch_config_path=_resolve_path(source_path, _required(root, "batch_config"), "batch_config"),
        flat_walking=flat_walking,
        flat_controller=flat_controller,
        ppo=ppo,
        output=output,
        smoke=smoke,
        scope=scope,
        stability_gate=stability_gate,
        source_digest=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
    )


def _validate_batch_for_flat_training(
    batch_config: WarpBatchConfig,
    config: FlatPpoTrainingConfig,
    task_config: Any,
) -> None:
    if batch_config.controller_backend != "raw_controls_only" or batch_config.ppo_training_enabled:
        raise WarpFlatPpoConfigError("flat PPO requires the conservative raw-control batch allocation config")
    if batch_config.num_worlds != config.stability_gate.required_num_worlds:
        raise WarpFlatPpoConfigError(
            f"flat PPO requires exactly {config.stability_gate.required_num_worlds} worlds; got {batch_config.num_worlds}"
        )
    if config.stability_gate.duration_seconds > task_config.episode_seconds + 1.0e-9:
        raise WarpFlatPpoConfigError("stability_gate.duration_seconds cannot exceed task episode_seconds")
    if config.ppo.total_timesteps % batch_config.num_worlds:
        raise WarpFlatPpoConfigError("ppo.total_timesteps must be divisible by num_worlds")
    if batch_config.safety.torque_fraction_of_rated > 0.80:
        raise WarpFlatPpoConfigError("batch safety torque fraction cannot exceed 0.80")
    if config.flat_controller.max_torque_fraction > batch_config.safety.torque_fraction_of_rated + 1.0e-9:
        raise WarpFlatPpoConfigError("flat controller torque fraction exceeds batch safety derating")


def _run_zero_residual_stability_gate(
    batch: WarpPhysicsBatch,
    task: Any,
    config: FlatPpoTrainingConfig,
) -> StabilityGateReport:
    """Run the required nominal fixed-gain episode before policy allocation.

    Per-step predicates remain resident on CUDA. The gate performs one host
    boundary after the full batch episode to decide whether PPO may start.
    """

    torch = batch._torch
    action_dt = float(task._time_step)
    if not math.isfinite(action_dt) or action_dt <= 0.0:
        raise WarpFlatPpoConfigError("task action timestep must be finite and positive")
    policy_steps = max(1, int(math.ceil(config.stability_gate.duration_seconds / action_dt)))
    zero_action = torch.zeros((batch.num_worlds, ACTION_SIZE), dtype=torch.float32, device=batch.device)
    terminated_seen = torch.zeros(batch.num_worlds, dtype=torch.bool, device=batch.device)
    overflow_seen = torch.zeros_like(terminated_seen)
    estopped_seen = torch.zeros_like(terminated_seen)
    task.reset()
    for _ in range(policy_steps):
        result = task.step(zero_action)
        terminated_seen.logical_or_(result.terminated)
        overflow_seen.logical_or_(batch.overflow.ne(0))
        estopped_seen.logical_or_(batch.estopped)
    finite_state = torch.isfinite(batch.qpos).all() & torch.isfinite(batch.qvel).all()
    summary = torch.stack((
        terminated_seen.sum(dtype=torch.int64),
        overflow_seen.sum(dtype=torch.int64),
        estopped_seen.sum(dtype=torch.int64),
        finite_state.to(dtype=torch.int64),
    ))
    torch.cuda.synchronize(batch.device)
    terminated_worlds, overflowed_worlds, estopped_worlds, finite_state_flag = (
        int(value) for value in summary.detach().cpu().tolist()
    )
    report = StabilityGateReport(
        requested_duration_seconds=config.stability_gate.duration_seconds,
        simulated_duration_seconds=policy_steps * action_dt,
        policy_steps=policy_steps,
        num_worlds=batch.num_worlds,
        terminated_worlds=terminated_worlds,
        overflowed_worlds=overflowed_worlds,
        estopped_worlds=estopped_worlds,
        finite_state=bool(finite_state_flag),
    )
    task.reset()
    if (
        (config.stability_gate.require_no_terminated and report.terminated_worlds)
        or (config.stability_gate.require_no_overflow and report.overflowed_worlds)
        or report.estopped_worlds
        or (config.stability_gate.require_finite_state and not report.finite_state)
    ):
        raise WarpFlatPpoConfigError(
            "fixed-gain nominal stability gate failed: "
            f"terminated={report.terminated_worlds}, overflowed={report.overflowed_worlds}, "
            f"estopped={report.estopped_worlds}, finite_state={report.finite_state}, "
            f"duration={report.simulated_duration_seconds:.6f}s"
        )
    return report


def _controller_calibration_metadata(calibration: Any) -> dict[str, Any]:
    payload = asdict(calibration)
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in payload.items()
    }


def _checkpoint_payload(
    *,
    policy: Any,
    optimizer: Any,
    timesteps: int,
    update_index: int,
    config: FlatPpoTrainingConfig,
    batch: WarpPhysicsBatch,
    calibration: Any,
    gate_report: StabilityGateReport,
    smoke: bool,
) -> dict[str, Any]:
    torch = batch._torch
    return {
        "format_version": FLAT_PPO_CHECKPOINT_FORMAT,
        "checkpoint_backend": FLAT_PPO_BACKEND,
        "algorithm": "ppo",
        "action_semantics": "seven_dimensional_fixed_gain_residual",
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "timesteps": int(timesteps),
        "update_index": int(update_index),
        "smoke": bool(smoke),
        "task_scope": {
            "task_mode": config.scope.task_mode,
            "terrain_enabled": False,
            "jump_enabled": False,
            "domain_randomization_enabled": False,
            "command_speed_mps": 0.0,
            "command_yaw_rate_rad_s": 0.0,
            "flat_terrain_features_zeroed": True,
            "jump_features_zeroed": True,
        },
        "policy_action_mask": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        "controller_scope": {
            "backend": FIXED_GAIN_CONTROLLER_BACKEND,
            "calibration_cpu_once": True,
            "dynamic_lqr_enabled": False,
            "seventh_leg_channel_enabled": False,
            "torque_fraction_of_rated": batch.config.safety.torque_fraction_of_rated,
            "flat_controller_config": asdict(config.flat_controller),
        },
        "stability_gate": {
            "requirements": asdict(config.stability_gate),
            "report": gate_report.as_dict(),
        },
        "flat_controller_calibration": _controller_calibration_metadata(calibration),
        "flat_walking_config": dict(config.flat_walking),
        "ppo_config": asdict(config.ppo),
        "experiment_config": {
            "source_path": str(config.source_path),
            "source_digest": config.source_digest,
            "batch_config_path": str(config.batch_config_path),
            "xml_path": str(batch.config.xml_path),
            "num_worlds": batch.num_worlds,
            "physics_substeps_per_action": batch.config.physics_substeps_per_action,
            "mujoco_version": batch._mujoco.__version__,
            "mujoco_warp_version": batch._mujoco_warp.__version__,
            "warp_version": batch._warp.__version__,
            "torch_version": torch.__version__,
            "device": str(batch.device),
        },
        "model_state_dict": {key: value.detach().clone() for key, value in policy.state_dict().items()},
        "actor_state_dict": {key: value.detach().clone() for key, value in policy.actor.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        },
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _append_metrics(path: Path, row: dict[str, float | int]) -> None:
    fields = (
        "update", "timesteps", "rollout_world_steps", "mean_reward", "completed_episodes",
        "mean_completed_return", "partial_mean_return", "policy_loss", "value_loss", "entropy",
        "aggregate_world_steps_per_second",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="ascii") as handle:
            existing = tuple(next(csv.reader(handle), []))
        if existing != fields:
            raise WarpFlatPpoConfigError(
                f"metrics schema mismatch at {path}; select a new fixed_gain_v2 metrics path"
            )
    with path.open("a", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_flat_training(config: FlatPpoTrainingConfig, *, smoke: bool) -> None:
    """Calibrate, gate, then run the explicit fixed-gain flat PPO scope."""

    from train_ppo import ActorCritic
    from warp_flat_controller import FixedGainFlatController, calibrate_flat_controller
    from warp_ppo import WarpPPOCollector, update_policy_cuda
    from warp_task import WarpFlatWalkingConfig, WarpFlatWalkingTask

    batch_config = load_warp_batch_config(config.batch_config_path)
    task_config = WarpFlatWalkingConfig.from_mapping(config.flat_walking)
    _validate_batch_for_flat_training(batch_config, config, task_config)

    # CPU calibration is one-time and completes before the CUDA batch exists.
    calibration = calibrate_flat_controller(batch_config, config.flat_controller)
    batch = WarpPhysicsBatch(batch_config)
    try:
        task = WarpFlatWalkingTask(
            batch,
            task_config,
            calibration=calibration.to_task_calibration(),
        )
        controller = FixedGainFlatController(calibration, task, config.flat_controller)
        task.set_feedback_controller(controller)
        gate_report = _run_zero_residual_stability_gate(batch, task, config)
        print(
            "Fixed-gain flat stability gate passed: "
            f"worlds={gate_report.num_worlds} duration={gate_report.simulated_duration_seconds:.6f}s "
            f"steps={gate_report.policy_steps} terminated={gate_report.terminated_worlds} "
            f"overflowed={gate_report.overflowed_worlds}"
        )

        torch = batch._torch
        torch.manual_seed(config.ppo.seed)
        torch.cuda.manual_seed_all(config.ppo.seed)
        policy = ActorCritic(
            OBSERVATION_SIZE,
            ACTION_SIZE,
            hidden_size=config.ppo.hidden_size,
            initial_action_std=config.ppo.initial_action_std,
        ).to(batch.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo.learning_rate)
        collector = WarpPPOCollector(task, policy, gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda)
        if smoke:
            rollout_steps = config.smoke.rollout_steps
            epochs = config.smoke.epochs
            minibatch_size = config.smoke.minibatch_size
            output_path = config.smoke.checkpoint_path
            metrics_path = config.smoke.metrics_path
            total_timesteps = rollout_steps * batch.num_worlds
            max_updates = 1
        else:
            rollout_steps = config.ppo.rollout_steps
            epochs = config.ppo.epochs
            minibatch_size = config.ppo.minibatch_size
            output_path = config.output.checkpoint_path
            metrics_path = config.output.metrics_path
            total_timesteps = config.ppo.total_timesteps
            max_updates = None

        observations = task.reset()
        episode_returns = torch.zeros(batch.num_worlds, dtype=torch.float32, device=batch.device)
        timesteps = 0
        update_index = 0
        while timesteps < total_timesteps:
            remaining_world_steps = (total_timesteps - timesteps) // batch.num_worlds
            current_rollout_steps = min(rollout_steps, remaining_world_steps)
            if current_rollout_steps < 1:
                raise RuntimeError("flat PPO loop reached a fractional vector update")
            started = perf_counter()
            rollout, observations, episode_returns = collector.collect(
                current_rollout_steps,
                observations=observations,
                episode_returns=episode_returns,
            )
            metrics = update_policy_cuda(
                policy,
                optimizer,
                rollout,
                epochs=epochs,
                minibatch_size=minibatch_size,
                clip_ratio=config.ppo.clip_ratio,
                value_coefficient=config.ppo.value_coefficient,
                entropy_coefficient=config.ppo.entropy_coefficient,
                max_gradient_norm=config.ppo.max_gradient_norm,
            )
            torch.cuda.synchronize(batch.device)
            elapsed = perf_counter() - started
            timesteps += current_rollout_steps * batch.num_worlds
            update_index += 1
            done = rollout.continuation_masks.eq(0.0)
            completed_count = int(done.sum().detach().cpu().item())
            mean_completed_return = (
                float(rollout.completed_episode_returns[done].mean().detach().cpu().item())
                if completed_count else float("nan")
            )
            mean_reward = float(rollout.rewards.mean().detach().cpu().item())
            partial_mean_return = float(episode_returns.mean().detach().cpu().item())
            aggregate_rate = (
                current_rollout_steps * batch.num_worlds / elapsed if elapsed > 0.0 else float("inf")
            )
            _append_metrics(metrics_path, {
                "update": update_index,
                "timesteps": timesteps,
                "rollout_world_steps": current_rollout_steps * batch.num_worlds,
                "mean_reward": mean_reward,
                "completed_episodes": completed_count,
                "mean_completed_return": mean_completed_return,
                "partial_mean_return": partial_mean_return,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "aggregate_world_steps_per_second": aggregate_rate,
            })
            payload = _checkpoint_payload(
                policy=policy,
                optimizer=optimizer,
                timesteps=timesteps,
                update_index=update_index,
                config=config,
                batch=batch,
                calibration=calibration,
                gate_report=gate_report,
                smoke=smoke,
            )
            final_update = timesteps == total_timesteps
            checkpoint_due = smoke or final_update or (update_index % config.output.checkpoint_interval_updates == 0)
            if checkpoint_due:
                _atomic_torch_save(payload, output_path, torch)
            print(
                "Warp fixed-gain flat PPO "
                f"update={update_index} timesteps={timesteps} mean_reward={mean_reward:.4f} "
                f"policy_loss={metrics['policy_loss']:.5f} value_loss={metrics['value_loss']:.5f} "
                f"entropy={metrics['entropy']:.5f} completed={completed_count} "
                f"steps_per_second={aggregate_rate:.1f} checkpoint={checkpoint_due}"
            )
            if max_updates is not None and update_index >= max_updates:
                break
        if smoke and update_index != 1:
            raise RuntimeError("fixed-gain PPO smoke must execute exactly one PPO update")
        print(
            "Warp fixed-gain flat PPO complete: "
            f"backend={FLAT_PPO_BACKEND} updates={update_index} timesteps={timesteps} checkpoint={output_path}"
        )
    finally:
        batch._safe_controls.zero_()
        batch._warp.copy(batch.data.ctrl, batch._safe_controls_warp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo-Warp physics preflight or fixed-gain flat CUDA PPO.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/warp_batch_preflight.yaml"),
        help="Physics preflight YAML used when --train is absent.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/warp_flat_ppo.yaml"),
        help="Strict v2 fixed-gain flat PPO experiment YAML used with --train.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fixed-gain flat PPO after the 128-world zero-residual stability gate.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Without --train run physics preflight; with --train run one gated PPO update.",
    )
    return parser.parse_args()


def _run_preflight(config_path: Path) -> None:
    try:
        report = run_warp_preflight(load_warp_batch_config(config_path))
    except WarpBatchError as error:
        raise SystemExit(f"MuJoCo-Warp batch preflight failed: {error}") from error
    print(
        "GPU batch physics ready for controller-parity work: "
        f"worlds={report.num_worlds}, physics_steps={report.physics_steps}, "
        f"aggregate_steps_per_second={report.aggregate_steps_per_second:.1f}, "
        f"terminated={report.terminated_worlds}, overflowed={report.overflowed_worlds}, "
        f"parity=(qpos:{report.parity_qpos_max_abs_error:.3e}, "
        f"qvel:{report.parity_qvel_max_abs_error:.3e}, "
        f"sensor:{report.parity_sensordata_max_abs_error:.3e}), "
        f"estop_probe={report.estop_probe_passed}."
    )
    if report.terminated_worlds or report.overflowed_worlds or not report.finite_state:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.train:
        try:
            _run_flat_training(load_flat_ppo_training_config(args.train_config), smoke=args.smoke)
        except (WarpFlatPpoConfigError, WarpBatchError, RuntimeError, ValueError) as error:
            raise SystemExit(f"MuJoCo-Warp fixed-gain flat PPO failed: {error}") from error
        return
    _run_preflight(args.config)


if __name__ == "__main__":
    main()
