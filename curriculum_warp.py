"""Capability-gated CUDA factory for the flat RMUC DR curriculum stage.

This module deliberately publishes one and only one GPU curriculum capability:
``rmuc_flat_dr``.  The stage keeps the proven flat support surface and
fixed-gain residual controller, while applying per-world vehicle/sensor/action
randomization through resident MuJoCo-Warp and Torch buffers.  It must not be
used as evidence that grades, steps, jumps, dog-hole traversal, or turning
have GPU parity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


STAGE_ID = "rmuc_flat_dr"
FLAT_TASK_MODE = "flat_vehicle_domain_randomization"
FIXED_GAIN_BACKEND = "fixed_gain_flat_controller_v2"
OBSERVATION_SIZE = 67
ACTION_SIZE = 7
REWARD_SCHEMA = "warp_flat_terrain_compensated_reward_v2"
_EXPECTED_ACTION_MASK = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
_PHYSICAL_DR_FIELDS = (
    "body_mass",
    "body_inertia",
    "dof_damping",
    "geom_friction",
    "actuator_strength",
)


class WarpCurriculumStageError(ValueError):
    """Raised before a GPU curriculum stage can violate its declared scope."""


@dataclass(frozen=True)
class ResolvedGpuTaskSettings:
    """Validated rollout settings read from the curriculum YAML manifest."""

    sensor_noise_std: float
    control_delay_steps: int
    stability_gate_seconds: float
    command_speed_gain_nm_per_mps: float
    command_yaw_gain_nm_per_rad_s: float
    command_wheel_feedforward_limit_nm: float


@dataclass
class WarpCurriculumStageBundle:
    """Objects consumed by :mod:`train_warp_curriculum` without CPU rollouts."""

    batch: Any
    task: Any
    controller: Any
    run_stability_gate: Callable[[], Mapping[str, Any]]
    close: Callable[[], None]


# ``train_warp_curriculum._coerce_capability`` accepts mappings, avoiding a
# circular import of its dataclass while this module is dynamically discovered.
GPU_CURRICULUM_CAPABILITIES: dict[str, dict[str, Any]] = {
    STAGE_ID: {
        "backend": FIXED_GAIN_BACKEND,
        "terrain": False,
        "steps": False,
        "jump": False,
        "domain_randomization": True,
        "speed_command": False,
        "yaw_command": False,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "reward_schema": REWARD_SCHEMA,
    }
}


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        if name not in source:
            raise WarpCurriculumStageError(f"missing required curriculum field: {name}")
        return source[name]
    if not hasattr(source, name):
        raise WarpCurriculumStageError(f"missing required curriculum field: {name}")
    return getattr(source, name)


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarpCurriculumStageError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise WarpCurriculumStageError(f"{name} must be {qualifier}")
    return result


def _default_gpu_task_mapping(config: Any) -> Mapping[str, Any]:
    """Read conservative defaults from YAML when a legacy config lacks them."""

    source = getattr(config, "source_path", None)
    candidates: list[Path] = []
    if source is not None:
        candidates.append(Path(source))
    candidates.append(Path(__file__).resolve().parent / "configs" / "warp_curriculum_ppo.yaml")
    for path in candidates:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(raw, Mapping) and isinstance(raw.get("gpu_task"), Mapping):
            return raw["gpu_task"]
    raise WarpCurriculumStageError("gpu_task settings are missing and no YAML defaults are readable")


def _resolve_gpu_task_settings(config: Any) -> ResolvedGpuTaskSettings:
    """Use parsed settings when available, otherwise the manifest's YAML block."""

    settings = getattr(config, "gpu_task", None)
    if settings is None:
        settings = _default_gpu_task_mapping(config)
    try:
        sensor_noise = _finite(_value(settings, "sensor_noise_std"), "gpu_task.sensor_noise_std", nonnegative=True)
        delay = _value(settings, "control_delay_steps")
        gate_seconds = _finite(_value(settings, "stability_gate_seconds"), "gpu_task.stability_gate_seconds")
        speed_gain = _finite(
            _value(settings, "command_speed_gain_nm_per_mps"),
            "gpu_task.command_speed_gain_nm_per_mps",
            nonnegative=True,
        )
        yaw_gain = _finite(
            _value(settings, "command_yaw_gain_nm_per_rad_s"),
            "gpu_task.command_yaw_gain_nm_per_rad_s",
            nonnegative=True,
        )
        wheel_limit = _finite(
            _value(settings, "command_wheel_feedforward_limit_nm"),
            "gpu_task.command_wheel_feedforward_limit_nm",
            nonnegative=True,
        )
    except WarpCurriculumStageError:
        raise
    if sensor_noise > 0.10:
        raise WarpCurriculumStageError("gpu_task.sensor_noise_std must be within [0, 0.10]")
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0 or delay > 2:
        raise WarpCurriculumStageError("gpu_task.control_delay_steps must be an integer within [0, 2]")
    if gate_seconds <= 0.0:
        raise WarpCurriculumStageError("gpu_task.stability_gate_seconds must be positive")
    return ResolvedGpuTaskSettings(
        sensor_noise_std=sensor_noise,
        control_delay_steps=int(delay),
        stability_gate_seconds=gate_seconds,
        command_speed_gain_nm_per_mps=speed_gain,
        command_yaw_gain_nm_per_rad_s=yaw_gain,
        command_wheel_feedforward_limit_nm=wheel_limit,
    )


def _validate_stage_contract(stage: Any) -> None:
    """Refuse every feature that lacks flat CUDA parity before allocation."""

    if _value(stage, "stage_id") != STAGE_ID:
        raise WarpCurriculumStageError(f"{STAGE_ID} factory cannot build stage {_value(stage, 'stage_id')!r}")
    if _value(stage, "task_mode") != FLAT_TASK_MODE:
        raise WarpCurriculumStageError(f"{STAGE_ID} must use task_mode={FLAT_TASK_MODE!r}")
    if _value(stage, "controller_backend") != FIXED_GAIN_BACKEND:
        raise WarpCurriculumStageError(f"{STAGE_ID} requires controller_backend={FIXED_GAIN_BACKEND!r}")
    expected_bools = {
        "terrain_enabled": False,
        "steps_enabled": False,
        "jump_enabled": False,
        "domain_randomization_enabled": True,
        "requires_gpu_parity": True,
    }
    for name, expected in expected_bools.items():
        if _value(stage, name) is not expected:
            raise WarpCurriculumStageError(f"{STAGE_ID}.{name} must be {str(expected).lower()}")
    if abs(_finite(_value(stage, "command_speed_mps"), "command_speed_mps")) > 1.0e-12:
        raise WarpCurriculumStageError(f"{STAGE_ID} accepts only a zero speed command")
    if abs(_finite(_value(stage, "command_yaw_rate_rad_s"), "command_yaw_rate_rad_s")) > 1.0e-12:
        raise WarpCurriculumStageError(f"{STAGE_ID} accepts only a zero yaw command")
    mask = _value(stage, "residual_action_mask")
    if not isinstance(mask, (tuple, list)) or len(mask) != ACTION_SIZE:
        raise WarpCurriculumStageError(f"{STAGE_ID} residual_action_mask must contain seven values")
    try:
        normalized_mask = tuple(float(value) for value in mask)
    except (TypeError, ValueError) as error:
        raise WarpCurriculumStageError(f"{STAGE_ID} residual_action_mask must be numeric") from error
    if any(not math.isfinite(value) for value in normalized_mask) or normalized_mask != _EXPECTED_ACTION_MASK:
        raise WarpCurriculumStageError(
            f"{STAGE_ID} must retain six residual actuator channels and mask channel seven"
        )


def _validate_batch_contract(batch_config: Any, stage: Any) -> None:
    """Validate mutable DR dimensions while keeping geometry immutable."""

    stage_xml = Path(_value(stage, "xml_path")).resolve()
    if Path(batch_config.xml_path).resolve() != stage_xml:
        raise WarpCurriculumStageError(f"{STAGE_ID} batch XML must match the stage XML")
    if not batch_config.domain_randomization.enabled:
        raise WarpCurriculumStageError(f"{STAGE_ID} requires domain_randomization.enabled=true")
    if batch_config.domain_randomization.terrain_geometry_randomization:
        raise WarpCurriculumStageError("terrain geometry randomization must remain disabled on the flat GPU stage")
    if batch_config.domain_randomization.delay.steps != 0:
        raise WarpCurriculumStageError(
            "physics-parameter DR delay must be zero until a reset-boundary scheduler is parity-validated"
        )
    if batch_config.safety.torque_fraction_of_rated > 0.80:
        raise WarpCurriculumStageError("batch torque_fraction_of_rated cannot exceed 0.80")
    ranges = batch_config.domain_randomization.ranges
    nonzero_ranges = 0
    for field in _PHYSICAL_DR_FIELDS:
        value = getattr(ranges, field)
        if not isinstance(value, tuple) or len(value) != 2:
            raise WarpCurriculumStageError(f"domain randomization range {field} is malformed")
        if value != (0.0, 0.0):
            nonzero_ranges += 1
    if nonzero_ranges == 0 and batch_config.domain_randomization.noise.std == 0.0:
        raise WarpCurriculumStageError(f"{STAGE_ID} requires at least one non-zero physical DR dimension")


def _make_stability_gate(batch: Any, task: Any, duration_seconds: float) -> Callable[[], Mapping[str, Any]]:
    """Build a nominal randomized gate that performs one host decision boundary."""

    if duration_seconds > float(task.config.episode_seconds) + 1.0e-9:
        raise WarpCurriculumStageError(
            "gpu_task.stability_gate_seconds cannot exceed the flat task episode_seconds"
        )
    action_dt = float(task._time_step)
    if not math.isfinite(action_dt) or action_dt <= 0.0:
        raise WarpCurriculumStageError("flat task action timestep must be finite and positive")
    cache: dict[str, Mapping[str, Any]] = {}

    def run() -> Mapping[str, Any]:
        if "report" in cache:
            return cache["report"]
        torch = batch._torch
        policy_steps = max(1, int(math.ceil(duration_seconds / action_dt)))
        zero_action = torch.zeros(
            (batch.num_worlds, ACTION_SIZE), dtype=torch.float32, device=batch.device
        )
        terminated_seen = torch.zeros(batch.num_worlds, dtype=torch.bool, device=batch.device)
        overflow_seen = torch.zeros_like(terminated_seen)
        estopped_seen = torch.zeros_like(terminated_seen)
        finite_reward = torch.ones((), dtype=torch.bool, device=batch.device)
        finite_reward_terms = torch.ones((), dtype=torch.bool, device=batch.device)
        finite_reward_values = torch.empty_like(terminated_seen)
        finite_reward_step = torch.empty((), dtype=torch.bool, device=batch.device)
        finite_reward_terms_step = torch.empty((), dtype=torch.bool, device=batch.device)
        task.reset()
        for _ in range(policy_steps):
            result = task.step(zero_action)
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
            terminated_seen.logical_or_(result.terminated)
            overflow_seen.logical_or_(batch.overflow.ne(0))
            estopped_seen.logical_or_(batch.estopped)
        finite_state = torch.isfinite(batch.qpos).all() & torch.isfinite(batch.qvel).all()
        summary = torch.stack((
            terminated_seen.sum(dtype=torch.int64),
            overflow_seen.sum(dtype=torch.int64),
            estopped_seen.sum(dtype=torch.int64),
            finite_state.to(dtype=torch.int64),
            finite_reward.to(dtype=torch.int64),
            finite_reward_terms.to(dtype=torch.int64),
        ))
        torch.cuda.synchronize(batch.device)
        terminated_worlds, overflowed_worlds, estopped_worlds, finite_state_flag, finite_reward_flag, finite_reward_terms_flag = (
            int(value) for value in summary.detach().cpu().tolist()
        )
        report: Mapping[str, Any] = {
            "stage_id": STAGE_ID,
            "requested_duration_seconds": duration_seconds,
            "simulated_duration_seconds": policy_steps * action_dt,
            "policy_steps": policy_steps,
            "num_worlds": int(batch.num_worlds),
            "terminated_worlds": terminated_worlds,
            "overflowed_worlds": overflowed_worlds,
            "estopped_worlds": estopped_worlds,
            "finite_state": bool(finite_state_flag),
            "finite_reward": bool(finite_reward_flag),
            "finite_reward_terms": bool(finite_reward_terms_flag),
            "zero_residual": True,
            "domain_randomization_enabled": True,
        }
        # Give the collector a fresh episode and freshly sampled per-world DR
        # profiles after the gate.  This is a CUDA reset-boundary operation,
        # not a per-substep model mutation.
        task.reset()
        if (
            terminated_worlds
            or overflowed_worlds
            or estopped_worlds
            or not finite_state_flag
            or not finite_reward_flag
            or not finite_reward_terms_flag
        ):
            raise WarpCurriculumStageError(
                "rmuc_flat_dr nominal stability gate failed: "
                f"terminated={terminated_worlds}, overflowed={overflowed_worlds}, "
                f"estopped={estopped_worlds}, finite_state={bool(finite_state_flag)}, "
                f"finite_reward={bool(finite_reward_flag)}, finite_reward_terms={bool(finite_reward_terms_flag)}, "
                f"duration={policy_steps * action_dt:.6f}s"
            )
        cache["report"] = report
        return report

    return run


def _make_close(batch: Any) -> Callable[[], None]:
    """Provide an idempotent independent estop/torque-clear close operation."""

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


def build_curriculum_stage(stage: Any, config: Any) -> WarpCurriculumStageBundle:
    """Allocate the parity-gated ``rmuc_flat_dr`` CUDA residual task.

    CPU LQR calibration occurs once before the MuJoCo-Warp model/data are
    allocated.  Thereafter the task, controller, randomization, rewards, and
    safety checks operate on resident CUDA tensors.
    """

    _validate_stage_contract(stage)
    settings = _resolve_gpu_task_settings(config)

    from train_warp_ppo import load_flat_ppo_training_config
    from warp_env import WarpPhysicsBatch, load_warp_batch_config
    from warp_flat_controller import FixedGainFlatController, calibrate_flat_controller
    from warp_task import WarpFlatWalkingConfig, WarpFlatWalkingTask

    flat = load_flat_ppo_training_config(_value(config, "flat_ppo_config_path"))
    batch_config = load_warp_batch_config(_value(config, "batch_config_path"))
    _validate_batch_contract(batch_config, stage)
    if flat.flat_controller.max_torque_fraction > batch_config.safety.torque_fraction_of_rated + 1.0e-9:
        raise WarpCurriculumStageError("flat controller torque limit exceeds the batch 80% derating")

    task_base = WarpFlatWalkingConfig.from_mapping(flat.flat_walking)
    task_config = replace(
        task_base,
        command_speed_mps=0.0,
        command_yaw_rate_rad_s=0.0,
        domain_randomization_enabled=True,
        sensor_noise_std=settings.sensor_noise_std,
        control_delay_steps=settings.control_delay_steps,
        domain_randomization_seed=int(batch_config.domain_randomization.seed) + 1,
    )
    controller_config = replace(
        flat.flat_controller,
        command_speed_mps=0.0,
        command_yaw_rate_rad_s=0.0,
        # This stage is a zero-command robustness exercise.  A wheel
        # command-feedforward term would turn passive calibration drift into
        # an active drive command and can trip the attitude guard before PPO
        # starts.  Non-zero route stages retain their YAML-owned feedforward.
        command_wheel_feedforward_limit_nm=0.0,
        command_wheel_accel_limit_nm=0.0,
        command_wheel_brake_limit_nm=0.0,
        command_speed_gain_nm_per_mps=settings.command_speed_gain_nm_per_mps,
        command_yaw_gain_nm_per_rad_s=settings.command_yaw_gain_nm_per_rad_s,
    )

    # Calibration intentionally happens before GPU allocation and never runs
    # inside reset, gate, controller, or physics loops.
    calibration = calibrate_flat_controller(batch_config, controller_config)
    batch = None
    try:
        batch = WarpPhysicsBatch(batch_config)
        task = WarpFlatWalkingTask(batch, task_config, calibration=calibration.to_task_calibration())
        controller = FixedGainFlatController(calibration, task, controller_config)
        task.set_feedback_controller(controller)
        return WarpCurriculumStageBundle(
            batch=batch,
            task=task,
            controller=controller,
            run_stability_gate=_make_stability_gate(batch, task, settings.stability_gate_seconds),
            close=_make_close(batch),
        )
    except Exception:
        if batch is not None:
            _make_close(batch)()
        raise


__all__ = [
    "ACTION_SIZE",
    "FIXED_GAIN_BACKEND",
    "FLAT_TASK_MODE",
    "GPU_CURRICULUM_CAPABILITIES",
    "OBSERVATION_SIZE",
    "REWARD_SCHEMA",
    "ResolvedGpuTaskSettings",
    "STAGE_ID",
    "WarpCurriculumStageBundle",
    "WarpCurriculumStageError",
    "build_curriculum_stage",
]
