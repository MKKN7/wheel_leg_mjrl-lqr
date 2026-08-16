"""Evaluate a residual PPO checkpoint against the LQR baseline in an MJCF scene."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from env import DEFAULT_EPISODE_SECONDS, DomainRandomizationConfig, WheelLegResidualEnv
import lqr_deploy as lqr
from policy_runtime import PpoPolicyRuntime, environment_compatibility_warnings, load_ppo_residual_policy


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIRECTORY = ROOT.parent / "artifacts"
DEFAULT_XML_PATH = ROOT / "rm_train_ground.xml"


@dataclass(frozen=True)
class EvaluationTask:
    name: str
    speed_mps: float
    yaw_delta_rad: float
    jump_at_s: float | None


@dataclass(frozen=True)
class EpisodeResult:
    controller: str
    task: str
    seed: int
    episode_return: float
    duration_s: float
    terminated: bool
    safety_reason: str | None
    speed_mae_mps: float
    yaw_mae_rad: float
    mean_residual_norm: float
    jump_succeeded: bool
    jump_landing_stable: bool
    jump_peak_mean_clearance_m: float
    jump_peak_min_clearance_m: float


def default_checkpoint_path() -> Path:
    """Prefer the highest saved training snapshot over a possibly overwritten latest file."""
    snapshots = list(CHECKPOINT_DIRECTORY.glob("ppo_jump_clearance_step_*.pt"))
    if snapshots:
        return max(snapshots, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    return CHECKPOINT_DIRECTORY / "ppo_jump_clearance.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a residual PPO checkpoint with zero-residual LQR on an MJCF scene."
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--episodes", type=int, default=8, help="Total evaluation episodes.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--scenario",
        choices=("mixed", "walk", "turn", "jump"),
        default="mixed",
        help="Evaluation task family; mixed cycles walk, left turn, right turn and jump.",
    )
    parser.add_argument("--speed", type=float, default=0.50, help="Walking command magnitude in m/s.")
    parser.add_argument("--turn-deg", type=float, default=20.0, help="Magnitude of turn command in degrees.")
    parser.add_argument("--jump-at", type=float, default=0.80)
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Evaluate robustness under the configured walking/jump domain randomization.",
    )
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample the Gaussian policy instead of evaluating its deterministic mean action.",
    )
    parser.add_argument(
        "--no-baseline",
        dest="compare_baseline",
        action="store_false",
        help="Skip the matched zero-residual LQR baseline rollout.",
    )
    parser.set_defaults(compare_baseline=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.xml_path = args.xml_path.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"--checkpoint does not exist: {args.checkpoint}")
    if not args.xml_path.is_file():
        parser.error(f"--xml-path does not exist: {args.xml_path}")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not np.isfinite(args.speed):
        parser.error("--speed must be finite")
    if not np.isfinite(args.turn_deg) or not 0.0 <= args.turn_deg <= 180.0:
        parser.error("--turn-deg must be within 0..180")
    if not np.isfinite(args.jump_at) or not 0.0 <= args.jump_at < DEFAULT_EPISODE_SECONDS:
        parser.error(f"--jump-at must be within 0..{DEFAULT_EPISODE_SECONDS:g}")
    return args


def checkpoint_forward_speed_limit(runtime: PpoPolicyRuntime) -> float:
    task_config = runtime.metadata.get("task_config")
    if not isinstance(task_config, dict):
        return lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS
    value = task_config.get("max_speed_mps", lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS)
    try:
        return lqr.validate_forward_speed_limit(float(value))
    except (TypeError, ValueError):
        return lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS


def build_tasks(args: argparse.Namespace, speed_limit_mps: float) -> list[EvaluationTask]:
    speed = min(abs(float(args.speed)), speed_limit_mps)
    turn_speed = min(speed, 0.35)
    turn = math.radians(args.turn_deg)
    templates = {
        "walk": (EvaluationTask("walk", speed, 0.0, None),),
        "turn": (
            EvaluationTask("turn_left", turn_speed, turn, None),
            EvaluationTask("turn_right", turn_speed, -turn, None),
        ),
        "jump": (EvaluationTask("jump", 0.0, 0.0, args.jump_at),),
        "mixed": (
            EvaluationTask("walk", speed, 0.0, None),
            EvaluationTask("turn_left", turn_speed, turn, None),
            EvaluationTask("turn_right", turn_speed, -turn, None),
            EvaluationTask("jump", 0.0, 0.0, args.jump_at),
        ),
    }
    selected = templates[args.scenario]
    return [selected[index % len(selected)] for index in range(args.episodes)]


def make_environment(
    args: argparse.Namespace,
    speed_limit_mps: float,
) -> WheelLegResidualEnv:
    domain_randomization = (
        DomainRandomizationConfig.training_defaults()
        if args.domain_randomization
        else DomainRandomizationConfig.disabled()
    )
    jump_domain_randomization = (
        DomainRandomizationConfig.jump_training_defaults()
        if args.domain_randomization
        else None
    )
    return WheelLegResidualEnv(
        xml_path=args.xml_path,
        episode_seconds=DEFAULT_EPISODE_SECONDS,
        randomize_command=False,
        randomize_leg_length=False,
        max_forward_speed=speed_limit_mps,
        max_command_yaw_delta_rad=0.0,
        jump_probability=0.0,
        jump_at=args.jump_at,
        domain_randomization=domain_randomization,
        jump_domain_randomization=jump_domain_randomization,
    )


def environment_interface(args: argparse.Namespace) -> tuple[int, int]:
    """Read the selected scene's policy interface before loading a checkpoint."""
    environment = make_environment(args, lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS)
    try:
        return environment.observation_space.shape[0], environment.action_space.shape[0]
    finally:
        environment.close()


def evaluate_episode(
    environment: WheelLegResidualEnv,
    task: EvaluationTask,
    seed: int,
    runtime: PpoPolicyRuntime | None,
    *,
    deterministic: bool,
) -> EpisodeResult:
    observation, _ = environment.reset(
        seed=seed,
        options={
            "command_speed": task.speed_mps,
            "command_leg_length": 0.244,
            "command_yaw_delta_rad": task.yaw_delta_rad,
            "jump_at": task.jump_at_s,
        },
    )
    zero_action = np.zeros(environment.action_space.shape, dtype=np.float32)
    episode_return = 0.0
    speed_errors: list[float] = []
    yaw_errors: list[float] = []
    residual_norms: list[float] = []
    final_info: dict[str, object] = {}
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = (
            zero_action
            if runtime is None
            else runtime.action(observation, deterministic=deterministic)
        )
        observation, reward, terminated, truncated, info = environment.step(action)
        final_info = info
        episode_return += float(reward)
        if bool(info.get("policy_action_applied", True)):
            speed_errors.append(abs(
                float(info["forward_speed_mps"]) - float(info["ramped_command_speed_mps"])
            ))
            yaw_errors.append(abs(float(info["yaw_error_rad"])))
            residual_norms.append(float(np.linalg.norm(action)))

    return EpisodeResult(
        controller="ppo" if runtime is not None else "lqr_zero_residual",
        task=task.name,
        seed=seed,
        episode_return=episode_return,
        duration_s=float(environment.data.time),
        terminated=terminated,
        safety_reason=None if final_info.get("safety_reason") is None else str(final_info["safety_reason"]),
        speed_mae_mps=float(np.mean(speed_errors)) if speed_errors else float("nan"),
        yaw_mae_rad=float(np.mean(yaw_errors)) if yaw_errors else float("nan"),
        mean_residual_norm=float(np.mean(residual_norms)) if residual_norms else 0.0,
        jump_succeeded=bool(final_info.get("jump_succeeded", False)),
        jump_landing_stable=bool(final_info.get("jump_landing_stable", False)),
        jump_peak_mean_clearance_m=float(final_info.get("jump_peak_mean_wheel_clearance_m", 0.0)),
        jump_peak_min_clearance_m=float(final_info.get("jump_peak_wheel_clearance_m", 0.0)),
    )


def evaluate_controller(
    args: argparse.Namespace,
    tasks: list[EvaluationTask],
    speed_limit_mps: float,
    runtime: PpoPolicyRuntime | None,
) -> list[EpisodeResult]:
    environment = make_environment(args, speed_limit_mps)
    try:
        return [
            evaluate_episode(
                environment,
                task,
                args.seed + index,
                runtime,
                deterministic=not args.stochastic_policy,
            )
            for index, task in enumerate(tasks)
        ]
    finally:
        environment.close()


def print_summary(label: str, results: list[EpisodeResult]) -> None:
    print(f"\n{label}")
    print(
        "task          safe  duration  return      speed_mae  yaw_mae_deg  "
        "residual_norm  jump_peak_mean/min  landing"
    )
    for result in results:
        safe = "yes" if not result.terminated else "no"
        yaw_mae_deg = math.degrees(result.yaw_mae_rad) if np.isfinite(result.yaw_mae_rad) else float("nan")
        landing = "-"
        if result.task == "jump":
            landing = "success" if result.jump_succeeded else (
                "stable" if result.jump_landing_stable else "failed"
            )
        print(
            f"{result.task:<13} {safe:<4} {result.duration_s:7.3f}s "
            f"{result.episode_return:9.2f} {result.speed_mae_mps:10.3f} "
            f"{yaw_mae_deg:11.2f} {result.mean_residual_norm:14.3f} "
            f"{result.jump_peak_mean_clearance_m:6.3f}/{result.jump_peak_min_clearance_m:6.3f} "
            f"{landing}"
        )
        if result.safety_reason is not None:
            print(f"  unsafe: {result.safety_reason}")

    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.task].append(result)
    print("summary")
    for task, task_results in grouped.items():
        safe_rate = float(np.mean([not result.terminated for result in task_results]))
        mean_return = float(np.mean([result.episode_return for result in task_results]))
        speed_mae = float(np.nanmean([result.speed_mae_mps for result in task_results]))
        yaw_mae = float(np.degrees(np.nanmean([result.yaw_mae_rad for result in task_results])))
        print(
            f"  {task}: safe_rate={safe_rate:.2f} mean_return={mean_return:.2f} "
            f"speed_mae={speed_mae:.3f}m/s yaw_mae={yaw_mae:.2f}deg"
        )
    unsafe_reasons = Counter(result.safety_reason for result in results if result.safety_reason is not None)
    if unsafe_reasons:
        print("unsafe counts: " + ", ".join(
            f"{reason}={count}" for reason, count in unsafe_reasons.items()
        ))


def main() -> None:
    args = parse_args()
    observation_size, action_size = environment_interface(args)
    runtime = load_ppo_residual_policy(
        args.checkpoint,
        observation_size,
        action_size,
        device=args.device,
    )
    speed_limit_mps = checkpoint_forward_speed_limit(runtime)
    print(
        f"Loaded PPO checkpoint: {runtime.checkpoint_path} "
        f"timesteps={runtime.timesteps} obs/action=({runtime.observation_size}, {runtime.action_size})"
    )
    warnings = environment_compatibility_warnings(
        runtime,
        xml_path=args.xml_path,
        lqr_source_path=Path(lqr.__file__),
    )
    for warning in warnings:
        print(f"Compatibility warning: {warning}")
    if not warnings:
        print("Compatibility: exact MJCF and LQR fingerprints match the checkpoint.")
    if args.domain_randomization:
        print("Evaluation domain randomization: enabled")
    else:
        print("Evaluation domain randomization: disabled (deterministic nominal terrain test)")
    tasks = build_tasks(args, speed_limit_mps)
    print("Tasks: " + ", ".join(task.name for task in tasks))

    ppo_results = evaluate_controller(args, tasks, speed_limit_mps, runtime)
    print_summary("PPO residual policy", ppo_results)
    if args.compare_baseline:
        baseline_results = evaluate_controller(args, tasks, speed_limit_mps, None)
        print_summary("LQR zero-residual baseline", baseline_results)
    if any(task.jump_at_s is not None for task in tasks):
        print(
            "Note: during active jump phases the environment masks residual actions, so jump height and "
            "landing primarily evaluate the LQR jump controller and its pre/post-jump recovery."
        )


if __name__ == "__main__":
    main()
