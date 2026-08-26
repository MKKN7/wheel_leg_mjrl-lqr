"""Evaluate a residual PPO checkpoint against the LQR baseline in an MJCF scene."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from env import DEFAULT_EPISODE_SECONDS, DomainRandomizationConfig, LocomotionCommand, WheelLegResidualEnv
import lqr_deploy as lqr
from policy_runtime import PpoPolicyRuntime, environment_compatibility_warnings, load_ppo_residual_policy
from terrain_curriculum import (
    TerrainCurriculumConfig,
    TerrainCurriculumError,
    TerrainCurriculumStage,
    load_terrain_curriculum,
    validate_scene_contract,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIRECTORY = ROOT.parent / "artifacts"
DEFAULT_XML_PATH = ROOT / "rm_train_ground.xml"


@dataclass(frozen=True)
class EvaluationTask:
    name: str
    speed_mps: float
    yaw_rate_rad_s: float
    jump_at_s: float | None
    terrain_task_id: str | None = None
    terrain_route_index: int | None = None
    terrain_route_id: str | None = None


@dataclass(frozen=True)
class EpisodeResult:
    controller: str
    task: str
    seed: int
    episode_return: float
    duration_s: float
    terminated: bool
    physically_safe: bool
    task_completed: bool
    task_succeeded: bool
    safety_reason: str | None
    speed_mae_mps: float
    yaw_mae_rad: float
    mean_residual_norm: float
    jump_succeeded: bool
    jump_landing_stable: bool
    jump_peak_mean_clearance_m: float
    jump_peak_min_clearance_m: float
    contact_recovery_count: int
    max_contact_loss_duration_s: float
    terrain_route: str | None = None
    terrain_progress_m: float = float("nan")
    terrain_required_distance_m: float = float("nan")


@dataclass(frozen=True)
class TerrainRouteGate:
    """One route-level acceptance result for a scene-bound terrain stage."""

    task_id: str
    route_id: str
    episodes: int
    completion_rate: float
    physical_unsafe_rate: float
    mean_speed_mae_mps: float
    mean_yaw_mae_rad: float
    maximum_success_streak: int
    required_success_streak: int
    passed: bool


def minimum_terrain_gate_episodes(
    curriculum: TerrainCurriculumConfig,
    stage: TerrainCurriculumStage,
) -> int:
    """Return enough cyclic rollouts to exercise every required success streak."""

    template_count = sum(
        len(curriculum.task(task_id).routes)
        for task_id in stage.task_ids
    )
    return template_count * max(
        curriculum.task(task_id).required_consecutive_successes
        for task_id in stage.task_ids
    )


def default_checkpoint_path() -> Path:
    """Prefer the highest saved training snapshot over a possibly overwritten latest file."""
    snapshots = list(CHECKPOINT_DIRECTORY.glob("ppo_locomotion_controller_step_*.pt"))
    if snapshots:
        return max(snapshots, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    return CHECKPOINT_DIRECTORY / "ppo_locomotion_controller.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a residual PPO checkpoint with zero-residual LQR on an MJCF scene."
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=None,
        help="MJCF scene; defaults to the scene bound by the selected curriculum.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Total evaluation episodes. Defaults to 4 for ordinary scenarios and to the "
            "selected curriculum stage's evaluation_episodes for terrain evaluation."
        ),
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--scenario",
        choices=("mixed", "walk", "turn", "jump"),
        default="mixed",
        help="Evaluation task family; mixed cycles walk, left turn, right turn and jump.",
    )
    parser.add_argument("--speed", type=float, default=0.25, help="Walking command magnitude in m/s.")
    parser.add_argument(
        "--turn-rate",
        type=float,
        default=0.20,
        help="Absolute high-level yaw-rate command for turn tasks in rad/s.",
    )
    parser.add_argument("--jump-at", type=float, default=0.80)
    parser.add_argument(
        "--terrain-curriculum",
        type=Path,
        default=None,
        help=(
            "Strict scene-bound terrain curriculum YAML. Each enabled task/route is reset "
            "independently; routes remain evaluation metadata, not policy inputs."
        ),
    )
    parser.add_argument(
        "--terrain-stage",
        default=None,
        help="Curriculum stage ID to evaluate; requires --terrain-curriculum.",
    )
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help=(
            "Return exit code 1 unless every enabled terrain route satisfies the stage's "
            "unsafe, completion, and consecutive-success requirements."
        ),
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Randomize vehicle dynamics, sensing and delay while keeping terrain geometry and friction fixed.",
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
    if not args.checkpoint.is_file():
        parser.error(f"--checkpoint does not exist: {args.checkpoint}")
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    if not np.isfinite(args.speed):
        parser.error("--speed must be finite")
    if not np.isfinite(args.turn_rate) or not 0.0 <= args.turn_rate <= lqr.MAX_YAW_RATE_RAD_S:
        parser.error(f"--turn-rate must be within 0..{lqr.MAX_YAW_RATE_RAD_S:.2f}")
    if not np.isfinite(args.jump_at) or not 0.0 <= args.jump_at < DEFAULT_EPISODE_SECONDS:
        parser.error(f"--jump-at must be within 0..{DEFAULT_EPISODE_SECONDS:g}")
    if (args.terrain_curriculum is None) != (args.terrain_stage is None):
        parser.error("--terrain-curriculum and --terrain-stage must be provided together")
    if args.require_gate and args.terrain_curriculum is None:
        parser.error("--require-gate requires --terrain-curriculum and --terrain-stage")
    args.terrain_curriculum_config: TerrainCurriculumConfig | None = None
    if args.terrain_curriculum is not None:
        args.terrain_curriculum = args.terrain_curriculum.resolve()
        try:
            args.terrain_curriculum_config = load_terrain_curriculum(args.terrain_curriculum)
            stage = args.terrain_curriculum_config.stage(args.terrain_stage)
        except (FileNotFoundError, RuntimeError, TerrainCurriculumError, KeyError) as error:
            parser.error(str(error))
        if args.xml_path is None:
            scene_filename = (
                args.terrain_curriculum_config.scene_contract.mjcf_filename
                if args.terrain_curriculum_config.scene_contract is not None
                else DEFAULT_XML_PATH.name
            )
            args.xml_path = ROOT / scene_filename
        if args.episodes is None:
            args.episodes = stage.evaluation_episodes
        route_count = sum(
            len(args.terrain_curriculum_config.task(task_id).routes)
            for task_id in stage.task_ids
        )
        if args.episodes < route_count:
            parser.error(
                f"--episodes must be at least {route_count} so every enabled terrain task/route "
                "is evaluated at least once"
            )
        if args.require_gate:
            required_episodes = minimum_terrain_gate_episodes(
                args.terrain_curriculum_config,
                stage,
            )
            if args.episodes < required_episodes:
                parser.error(
                    f"--require-gate needs at least {required_episodes} episodes for the "
                    "stage's per-route consecutive-success requirements"
                )
    elif args.episodes is None:
        args.episodes = 4
    if args.xml_path is None:
        args.xml_path = DEFAULT_XML_PATH
    args.xml_path = args.xml_path.expanduser().resolve()
    if not args.xml_path.is_file():
        parser.error(f"--xml-path does not exist: {args.xml_path}")
    if args.terrain_curriculum_config is not None:
        try:
            validate_scene_contract(
                args.terrain_curriculum_config,
                args.xml_path,
                curriculum_path=args.terrain_curriculum,
            )
        except TerrainCurriculumError as error:
            parser.error(str(error))
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


def checkpoint_yaw_rate_limit(runtime: PpoPolicyRuntime) -> float:
    task_config = runtime.metadata.get("task_config")
    if not isinstance(task_config, dict):
        return lqr.MAX_YAW_RATE_RAD_S
    value = task_config.get("max_yaw_rate_rad_s", lqr.MAX_YAW_RATE_RAD_S)
    try:
        return float(np.clip(float(value), 0.0, lqr.MAX_YAW_RATE_RAD_S))
    except (TypeError, ValueError):
        return lqr.MAX_YAW_RATE_RAD_S


def terrain_curriculum_compatibility_warnings(
    runtime: PpoPolicyRuntime,
    args: argparse.Namespace,
) -> list[str]:
    """Describe a curriculum/stage mismatch without blocking transfer tests."""
    curriculum = args.terrain_curriculum_config
    if curriculum is None:
        return []
    task_config = runtime.metadata.get("task_config")
    saved = task_config.get("terrain_curriculum") if isinstance(task_config, dict) else None
    if not isinstance(saved, dict):
        return [
            "checkpoint has no RMUC terrain-curriculum identity; results are a cross-curriculum transfer evaluation"
        ]
    expected = {
        "yaml_sha256": hashlib.sha256(args.terrain_curriculum.read_bytes()).hexdigest(),
        "schema_version": int(curriculum.schema_version),
        "stage_id": str(args.terrain_stage),
        "stage_task_ids": list(curriculum.stage(args.terrain_stage).task_ids),
        "stage_gate": {
            "maximum_unsafe_rate": curriculum.stage(args.terrain_stage).maximum_unsafe_rate,
            "minimum_completion_rate": curriculum.stage(args.terrain_stage).minimum_completion_rate,
            "maximum_speed_mae_mps": curriculum.stage(args.terrain_stage).maximum_speed_mae_mps,
            "maximum_yaw_mae_rad": curriculum.stage(args.terrain_stage).maximum_yaw_mae_rad,
        },
    }
    if curriculum.scene_contract is not None:
        expected["scene_id"] = curriculum.scene_id
        expected["scene_contract"] = {
            "mjcf_filename": curriculum.scene_contract.mjcf_filename,
            "mjcf_model": curriculum.scene_contract.mjcf_model,
            "terrain_spec_filename": curriculum.scene_contract.terrain_spec_filename,
            "support_geoms": list(curriculum.scene_contract.support_geoms),
            "obstacle_geoms": list(curriculum.scene_contract.obstacle_geoms),
        }
    mismatches = [
        key for key, value in expected.items()
        if saved.get(key) != value
    ]
    if not mismatches:
        return []
    return [
        "terrain curriculum differs from the checkpoint "
        f"({', '.join(mismatches)}); results are a cross-stage/scene transfer evaluation"
    ]


def build_terrain_tasks(args: argparse.Namespace) -> list[EvaluationTask]:
    """Build a balanced fixed-route terrain evaluation set.

    A route selects a deterministic reset pose and defines completion
    bookkeeping only.  The policy still receives only the environment's
    proprioceptive observation and the scaled speed/yaw-rate/jump command.
    """
    curriculum = args.terrain_curriculum_config
    if curriculum is None:
        raise RuntimeError("terrain tasks requested without a terrain curriculum")
    stage = curriculum.stage(args.terrain_stage)
    templates: list[EvaluationTask] = []
    for task_id in stage.task_ids:
        task = curriculum.task(task_id)
        command = stage.command_for(task)
        for route_index, route in enumerate(task.routes):
            templates.append(EvaluationTask(
                name=task.task_id,
                speed_mps=command.forward_speed_mps,
                yaw_rate_rad_s=command.yaw_rate_rad_s,
                jump_at_s=None,
                terrain_task_id=task.task_id,
                terrain_route_index=route_index,
                terrain_route_id=route.route_id,
            ))
    if args.episodes < len(templates):
        raise ValueError(
            f"--episodes={args.episodes} is insufficient for terrain stage {stage.stage_id!r}; "
            f"at least {len(templates)} rollouts are required to reset every enabled task/route"
        )
    return [templates[index % len(templates)] for index in range(args.episodes)]


def build_tasks(
    args: argparse.Namespace,
    speed_limit_mps: float,
    yaw_rate_limit_rad_s: float,
) -> list[EvaluationTask]:
    if args.terrain_curriculum_config is not None:
        return build_terrain_tasks(args)
    speed = min(abs(float(args.speed)), speed_limit_mps)
    turn_speed = min(speed, 0.35)
    turn_rate = min(abs(float(args.turn_rate)), yaw_rate_limit_rad_s)
    jump_speed = min(speed, 0.08)
    templates = {
        "walk": (EvaluationTask("walk", speed, 0.0, None),),
        "turn": (
            EvaluationTask("turn_left", turn_speed, turn_rate, None),
            EvaluationTask("turn_right", turn_speed, -turn_rate, None),
        ),
        "jump": (EvaluationTask("jump", jump_speed, 0.0, args.jump_at),),
        "mixed": (
            EvaluationTask("walk", speed, 0.0, None),
            EvaluationTask("turn_left", turn_speed, turn_rate, None),
            EvaluationTask("turn_right", turn_speed, -turn_rate, None),
            EvaluationTask("jump", jump_speed, 0.0, args.jump_at),
        ),
    }
    selected = templates[args.scenario]
    return [selected[index % len(selected)] for index in range(args.episodes)]


def episode_seconds_for_args(args: argparse.Namespace) -> float:
    """Honor long YAML horizons for slow grade and step evaluation tasks."""
    curriculum = args.terrain_curriculum_config
    if curriculum is None:
        return float(DEFAULT_EPISODE_SECONDS)
    return max(
        float(DEFAULT_EPISODE_SECONDS),
        float(curriculum.stage_max_episode_seconds(args.terrain_stage)),
    )


def make_environment(
    args: argparse.Namespace,
    speed_limit_mps: float,
    yaw_rate_limit_rad_s: float = lqr.MAX_YAW_RATE_RAD_S,
) -> WheelLegResidualEnv:
    terrain_curriculum = args.terrain_curriculum_config
    terrain_stage_id = args.terrain_stage if terrain_curriculum is not None else None
    if terrain_curriculum is not None:
        # The environment validates all declared curriculum commands, not
        # just the selected stage.  Keep the physical command bounds wide
        # enough for that validation while the stage itself supplies the
        # actual high-level command at reset.
        speed_limit_mps = max(
            speed_limit_mps,
            terrain_curriculum.limits.max_forward_speed_mps,
        )
        yaw_rate_limit_rad_s = max(
            yaw_rate_limit_rad_s,
            terrain_curriculum.limits.max_yaw_rate_rad_s,
        )
        if yaw_rate_limit_rad_s > lqr.MAX_YAW_RATE_RAD_S + 1e-9:
            raise ValueError(
                "terrain curriculum yaw-rate limit exceeds the physical LQR yaw-rate limit"
            )
    domain_randomization = (
        DomainRandomizationConfig.terrain_vehicle_only_defaults()
        if args.domain_randomization and terrain_curriculum is not None
        else (
            DomainRandomizationConfig.vehicle_only_defaults()
            if args.domain_randomization
            else DomainRandomizationConfig.disabled()
        )
    )
    jump_domain_randomization = (
        DomainRandomizationConfig.jump_vehicle_only_defaults()
        if args.domain_randomization
        else None
    )
    return WheelLegResidualEnv(
        xml_path=args.xml_path,
        episode_seconds=episode_seconds_for_args(args),
        randomize_command=False,
        randomize_leg_length=False,
        max_forward_speed=speed_limit_mps,
        command_speed_limit_mps=speed_limit_mps,
        max_command_yaw_delta_rad=0.0,
        max_command_yaw_rate_rad_s=yaw_rate_limit_rad_s,
        jump_probability=0.0,
        jump_at=args.jump_at,
        domain_randomization=domain_randomization,
        jump_domain_randomization=jump_domain_randomization,
        terrain_curriculum=terrain_curriculum,
        terrain_stage_id=terrain_stage_id,
        terrain_evaluation=terrain_curriculum is not None,
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
    reset_options: dict[str, object] = {"command_leg_length": 0.244}
    if task.terrain_task_id is None:
        reset_options.update({
            "locomotion_command": LocomotionCommand(task.speed_mps, task.yaw_rate_rad_s),
            "jump_at": task.jump_at_s,
        })
    else:
        # The curriculum owns the high-level command.  Passing only this
        # fixed reset selector keeps the route out of policy observations and
        # avoids turning the evaluator into a navigation layer.
        reset_options.update({
            "terrain_task_id": task.terrain_task_id,
            "terrain_route_index": task.terrain_route_index,
        })
    observation, _ = environment.reset(seed=seed, options=reset_options)
    zero_action = np.zeros(environment.action_space.shape, dtype=np.float32)
    episode_return = 0.0
    speed_errors: list[float] = []
    yaw_errors: list[float] = []
    residual_norms: list[float] = []
    max_contact_loss_duration_s = 0.0
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
        max_contact_loss_duration_s = max(
            max_contact_loss_duration_s,
            float(info.get("contact_loss_duration_s", 0.0)),
        )
        if bool(info.get("policy_action_applied", True)):
            speed_errors.append(abs(
                float(info["forward_speed_mps"]) - float(info["ramped_command_speed_mps"])
            ))
            yaw_errors.append(abs(float(info["yaw_error_rad"])))
            residual_norms.append(float(np.linalg.norm(info.get("effective_policy_action", action))))

    physically_safe = not bool(final_info.get("physical_unsafe", terminated))
    if task.terrain_task_id is None:
        task_completed = bool(truncated or final_info.get("jump_succeeded", False))
    else:
        completion_marker = final_info.get("terrain_task_completed")
        task_completed = (
            bool(completion_marker)
            if completion_marker is not None
            else bool(truncated and not final_info.get("terrain_task_failed", False))
        )
    return EpisodeResult(
        controller="ppo" if runtime is not None else "lqr_zero_residual",
        task=task.name,
        seed=seed,
        episode_return=episode_return,
        duration_s=float(environment.data.time),
        terminated=terminated,
        physically_safe=physically_safe,
        task_completed=task_completed,
        task_succeeded=bool(physically_safe and task_completed),
        safety_reason=None if final_info.get("safety_reason") is None else str(final_info["safety_reason"]),
        speed_mae_mps=float(np.mean(speed_errors)) if speed_errors else float("nan"),
        yaw_mae_rad=float(np.mean(yaw_errors)) if yaw_errors else float("nan"),
        mean_residual_norm=float(np.mean(residual_norms)) if residual_norms else 0.0,
        jump_succeeded=bool(final_info.get("jump_succeeded", False)),
        jump_landing_stable=bool(final_info.get("jump_landing_stable", False)),
        jump_peak_mean_clearance_m=float(final_info.get("jump_peak_mean_wheel_clearance_m", 0.0)),
        jump_peak_min_clearance_m=float(final_info.get("jump_peak_wheel_clearance_m", 0.0)),
        contact_recovery_count=int(final_info.get("contact_recovery_count", 0)),
        max_contact_loss_duration_s=max_contact_loss_duration_s,
        terrain_route=task.terrain_route_id,
        terrain_progress_m=float(final_info.get("terrain_progress_m", float("nan"))),
        terrain_required_distance_m=float(
            final_info.get("terrain_required_distance_m", float("nan"))
        ),
    )


def evaluate_controller(
    args: argparse.Namespace,
    tasks: list[EvaluationTask],
    speed_limit_mps: float,
    yaw_rate_limit_rad_s: float,
    runtime: PpoPolicyRuntime | None,
) -> list[EpisodeResult]:
    environment = make_environment(args, speed_limit_mps, yaw_rate_limit_rad_s)
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


def maximum_success_streak(results: list[EpisodeResult]) -> int:
    """Return the longest consecutive sequence of safe, completed rollouts."""

    longest = 0
    current = 0
    for result in results:
        if result.task_succeeded:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def terrain_route_gates(
    results: list[EpisodeResult],
    curriculum: TerrainCurriculumConfig,
    stage: TerrainCurriculumStage,
) -> tuple[TerrainRouteGate, ...]:
    """Evaluate every declared route independently, including lab success streaks."""

    grouped: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        if result.terrain_route is not None:
            grouped[(result.task, result.terrain_route)].append(result)

    reports: list[TerrainRouteGate] = []
    for task_id in stage.task_ids:
        task = curriculum.task(task_id)
        for route in task.routes:
            route_results = grouped.get((task_id, route.route_id), [])
            episodes = len(route_results)
            completion_rate = (
                float(np.mean([result.task_completed for result in route_results]))
                if route_results
                else 0.0
            )
            unsafe_rate = (
                float(np.mean([not result.physically_safe for result in route_results]))
                if route_results
                else 1.0
            )
            speed_mae = (
                float(np.nanmean([result.speed_mae_mps for result in route_results]))
                if route_results
                else float("nan")
            )
            yaw_mae = (
                float(np.nanmean([result.yaw_mae_rad for result in route_results]))
                if route_results
                else float("nan")
            )
            streak = maximum_success_streak(route_results)
            speed_tracking_passed = (
                stage.maximum_speed_mae_mps is None
                or (
                    np.isfinite(speed_mae)
                    and speed_mae <= stage.maximum_speed_mae_mps + 1e-12
                )
            )
            yaw_tracking_passed = (
                stage.maximum_yaw_mae_rad is None
                or (
                    np.isfinite(yaw_mae)
                    and yaw_mae <= stage.maximum_yaw_mae_rad + 1e-12
                )
            )
            passed = bool(
                episodes > 0
                and unsafe_rate <= stage.maximum_unsafe_rate + 1e-12
                and completion_rate >= stage.minimum_completion_rate - 1e-12
                and speed_tracking_passed
                and yaw_tracking_passed
                and streak >= task.required_consecutive_successes
            )
            reports.append(
                TerrainRouteGate(
                    task_id=task_id,
                    route_id=route.route_id,
                    episodes=episodes,
                    completion_rate=completion_rate,
                    physical_unsafe_rate=unsafe_rate,
                    mean_speed_mae_mps=speed_mae,
                    mean_yaw_mae_rad=yaw_mae,
                    maximum_success_streak=streak,
                    required_success_streak=task.required_consecutive_successes,
                    passed=passed,
                )
            )
    return tuple(reports)


def print_summary(
    label: str,
    results: list[EpisodeResult],
    *,
    maximum_unsafe_rate: float | None = None,
    minimum_completion_rate: float | None = None,
    route_gates: tuple[TerrainRouteGate, ...] = (),
) -> None:
    print(f"\n{label}")
    show_episode_rows = len(results) <= 32
    if show_episode_rows:
        print(
            "task/route                  safe  complete success duration  return      speed_mae  "
            "yaw_mae_deg  residual_norm  jump_peak_mean/min  landing   recovery"
        )
    else:
        print(f"{len(results)} episodes; per-episode rows suppressed, reporting route/task aggregates.")
    for result in (results if show_episode_rows else ()):
        safe = "yes" if result.physically_safe else "no"
        completed = "yes" if result.task_completed else "no"
        succeeded = "yes" if result.task_succeeded else "no"
        yaw_mae_deg = math.degrees(result.yaw_mae_rad) if np.isfinite(result.yaw_mae_rad) else float("nan")
        landing = "-"
        if result.task == "jump":
            landing = "success" if result.jump_succeeded else (
                "stable" if result.jump_landing_stable else "failed"
            )
        route_label = result.task if result.terrain_route is None else f"{result.task}/{result.terrain_route}"
        print(
            f"{route_label:<27} {safe:<4} {completed:<8} {succeeded:<7} {result.duration_s:7.3f}s "
            f"{result.episode_return:9.2f} {result.speed_mae_mps:10.3f} "
            f"{yaw_mae_deg:11.2f} {result.mean_residual_norm:14.3f} "
            f"{result.jump_peak_mean_clearance_m:6.3f}/{result.jump_peak_min_clearance_m:6.3f} "
            f"{landing:<8} {result.contact_recovery_count:2d}/"
            f"{1000.0 * result.max_contact_loss_duration_s:4.0f}ms"
        )
        if result.terrain_route is not None and np.isfinite(result.terrain_progress_m):
            required = (
                f"/{result.terrain_required_distance_m:.3f}m"
                if np.isfinite(result.terrain_required_distance_m)
                else ""
            )
            print(f"  terrain progress: {result.terrain_progress_m:.3f}m{required}")
        if result.safety_reason is not None:
            reason_label = "unsafe" if not result.physically_safe else "task failure"
            print(f"  {reason_label}: {result.safety_reason}")

    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.task].append(result)
    route_grouped: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        if result.terrain_route is not None:
            route_grouped[(result.task, result.terrain_route)].append(result)
    route_gate_index = {
        (gate.task_id, gate.route_id): gate
        for gate in route_gates
    }

    def print_group(
        prefix: str,
        group_results: list[EpisodeResult],
        route_gate: TerrainRouteGate | None = None,
    ) -> None:
        completed_rate = float(np.mean([result.task_completed for result in group_results]))
        success_rate = float(np.mean([result.task_succeeded for result in group_results]))
        physical_unsafe_rate = float(np.mean([
            not result.physically_safe for result in group_results
        ]))
        mean_return = float(np.mean([result.episode_return for result in group_results]))
        speed_mae = float(np.nanmean([result.speed_mae_mps for result in group_results]))
        yaw_mae = float(np.degrees(np.nanmean([result.yaw_mae_rad for result in group_results])))
        recovery_rate = float(np.mean([
            result.contact_recovery_count > 0 for result in group_results
        ]))
        gate = ""
        if route_gate is not None:
            gate = (
                f" gate={'PASS' if route_gate.passed else 'FAIL'}"
                f" consecutive={route_gate.maximum_success_streak}/"
                f"{route_gate.required_success_streak}"
            )
            if maximum_unsafe_rate is not None and minimum_completion_rate is not None:
                tracking = (
                    f" gate_speed_mae={route_gate.mean_speed_mae_mps:.3f}m/s"
                    f" gate_yaw_mae={math.degrees(route_gate.mean_yaw_mae_rad):.2f}deg"
                )
                gate += tracking
        elif (
            not route_gates
            and maximum_unsafe_rate is not None
            and minimum_completion_rate is not None
        ):
            passed = (
                physical_unsafe_rate <= maximum_unsafe_rate + 1e-12
                and completed_rate >= minimum_completion_rate - 1e-12
            )
            gate = f" gate={'PASS' if passed else 'FAIL'}"
        print(
            f"  {prefix}: completed={completed_rate:.2%} task_success={success_rate:.2%} "
            f"physical_unsafe={physical_unsafe_rate:.2%} mean_return={mean_return:.2f} "
            f"speed_mae={speed_mae:.3f}m/s yaw_mae={yaw_mae:.2f}deg "
            f"contact_recovery_rate={recovery_rate:.2%}{gate}"
        )
        unsafe_reasons = Counter(
            result.safety_reason
            for result in group_results
            if result.safety_reason is not None and not result.physically_safe
        )
        if unsafe_reasons:
            print("    unsafe reasons: " + ", ".join(
                f"{reason}={count}" for reason, count in unsafe_reasons.items()
            ))
        task_failures = Counter(
            result.safety_reason
            for result in group_results
            if result.safety_reason is not None and result.physically_safe and not result.task_completed
        )
        if task_failures:
            print("    task failure reasons: " + ", ".join(
                f"{reason}={count}" for reason, count in task_failures.items()
            ))

    if route_grouped:
        print("route summary")
        for (task, route), route_results in route_grouped.items():
            print_group(
                f"{task}/{route}",
                route_results,
                route_gate_index.get((task, route)),
            )
    print("task summary")
    for task, task_results in grouped.items():
        print_group(task, task_results)


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
    yaw_rate_limit_rad_s = checkpoint_yaw_rate_limit(runtime)
    print(
        f"Loaded PPO checkpoint: {runtime.checkpoint_path} "
        f"timesteps={runtime.timesteps} obs/action=({runtime.observation_size}, {runtime.action_size})"
    )
    warnings = environment_compatibility_warnings(
        runtime,
        xml_path=args.xml_path,
        lqr_source_path=Path(lqr.__file__),
    )
    warnings.extend(terrain_curriculum_compatibility_warnings(runtime, args))
    for warning in warnings:
        print(f"Compatibility warning: {warning}")
    if not warnings:
        print("Compatibility: exact MJCF and LQR fingerprints match the checkpoint.")
    if args.domain_randomization:
        print("Evaluation domain randomization: vehicle-only; terrain geometry and friction are fixed")
    else:
        print("Evaluation domain randomization: disabled (deterministic nominal terrain test)")
    tasks = build_tasks(args, speed_limit_mps, yaw_rate_limit_rad_s)
    if args.terrain_curriculum_config is not None:
        curriculum = args.terrain_curriculum_config
        stage = curriculum.stage(args.terrain_stage)
        max_stage_speed = max(
            abs(stage.command_for(curriculum.task(task_id)).forward_speed_mps)
            for task_id in stage.task_ids
        )
        max_stage_yaw_rate = max(
            abs(stage.command_for(curriculum.task(task_id)).yaw_rate_rad_s)
            for task_id in stage.task_ids
        )
        print(
            f"Terrain curriculum: {curriculum.name} stage={stage.stage_id} "
            f"episodes={len(tasks)} target_unsafe<={stage.maximum_unsafe_rate:.0%} "
            f"target_completion>={stage.minimum_completion_rate:.0%}"
        )
        if stage.maximum_speed_mae_mps is not None or stage.maximum_yaw_mae_rad is not None:
            speed_target = (
                "unbounded"
                if stage.maximum_speed_mae_mps is None
                else f"<={stage.maximum_speed_mae_mps:.3f}m/s"
            )
            yaw_target = (
                "unbounded"
                if stage.maximum_yaw_mae_rad is None
                else f"<={math.degrees(stage.maximum_yaw_mae_rad):.2f}deg"
            )
            print(f"Stage tracking gates: speed_mae{speed_target} yaw_mae{yaw_target}")
        print(
            f"Stage high-level limits: speed<={max_stage_speed:.3f}m/s "
            f"yaw_rate<={max_stage_yaw_rate:.3f}rad/s; fixed route endpoints are reset/evaluation metadata only."
        )
        if max_stage_speed > speed_limit_mps + 1e-9:
            print(
                "Compatibility warning: selected terrain command speed exceeds the checkpoint's "
                "recorded training speed limit; this is a transfer evaluation."
            )
        if max_stage_yaw_rate > yaw_rate_limit_rad_s + 1e-9:
            print(
                "Compatibility warning: selected terrain yaw-rate exceeds the checkpoint's "
                "recorded training yaw-rate limit; this is a transfer evaluation."
            )
        terrain_cases = list(dict.fromkeys(
            f"{task.name}/{task.terrain_route_id}" for task in tasks
        ))
        print("Terrain cases: " + ", ".join(terrain_cases))
    else:
        print("Tasks: " + ", ".join(task.name for task in tasks))
    print(
        "Physical unsafe rate counts falls/invalid contact states only; corridor and deadline misses "
        "remain separate task failures."
    )

    ppo_results = evaluate_controller(args, tasks, speed_limit_mps, yaw_rate_limit_rad_s, runtime)
    stage = (
        None
        if args.terrain_curriculum_config is None
        else args.terrain_curriculum_config.stage(args.terrain_stage)
    )
    ppo_route_gates = (
        ()
        if stage is None
        else terrain_route_gates(ppo_results, args.terrain_curriculum_config, stage)
    )
    print_summary(
        "PPO residual policy",
        ppo_results,
        maximum_unsafe_rate=None if stage is None else stage.maximum_unsafe_rate,
        minimum_completion_rate=None if stage is None else stage.minimum_completion_rate,
        route_gates=ppo_route_gates,
    )
    ppo_gate_passed = bool(ppo_route_gates) and all(gate.passed for gate in ppo_route_gates)
    if stage is not None:
        print(f"PPO terrain acceptance gate: {'PASS' if ppo_gate_passed else 'FAIL'}")
    if args.compare_baseline:
        baseline_results = evaluate_controller(args, tasks, speed_limit_mps, yaw_rate_limit_rad_s, None)
        print_summary(
            "LQR zero-residual baseline",
            baseline_results,
            maximum_unsafe_rate=None if stage is None else stage.maximum_unsafe_rate,
            minimum_completion_rate=None if stage is None else stage.minimum_completion_rate,
            route_gates=(
                ()
                if stage is None
                else terrain_route_gates(baseline_results, args.terrain_curriculum_config, stage)
            ),
        )
    if any(task.jump_at_s is not None for task in tasks):
        print(
            "Note: the LQR owns the common leg trajectory and wheel-speed loop during jumps; "
            "the policy retains only tightly bounded hip residual authority in contact phases."
        )
    if args.require_gate and not ppo_gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
