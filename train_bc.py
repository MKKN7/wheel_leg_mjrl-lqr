"""Behaviour-cloning trainer for the safe PhysicalLqr residual interface.

The environment always executes ``PhysicalLqr.command`` and interprets policy
outputs as bounded residual corrections.  BC therefore learns the neutral
residual action, producing a checkpoint that is directly compatible with
``train_ppo.py --bc-checkpoint``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

import env as environment_definition
from env import DEFAULT_EPISODE_SECONDS, DomainRandomizationConfig, WheelLegResidualEnv
import lqr_deploy as lqr
from terrain_curriculum import (
    TerrainCurriculumConfig,
    TerrainCurriculumError,
    load_terrain_curriculum,
    validate_scene_contract,
)


CHECKPOINT_FORMAT_VERSION = 21
REWARD_SCHEMA = "command_tracking_jump_clearance_terrain_command_speed_v13"


class BehaviorCloningPolicy(nn.Module):
    """Standalone tanh actor with the same actor layout as the PPO policy."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return torch.tanh(self.actor(observations))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone safe command-conditioned PhysicalLqr residual demonstrations with PyTorch."
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--xml-path",
        type=Path,
        help=(
            "MJCF scene to train in; defaults to wheeled_infantry.xml, or "
            "the curriculum-bound scene when --terrain-curriculum is supplied."
        ),
    )
    parser.add_argument(
        "--terrain-curriculum",
        type=Path,
        help=(
            "Strict scene-bound terrain curriculum YAML. When supplied, --terrain-stage selects the "
            "fixed non-navigating task stage used at environment reset."
        ),
    )
    parser.add_argument(
        "--terrain-stage",
        help="Stage id from --terrain-curriculum; required when a terrain curriculum is selected.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=None,
        help=(
            "Optional forward speed limit in m/s; valid range is "
            f"{lqr.MIN_FORWARD_SPEED_LIMIT_MPS:.1f}..{lqr.MAX_FORWARD_SPEED_MPS:.1f}."
        ),
    )
    parser.add_argument(
        "--command-speed-limit",
        type=float,
        default=None,
        help="Maximum magnitude sampled for high-level speed commands; defaults to --max-speed.",
    )
    parser.add_argument(
        "--command-speed-fraction",
        type=float,
        default=environment_definition.RL_COMMAND_SPEED_FRACTION,
        help=(
            "Fraction of --command-speed-limit used by randomly sampled high-level commands "
            f"(0..1, default {environment_definition.RL_COMMAND_SPEED_FRACTION:g})."
        ),
    )
    parser.add_argument(
        "--command-resample-seconds",
        type=float,
        default=0.75,
        help="Seconds between sampled high-level speed/yaw-rate commands; 0 holds one command per episode.",
    )
    parser.add_argument(
        "--yaw-range-deg",
        type=float,
        default=0.0,
        help="Optional initial heading offset range in degrees; yaw-rate commands drive normal turn training.",
    )
    parser.add_argument(
        "--max-yaw-rate",
        type=float,
        default=lqr.MAX_YAW_RATE_RAD_S,
        help=f"Maximum absolute high-level yaw-rate command in rad/s (0..{lqr.MAX_YAW_RATE_RAD_S:.2f}).",
    )
    parser.add_argument(
        "--jump-probability",
        type=float,
        default=0.5,
        help="Fraction of episodes that execute the LQR jump sequence.",
    )
    parser.add_argument(
        "--jump-at",
        type=float,
        default=0.80,
        help="Simulation time in seconds at which a scheduled jump begins.",
    )
    parser.add_argument(
        "--domain-randomization",
        dest="domain_randomization",
        action="store_true",
        default=True,
        help="Randomize physical dynamics, sensors and control delay at every episode reset (default).",
    )
    parser.add_argument(
        "--no-domain-randomization",
        dest="domain_randomization",
        action="store_false",
        help="Disable dynamics/random-sensor randomization for a nominal-model run.",
    )
    parser.add_argument(
        "--vehicle-only-domain-randomization",
        action="store_true",
        help="Keep terrain geometry and friction fixed while randomizing robot dynamics, sensing and delay.",
    )
    parser.add_argument(
        "--target",
        choices=("residual",),
        default="residual",
        help="Only residual targets are valid for WheelLegResidualEnv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "bc_locomotion_controller.pt",
        help="BC checkpoint path for the high-level-command residual locomotion interface.",
    )
    parser.add_argument("--smoke", action="store_true", help="Collect a small dataset and run one optimization epoch.")
    args = parser.parse_args()
    if args.max_speed is not None:
        try:
            args.max_speed = lqr.validate_forward_speed_limit(args.max_speed)
        except ValueError as error:
            parser.error(str(error))
    physical_speed_limit = float(
        args.max_speed if args.max_speed is not None else lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS
    )
    if args.command_speed_limit is not None:
        if (
            not np.isfinite(args.command_speed_limit)
            or args.command_speed_limit <= 0.0
            or args.command_speed_limit > physical_speed_limit
        ):
            parser.error(
                "--command-speed-limit must be positive and no greater than the selected physical --max-speed"
            )
    if not np.isfinite(args.command_speed_fraction) or not 0.0 < args.command_speed_fraction <= 1.0:
        parser.error("--command-speed-fraction must be within (0, 1]")
    if not np.isfinite(args.command_resample_seconds) or args.command_resample_seconds < 0.0:
        parser.error("--command-resample-seconds must be finite and non-negative")
    if args.xml_path is not None:
        args.xml_path = args.xml_path.resolve()
        if not args.xml_path.is_file():
            parser.error(f"--xml-path does not exist: {args.xml_path}")
    if not 0.0 <= args.yaw_range_deg <= 180.0:
        parser.error("--yaw-range-deg must be within 0..180")
    if not 0.0 <= args.max_yaw_rate <= lqr.MAX_YAW_RATE_RAD_S:
        parser.error(f"--max-yaw-rate must be within 0..{lqr.MAX_YAW_RATE_RAD_S:.3f}")
    if not 0.0 <= args.jump_probability <= 1.0:
        parser.error("--jump-probability must be within 0..1")
    if not 0.0 <= args.jump_at < DEFAULT_EPISODE_SECONDS:
        parser.error(f"--jump-at must be within 0..{DEFAULT_EPISODE_SECONDS:g} seconds")
    if args.vehicle_only_domain_randomization and not args.domain_randomization:
        parser.error("--vehicle-only-domain-randomization requires --domain-randomization")
    resolve_terrain_curriculum_args(args, parser)
    return args


def command_speed_limit_for_args(args: argparse.Namespace) -> float:
    if args.command_speed_limit is not None:
        return float(args.command_speed_limit)
    if args.max_speed is not None:
        return float(args.max_speed)
    return float(lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS)


def command_resample_seconds_for_args(args: argparse.Namespace) -> float | None:
    return None if args.command_resample_seconds == 0.0 else float(args.command_resample_seconds)


def resolve_terrain_curriculum_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Load and validate an optional fixed-stage terrain curriculum before training starts."""
    if args.terrain_curriculum is None:
        if args.terrain_stage is not None:
            parser.error("--terrain-stage requires --terrain-curriculum")
        args.terrain_curriculum_config = None
        args.terrain_stage_id = None
        return

    curriculum_path = args.terrain_curriculum.expanduser().resolve()
    try:
        curriculum = load_terrain_curriculum(curriculum_path)
    except (FileNotFoundError, RuntimeError, TerrainCurriculumError) as error:
        parser.error(str(error))
    if args.terrain_stage is None:
        parser.error("--terrain-curriculum requires --terrain-stage")
    try:
        stage = curriculum.stage(args.terrain_stage)
    except KeyError:
        parser.error(
            f"unknown --terrain-stage {args.terrain_stage!r}; available stages: "
            + ", ".join(item.stage_id for item in curriculum.stages)
        )

    physical_speed_limit = float(
        args.max_speed if args.max_speed is not None else lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS
    )
    if curriculum.limits.max_forward_speed_mps > physical_speed_limit + 1e-9:
        parser.error(
            "terrain curriculum max_forward_speed_mps exceeds the selected physical --max-speed: "
            f"{curriculum.limits.max_forward_speed_mps:.3f} > {physical_speed_limit:.3f}"
        )
    if curriculum.limits.max_yaw_rate_rad_s > float(args.max_yaw_rate) + 1e-9:
        parser.error(
            "terrain curriculum max_yaw_rate_rad_s exceeds --max-yaw-rate: "
            f"{curriculum.limits.max_yaw_rate_rad_s:.3f} > {float(args.max_yaw_rate):.3f}"
        )

    command_speed_limit = command_speed_limit_for_args(args)
    for task_id in stage.task_ids:
        command = stage.command_for(curriculum.task(task_id))
        if abs(command.forward_speed_mps) > command_speed_limit + 1e-9:
            parser.error(
                f"terrain stage {stage.stage_id!r} task {task_id!r} command speed "
                f"{command.forward_speed_mps:.3f} exceeds --command-speed-limit "
                f"{command_speed_limit:.3f}"
            )

    args.terrain_curriculum = curriculum_path
    args.terrain_curriculum_config = curriculum
    args.terrain_stage_id = stage.stage_id
    if args.xml_path is None:
        scene_filename = (
            curriculum.scene_contract.mjcf_filename
            if curriculum.scene_contract is not None
            else "rm_train_ground.xml"
        )
        scene_xml_path = Path(__file__).with_name(scene_filename)
        if scene_xml_path.is_file():
            args.xml_path = scene_xml_path.resolve()
    if args.xml_path is None:
        parser.error("terrain curriculum scene XML could not be resolved")
    try:
        validate_scene_contract(
            curriculum,
            args.xml_path,
            curriculum_path=curriculum_path,
        )
    except TerrainCurriculumError as error:
        parser.error(str(error))


def terrain_curriculum_metadata_for_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """Stable curriculum identity embedded in PPO and BC compatibility metadata."""
    curriculum = getattr(args, "terrain_curriculum_config", None)
    stage_id = getattr(args, "terrain_stage_id", None)
    if curriculum is None:
        return None
    if not isinstance(curriculum, TerrainCurriculumConfig) or not isinstance(stage_id, str):
        raise ValueError("terrain curriculum arguments were not resolved before metadata generation")
    curriculum_path = getattr(args, "terrain_curriculum", None)
    if not isinstance(curriculum_path, Path):
        raise ValueError("terrain curriculum path is unavailable")
    stage = curriculum.stage(stage_id)
    metadata = {
        "yaml_sha256": hashlib.sha256(curriculum_path.read_bytes()).hexdigest(),
        "schema_version": int(curriculum.schema_version),
        "stage_id": stage.stage_id,
        "stage_task_ids": list(stage.task_ids),
        "stage_gate": {
            "maximum_unsafe_rate": stage.maximum_unsafe_rate,
            "minimum_completion_rate": stage.minimum_completion_rate,
            "maximum_speed_mae_mps": stage.maximum_speed_mae_mps,
            "maximum_yaw_mae_rad": stage.maximum_yaw_mae_rad,
        },
    }
    if curriculum.scene_contract is not None:
        metadata["scene_id"] = curriculum.scene_id
        metadata["scene_contract"] = {
            "mjcf_filename": curriculum.scene_contract.mjcf_filename,
            "mjcf_model": curriculum.scene_contract.mjcf_model,
            "terrain_spec_filename": curriculum.scene_contract.terrain_spec_filename,
            "support_geoms": list(curriculum.scene_contract.support_geoms),
            "obstacle_geoms": list(curriculum.scene_contract.obstacle_geoms),
        }
    return metadata


def episode_seconds_for_args(args: argparse.Namespace) -> float:
    """Use the YAML horizon so slow fixed terrain tasks are not truncated at 8 s."""
    curriculum = getattr(args, "terrain_curriculum_config", None)
    stage_id = getattr(args, "terrain_stage_id", None)
    if curriculum is None:
        return float(DEFAULT_EPISODE_SECONDS)
    if not isinstance(curriculum, TerrainCurriculumConfig) or not isinstance(stage_id, str):
        raise ValueError("terrain curriculum arguments were not resolved before episode duration selection")
    return max(float(DEFAULT_EPISODE_SECONDS), float(curriculum.stage_max_episode_seconds(stage_id)))


def domain_randomization_for_args(
    args: argparse.Namespace,
) -> tuple[DomainRandomizationConfig, DomainRandomizationConfig | None]:
    if not args.domain_randomization:
        return DomainRandomizationConfig.disabled(), None
    if getattr(args, "terrain_curriculum_config", None) is not None:
        walking = DomainRandomizationConfig.terrain_vehicle_only_defaults()
        jumping = DomainRandomizationConfig.jump_vehicle_only_defaults()
        return walking, jumping
    elif args.vehicle_only_domain_randomization:
        walking = DomainRandomizationConfig.vehicle_only_defaults()
        jumping = DomainRandomizationConfig.jump_vehicle_only_defaults()
    else:
        walking = DomainRandomizationConfig.training_defaults()
        jumping = DomainRandomizationConfig.jump_training_defaults()
    return walking, jumping if args.jump_probability > 0.0 else None


def collect_demonstrations(
    environment: WheelLegResidualEnv,
    samples: int,
    target_kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if target_kind != "residual":
        raise ValueError("only residual demonstrations are valid for this environment")
    rng = np.random.default_rng(seed)
    observation, _ = environment.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    zero_residual = np.zeros(environment.action_space.shape, dtype=np.float32)
    for _ in range(samples):
        observations.append(observation.copy())
        actions.append(environment.expert_action())
        observation, _, terminated, truncated, info = environment.step(zero_residual)
        action_masks.append(np.asarray(
            info.get("policy_action_mask", environment.policy_action_mask()),
            dtype=np.float32,
        ).copy())
        if terminated or truncated:
            if environment.terrain_curriculum is None:
                speed = environment.sample_command_speed(rng)
                observation, _ = environment.reset(options={"command_speed": speed})
            else:
                observation, _ = environment.reset()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(action_masks, dtype=np.float32),
    )


def train_bc(
    observations: np.ndarray,
    expert_actions: np.ndarray,
    action_masks: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> BehaviorCloningPolicy:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    policy = BehaviorCloningPolicy(observations.shape[1], expert_actions.shape[1]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(expert_actions, dtype=torch.float32, device=device)
    action_mask_tensor = torch.as_tensor(action_masks, dtype=torch.float32, device=device)
    if action_mask_tensor.shape != target_tensor.shape:
        raise ValueError("action_masks must match expert_actions shape")
    sample_count = observation_tensor.shape[0]

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(sample_count, device=device)
        losses: list[float] = []
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size]
            prediction = policy(observation_tensor[indices])
            mask = action_mask_tensor[indices]
            loss = ((prediction - target_tensor[indices]).square() * mask).sum() / mask.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} bc_mse={np.mean(losses):.6f}")
    return policy


def save_policy(
    path: Path,
    policy: BehaviorCloningPolicy,
    observation_size: int,
    action_size: int,
    target_kind: str,
    task_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "algorithm": "behavior_cloning",
            "target_kind": target_kind,
            "observation_size": observation_size,
            "action_size": action_size,
            "task_config": task_config,
            "actor_state_dict": policy.actor.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.samples = min(args.samples, 32)
        args.epochs = 1
        args.batch_size = min(args.batch_size, args.samples)
        args.output = Path("artifacts") / f"bc_locomotion_controller_{args.target}_smoke.pt"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    xml_path = (lqr.XML_PATH if args.xml_path is None else args.xml_path).resolve()
    domain_randomization, jump_domain_randomization = domain_randomization_for_args(args)
    terrain_curriculum = terrain_curriculum_metadata_for_args(args)
    task_config = {
        "max_speed_mps": float(
            args.max_speed if args.max_speed is not None else lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS
        ),
        "command_speed_limit_mps": command_speed_limit_for_args(args),
        "command_speed_fraction": float(args.command_speed_fraction),
        "command_resample_seconds": float(args.command_resample_seconds),
        "yaw_range_deg": float(args.yaw_range_deg),
        "max_yaw_rate_rad_s": float(args.max_yaw_rate),
        "jump_probability": float(args.jump_probability),
        "jump_at_s": float(args.jump_at),
        "terrain_curriculum": terrain_curriculum,
        "locomotion_command_conditioning": environment_definition.locomotion_command_conditioning_config(),
        "locomotion_command_schema": environment_definition.LOCOMOTION_COMMAND_SCHEMA,
        "residual_authority_schema": environment_definition.RESIDUAL_AUTHORITY_SCHEMA,
        "reward_schema": REWARD_SCHEMA,
        "reward_config": {
            "jump_target_clearance_m": float(environment_definition.JUMP_TARGET_CLEARANCE_M),
            "jump_max_reward_clearance_m": float(environment_definition.JUMP_MAX_REWARD_CLEARANCE_M),
            "jump_min_clearance_success_m": float(environment_definition.JUMP_MIN_CLEARANCE_SUCCESS_M),
            "jump_peak_clearance_scale_m": float(environment_definition.JUMP_PEAK_CLEARANCE_SCALE_M),
            "jump_peak_clearance_weight": float(environment_definition.JUMP_HEIGHT_REWARD_WEIGHT),
            "jump_success_reward": float(environment_definition.JUMP_SUCCESS_REWARD),
            "jump_abort_penalty": float(environment_definition.JUMP_ABORT_PENALTY),
            "jump_landing_fall_penalty": float(environment_definition.JUMP_LANDING_FALL_PENALTY),
            "jump_landing_guard_seconds": float(environment_definition.JUMP_LANDING_GUARD_SECONDS),
            "jump_stable_attitude_limit_rad": float(
                environment_definition.JUMP_STABLE_ATTITUDE_LIMIT_RAD
            ),
            "jump_stable_vertical_speed_mps": float(
                environment_definition.JUMP_STABLE_VERTICAL_SPEED_MPS
            ),
            "jump_stable_angular_speed_rad_s": float(
                environment_definition.JUMP_STABLE_ANGULAR_SPEED_RAD_S
            ),
            "terrain_progress_reward_per_m": float(
                environment_definition.TERRAIN_PROGRESS_REWARD_PER_M
            ),
            "terrain_completion_reward": float(environment_definition.TERRAIN_COMPLETION_REWARD),
            "terrain_corridor_penalty_per_m": float(
                environment_definition.TERRAIN_CORRIDOR_PENALTY_PER_M
            ),
            "terrain_dense_reward_rate_scale": float(
                environment_definition.TERRAIN_DENSE_REWARD_RATE_SCALE
            ),
            "terrain_task_timeout_penalty": float(
                environment_definition.TERRAIN_TASK_TIMEOUT_PENALTY
            ),
        },
        "environment_config": {
            "episode_seconds": episode_seconds_for_args(args),
            "control_decimation": int(environment_definition.DEFAULT_CONTROL_DECIMATION),
            "command_speed_limit_mps": command_speed_limit_for_args(args),
            "command_speed_fraction": float(args.command_speed_fraction),
            "command_resample_seconds": float(args.command_resample_seconds),
            "mjcf_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
            "lqr_source_sha256": hashlib.sha256(Path(lqr.__file__).read_bytes()).hexdigest(),
            "jump_controller": lqr.jump_controller_config(),
            "terrain_controller": lqr.terrain_controller_config(),
            "locomotion_command_conditioning": environment_definition.locomotion_command_conditioning_config(),
            "locomotion_command_schema": environment_definition.LOCOMOTION_COMMAND_SCHEMA,
            "residual_authority_schema": environment_definition.RESIDUAL_AUTHORITY_SCHEMA,
            "domain_randomization": domain_randomization.as_dict(),
            "jump_domain_randomization": (
                None if jump_domain_randomization is None else jump_domain_randomization.as_dict()
            ),
            "terrain_curriculum": terrain_curriculum,
            "terrain_evaluation": False,
        },
        "domain_randomization": domain_randomization.as_dict(),
        "jump_domain_randomization": (
            None if jump_domain_randomization is None else jump_domain_randomization.as_dict()
        ),
    }
    environment = WheelLegResidualEnv(
        xml_path=args.xml_path,
        episode_seconds=episode_seconds_for_args(args),
        max_forward_speed=getattr(args, "max_speed", None),
        command_speed_limit_mps=command_speed_limit_for_args(args),
        command_resample_seconds=command_resample_seconds_for_args(args),
        command_speed_fraction=float(args.command_speed_fraction),
        max_command_yaw_delta_rad=np.deg2rad(args.yaw_range_deg),
        max_command_yaw_rate_rad_s=args.max_yaw_rate,
        jump_probability=args.jump_probability,
        jump_at=args.jump_at,
        domain_randomization=domain_randomization,
        jump_domain_randomization=jump_domain_randomization,
        terrain_curriculum=getattr(args, "terrain_curriculum_config", None),
        terrain_stage_id=getattr(args, "terrain_stage_id", None),
        terrain_evaluation=False,
    )
    try:
        observations, expert_actions, action_masks = collect_demonstrations(
            environment,
            args.samples,
            args.target,
            args.seed,
        )
        policy = train_bc(
            observations,
            expert_actions,
            action_masks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=torch.device(args.device),
        )
        save_policy(
            args.output,
            policy,
            observations.shape[1],
            expert_actions.shape[1],
            args.target,
            task_config,
        )
        print(f"saved BC {args.target} policy: {args.output}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
