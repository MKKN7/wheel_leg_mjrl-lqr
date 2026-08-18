"""Pure-PyTorch PPO trainer for :class:`env.WheelLegResidualEnv`.

This entry point intentionally has no Stable-Baselines3 dependency.  The
policy learns bounded torque residuals and a common leg-length rate command on
top of the physical low-centre ``PhysicalLqr`` baseline supplied by the
environment.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

import env as environment_definition
from env import DEFAULT_EPISODE_SECONDS, DomainRandomizationConfig, WheelLegResidualEnv
import lqr_deploy as lqr
from terrain_curriculum import (
    TerrainCurriculumConfig,
    TerrainCurriculumError,
    load_terrain_curriculum,
)


CHECKPOINT_FORMAT_VERSION = 12
REWARD_SCHEMA = "command_tracking_jump_clearance_terrain_safe_terminal_v12"


@dataclass
class Rollout:
    observations: Tensor
    actions: Tensor
    policy_action_masks: Tensor
    log_probabilities: Tensor
    rewards: Tensor
    values: Tensor
    bootstrap_values: Tensor
    continuation_masks: Tensor
    advantages: Tensor
    returns: Tensor


@dataclass(frozen=True)
class ResumeState:
    """Persistent PPO state restored at a completed-update boundary."""

    timesteps: int
    update_index: int


class ActorCritic(nn.Module):
    """Tanh-squashed Gaussian residual actor and scalar value critic."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_size,), -1.0))

    def _distribution(self, observations: Tensor) -> Normal:
        mean = self.actor(observations)
        standard_deviation = self.log_std.exp().expand_as(mean)
        return Normal(mean, standard_deviation)

    def sample_action(self, observations: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self._distribution(observations)
        latent_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(latent_action)
        log_probability = self._squashed_log_probability(distribution, latent_action, action)
        value = self.critic(observations).squeeze(-1)
        return action, log_probability, value

    def evaluate_actions(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bounded_actions = actions.clamp(-0.999999, 0.999999)
        latent_actions = 0.5 * (torch.log1p(bounded_actions) - torch.log1p(-bounded_actions))
        distribution = self._distribution(observations)
        log_probability = self._squashed_log_probability(distribution, latent_actions, bounded_actions)
        entropy = distribution.entropy()
        value = self.critic(observations).squeeze(-1)
        return log_probability, entropy, value

    @staticmethod
    def _squashed_log_probability(distribution: Normal, latent: Tensor, action: Tensor) -> Tensor:
        correction = torch.log(1.0 - action.square() + 1e-6)
        return distribution.log_prob(latent) - correction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a command-conditioned PPO residual locomotion controller for the wheeled-leg robot."
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--max-gradient-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", help="PyTorch device, such as cpu or cuda.")
    parser.add_argument(
        "--xml-path",
        type=Path,
        help=(
            "MJCF scene to train in; defaults to wheeled_infantry.xml, or "
            "rm_train_ground.xml when --terrain-curriculum is supplied."
        ),
    )
    parser.add_argument(
        "--terrain-curriculum",
        type=Path,
        help=(
            "Strict RMUC terrain curriculum YAML. When supplied, --terrain-stage selects the "
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
        help=(
            "Maximum magnitude sampled for high-level speed commands. Defaults to --max-speed; "
            "use a lower value for an RMUC locomotion curriculum."
        ),
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
        "--output",
        type=Path,
        default=Path("artifacts") / "ppo_locomotion_controller.pt",
        help="Latest checkpoint for the high-level-command residual locomotion controller.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Save a step-numbered checkpoint snapshot every N completed PPO updates.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help="CSV update metrics path; defaults to <output>.metrics.csv.",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        help="Optional BC checkpoint trained with --target residual for a safe residual warm start.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Resume PPO model and optimizer state. --total-timesteps is the final cumulative target, "
            "not additional steps."
        ),
    )
    parser.add_argument(
        "--resume-stage-transfer",
        action="store_true",
        help=(
            "Permit an explicit resume from an earlier stage of the same RMUC curriculum YAML. "
            "MJCF, LQR, reward, command, action-authority, and domain-randomization fingerprints "
            "remain strict."
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="Run one short PPO update for dependency verification.")
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
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")
    if args.vehicle_only_domain_randomization and not args.domain_randomization:
        parser.error("--vehicle-only-domain-randomization requires --domain-randomization")
    if args.bc_checkpoint is not None and args.resume_checkpoint is not None:
        parser.error("--bc-checkpoint and --resume-checkpoint cannot be used together")
    resolve_terrain_curriculum_args(args, parser)
    if args.resume_stage_transfer and args.resume_checkpoint is None:
        parser.error("--resume-stage-transfer requires --resume-checkpoint")
    if args.resume_stage_transfer and args.terrain_curriculum_config is None:
        parser.error("--resume-stage-transfer requires --terrain-curriculum and --terrain-stage")
    return args


def tensor_from_observation(observation: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


def collect_rollout(
    environment: WheelLegResidualEnv,
    policy: ActorCritic,
    observation: np.ndarray,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    episode_return: float,
) -> tuple[Rollout, np.ndarray, list[float], list[dict[str, Any]], float]:
    observation_size = environment.observation_space.shape[0]
    action_size = environment.action_space.shape[0]
    observations = torch.empty((rollout_steps, observation_size), dtype=torch.float32, device=device)
    actions = torch.empty((rollout_steps, action_size), dtype=torch.float32, device=device)
    policy_action_masks = torch.empty(
        (rollout_steps, action_size), dtype=torch.float32, device=device
    )
    log_probabilities = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    rewards = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    values = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    bootstrap_values = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    continuation_masks = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    completed_returns: list[float] = []
    completed_infos: list[dict[str, Any]] = []

    for index in range(rollout_steps):
        observation_tensor = tensor_from_observation(observation, device)
        with torch.no_grad():
            action_tensor, _, value_tensor = policy.sample_action(observation_tensor)
        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        with torch.no_grad():
            next_value_tensor = policy.critic(tensor_from_observation(next_observation, device)).squeeze()
        action_mask = np.asarray(
            info.get("policy_action_mask", np.ones(action_size, dtype=np.float32)),
            dtype=np.float32,
        )
        if action_mask.shape != (action_size,):
            raise RuntimeError(
                "environment returned invalid policy_action_mask shape "
                f"{action_mask.shape}; expected ({action_size},)"
            )
        if not np.all(np.isfinite(action_mask)) or np.any((action_mask < 0.0) | (action_mask > 1.0)):
            raise RuntimeError("environment returned an invalid policy_action_mask")
        action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=device)
        with torch.no_grad():
            log_probability_dimensions, _, _ = policy.evaluate_actions(
                observation_tensor, action_tensor
            )
            log_probability_tensor = (log_probability_dimensions * action_mask_tensor).sum(dim=-1)

        observations[index] = observation_tensor.squeeze(0)
        actions[index] = action_tensor.squeeze(0)
        policy_action_masks[index] = action_mask_tensor
        log_probabilities[index] = log_probability_tensor.squeeze(0)
        rewards[index] = float(reward)
        values[index] = value_tensor.squeeze(0)
        bootstrap_values[index] = 0.0 if terminated else next_value_tensor
        continuation_masks[index] = 0.0 if done else 1.0
        episode_return += float(reward)

        if done:
            completed_returns.append(episode_return)
            completed_infos.append(info)
            episode_return = 0.0
            next_observation, _ = environment.reset()
        observation = next_observation

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), dtype=torch.float32, device=device)
    for index in range(rollout_steps - 1, -1, -1):
        temporal_difference = rewards[index] + gamma * bootstrap_values[index] - values[index]
        last_advantage = (
            temporal_difference
            + gamma * gae_lambda * continuation_masks[index] * last_advantage
        )
        advantages[index] = last_advantage
    returns = advantages + values
    return (
        Rollout(
            observations=observations,
            actions=actions,
            policy_action_masks=policy_action_masks,
            log_probabilities=log_probabilities,
            rewards=rewards,
            values=values,
            bootstrap_values=bootstrap_values,
            continuation_masks=continuation_masks,
            advantages=advantages,
            returns=returns,
        ),
        observation,
        completed_returns,
        completed_infos,
        episode_return,
    )


def update_policy(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    *,
    epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_gradient_norm: float,
) -> dict[str, float]:
    advantages = torch.zeros_like(rollout.advantages)
    active_samples = rollout.policy_action_masks.any(dim=-1)
    if bool(active_samples.any()):
        active_advantages = rollout.advantages[active_samples]
        advantages[active_samples] = (
            active_advantages - active_advantages.mean()
        ) / (active_advantages.std(unbiased=False) + 1e-8)
    batch_size = rollout.observations.shape[0]
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    for _ in range(epochs):
        indices = torch.randperm(batch_size, device=rollout.observations.device)
        for start in range(0, batch_size, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            log_probability_dimensions, entropy_dimensions, predicted_values = policy.evaluate_actions(
                rollout.observations[batch_indices], rollout.actions[batch_indices]
            )
            action_mask = rollout.policy_action_masks[batch_indices]
            new_log_probabilities = (log_probability_dimensions * action_mask).sum(dim=-1)
            ratio = torch.exp(new_log_probabilities - rollout.log_probabilities[batch_indices])
            unclipped_objective = ratio * advantages[batch_indices]
            clipped_objective = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[batch_indices]
            active_samples = action_mask.any(dim=-1)
            if bool(active_samples.any()):
                policy_loss = -torch.minimum(
                    unclipped_objective[active_samples], clipped_objective[active_samples]
                ).mean()
                entropy_bonus = (
                    entropy_dimensions * action_mask
                ).sum() / action_mask.sum().clamp_min(1.0)
            else:
                policy_loss = (new_log_probabilities * 0.0).sum()
                entropy_bonus = (entropy_dimensions * 0.0).sum()
            value_loss = torch.nn.functional.mse_loss(predicted_values, rollout.returns[batch_indices])
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_gradient_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy_bonus.detach().cpu()))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }


def default_metrics_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_metrics.csv")


def command_speed_limit_for_args(args: argparse.Namespace) -> float:
    if args.command_speed_limit is not None:
        return float(args.command_speed_limit)
    if args.max_speed is not None:
        return float(args.max_speed)
    return float(lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS)


def command_resample_seconds_for_args(args: argparse.Namespace) -> float | None:
    return None if args.command_resample_seconds == 0.0 else float(args.command_resample_seconds)


def resolve_terrain_curriculum_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Load and validate an optional fixed-stage RMUC curriculum before training starts."""
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
        rmuc_xml_path = Path(__file__).with_name("rm_train_ground.xml")
        if rmuc_xml_path.is_file():
            # A terrain curriculum is explicitly an RMUC-scene training run;
            # avoid silently falling back to the flat wheeled XML.
            args.xml_path = rmuc_xml_path.resolve()


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
    return {
        "yaml_sha256": hashlib.sha256(curriculum_path.read_bytes()).hexdigest(),
        "schema_version": int(curriculum.schema_version),
        "stage_id": stage.stage_id,
        "stage_task_ids": list(stage.task_ids),
    }


def episode_seconds_for_args(args: argparse.Namespace) -> float:
    """Use the YAML horizon so slow fixed terrain tasks are not truncated at 8 s."""
    curriculum = getattr(args, "terrain_curriculum_config", None)
    stage_id = getattr(args, "terrain_stage_id", None)
    if curriculum is None:
        return float(DEFAULT_EPISODE_SECONDS)
    if not isinstance(curriculum, TerrainCurriculumConfig) or not isinstance(stage_id, str):
        raise ValueError("terrain curriculum arguments were not resolved before episode duration selection")
    return max(float(DEFAULT_EPISODE_SECONDS), float(curriculum.stage_max_episode_seconds(stage_id)))


def terrain_stage_uses_progress_jump(args: argparse.Namespace) -> bool:
    """Whether a selected fixed terrain stage issues a future jump edge."""
    curriculum = getattr(args, "terrain_curriculum_config", None)
    stage_id = getattr(args, "terrain_stage_id", None)
    if curriculum is None:
        return False
    if not isinstance(curriculum, TerrainCurriculumConfig) or not isinstance(stage_id, str):
        raise ValueError("terrain curriculum arguments were not resolved before jump-profile selection")
    return any(
        curriculum.task(task_id).has_progress_jump_trigger
        for task_id in curriculum.stage(stage_id).task_ids
    )


def domain_randomization_for_args(
    args: argparse.Namespace,
) -> tuple[DomainRandomizationConfig, DomainRandomizationConfig | None]:
    if not args.domain_randomization:
        return DomainRandomizationConfig.disabled(), None
    if getattr(args, "terrain_curriculum_config", None) is not None:
        # RMUC terrain geometry and support friction are fixed.  Randomize the
        # vehicle with the staged profile; jump episodes still use the
        # conservative landing profile selected by the environment.
        walking = DomainRandomizationConfig.terrain_vehicle_only_defaults()
        jumping = DomainRandomizationConfig.jump_vehicle_only_defaults()
    elif args.vehicle_only_domain_randomization:
        walking = DomainRandomizationConfig.vehicle_only_defaults()
        jumping = DomainRandomizationConfig.jump_vehicle_only_defaults()
    else:
        walking = DomainRandomizationConfig.training_defaults()
        jumping = DomainRandomizationConfig.jump_training_defaults()
    return walking, jumping if (
        args.jump_probability > 0.0 or terrain_stage_uses_progress_jump(args)
    ) else None


def task_config_for_args(args: argparse.Namespace) -> dict[str, Any]:
    xml_path = (lqr.XML_PATH if args.xml_path is None else args.xml_path).resolve()
    domain_randomization, jump_domain_randomization = domain_randomization_for_args(args)
    terrain_curriculum = terrain_curriculum_metadata_for_args(args)
    return {
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
        "locomotion_command_schema": environment_definition.LOCOMOTION_COMMAND_SCHEMA,
        "residual_authority_schema": environment_definition.RESIDUAL_AUTHORITY_SCHEMA,
        "domain_randomization": domain_randomization.as_dict(),
        "jump_domain_randomization": (
            None if jump_domain_randomization is None else jump_domain_randomization.as_dict()
        ),
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
            "locomotion_command_schema": environment_definition.LOCOMOTION_COMMAND_SCHEMA,
            "residual_authority_schema": environment_definition.RESIDUAL_AUTHORITY_SCHEMA,
            "domain_randomization": domain_randomization.as_dict(),
            "jump_domain_randomization": (
                None if jump_domain_randomization is None else jump_domain_randomization.as_dict()
            ),
            "terrain_curriculum": terrain_curriculum,
            "terrain_evaluation": False,
        },
    }


def training_config_for_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "rollout_steps": int(args.rollout_steps),
        "epochs": int(args.epochs),
        "minibatch_size": int(args.minibatch_size),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_ratio": float(args.clip_ratio),
        "value_coefficient": float(args.value_coefficient),
        "entropy_coefficient": float(args.entropy_coefficient),
        "max_gradient_norm": float(args.max_gradient_norm),
    }


def capture_rng_state(environment: WheelLegResidualEnv | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state().clone(),
        "numpy": copy.deepcopy(np.random.get_state()),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    if environment is not None:
        state["environment_np_random"] = copy.deepcopy(environment.np_random.bit_generator.state)
    return state


def restore_rng_state(state: Any, environment: WheelLegResidualEnv | None = None) -> None:
    if not isinstance(state, dict):
        print("Resume checkpoint has no RNG state; continuing from a fresh random sequence.")
        return
    torch_state = state.get("torch")
    numpy_state = state.get("numpy")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    environment_state = state.get("environment_np_random")
    if environment is not None and environment_state is not None:
        environment.np_random.bit_generator.state = copy.deepcopy(environment_state)


def build_checkpoint_payload(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observation_size: int,
    action_size: int,
    timesteps: int,
    update_index: int,
    task_config: dict[str, Any],
    training_config: dict[str, Any],
    environment: WheelLegResidualEnv | None = None,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "algorithm": "ppo",
        "action_semantics": "residual",
        "resume_semantics": "fresh_episode_at_completed_update_boundary",
        "observation_size": observation_size,
        "action_size": action_size,
        "task_config": copy.deepcopy(task_config),
        "training_config": copy.deepcopy(training_config),
        "model_state_dict": copy.deepcopy(policy.state_dict()),
        "actor_state_dict": copy.deepcopy(policy.actor.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "rng_state": capture_rng_state(environment),
        "timesteps": int(timesteps),
        "update_index": int(update_index),
    }


def checkpoint_snapshot_path(path: Path, timesteps: int) -> Path:
    return path.with_name(f"{path.stem}_step_{timesteps:09d}{path.suffix}")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_checkpoint(
    path: Path,
    payload: dict[str, Any],
    *,
    save_snapshot: bool,
) -> Path | None:
    atomic_torch_save(payload, path)
    if not save_snapshot:
        return None
    snapshot_path = checkpoint_snapshot_path(path, int(payload["timesteps"]))
    atomic_torch_save(payload, snapshot_path)
    return snapshot_path


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def validate_terrain_stage_transfer(
    saved_metadata: Any,
    current_metadata: Any,
    curriculum: TerrainCurriculumConfig | None,
) -> tuple[str, str]:
    """Validate an explicit monotonic RMUC curriculum-stage continuation.

    Checkpoint metadata carries the YAML digest rather than the full task
    catalog.  Re-reading the selected YAML here lets us prove that both the
    saved and requested stage IDs/tasks belong to exactly the same curriculum
    before allowing the episode horizon to change.
    """
    if not isinstance(saved_metadata, dict) or not isinstance(current_metadata, dict):
        raise ValueError(
            "--resume-stage-transfer requires terrain curriculum metadata in both checkpoint and request"
        )
    if curriculum is None:
        raise ValueError("--resume-stage-transfer requires a resolved terrain curriculum")
    for key in ("yaml_sha256", "schema_version"):
        if saved_metadata.get(key) != current_metadata.get(key):
            raise ValueError(
                "PPO stage transfer requires the identical terrain curriculum YAML: "
                f"{key} checkpoint={saved_metadata.get(key)!r}, requested={current_metadata.get(key)!r}"
            )
    saved_stage_id = saved_metadata.get("stage_id")
    current_stage_id = current_metadata.get("stage_id")
    if not isinstance(saved_stage_id, str) or not isinstance(current_stage_id, str):
        raise ValueError("PPO stage transfer metadata is missing a valid stage_id")
    stage_ids = tuple(stage.stage_id for stage in curriculum.stages)
    if saved_stage_id not in stage_ids or current_stage_id not in stage_ids:
        raise ValueError(
            "PPO stage transfer stage is not present in the selected terrain curriculum: "
            f"checkpoint={saved_stage_id!r}, requested={current_stage_id!r}"
        )
    expected_saved_tasks = list(curriculum.stage(saved_stage_id).task_ids)
    expected_current_tasks = list(curriculum.stage(current_stage_id).task_ids)
    saved_tasks = saved_metadata.get("stage_task_ids")
    current_tasks = current_metadata.get("stage_task_ids")
    if not isinstance(saved_tasks, (list, tuple)) or not isinstance(current_tasks, (list, tuple)):
        raise ValueError("PPO stage transfer metadata is missing stage_task_ids")
    if list(saved_tasks) != expected_saved_tasks:
        raise ValueError(
            "PPO stage transfer checkpoint task list does not match the selected curriculum YAML"
        )
    if list(current_tasks) != expected_current_tasks:
        raise ValueError(
            "PPO stage transfer requested task list does not match the selected curriculum YAML"
        )
    saved_index = stage_ids.index(saved_stage_id)
    current_index = stage_ids.index(current_stage_id)
    if current_index <= saved_index:
        raise ValueError(
            "--resume-stage-transfer only permits a later curriculum stage: "
            f"checkpoint={saved_stage_id!r}, requested={current_stage_id!r}"
        )
    return saved_stage_id, current_stage_id


def validate_resume_task_config(
    checkpoint_config: Any,
    current_config: dict[str, Any],
    *,
    allow_terrain_stage_transfer: bool = False,
    terrain_curriculum: TerrainCurriculumConfig | None = None,
) -> None:
    if not isinstance(checkpoint_config, dict):
        raise ValueError("PPO resume checkpoint is missing task_config")
    required_keys = (
        "max_speed_mps",
        "command_speed_limit_mps",
        "command_speed_fraction",
        "command_resample_seconds",
        "yaw_range_deg",
        "max_yaw_rate_rad_s",
        "jump_probability",
        "jump_at_s",
    )
    mismatches: list[str] = []
    for key in required_keys:
        saved_value = checkpoint_config.get(key)
        current_value = current_config[key]
        if saved_value is None or not np.isclose(float(saved_value), float(current_value)):
            mismatches.append(f"{key}: checkpoint={saved_value!r}, requested={current_value!r}")
    if mismatches:
        raise ValueError("PPO resume task configuration mismatch: " + "; ".join(mismatches))
    saved_terrain_curriculum = checkpoint_config.get("terrain_curriculum")
    current_terrain_curriculum = current_config.get("terrain_curriculum")
    stage_transfer: tuple[str, str] | None = None
    if saved_terrain_curriculum != current_terrain_curriculum:
        if not allow_terrain_stage_transfer:
            raise ValueError(
                "PPO resume terrain curriculum mismatch: "
                f"checkpoint={saved_terrain_curriculum!r}, current={current_terrain_curriculum!r}"
            )
        stage_transfer = validate_terrain_stage_transfer(
            saved_terrain_curriculum,
            current_terrain_curriculum,
            terrain_curriculum,
        )
    saved_reward_schema = checkpoint_config.get("reward_schema")
    if saved_reward_schema is None:
        print(
            "Resume checkpoint predates reward metadata; using the current reward definition. "
            "This is not an exact continuation if the reward changed."
        )
    elif saved_reward_schema != current_config["reward_schema"]:
        raise ValueError(
            "PPO resume reward schema mismatch: "
            f"checkpoint={saved_reward_schema!r}, current={current_config['reward_schema']!r}"
        )
    saved_reward_config = checkpoint_config.get("reward_config")
    current_reward_config = current_config["reward_config"]
    if saved_reward_config is None:
        print("Resume checkpoint has no reward parameters; using the current reward parameters.")
    elif not isinstance(saved_reward_config, dict):
        raise ValueError("PPO resume checkpoint has an invalid reward configuration")
    else:
        reward_mismatches = [
            f"{key}: checkpoint={saved_reward_config.get(key)!r}, current={value!r}"
            for key, value in current_reward_config.items()
            if key not in saved_reward_config
            or not np.isclose(float(saved_reward_config[key]), float(value))
        ]
        if reward_mismatches:
            raise ValueError("PPO resume reward configuration mismatch: " + "; ".join(reward_mismatches))
    saved_environment_config = checkpoint_config.get("environment_config")
    current_environment_config = current_config["environment_config"]
    if saved_environment_config is None:
        current_domain_config = current_environment_config.get("domain_randomization")
        if isinstance(current_domain_config, dict) and current_domain_config.get("enabled"):
            raise ValueError(
                "PPO resume checkpoint has no domain-randomization metadata; "
                "start a new PPO run or resume with --no-domain-randomization explicitly."
            )
        print("Resume checkpoint has no environment fingerprint; using the current MJCF and environment settings.")
    elif not isinstance(saved_environment_config, dict):
        raise ValueError("PPO resume checkpoint has an invalid environment configuration")
    else:
        if stage_transfer is not None:
            saved_environment_terrain = saved_environment_config.get("terrain_curriculum")
            current_environment_terrain = current_environment_config.get("terrain_curriculum")
            if (
                saved_environment_terrain != saved_terrain_curriculum
                or current_environment_terrain != current_terrain_curriculum
            ):
                raise ValueError(
                    "PPO stage transfer environment/task curriculum metadata is inconsistent"
                )
        stage_environment_keys = {"terrain_curriculum", "episode_seconds"} if stage_transfer else set()
        environment_mismatches = [
            f"{key}: checkpoint={saved_environment_config.get(key)!r}, current={value!r}"
            for key, value in current_environment_config.items()
            if key not in stage_environment_keys
            and not (
                key == "terrain_curriculum"
                and key not in saved_environment_config
                and value is None
            )
            and saved_environment_config.get(key) != value
        ]
        if environment_mismatches:
            raise ValueError(
                "PPO resume environment configuration mismatch: " + "; ".join(environment_mismatches)
            )
    if stage_transfer is not None:
        print(
            "Accepted PPO curriculum stage transfer: "
            f"{stage_transfer[0]} -> {stage_transfer[1]}. "
            "The next rollout starts from fresh episodes at the later stage."
        )


def report_resume_training_config(checkpoint_config: Any, current_config: dict[str, Any]) -> None:
    if not isinstance(checkpoint_config, dict):
        print("Resume checkpoint has no training configuration; using the requested PPO update settings.")
        return
    changed = [
        key
        for key, current_value in current_config.items()
        if checkpoint_config.get(key) != current_value
    ]
    if changed:
        print(
            "Resume uses the requested PPO update settings; checkpoint values differ for: "
            + ", ".join(changed)
        )


def load_ppo_resume(
    path: Path,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observation_size: int,
    action_size: int,
    task_config: dict[str, Any],
    training_config: dict[str, Any],
    environment: WheelLegResidualEnv,
    device: torch.device,
    *,
    allow_terrain_stage_transfer: bool = False,
    terrain_curriculum: TerrainCurriculumConfig | None = None,
) -> ResumeState:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "PPO resume checkpoint uses an incompatible locomotion-command/action-authority schema; "
            "start a new run or use a checkpoint produced by this trainer version."
        )
    if checkpoint.get("algorithm") != "ppo" or checkpoint.get("action_semantics") != "residual":
        raise ValueError("--resume-checkpoint must contain a residual PPO checkpoint")
    if checkpoint.get("observation_size") != observation_size or checkpoint.get("action_size") != action_size:
        raise ValueError("PPO resume checkpoint dimensions do not match the current environment")
    validate_resume_task_config(
        checkpoint.get("task_config"),
        task_config,
        allow_terrain_stage_transfer=allow_terrain_stage_transfer,
        terrain_curriculum=terrain_curriculum,
    )
    report_resume_training_config(checkpoint.get("training_config"), training_config)
    timesteps = checkpoint.get("timesteps")
    if not isinstance(timesteps, int) or timesteps < 0:
        raise ValueError("PPO resume checkpoint has an invalid timesteps value")
    if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint:
        raise ValueError("PPO resume checkpoint is missing model or optimizer state")
    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer_to_device(optimizer, device)
    restore_rng_state(checkpoint.get("rng_state"), environment)
    update_index = checkpoint.get("update_index")
    if not isinstance(update_index, int) or update_index < 0:
        rollout_steps = training_config["rollout_steps"]
        update_index = (timesteps + rollout_steps - 1) // rollout_steps
        print(
            "Resume checkpoint has no update index; inferred "
            f"update={update_index} from timesteps and the requested rollout length."
        )
    print(f"Resumed PPO checkpoint: {path} at timesteps={timesteps}, update={update_index}")
    return ResumeState(timesteps=timesteps, update_index=update_index)


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "update",
        "timesteps",
        "completed_episodes",
        "mean_completed_return",
        "partial_return",
        "policy_loss",
        "value_loss",
        "entropy",
        "jump_episodes",
        "jump_successes",
        "jump_trigger_rate",
        "jump_height_target_rate",
        "jump_stable_landing_rate",
        "mean_jump_peak_clearance_m",
        "mean_jump_min_clearance_m",
        "physical_unsafe_episodes",
        "physical_unsafe_rate",
        "terrain_episodes",
        "terrain_completions",
        "terrain_completion_rate",
        "terrain_physical_unsafe_episodes",
        "terrain_physical_unsafe_rate",
    )
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open("r", newline="", encoding="ascii") as handle:
            existing_header = next(csv.reader(handle), [])
        if tuple(existing_header) != fieldnames:
            raise ValueError(
                f"metrics CSV schema mismatch at {path}; choose a new --metrics-path or remove the old file"
            )
    with path.open("a", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def summarize_jump_episodes(infos: list[dict[str, Any]]) -> dict[str, float | int]:
    scheduled_infos = [info for info in infos if bool(info.get("jump_scheduled"))]
    triggered_infos = [info for info in scheduled_infos if bool(info.get("jump_triggered"))]
    if not scheduled_infos:
        return {
            "jump_episodes": 0,
            "jump_successes": 0,
            "jump_trigger_rate": float("nan"),
            "jump_height_target_rate": float("nan"),
            "jump_stable_landing_rate": float("nan"),
            "mean_jump_peak_clearance_m": float("nan"),
            "mean_jump_min_clearance_m": float("nan"),
        }
    peaks = np.asarray(
        [float(info["jump_peak_mean_wheel_clearance_m"]) for info in scheduled_infos],
        dtype=np.float64,
    )
    minimums = np.asarray(
        [float(info["jump_peak_wheel_clearance_m"]) for info in scheduled_infos],
        dtype=np.float64,
    )
    height_reached = np.asarray(
        [bool(info["jump_height_target_reached"]) for info in scheduled_infos],
        dtype=np.float64,
    )
    stable = np.asarray(
        [bool(info.get("jump_landing_stable", info["jump_succeeded"])) for info in scheduled_infos],
        dtype=np.float64,
    )
    successes = np.asarray([bool(info["jump_succeeded"]) for info in scheduled_infos], dtype=np.float64)
    return {
        "jump_episodes": len(scheduled_infos),
        "jump_successes": int(np.sum(successes)),
        "jump_trigger_rate": float(len(triggered_infos) / len(scheduled_infos)),
        "jump_height_target_rate": float(np.mean(height_reached)),
        "jump_stable_landing_rate": float(np.mean(stable)),
        "mean_jump_peak_clearance_m": float(np.mean(peaks)),
        "mean_jump_min_clearance_m": float(np.mean(minimums)),
    }


def summarize_safety_and_terrain_episodes(infos: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate terminal physical safety and fixed-terrain task outcomes.

    ``physical_unsafe`` deliberately excludes a curriculum corridor miss and
    other task-level failures.  That keeps the reported unsafe rate aligned
    with the robot-fall target while terrain completion remains separately
    visible in the same metrics CSV.
    """
    physical_unsafe_episodes = sum(bool(info.get("physical_unsafe", False)) for info in infos)
    terrain_infos = [info for info in infos if info.get("terrain_task_id") is not None]
    terrain_completions = sum(bool(info.get("terrain_task_completed", False)) for info in terrain_infos)
    terrain_physical_unsafe = sum(
        bool(info.get("physical_unsafe", False)) for info in terrain_infos
    )
    episode_count = len(infos)
    terrain_episode_count = len(terrain_infos)
    return {
        "physical_unsafe_episodes": physical_unsafe_episodes,
        "physical_unsafe_rate": (
            float(physical_unsafe_episodes / episode_count)
            if episode_count
            else float("nan")
        ),
        "terrain_episodes": terrain_episode_count,
        "terrain_completions": terrain_completions,
        "terrain_completion_rate": (
            float(terrain_completions / terrain_episode_count)
            if terrain_episode_count
            else float("nan")
        ),
        "terrain_physical_unsafe_episodes": terrain_physical_unsafe,
        "terrain_physical_unsafe_rate": (
            float(terrain_physical_unsafe / terrain_episode_count)
            if terrain_episode_count
            else float("nan")
        ),
    }


def load_residual_bc_warm_start(
    path: Path,
    policy: ActorCritic,
    observation_size: int,
    action_size: int,
    task_config: dict[str, Any] | None = None,
) -> None:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "BC checkpoint uses an incompatible locomotion-command/action-authority schema; "
            "regenerate BC with the current environment."
        )
    if checkpoint.get("target_kind") != "residual":
        raise ValueError("BC warm start requires a checkpoint trained with '--target residual'")
    if checkpoint.get("observation_size") != observation_size or checkpoint.get("action_size") != action_size:
        raise ValueError("BC checkpoint dimensions do not match the residual environment")
    if task_config is not None:
        saved_task_config = checkpoint.get("task_config")
        if not isinstance(saved_task_config, dict):
            raise ValueError("BC checkpoint is missing task metadata; regenerate BC for the current PPO task.")
        task_mismatches: list[str] = []
        for key in (
            "max_speed_mps",
            "command_speed_limit_mps",
            "command_speed_fraction",
            "command_resample_seconds",
            "yaw_range_deg",
            "max_yaw_rate_rad_s",
            "jump_probability",
            "jump_at_s",
        ):
            saved_value = saved_task_config.get(key)
            current_value = task_config.get(key)
            if saved_value is None or current_value is None or not np.isclose(
                float(saved_value), float(current_value)
            ):
                task_mismatches.append(
                    f"{key}: checkpoint={saved_value!r}, current={current_value!r}"
                )
        for key in ("locomotion_command_schema", "residual_authority_schema"):
            if saved_task_config.get(key) != task_config.get(key):
                task_mismatches.append(
                    f"{key}: checkpoint={saved_task_config.get(key)!r}, current={task_config.get(key)!r}"
                )
        if saved_task_config.get("reward_schema") != task_config.get("reward_schema"):
            task_mismatches.append(
                "reward_schema: "
                f"checkpoint={saved_task_config.get('reward_schema')!r}, "
                f"current={task_config.get('reward_schema')!r}"
            )
        saved_reward_config = saved_task_config.get("reward_config")
        current_reward_config = task_config.get("reward_config")
        if saved_reward_config != current_reward_config:
            task_mismatches.append("reward_config differs between BC and PPO")
        saved_terrain_curriculum = saved_task_config.get("terrain_curriculum")
        current_terrain_curriculum = task_config.get("terrain_curriculum")
        if saved_terrain_curriculum != current_terrain_curriculum:
            task_mismatches.append(
                "terrain_curriculum: "
                f"checkpoint={saved_terrain_curriculum!r}, current={current_terrain_curriculum!r}"
            )
        if task_mismatches:
            raise ValueError("BC checkpoint task configuration does not match PPO: " + "; ".join(task_mismatches))
        saved_domain = saved_task_config.get("domain_randomization") if isinstance(saved_task_config, dict) else None
        current_domain = task_config.get("domain_randomization")
        if saved_domain is None and isinstance(current_domain, dict) and current_domain.get("enabled"):
            raise ValueError(
                "BC checkpoint has no domain-randomization metadata; regenerate BC with the current DR settings."
            )
        if saved_domain is not None and saved_domain != current_domain:
            raise ValueError("BC checkpoint domain-randomization configuration does not match PPO")
        saved_environment = (
            saved_task_config.get("environment_config") if isinstance(saved_task_config, dict) else None
        )
        current_environment = task_config.get("environment_config")
        if saved_environment is None:
            raise ValueError(
                "BC checkpoint has no environment fingerprint; regenerate BC with the current environment."
            )
        if not isinstance(saved_environment, dict) or not isinstance(current_environment, dict):
            raise ValueError("BC checkpoint has an invalid environment fingerprint")
        normalized_saved_environment = dict(saved_environment)
        if current_environment.get("terrain_curriculum") is None:
            normalized_saved_environment.setdefault("terrain_curriculum", None)
        if normalized_saved_environment != current_environment:
            raise ValueError("BC checkpoint environment configuration does not match PPO")
    policy.actor.load_state_dict(checkpoint["actor_state_dict"])


def train(args: argparse.Namespace) -> None:
    if args.total_timesteps < 1 or args.rollout_steps < 1:
        raise ValueError("total-timesteps and rollout-steps must be positive")
    if args.epochs < 1 or args.minibatch_size < 1:
        raise ValueError("epochs and minibatch-size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    task_config = task_config_for_args(args)
    training_config = training_config_for_args(args)
    metrics_path = args.metrics_path if args.metrics_path is not None else default_metrics_path(args.output)
    domain_randomization, jump_domain_randomization = domain_randomization_for_args(args)
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
    policy: ActorCritic | None = None
    optimizer: torch.optim.Optimizer | None = None
    observation_size = 0
    action_size = 0
    timesteps = 0
    update_index = 0
    completed_payload: dict[str, Any] | None = None
    try:
        observation, _ = environment.reset(seed=args.seed)
        observation_size = environment.observation_space.shape[0]
        action_size = environment.action_space.shape[0]
        policy = ActorCritic(observation_size, action_size).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        if args.resume_checkpoint is not None:
            resume_state = load_ppo_resume(
                args.resume_checkpoint,
                policy,
                optimizer,
                observation_size,
                action_size,
                task_config,
                training_config,
                environment,
                device,
                allow_terrain_stage_transfer=args.resume_stage_transfer,
                terrain_curriculum=getattr(args, "terrain_curriculum_config", None),
            )
            timesteps = resume_state.timesteps
            update_index = resume_state.update_index
            if timesteps >= args.total_timesteps:
                raise ValueError(
                    "--total-timesteps must be greater than the resumed checkpoint's "
                    f"timesteps ({timesteps})"
                )
            # The checkpoint stores RNG state at a completed-update boundary;
            # reset once after restoring it so the first rollout starts from
            # the next reproducible randomized episode.
            observation, _ = environment.reset()
        elif args.bc_checkpoint is not None:
            load_residual_bc_warm_start(
                args.bc_checkpoint,
                policy,
                observation_size,
                action_size,
                task_config,
            )

        episode_return = 0.0
        while timesteps < args.total_timesteps:
            current_rollout_steps = min(args.rollout_steps, args.total_timesteps - timesteps)
            rollout, observation, completed_returns, completed_infos, episode_return = collect_rollout(
                environment,
                policy,
                observation,
                current_rollout_steps,
                args.gamma,
                args.gae_lambda,
                device,
                episode_return,
            )
            metrics = update_policy(
                policy,
                optimizer,
                rollout,
                epochs=args.epochs,
                minibatch_size=args.minibatch_size,
                clip_ratio=args.clip_ratio,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                max_gradient_norm=args.max_gradient_norm,
            )
            timesteps += current_rollout_steps
            update_index += 1
            completed_payload = build_checkpoint_payload(
                policy,
                optimizer,
                observation_size,
                action_size,
                timesteps,
                update_index,
                task_config,
                training_config,
                environment,
            )
            if completed_returns:
                return_text = f"return={float(np.mean(completed_returns)):.3f}"
                mean_completed_return = float(np.mean(completed_returns))
            else:
                return_text = f"partial_return={episode_return:.3f}"
                mean_completed_return = float("nan")
            jump_metrics = summarize_jump_episodes(completed_infos)
            safety_metrics = summarize_safety_and_terrain_episodes(completed_infos)
            append_metrics(
                metrics_path,
                {
                    "update": update_index,
                    "timesteps": timesteps,
                    "completed_episodes": len(completed_returns),
                    "mean_completed_return": mean_completed_return,
                    "partial_return": float(episode_return),
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    **jump_metrics,
                    **safety_metrics,
                },
            )
            jump_text = ""
            if jump_metrics["jump_episodes"]:
                jump_text = (
                    f" jump_height={jump_metrics['mean_jump_peak_clearance_m']:.3f}m"
                    f" trigger_rate={jump_metrics['jump_trigger_rate']:.2f}"
                    f" target_rate={jump_metrics['jump_height_target_rate']:.2f}"
                    f" landing_rate={jump_metrics['jump_stable_landing_rate']:.2f}"
                )
            terrain_text = ""
            if safety_metrics["terrain_episodes"]:
                terrain_text = (
                    f" terrain_complete={safety_metrics['terrain_completion_rate']:.2f}"
                    f" terrain_unsafe={safety_metrics['terrain_physical_unsafe_rate']:.2f}"
                )
            safety_text = ""
            if completed_returns:
                safety_text = f" unsafe={safety_metrics['physical_unsafe_rate']:.2f}"
            print(
                f"update={update_index} timesteps={timesteps} {return_text} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}{jump_text}{safety_text}{terrain_text}"
            )
            snapshot_path = save_checkpoint(
                args.output,
                completed_payload,
                save_snapshot=update_index % args.checkpoint_interval == 0,
            )
            if snapshot_path is not None:
                print(f"saved PPO checkpoint: latest={args.output}, snapshot={snapshot_path}")
        if completed_payload is None:
            raise RuntimeError("PPO training completed without a full update")
        snapshot_path = save_checkpoint(args.output, completed_payload, save_snapshot=True)
        print(f"saved PPO residual policy: latest={args.output}, snapshot={snapshot_path}")
    except KeyboardInterrupt:
        if completed_payload is not None:
            snapshot_path = save_checkpoint(args.output, completed_payload, save_snapshot=True)
            print(
                "Interrupted after a completed PPO update; saved checkpoint: "
                f"latest={args.output}, snapshot={snapshot_path}"
            )
        else:
            print("Interrupted before the first completed PPO update; no partial checkpoint was written.")
    finally:
        environment.close()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.total_timesteps = min(args.total_timesteps, 32)
        args.rollout_steps = min(args.rollout_steps, 16)
        args.epochs = 1
        args.minibatch_size = min(args.minibatch_size, args.rollout_steps)
        args.output = Path("artifacts") / "ppo_locomotion_controller_smoke.pt"
    train(args)


if __name__ == "__main__":
    main()
