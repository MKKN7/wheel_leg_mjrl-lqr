"""Pure-PyTorch PPO trainer for :class:`env.WheelLegResidualEnv`.

This entry point intentionally has no Stable-Baselines3 dependency.  The
policy learns bounded torque residuals and a common leg-length rate command on
top of the physical low-centre ``PhysicalLqr`` baseline supplied by the
environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from env import WheelLegResidualEnv


@dataclass
class Rollout:
    observations: Tensor
    actions: Tensor
    log_probabilities: Tensor
    rewards: Tensor
    values: Tensor
    bootstrap_values: Tensor
    continuation_masks: Tensor
    advantages: Tensor
    returns: Tensor


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
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observations).squeeze(-1)
        return log_probability, entropy, value

    @staticmethod
    def _squashed_log_probability(distribution: Normal, latent: Tensor, action: Tensor) -> Tensor:
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (distribution.log_prob(latent) - correction).sum(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch PPO residual policy for the wheeled-leg robot.")
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
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "ppo_residual.pt")
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        help="Optional BC checkpoint trained with --target residual for a safe residual warm start.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run one short PPO update for dependency verification.")
    return parser.parse_args()


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
) -> tuple[Rollout, np.ndarray, list[float], float]:
    observation_size = environment.observation_space.shape[0]
    action_size = environment.action_space.shape[0]
    observations = torch.empty((rollout_steps, observation_size), dtype=torch.float32, device=device)
    actions = torch.empty((rollout_steps, action_size), dtype=torch.float32, device=device)
    log_probabilities = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    rewards = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    values = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    bootstrap_values = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    continuation_masks = torch.empty(rollout_steps, dtype=torch.float32, device=device)
    completed_returns: list[float] = []

    for index in range(rollout_steps):
        observation_tensor = tensor_from_observation(observation, device)
        with torch.no_grad():
            action_tensor, log_probability_tensor, value_tensor = policy.sample_action(observation_tensor)
        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated
        with torch.no_grad():
            next_value_tensor = policy.critic(tensor_from_observation(next_observation, device)).squeeze()

        observations[index] = observation_tensor.squeeze(0)
        actions[index] = action_tensor.squeeze(0)
        log_probabilities[index] = log_probability_tensor.squeeze(0)
        rewards[index] = float(reward)
        values[index] = value_tensor.squeeze(0)
        bootstrap_values[index] = 0.0 if terminated else next_value_tensor
        continuation_masks[index] = 0.0 if done else 1.0
        episode_return += float(reward)

        if done:
            completed_returns.append(episode_return)
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
    advantages = rollout.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = rollout.observations.shape[0]
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    for _ in range(epochs):
        indices = torch.randperm(batch_size, device=rollout.observations.device)
        for start in range(0, batch_size, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            new_log_probabilities, entropy, predicted_values = policy.evaluate_actions(
                rollout.observations[batch_indices], rollout.actions[batch_indices]
            )
            ratio = torch.exp(new_log_probabilities - rollout.log_probabilities[batch_indices])
            unclipped_objective = ratio * advantages[batch_indices]
            clipped_objective = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[batch_indices]
            policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
            value_loss = torch.nn.functional.mse_loss(predicted_values, rollout.returns[batch_indices])
            entropy_bonus = entropy.mean()
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


def save_checkpoint(
    path: Path,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observation_size: int,
    action_size: int,
    timesteps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "algorithm": "ppo",
            "action_semantics": "residual",
            "observation_size": observation_size,
            "action_size": action_size,
            "model_state_dict": policy.state_dict(),
            "actor_state_dict": policy.actor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timesteps": timesteps,
        },
        path,
    )


def load_residual_bc_warm_start(path: Path, policy: ActorCritic, observation_size: int, action_size: int) -> None:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("target_kind") != "residual":
        raise ValueError("BC warm start requires a checkpoint trained with '--target residual'")
    if checkpoint.get("observation_size") != observation_size or checkpoint.get("action_size") != action_size:
        raise ValueError("BC checkpoint dimensions do not match the residual environment")
    policy.actor.load_state_dict(checkpoint["actor_state_dict"])


def train(args: argparse.Namespace) -> None:
    if args.total_timesteps < 1 or args.rollout_steps < 1:
        raise ValueError("total-timesteps and rollout-steps must be positive")
    if args.epochs < 1 or args.minibatch_size < 1:
        raise ValueError("epochs and minibatch-size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    environment = WheelLegResidualEnv()
    try:
        observation, _ = environment.reset(seed=args.seed)
        observation_size = environment.observation_space.shape[0]
        action_size = environment.action_space.shape[0]
        policy = ActorCritic(observation_size, action_size).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        if args.bc_checkpoint is not None:
            load_residual_bc_warm_start(args.bc_checkpoint, policy, observation_size, action_size)

        timesteps = 0
        update_index = 0
        episode_return = 0.0
        while timesteps < args.total_timesteps:
            current_rollout_steps = min(args.rollout_steps, args.total_timesteps - timesteps)
            rollout, observation, completed_returns, episode_return = collect_rollout(
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
            if completed_returns:
                return_text = f"return={float(np.mean(completed_returns)):.3f}"
            else:
                return_text = f"partial_return={episode_return:.3f}"
            print(
                f"update={update_index} timesteps={timesteps} {return_text} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}"
            )
            if update_index % args.checkpoint_interval == 0:
                save_checkpoint(args.output, policy, optimizer, observation_size, action_size, timesteps)
        save_checkpoint(args.output, policy, optimizer, observation_size, action_size, timesteps)
        print(f"saved PPO residual policy: {args.output}")
    finally:
        environment.close()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.total_timesteps = min(args.total_timesteps, 32)
        args.rollout_steps = min(args.rollout_steps, 16)
        args.epochs = 1
        args.minibatch_size = min(args.minibatch_size, args.rollout_steps)
        args.output = Path("artifacts") / "ppo_residual_smoke.pt"
    train(args)


if __name__ == "__main__":
    main()
