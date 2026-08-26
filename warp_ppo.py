"""CUDA-native vector PPO collector and updater for the Warp backend.

The module deliberately owns only the RL plumbing.  A backend supplies the
GPU observation/reward/reset contract and is responsible for the validated
controller, actuator derating, and safety state machine.  This keeps the
policy path honest: observations are exactly the repository's 67-element
contract, actions are exactly the seven residual channels, and an authority
mask can remove channels without a CPU round trip.

Terrain routes, jumps, and raw six-actuator controls are intentionally not
implemented here.  A backend must reject those modes until their controller
parity has been established.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


WARP_OBSERVATION_SIZE = 67
WARP_ACTION_SIZE = 7


class WarpVectorEnv(Protocol):
    """Minimal GPU-only environment contract used by :class:`WarpPPOCollector`.

    ``reset`` must return a full ``[num_worlds, 67]`` CUDA observation tensor,
    including when ``world_mask`` selects only a subset.  ``step_policy`` must
    apply the policy's seven residual channels through the backend controller
    and return a full post-step tensor.  The backend may keep any controller
    state internally, but it must not copy per-world data through Python.
    """

    num_worlds: int
    device: torch.device

    def reset(self, world_mask: Tensor | None = None) -> Tensor:
        ...

    def step_policy(self, actions: Tensor) -> "WarpVectorStep":
        ...


@dataclass(frozen=True)
class WarpVectorStep:
    """GPU tensors returned by one vector policy action."""

    observations: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    policy_action_masks: Tensor


@dataclass(frozen=True)
class WarpVectorRollout:
    """Time-major CUDA rollout storage.

    The first two dimensions are ``[time, world]``.  Keeping that layout until
    the PPO update makes per-world GAE/reset semantics explicit and avoids
    accidental CPU flattening in the collector.
    """

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
    completed_episode_returns: Tensor | None = None

    @property
    def time_steps(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_worlds(self) -> int:
        return int(self.observations.shape[1])

    def flatten(self) -> "WarpVectorRollout":
        """Flatten time/world dimensions for minibatch PPO updates."""

        time_steps, num_worlds = self.observations.shape[:2]
        flat = time_steps * num_worlds

        def flatten_tensor(value: Tensor) -> Tensor:
            return value.reshape(flat, *value.shape[2:])

        return WarpVectorRollout(
            observations=flatten_tensor(self.observations),
            actions=flatten_tensor(self.actions),
            policy_action_masks=flatten_tensor(self.policy_action_masks),
            log_probabilities=flatten_tensor(self.log_probabilities),
            rewards=flatten_tensor(self.rewards),
            values=flatten_tensor(self.values),
            bootstrap_values=flatten_tensor(self.bootstrap_values),
            continuation_masks=flatten_tensor(self.continuation_masks),
            advantages=flatten_tensor(self.advantages),
            returns=flatten_tensor(self.returns),
            completed_episode_returns=(
                None
                if self.completed_episode_returns is None
                else flatten_tensor(self.completed_episode_returns)
            ),
        )


def _require_cuda_tensor(
    value: Tensor,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_cuda:
        raise ValueError(f"{name} must remain on CUDA; got {value.device}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}; expected {shape}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} has dtype {value.dtype}; expected {dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} is on {value.device}; expected {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_step(step: WarpVectorStep, *, num_worlds: int, device: torch.device) -> None:
    _require_cuda_tensor(
        step.observations,
        name="step.observations",
        shape=(num_worlds, WARP_OBSERVATION_SIZE),
        dtype=torch.float32,
        device=device,
    )
    _require_cuda_tensor(
        step.rewards,
        name="step.rewards",
        shape=(num_worlds,),
        dtype=torch.float32,
        device=device,
    )
    for name, value in (("terminated", step.terminated), ("truncated", step.truncated)):
        _require_cuda_tensor(
            value,
            name=f"step.{name}",
            shape=(num_worlds,),
            dtype=torch.bool,
            device=device,
        )
    _require_cuda_tensor(
        step.policy_action_masks,
        name="step.policy_action_masks",
        shape=(num_worlds, WARP_ACTION_SIZE),
        dtype=torch.float32,
        device=device,
    )


def _sanitize_step(step: WarpVectorStep, invalid_policy_worlds: Tensor) -> WarpVectorStep:
    """Replace malformed GPU outputs and terminate only affected worlds.

    All predicates stay as CUDA tensors.  This is intentionally a data-path
    guard rather than a Python exception: a malformed policy or sensor sample
    must not synchronize every world or poison the PPO loss.  The physics
    backend's independent estop remains authoritative for the mechanism.  A
    finite physical terminal transition remains an actor sample; only a
    malformed policy/action/environment output loses actor authority.
    """

    finite_observation = torch.isfinite(step.observations).all(dim=1)
    finite_reward = torch.isfinite(step.rewards)
    finite_mask = torch.isfinite(step.policy_action_masks).all(dim=1)
    mask_in_range = (
        (step.policy_action_masks >= 0.0) & (step.policy_action_masks <= 1.0)
    ).all(dim=1)
    invalid = invalid_policy_worlds | ~finite_observation | ~finite_reward | ~finite_mask | ~mask_in_range
    observations = torch.nan_to_num(
        step.observations, nan=0.0, posinf=10.0, neginf=-10.0
    ).clamp(-10.0, 10.0)
    rewards = torch.nan_to_num(
        step.rewards, nan=-30.0, posinf=-30.0, neginf=-30.0
    )
    rewards = torch.where(invalid, torch.full_like(rewards, -30.0), rewards)
    masks = torch.nan_to_num(
        step.policy_action_masks, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    terminated = step.terminated | invalid
    truncated = step.truncated & ~terminated
    # Terminal transitions already stop GAE via ``continuation`` in the
    # collector.  Preserve the action likelihood for a finite safety/fall
    # transition so the actor learns from the action that caused it.  Invalid
    # policy actions or malformed environment outputs instead fail closed.
    masks = masks * (~invalid).to(dtype=torch.float32).unsqueeze(1)
    return WarpVectorStep(
        observations=observations.contiguous(),
        rewards=rewards.contiguous(),
        terminated=terminated.contiguous(),
        truncated=truncated.contiguous(),
        policy_action_masks=masks.contiguous(),
    )


def compute_vector_gae(
    rewards: Tensor,
    values: Tensor,
    bootstrap_values: Tensor,
    continuation_masks: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute per-world GAE without leaving the policy device."""

    if rewards.ndim != 2:
        raise ValueError("rewards must have [time, world] shape")
    if values.shape != rewards.shape or bootstrap_values.shape != rewards.shape:
        raise ValueError("values and bootstrap_values must match rewards shape")
    if continuation_masks.shape != rewards.shape:
        raise ValueError("continuation_masks must match rewards shape")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be within [0, 1]")
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
    gamma_value = float(gamma)
    gae_value = float(gae_lambda)
    for index in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[index] + gamma_value * bootstrap_values[index] - values[index]
        last_advantage = delta + gamma_value * gae_value * continuation_masks[index] * last_advantage
        advantages[index] = last_advantage
    return advantages, advantages + values


class WarpPPOCollector:
    """Collect a fixed-length CUDA rollout from a vector backend."""

    def __init__(
        self,
        environment: WarpVectorEnv,
        policy: nn.Module,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if not isinstance(environment.num_worlds, int) or environment.num_worlds < 1:
            raise ValueError("environment.num_worlds must be a positive integer")
        device = torch.device(environment.device)
        if device.type != "cuda":
            raise ValueError("WarpPPOCollector requires a CUDA environment device")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be within [0, 1]")
        self.environment = environment
        self.policy = policy
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.device = device
        self.num_worlds = int(environment.num_worlds)

    def collect(
        self,
        rollout_steps: int,
        *,
        observations: Tensor | None = None,
        episode_returns: Tensor | None = None,
    ) -> tuple[WarpVectorRollout, Tensor, Tensor]:
        if isinstance(rollout_steps, bool) or rollout_steps < 1:
            raise ValueError("rollout_steps must be a positive integer")
        if observations is None:
            observations = self.environment.reset()
        _require_cuda_tensor(
            observations,
            name="observations",
            shape=(self.num_worlds, WARP_OBSERVATION_SIZE),
            dtype=torch.float32,
            device=self.device,
        )
        if episode_returns is None:
            episode_returns = torch.zeros(self.num_worlds, dtype=torch.float32, device=self.device)
        _require_cuda_tensor(
            episode_returns,
            name="episode_returns",
            shape=(self.num_worlds,),
            dtype=torch.float32,
            device=self.device,
        )

        storage_shape = (rollout_steps, self.num_worlds)
        observations_storage = torch.empty(
            (*storage_shape, WARP_OBSERVATION_SIZE), dtype=torch.float32, device=self.device
        )
        actions_storage = torch.empty(
            (*storage_shape, WARP_ACTION_SIZE), dtype=torch.float32, device=self.device
        )
        masks_storage = torch.empty_like(actions_storage)
        log_probabilities_storage = torch.empty(storage_shape, dtype=torch.float32, device=self.device)
        rewards_storage = torch.empty(storage_shape, dtype=torch.float32, device=self.device)
        values_storage = torch.empty(storage_shape, dtype=torch.float32, device=self.device)
        bootstrap_storage = torch.empty(storage_shape, dtype=torch.float32, device=self.device)
        continuation_storage = torch.empty(storage_shape, dtype=torch.float32, device=self.device)
        completed_returns_storage = torch.zeros(
            storage_shape, dtype=torch.float32, device=self.device
        )

        for index in range(rollout_steps):
            with torch.no_grad():
                actions, log_probability_dimensions, values = self.policy.sample_action(observations)
            _require_cuda_tensor(
                actions,
                name="policy actions",
                shape=(self.num_worlds, WARP_ACTION_SIZE),
                dtype=torch.float32,
                device=self.device,
            )
            _require_cuda_tensor(
                log_probability_dimensions,
                name="policy log probabilities",
                shape=(self.num_worlds, WARP_ACTION_SIZE),
                dtype=torch.float32,
                device=self.device,
            )
            _require_cuda_tensor(
                values,
                name="policy values",
                shape=(self.num_worlds,),
                dtype=torch.float32,
                device=self.device,
            )
            # ActorCritic already emits tanh-bounded actions.  Keep a final
            # device-side clip here so a custom policy cannot bypass the
            # normalized action contract before the backend applies its
            # actuator and 80%-rated-torque limits.  Non-finite policy rows
            # are replaced by a zero safe action and terminated after this
            # masked step, without a host synchronization.
            policy_finite = (
                torch.isfinite(actions).all(dim=1)
                & torch.isfinite(log_probability_dimensions).all(dim=1)
                & torch.isfinite(values)
            )
            actions = torch.nan_to_num(
                actions, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0).contiguous()
            log_probability_dimensions = torch.nan_to_num(
                log_probability_dimensions, nan=0.0, posinf=0.0, neginf=0.0
            )
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            step = self.environment.step_policy(actions)
            if not isinstance(step, WarpVectorStep):
                raise TypeError("environment.step_policy must return WarpVectorStep")
            _validate_step(step, num_worlds=self.num_worlds, device=self.device)
            step = _sanitize_step(step, ~policy_finite)
            masks = step.policy_action_masks
            terminated = step.terminated
            truncated = step.truncated
            done = terminated | truncated

            with torch.no_grad():
                next_values = self.policy.critic(step.observations).squeeze(-1)
            log_probability = (log_probability_dimensions * masks).sum(dim=-1)
            bootstrap_values = torch.where(terminated, torch.zeros_like(next_values), next_values)
            continuation = (~done).to(dtype=torch.float32)

            observations_storage[index].copy_(observations)
            actions_storage[index].copy_(actions)
            masks_storage[index].copy_(masks)
            log_probabilities_storage[index].copy_(log_probability)
            rewards_storage[index].copy_(step.rewards)
            values_storage[index].copy_(values)
            bootstrap_storage[index].copy_(bootstrap_values)
            continuation_storage[index].copy_(continuation)

            episode_returns.add_(step.rewards)
            completed_returns_storage[index].copy_(
                torch.where(done, episode_returns, torch.zeros_like(episode_returns))
            )
            # Always issue a masked reset.  An all-false mask is a no-op in the
            # backend, and this avoids a host ``done.any().item()`` sync on
            # every policy step.  The reset API returns the full batch.
            reset_observations = self.environment.reset(done)
            _require_cuda_tensor(
                reset_observations,
                name="reset observations",
                shape=(self.num_worlds, WARP_OBSERVATION_SIZE),
                dtype=torch.float32,
                device=self.device,
            )
            observations = torch.where(done.unsqueeze(-1), reset_observations, step.observations)
            episode_returns.masked_fill_(done, 0.0)

        advantages, returns = compute_vector_gae(
            rewards_storage,
            values_storage,
            bootstrap_storage,
            continuation_storage,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        return (
            WarpVectorRollout(
                observations=observations_storage,
                actions=actions_storage,
                policy_action_masks=masks_storage,
                log_probabilities=log_probabilities_storage,
                rewards=rewards_storage,
                values=values_storage,
                bootstrap_values=bootstrap_storage,
                continuation_masks=continuation_storage,
                advantages=advantages,
                returns=returns,
                completed_episode_returns=completed_returns_storage,
            ),
            observations,
            episode_returns,
        )


def update_policy_cuda(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: WarpVectorRollout,
    *,
    epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_gradient_norm: float,
) -> dict[str, float]:
    """Run masked PPO updates on CUDA tensors.

    This mirrors ``train_ppo.update_policy`` but avoids flattening or
    normalizing advantages on the host.  Samples with no granted authority
    contribute neither policy loss nor entropy, while their value target is
    retained for critic learning.
    """

    if epochs < 1 or minibatch_size < 1:
        raise ValueError("epochs and minibatch_size must be positive")
    if not 0.0 < clip_ratio < 1.0:
        raise ValueError("clip_ratio must be within (0, 1)")
    flat = rollout.flatten()
    device = flat.observations.device
    if not device.type == "cuda":
        raise ValueError("update_policy_cuda requires CUDA rollout tensors")
    batch_size = int(flat.observations.shape[0])
    if batch_size < 1:
        raise ValueError("rollout cannot be empty")

    active = flat.policy_action_masks.any(dim=-1).to(dtype=torch.float32)
    active_count = active.sum().clamp_min(1.0)
    mean_advantage = (flat.advantages * active).sum() / active_count
    centered = (flat.advantages - mean_advantage) * active
    variance = centered.square().sum() / active_count
    normalized_advantages = centered / torch.sqrt(variance + 1e-8)

    policy_losses: list[Tensor] = []
    value_losses: list[Tensor] = []
    entropy_values: list[Tensor] = []
    for _ in range(epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            log_probability_dimensions, entropy_dimensions, predicted_values = policy.evaluate_actions(
                flat.observations[batch_indices], flat.actions[batch_indices]
            )
            action_mask = flat.policy_action_masks[batch_indices]
            new_log_probabilities = (log_probability_dimensions * action_mask).sum(dim=-1)
            ratio = torch.exp(new_log_probabilities - flat.log_probabilities[batch_indices])
            objective = ratio * normalized_advantages[batch_indices]
            clipped_objective = torch.clamp(
                ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * normalized_advantages[batch_indices]
            sample_active = action_mask.any(dim=-1).to(dtype=torch.float32)
            sample_count = sample_active.sum().clamp_min(1.0)
            policy_loss = -(
                torch.minimum(objective, clipped_objective) * sample_active
            ).sum() / sample_count
            entropy_denominator = action_mask.sum().clamp_min(1.0)
            entropy_bonus = (entropy_dimensions * action_mask).sum() / entropy_denominator
            value_loss = torch.nn.functional.mse_loss(
                predicted_values, flat.returns[batch_indices]
            )
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_gradient_norm)
            optimizer.step()
            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropy_values.append(entropy_bonus.detach())

    return {
        "policy_loss": float(torch.stack(policy_losses).mean().detach().cpu().item()),
        "value_loss": float(torch.stack(value_losses).mean().detach().cpu().item()),
        "entropy": float(torch.stack(entropy_values).mean().detach().cpu().item()),
    }


__all__ = [
    "WARP_OBSERVATION_SIZE",
    "WARP_ACTION_SIZE",
    "WarpVectorEnv",
    "WarpVectorStep",
    "WarpVectorRollout",
    "WarpPPOCollector",
    "compute_vector_gae",
    "update_policy_cuda",
]
