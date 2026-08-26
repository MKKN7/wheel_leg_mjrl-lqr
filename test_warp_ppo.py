"""Focused tests for CUDA vector PPO storage, masking, and GAE semantics."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - the robot7 environment supplies torch
    torch = None  # type: ignore[assignment]

if torch is not None:
    from train_ppo import ActorCritic
    from warp_ppo import (
        WARP_ACTION_SIZE,
        WARP_OBSERVATION_SIZE,
        WarpPPOCollector,
        WarpVectorStep,
        _sanitize_step,
        compute_vector_gae,
        update_policy_cuda,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class WarpPPOContractTest(unittest.TestCase):
    def test_vector_gae_resets_at_terminated_world(self) -> None:
        device = torch.device("cpu")
        rewards = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device)
        values = torch.zeros_like(rewards)
        bootstrap = torch.tensor([[0.0, 2.0], [0.0, 3.0]], device=device)
        continuation = torch.tensor([[0.0, 1.0], [1.0, 1.0]], device=device)
        advantages, returns = compute_vector_gae(
            rewards,
            values,
            bootstrap,
            continuation,
            gamma=0.9,
            gae_lambda=0.95,
        )
        # World 0 is terminated at t=0, so its later transition cannot leak
        # backward through GAE. World 1 remains connected across both steps.
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0)
        self.assertGreater(float(advantages[0, 1]), float(advantages[0, 0]))
        torch.testing.assert_close(returns, advantages)

    def test_terminal_actor_masks_preserve_safety_samples_but_reject_invalid_data(self) -> None:
        """Physical safety termination is learnable; malformed data is not."""

        worlds = 2
        masks = torch.zeros((worlds, WARP_ACTION_SIZE), dtype=torch.float32)
        masks[:, :6] = 1.0
        valid_terminal_step = WarpVectorStep(
            observations=torch.zeros((worlds, WARP_OBSERVATION_SIZE), dtype=torch.float32),
            rewards=torch.ones(worlds, dtype=torch.float32),
            # World zero represents a finite safety/fall termination.
            terminated=torch.tensor([True, False]),
            truncated=torch.zeros(worlds, dtype=torch.bool),
            policy_action_masks=masks,
        )

        safety_terminal = _sanitize_step(
            valid_terminal_step,
            torch.zeros(worlds, dtype=torch.bool),
        )
        self.assertTrue(bool(safety_terminal.terminated[0]))
        torch.testing.assert_close(
            safety_terminal.policy_action_masks[0, :6],
            torch.ones(6, dtype=torch.float32),
        )
        self.assertEqual(float(safety_terminal.policy_action_masks[0, 6]), 0.0)

        invalid_policy_action = _sanitize_step(
            valid_terminal_step,
            torch.tensor([False, True]),
        )
        self.assertTrue(bool(invalid_policy_action.terminated[1]))
        torch.testing.assert_close(
            invalid_policy_action.policy_action_masks[1],
            torch.zeros(WARP_ACTION_SIZE, dtype=torch.float32),
        )

        malformed_observation = valid_terminal_step.observations.clone()
        malformed_observation[1, 0] = float("nan")
        invalid_environment_output = _sanitize_step(
            WarpVectorStep(
                observations=malformed_observation,
                rewards=valid_terminal_step.rewards,
                terminated=valid_terminal_step.terminated,
                truncated=valid_terminal_step.truncated,
                policy_action_masks=valid_terminal_step.policy_action_masks,
            ),
            torch.zeros(worlds, dtype=torch.bool),
        )
        self.assertTrue(bool(invalid_environment_output.terminated[1]))
        torch.testing.assert_close(
            invalid_environment_output.policy_action_masks[1],
            torch.zeros(WARP_ACTION_SIZE, dtype=torch.float32),
        )

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required")
    def test_collector_preserves_67_7_and_masked_reset_on_cuda(self) -> None:
        device = torch.device("cuda:0")

        class FakeWarpEnv:
            num_worlds = 4

            def __init__(self) -> None:
                self.device = device
                self.state = torch.zeros(
                    (self.num_worlds, WARP_OBSERVATION_SIZE), device=device, dtype=torch.float32
                )
                self.steps = torch.zeros(self.num_worlds, device=device, dtype=torch.int32)
                self.reset_masks: list[torch.Tensor] = []

            def reset(self, world_mask: torch.Tensor | None = None) -> torch.Tensor:
                if world_mask is None:
                    self.state.zero_()
                    self.steps.zero_()
                else:
                    self.reset_masks.append(world_mask.detach().clone())
                    self.state.masked_fill_(world_mask.unsqueeze(-1), 0.0)
                    self.steps.masked_fill_(world_mask, 0)
                return self.state.contiguous()

            def step_policy(self, actions: torch.Tensor) -> WarpVectorStep:
                self.steps.add_(1)
                self.state[:, :WARP_ACTION_SIZE].add_(actions)
                terminated = self.steps.eq(2)
                truncated = torch.zeros(self.num_worlds, device=device, dtype=torch.bool)
                masks = torch.ones(
                    (self.num_worlds, WARP_ACTION_SIZE), device=device, dtype=torch.float32
                )
                masks[0, 3:] = 0.0
                return WarpVectorStep(
                    observations=self.state.contiguous(),
                    rewards=torch.ones(self.num_worlds, device=device, dtype=torch.float32),
                    terminated=terminated,
                    truncated=truncated,
                    policy_action_masks=masks.contiguous(),
                )

        environment = FakeWarpEnv()
        policy = ActorCritic(WARP_OBSERVATION_SIZE, WARP_ACTION_SIZE, hidden_size=32).to(device)
        collector = WarpPPOCollector(environment, policy, gamma=0.99, gae_lambda=0.95)
        rollout, next_observations, episode_returns = collector.collect(4)
        self.assertEqual(tuple(rollout.observations.shape), (4, 4, 67))
        self.assertEqual(tuple(rollout.actions.shape), (4, 4, 7))
        self.assertEqual(tuple(rollout.policy_action_masks.shape), (4, 4, 7))
        self.assertTrue(rollout.observations.is_cuda)
        self.assertTrue(torch.all(rollout.policy_action_masks[0, 0, 3:] == 0))
        self.assertTrue(next_observations.is_cuda)
        self.assertTrue(episode_returns.is_cuda)
        self.assertTrue(any(bool(mask.any().item()) for mask in environment.reset_masks))

        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = update_policy_cuda(
            policy,
            optimizer,
            rollout,
            epochs=1,
            minibatch_size=8,
            clip_ratio=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.001,
            max_gradient_norm=0.5,
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

        class NonFinitePolicy(ActorCritic):
            def sample_action(self, observations: torch.Tensor, deterministic: bool = False):
                action, log_probability, value = super().sample_action(observations, deterministic)
                action = action.clone()
                action[0, 0] = float("nan")
                return action, log_probability, value

        malformed_environment = FakeWarpEnv()
        malformed_policy = NonFinitePolicy(
            WARP_OBSERVATION_SIZE, WARP_ACTION_SIZE, hidden_size=32
        ).to(device)
        malformed_rollout, _, _ = WarpPPOCollector(
            malformed_environment, malformed_policy
        ).collect(1)
        self.assertTrue(torch.all(malformed_rollout.policy_action_masks[0, 0] == 0.0))
        self.assertAlmostEqual(float(malformed_rollout.rewards[0, 0].item()), -30.0)
        self.assertEqual(float(malformed_rollout.continuation_masks[0, 0].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
