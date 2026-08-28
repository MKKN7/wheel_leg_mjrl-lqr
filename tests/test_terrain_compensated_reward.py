"""CPU contracts for terrain-compensated leg and attitude rewards.

These tests deliberately exercise only resident-tensor reward math.  They do
not construct a MuJoCo model, so they catch a sign/validity regression before
an expensive CUDA curriculum run is started.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from warp_task import (
    TerrainCompensatedLegRewardSettings,
    WarpFlatWalkingConfig,
    WarpFlatWalkingTask,
    terrain_adaptive_attitude_weight,
    terrain_compensated_leg_difference_cost,
)


class TerrainCompensatedRewardContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = TerrainCompensatedLegRewardSettings(
            enabled=True,
            support_height_to_leg_difference_gain=-1.0,
            target_leg_difference_limit_m=0.200,
            reward_error_scale_m=0.050,
            flat_leg_difference_penalty=2.0,
            uneven_leg_difference_penalty=0.50,
            turning_leg_penalty_fraction=0.25,
            relief_start_m=0.010,
            relief_full_m=0.150,
            flat_attitude_reward_weight=0.30,
            uneven_attitude_reward_weight=0.50,
            # The reward target is deliberately independent of the P0 raw
            # spread limit.  A sudden valid step sample may request a large
            # eventual compensation, but it must not become an instantaneous
            # target-error estop while the legs are moving safely.
            terrain_raw_leg_difference_limit_m=0.015,
        )

    def test_flat_support_preserves_a_zero_leg_difference_target(self) -> None:
        leg_lengths = torch.tensor(((0.250, 0.250),), dtype=torch.float32)
        support_heights = torch.zeros((1, 2), dtype=torch.float32)
        support_valid = torch.ones((1, 2), dtype=torch.bool)

        cost, target_difference, compensation_valid = terrain_compensated_leg_difference_cost(
            torch,
            leg_lengths,
            support_heights,
            support_valid,
            self.settings,
        )

        torch.testing.assert_close(target_difference, torch.zeros(1, dtype=torch.float32))
        torch.testing.assert_close(cost, torch.zeros(1, dtype=torch.float32))
        self.assertTrue(compensation_valid.item())

    def test_raised_left_support_prefers_the_matching_signed_leg_difference(self) -> None:
        # A higher left support needs a shorter left leg to keep the chassis
        # level.  The target is therefore support_right - support_left.
        support_heights = torch.tensor(((0.070, 0.000),), dtype=torch.float32)
        support_valid = torch.ones((1, 2), dtype=torch.bool)
        matching_legs = torch.tensor(((0.215, 0.285),), dtype=torch.float32)
        equal_legs = torch.tensor(((0.280, 0.280),), dtype=torch.float32)

        matching_cost, target_difference, matching_valid = terrain_compensated_leg_difference_cost(
            torch,
            matching_legs,
            support_heights,
            support_valid,
            self.settings,
        )
        equal_cost, _, equal_valid = terrain_compensated_leg_difference_cost(
            torch,
            equal_legs,
            support_heights,
            support_valid,
            self.settings,
        )

        torch.testing.assert_close(target_difference, torch.tensor((-0.070,), dtype=torch.float32))
        self.assertLess(matching_cost.item(), equal_cost.item())
        self.assertTrue(matching_valid.item())
        self.assertTrue(equal_valid.item())

    def test_invalid_or_lost_support_cannot_grant_compensation_reward(self) -> None:
        support_heights = torch.tensor(((0.070, 0.000),), dtype=torch.float32)
        matching_legs = torch.tensor(((0.215, 0.285),), dtype=torch.float32)
        support_lost = torch.tensor(((False, True),), dtype=torch.bool)

        cost, target_difference, compensation_valid = terrain_compensated_leg_difference_cost(
            torch,
            matching_legs,
            support_heights,
            support_lost,
            self.settings,
        )

        # No valid two-sided support means no terrain target.  The cost must
        # fall back to the raw asymmetry, rather than rewarding a potentially
        # stale terrain sample while the robot is losing support.
        torch.testing.assert_close(target_difference, torch.zeros(1, dtype=torch.float32))
        self.assertGreater(cost.item(), 0.0)
        self.assertFalse(compensation_valid.item())

    def test_target_difference_is_capped_before_it_can_drive_an_unsafe_spread(self) -> None:
        leg_lengths = torch.tensor(((0.205, 0.300),), dtype=torch.float32)
        support_heights = torch.tensor(((1.000, 0.000),), dtype=torch.float32)
        support_valid = torch.ones((1, 2), dtype=torch.bool)

        _, target_difference, compensation_valid = terrain_compensated_leg_difference_cost(
            torch,
            leg_lengths,
            support_heights,
            support_valid,
            self.settings,
        )

        torch.testing.assert_close(target_difference, torch.tensor((-0.200,), dtype=torch.float32))
        self.assertTrue(compensation_valid.item())

    def test_attitude_weight_increases_with_cross_slope_and_stays_bounded(self) -> None:
        supports = torch.tensor(
            ((0.000, 0.000), (0.050, 0.000), (0.150, 0.000), (0.500, 0.000)),
            dtype=torch.float32,
        )
        valid = torch.ones((4, 2), dtype=torch.bool)

        weights = terrain_adaptive_attitude_weight(torch, supports, valid, self.settings)

        self.assertAlmostEqual(weights[0].item(), self.settings.flat_attitude_reward_weight, places=6)
        self.assertGreater(weights[1].item(), weights[0].item())
        self.assertGreaterEqual(weights[2].item(), weights[1].item())
        self.assertLessEqual(weights.max().item(), self.settings.uneven_attitude_reward_weight)
        self.assertGreaterEqual(weights.min().item(), self.settings.flat_attitude_reward_weight)

    def test_invalid_support_uses_flat_attitude_weight(self) -> None:
        supports = torch.tensor(((0.150, 0.000),), dtype=torch.float32)
        invalid = torch.tensor(((True, False),), dtype=torch.bool)

        weights = terrain_adaptive_attitude_weight(torch, supports, invalid, self.settings)

        torch.testing.assert_close(
            weights,
            torch.tensor((self.settings.flat_attitude_reward_weight,), dtype=torch.float32),
        )

    def test_config_parser_rejects_unsafe_or_malformed_terrain_reward_settings(self) -> None:
        valid = {
            "terrain_compensated_leg_reward": {
                "enabled": True,
                "support_height_to_leg_difference_gain": -1.0,
                "target_leg_difference_limit_m": 0.200,
                "reward_error_scale_m": 0.050,
                "flat_leg_difference_penalty": 2.0,
                "uneven_leg_difference_penalty": 0.50,
                "turning_leg_penalty_fraction": 0.25,
                "relief_start_m": 0.010,
                "relief_full_m": 0.150,
                "flat_attitude_reward_weight": 0.30,
                "uneven_attitude_reward_weight": 0.50,
                "terrain_raw_leg_difference_limit_m": 0.015,
            }
        }
        parsed = WarpFlatWalkingConfig.from_mapping(valid)
        self.assertEqual(parsed.terrain_compensated_leg_reward, self.settings)

        zero_target_limit = {
            "terrain_compensated_leg_reward": {
                **valid["terrain_compensated_leg_reward"],
                "target_leg_difference_limit_m": 0.0,
            }
        }
        inverted_weights = {
            "terrain_compensated_leg_reward": {
                **valid["terrain_compensated_leg_reward"],
                "flat_attitude_reward_weight": 0.51,
            }
        }
        zero_raw_limit = {
            "terrain_compensated_leg_reward": {
                **valid["terrain_compensated_leg_reward"],
                "terrain_raw_leg_difference_limit_m": 0.0,
            }
        }
        malformed = {"terrain_compensated_leg_reward": []}
        for candidate in (zero_target_limit, inverted_weights, zero_raw_limit, malformed):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    WarpFlatWalkingConfig.from_mapping(candidate)

    def test_parser_allows_reward_target_larger_than_the_raw_p0_spread_limit(self) -> None:
        config = WarpFlatWalkingConfig.from_mapping(
            {
                "terrain_compensated_leg_reward": {
                    "enabled": True,
                    "support_height_to_leg_difference_gain": -1.0,
                    "target_leg_difference_limit_m": 0.200,
                    "reward_error_scale_m": 0.050,
                    "flat_leg_difference_penalty": 2.0,
                    "uneven_leg_difference_penalty": 0.50,
                    "turning_leg_penalty_fraction": 0.25,
                    "relief_start_m": 0.010,
                    "relief_full_m": 0.150,
                    "flat_attitude_reward_weight": 0.30,
                    "uneven_attitude_reward_weight": 0.50,
                    "terrain_raw_leg_difference_limit_m": 0.015,
                }
            }
        )

        settings = config.terrain_compensated_leg_reward
        self.assertGreater(settings.target_leg_difference_limit_m, settings.terrain_raw_leg_difference_limit_m)

    def test_p0_spread_envelope_allows_only_known_uneven_support_or_bounded_flight(self) -> None:
        """A terrain target never becomes an instantaneous one-substep estop."""

        worlds = 4
        task = object.__new__(WarpFlatWalkingTask)
        task.torch = torch
        task.config = SimpleNamespace(
            terrain_compensated_leg_reward=TerrainCompensatedLegRewardSettings(
                enabled=True,
                terrain_raw_leg_difference_limit_m=0.200,
            ),
            max_leg_length_difference_m=0.015,
        )
        task._terrain_leg_raw_difference_m = torch.tensor((0.020, 0.160, 0.210, 0.160))
        task._terrain_leg_known_uneven_support = torch.tensor((False, True, True, False))
        task._contact_loss_exempt = torch.tensor((False, False, False, True))
        task._terrain_leg_raw_violation = torch.zeros(worlds, dtype=torch.bool)
        task._terrain_leg_fallback_violation = torch.zeros(worlds, dtype=torch.bool)
        task._terrain_leg_safety_violation = torch.zeros(worlds, dtype=torch.bool)
        task._terrain_leg_reason_mask = torch.zeros(worlds, dtype=torch.bool)
        result = SimpleNamespace(
            safe_controls=torch.ones((worlds, 2), dtype=torch.float32),
            leg_limit=torch.zeros(worlds, dtype=torch.bool),
            failure=torch.zeros(worlds, dtype=torch.bool),
            terminated=torch.zeros(worlds, dtype=torch.bool),
            reason_code=torch.zeros(worlds, dtype=torch.int64),
        )

        WarpFlatWalkingTask._apply_terrain_leg_safety(task, result)

        # Unknown/flat spread over 15 mm stops. Known uneven support accepts
        # the bounded 160 mm transition; >200 mm still stops. A supervisor-
        # bounded flight uses that same absolute envelope rather than a stale
        # terrain target, so its 160 mm spread remains admissible.
        self.assertEqual(result.terminated.tolist(), [True, False, True, False])
        self.assertEqual(result.leg_limit.tolist(), [True, False, True, False])
        torch.testing.assert_close(result.safe_controls[0], torch.zeros(2))
        torch.testing.assert_close(result.safe_controls[2], torch.zeros(2))
        torch.testing.assert_close(result.safe_controls[1], torch.ones(2))
        torch.testing.assert_close(result.safe_controls[3], torch.ones(2))


if __name__ == "__main__":
    unittest.main()
