"""Static contract tests for the v2 fixed-gain MuJoCo-Warp PPO entry."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase, main

import yaml

import train_warp_ppo
from train_warp_ppo import (
    FIXED_GAIN_CONTROLLER_BACKEND,
    FLAT_PPO_BACKEND,
    REWARD_SCHEMA,
    WarpFlatPpoConfigError,
    _load_flat_controller_config,
    _load_flat_walking_config,
    _load_scope_config,
    _validate_batch_for_flat_training,
    load_flat_ppo_training_config,
)
from warp_env import load_warp_batch_config
from warp_task import WarpFlatWalkingConfig


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = ROOT / "configs" / "warp_flat_ppo.yaml"


class FixedGainFlatPpoManifestTest(TestCase):
    def test_v2_manifest_is_fixed_gain_flat_only(self) -> None:
        config = load_flat_ppo_training_config(TRAIN_CONFIG)

        self.assertEqual(config.backend, FLAT_PPO_BACKEND)
        self.assertEqual(config.reward_schema, REWARD_SCHEMA)
        self.assertEqual(config.scope.controller_backend, FIXED_GAIN_CONTROLLER_BACKEND)
        self.assertEqual(config.scope.task_mode, "flat_walking_only")
        self.assertFalse(config.scope.terrain_enabled)
        self.assertFalse(config.scope.jump_enabled)
        self.assertFalse(config.scope.domain_randomization_enabled)
        self.assertFalse(config.scope.dynamic_lqr_enabled)
        self.assertTrue(config.scope.zero_command_only)
        self.assertFalse(config.scope.leg_action_enabled)
        self.assertEqual(config.flat_walking["command_speed_mps"], 0.0)
        self.assertEqual(config.flat_walking["command_yaw_rate_rad_s"], 0.0)
        self.assertFalse(config.flat_walking["direct_control_mode"])
        self.assertFalse(config.flat_walking["leg_action_enabled"])
        self.assertEqual(tuple(config.flat_walking["residual_limits"])[6], 0.0)
        self.assertTrue(config.flat_walking["terrain_compensated_leg_reward"]["enabled"])
        self.assertEqual(config.flat_controller.command_speed_mps, 0.0)
        self.assertEqual(config.flat_controller.command_yaw_rate_rad_s, 0.0)
        self.assertLessEqual(config.flat_controller.max_torque_fraction, 0.80)
        self.assertLessEqual(
            config.flat_controller.gas_spring_torque_nm,
            config.flat_controller.gas_spring_max_abs_generalized_force_nm,
        )
        self.assertEqual(config.stability_gate.duration_seconds, 8.0)
        self.assertEqual(config.stability_gate.required_num_worlds, 128)
        self.assertTrue(config.stability_gate.zero_residual)
        self.assertTrue(config.stability_gate.require_no_terminated)
        self.assertTrue(config.stability_gate.require_no_overflow)
        self.assertTrue(config.stability_gate.require_finite_state)
        self.assertIn("fixed_gain_v2", config.output.checkpoint_path.name)
        self.assertIn("fixed_gain_v2", config.smoke.checkpoint_path.name)
        self.assertFalse(hasattr(train_warp_ppo, "_CalibratedFlatResidualVectorEnv"))

    def test_batch_matches_the_required_128_world_gate(self) -> None:
        config = load_flat_ppo_training_config(TRAIN_CONFIG)
        batch_config = load_warp_batch_config(config.batch_config_path)
        task_config = WarpFlatWalkingConfig.from_mapping(config.flat_walking)

        _validate_batch_for_flat_training(batch_config, config, task_config)
        self.assertEqual(batch_config.num_worlds, 128)
        self.assertLessEqual(batch_config.safety.torque_fraction_of_rated, 0.80)

    def test_strict_controller_section_rejects_missing_parameter(self) -> None:
        payload = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
        payload["flat_controller"].pop("yaw_alignment_enabled")

        with self.assertRaisesRegex(WarpFlatPpoConfigError, "flat_controller keys are invalid"):
            _load_flat_controller_config(payload["flat_controller"])

    def test_scope_rejects_terrain_and_leg_policy_authority(self) -> None:
        terrain_payload = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
        terrain_payload["scope"]["terrain_enabled"] = True
        with self.assertRaisesRegex(WarpFlatPpoConfigError, "terrain, jump, and domain randomization"):
            _load_scope_config(terrain_payload["scope"])

        leg_payload = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
        leg_payload["flat_walking"]["leg_action_enabled"] = True
        with self.assertRaisesRegex(WarpFlatPpoConfigError, "leg_action_enabled must be false"):
            _load_flat_walking_config(leg_payload["flat_walking"])


if __name__ == "__main__":
    main()
