"""Parser, capability, and checkpoint metadata tests for GPU curriculum PPO."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from train_warp_curriculum import (
    ACTION_SIZE,
    CURRICULUM_BACKEND,
    CURRICULUM_CHECKPOINT_FORMAT,
    OBSERVATION_SIZE,
    REWARD_SCHEMA,
    WarpCurriculumConfigError,
    load_curriculum_config,
    parse_args,
    _validate_runtime_gate_report,
    validate_checkpoint_metadata,
    validate_stage_capability,
)


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "warp_curriculum_ppo.yaml"


class WarpCurriculumEntryTest(unittest.TestCase):
    def test_manifest_has_strict_67_7_stage_contract(self) -> None:
        config = load_curriculum_config(CONFIG)
        self.assertEqual(config.observation_size, OBSERVATION_SIZE)
        self.assertEqual(config.action_size, ACTION_SIZE)
        self.assertEqual(config.reward_schema, REWARD_SCHEMA)
        self.assertEqual(
            {stage.stage_id for stage in config.stages},
            {
                "rmuc_flat",
                "rmuc_flat_dr",
                "grades",
                "low_speed_turning",
                "official_grade15_up",
                "official_standard_full",
            },
        )
        flat = config.stage("rmuc_flat")
        self.assertEqual(flat.residual_action_mask[-1], 0.0)
        self.assertTrue(config.gpu_task.allow_cpu_residual_warm_start)
        self.assertEqual(config.gpu_task.control_delay_steps, 1)
        validate_stage_capability(config, flat)
        dr_stage = config.stage("rmuc_flat_dr")
        validate_stage_capability(config, dr_stage)

    def test_parser_requires_stage_and_accepts_warm_start(self) -> None:
        args = parse_args([
            "--curriculum", str(CONFIG), "--stage", "rmuc_flat",
            "--init-residual-checkpoint", "warm_start.pt", "--smoke",
        ])
        self.assertEqual(args.stage, "rmuc_flat")
        self.assertEqual(args.init_residual_checkpoint, Path("warm_start.pt"))
        self.assertTrue(args.smoke)

    def test_unproved_grades_are_blocked_before_gpu_setup(self) -> None:
        config = load_curriculum_config(CONFIG)
        with self.assertRaisesRegex(WarpCurriculumConfigError, "no GPU parity"):
            validate_stage_capability(config, config.stage("grades"))

    def test_conditional_gate_report_cannot_be_static_or_incomplete(self) -> None:
        config = load_curriculum_config(CONFIG)
        stage = config.stage("official_grade15_up")
        capability = validate_stage_capability(config, stage)
        report = {
            "stage_id": stage.stage_id,
            "conditional_capability": True,
            "passed": True,
            "num_worlds": 128,
            "terminated_worlds": 0,
            "overflowed_worlds": 0,
            "estopped_worlds": 0,
            "finite_state": True,
            "zero_residual": True,
            "minimum_progress_m": 1.0,
            "speed_mae_mps": 0.1,
            "unsafe_rate": 0.0,
            "first_fault_step": -1,
            "first_fault_reason_code": 0,
        }
        self.assertEqual(_validate_runtime_gate_report(report, stage=stage, capability=capability), report)
        for field in ("passed", "finite_state", "num_worlds"):
            invalid = dict(report)
            invalid.pop(field)
            with self.subTest(field=field):
                with self.assertRaisesRegex(WarpCurriculumConfigError, "conditional GPU gate"):
                    _validate_runtime_gate_report(invalid, stage=stage, capability=capability)

    def test_external_gate_failure_latches_bundle_before_propagating(self) -> None:
        from train_warp_curriculum import run_curriculum_training

        config = load_curriculum_config(CONFIG)
        closed = []

        def fail_gate() -> None:
            raise RuntimeError("intentional gate fault")

        bundle = SimpleNamespace(
            batch=object(),
            task=object(),
            run_stability_gate=fail_gate,
            close=lambda: closed.append(True),
        )
        with (
            mock.patch("train_warp_curriculum.validate_stage_capability"),
            mock.patch("train_warp_curriculum._external_stage_bundle", return_value=bundle),
            self.assertRaisesRegex(RuntimeError, "intentional gate fault"),
        ):
            run_curriculum_training(config, stage_id="official_grade15_up")
        self.assertEqual(closed, [True])

    def test_unknown_root_key_is_rejected(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        path = ROOT / "configs" / ".curriculum_invalid_test.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        try:
            with self.assertRaisesRegex(WarpCurriculumConfigError, "keys are invalid"):
                load_curriculum_config(path)
        finally:
            path.unlink(missing_ok=True)

    def test_checkpoint_metadata_requires_exact_shape_and_authority(self) -> None:
        config = load_curriculum_config(CONFIG)
        stage = config.stage("rmuc_flat")
        metadata = {
            "format_version": CURRICULUM_CHECKPOINT_FORMAT,
            "checkpoint_backend": CURRICULUM_BACKEND,
            "algorithm": "ppo",
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "action_semantics": "seven_dimensional_fixed_gain_residual",
            "reward_schema": REWARD_SCHEMA,
            "policy_action_mask": list(stage.residual_action_mask),
            "model_state_dict": {"actor.0.weight": object()},
        }
        validate_checkpoint_metadata(metadata, config=config, stage=stage)
        metadata["action_size"] = 6
        with self.assertRaisesRegex(WarpCurriculumConfigError, "observation/action"):
            validate_checkpoint_metadata(metadata, config=config, stage=stage)

    def test_legacy_flat_checkpoint_needs_provable_reward_scope(self) -> None:
        config = load_curriculum_config(CONFIG)
        stage = config.stage("rmuc_flat")
        legacy = {
            "format_version": 2,
            "checkpoint_backend": "mujoco_warp_flat_ppo_fixed_gain_v2",
            "algorithm": "ppo",
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "action_semantics": "seven_dimensional_fixed_gain_residual",
            "policy_action_mask": list(stage.residual_action_mask),
            "task_scope": {
                "task_mode": "flat_walking_only",
                "terrain_enabled": False,
                "jump_enabled": False,
                "domain_randomization_enabled": False,
                "flat_terrain_features_zeroed": True,
                "jump_features_zeroed": True,
            },
            "model_state_dict": {"actor.0.weight": object()},
        }
        validate_checkpoint_metadata(legacy, config=config, stage=stage)
        legacy["task_scope"] = {}
        with self.assertRaisesRegex(WarpCurriculumConfigError, "reward schema"):
            validate_checkpoint_metadata(legacy, config=config, stage=stage)

    def test_cpu_residual_checkpoint_is_explicit_cross_backend_warm_start(self) -> None:
        import env
        import train_ppo

        config = load_curriculum_config(CONFIG)
        stage = config.stage("rmuc_flat_dr")
        cpu_checkpoint = {
            "format_version": train_ppo.CHECKPOINT_FORMAT_VERSION,
            "algorithm": "ppo",
            "action_semantics": "residual",
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "task_config": {
                "locomotion_command_schema": env.LOCOMOTION_COMMAND_SCHEMA,
                "residual_authority_schema": env.RESIDUAL_AUTHORITY_SCHEMA,
                "reward_schema": train_ppo.REWARD_SCHEMA,
            },
            "model_state_dict": {"actor.0.weight": object()},
        }
        validate_checkpoint_metadata(cpu_checkpoint, config=config, stage=stage)
        cpu_checkpoint["task_config"]["residual_authority_schema"] = "bad"
        with self.assertRaisesRegex(WarpCurriculumConfigError, "authority schema"):
            validate_checkpoint_metadata(cpu_checkpoint, config=config, stage=stage)


if __name__ == "__main__":
    unittest.main()
