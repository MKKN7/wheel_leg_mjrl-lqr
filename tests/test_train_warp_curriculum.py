"""Parser, capability, and checkpoint metadata tests for GPU curriculum PPO."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from train_warp_curriculum import (
    ACTION_SIZE,
    CURRICULUM_BACKEND,
    CURRICULUM_CHECKPOINT_FORMAT,
    GATE_EVIDENCE_SCHEMA,
    OBSERVATION_SIZE,
    REWARD_SCHEMA,
    WarpCurriculumConfigError,
    build_checkpoint_metadata,
    load_curriculum_config,
    parse_args,
    _validate_runtime_gate_report,
    _run_post_training_evaluation,
    _stage_requires_promotion,
    validate_checkpoint_metadata,
    validate_stage_capability,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "warp_curriculum_ppo.yaml"


def _valid_dual_gate_report(config, stage) -> dict:
    """Return complete versioned evidence for parser-level fail-closed tests."""

    digest = hashlib.sha256(stage.adapter_config_path.read_bytes()).hexdigest()

    def evidence(domain_randomization_active: bool) -> dict:
        return {
            "passed": True,
            "requested_duration_seconds": config.gpu_task.stability_gate_seconds,
            "simulated_duration_seconds": config.gpu_task.stability_gate_seconds,
            "policy_steps": 800,
            "num_worlds": 128,
            "terminated_worlds": 0,
            "overflowed_worlds": 0,
            "estopped_worlds": 0,
            "finite_state": True,
            "finite_reward": True,
            "finite_reward_terms": True,
            "zero_residual": True,
            "domain_randomization_active": domain_randomization_active,
            "physical_parameter_randomization": domain_randomization_active,
            "terrain_geometry_randomization": False,
            "sensor_noise_std": config.gpu_task.sensor_noise_std,
            "control_delay_steps": config.gpu_task.control_delay_steps,
            "minimum_progress_m": 1.0,
            "speed_mae_mps": 0.1,
            "unsafe_rate": 0.0,
            "first_fault_step": -1,
            "first_fault_reason_code": 0,
            "obstacle_guard_verified": True,
        }

    deterministic = evidence(False)
    randomized = evidence(True)
    if stage.jump_enabled:
        for pass_report in (deterministic, randomized):
            pass_report.update({
                "jump_supervisor_verified": True,
                "jump_triggered_worlds": 128,
                "landing_confirmed_worlds": 128,
                "jump_minimum_peak_worlds": 128,
                "landing_kinematics_worlds": 128,
                "minimum_flight_seconds": config.gpu_task.stability_gate_seconds / 800.0,
                "landing_preload_seconds": 0.050,
            })
    return {
        "stage_id": stage.stage_id,
        "conditional_capability": True,
        "gate_evidence_schema": GATE_EVIDENCE_SCHEMA,
        "gate_config_sha256": digest,
        "threshold_config_sha256": digest,
        "passed": True,
        "requested_duration_seconds": randomized["requested_duration_seconds"],
        "simulated_duration_seconds": randomized["simulated_duration_seconds"],
        "policy_steps": randomized["policy_steps"],
        "num_worlds": randomized["num_worlds"],
        "terminated_worlds": randomized["terminated_worlds"],
        "overflowed_worlds": randomized["overflowed_worlds"],
        "estopped_worlds": randomized["estopped_worlds"],
        "finite_state": randomized["finite_state"],
        "finite_reward": randomized["finite_reward"],
        "finite_reward_terms": randomized["finite_reward_terms"],
        "zero_residual": True,
        "domain_randomization_enabled": True,
        "minimum_progress_m": randomized["minimum_progress_m"],
        "speed_mae_mps": randomized["speed_mae_mps"],
        "unsafe_rate": randomized["unsafe_rate"],
        "first_fault_step": randomized["first_fault_step"],
        "first_fault_reason_code": randomized["first_fault_reason_code"],
        "obstacle_guard_verified": randomized["obstacle_guard_verified"],
        "deterministic_baseline": deterministic,
        "domain_randomization_stress": randomized,
        "deterministic_baseline_passed": True,
        "domain_randomization_stress_passed": True,
    }


def _certified_predecessor_metadata(config, target_stage) -> dict:
    """Minimal certified predecessor used to test warm-start fail-closed rules."""

    source_stage_id = target_stage.prerequisite_stage_ids[-1]
    source_stage = config.stage(source_stage_id)
    return {
        "format_version": CURRICULUM_CHECKPOINT_FORMAT,
        "checkpoint_backend": CURRICULUM_BACKEND,
        "algorithm": "ppo",
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "action_semantics": "seven_dimensional_fixed_gain_residual",
        "reward_schema": source_stage.reward_schema or config.reward_schema,
        "stage_id": source_stage_id,
        "policy_action_mask": list(target_stage.residual_action_mask),
        "artifact_status": "certified",
        "course_evaluation": {"stage_id": source_stage_id, "passed": True},
        "course_certificate": {
            "certificate_schema": 1,
            "passed": True,
            "stage_id": source_stage_id,
            "curriculum_config_sha256": config.source_digest,
            "adapter_config_sha256": hashlib.sha256(source_stage.adapter_config_path.read_bytes()).hexdigest(),
        },
        "stage_scope": {
            "task_mode": "rmuc_stair_jump",
            "xml_path": "source.xml",
            "scene_variant": "canonical",
            "terrain_curriculum_path": "source.yaml",
            "terrain_stage_id": "stair_jump",
            "controller_backend": "rmuc_route_controller_v1",
        },
        "experiment_config": {
            "xml_sha256": "c" * 64,
            "terrain_curriculum_sha256": "d" * 64,
            "adapter_config_sha256": "e" * 64,
            "flat_ppo_config_sha256": "f" * 64,
        },
        "model_state_dict": {"actor.0.weight": object()},
    }


def _certified_exact_stage_metadata(config, stage) -> dict:
    """Minimal exact-stage artifact with the full CUDA certification contract."""

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "format_version": CURRICULUM_CHECKPOINT_FORMAT,
        "checkpoint_backend": CURRICULUM_BACKEND,
        "algorithm": "ppo",
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "action_semantics": "seven_dimensional_fixed_gain_residual",
        "reward_schema": stage.reward_schema or config.reward_schema,
        "stage_id": stage.stage_id,
        "policy_action_mask": list(stage.residual_action_mask),
        "artifact_status": "certified",
        "course_evaluation": {"stage_id": stage.stage_id, "passed": True},
        "course_certificate": {
            "certificate_schema": 1,
            "passed": True,
            "stage_id": stage.stage_id,
            "curriculum_config_sha256": config.source_digest,
            "adapter_config_sha256": (
                None if stage.adapter_config_path is None else digest(stage.adapter_config_path)
            ),
        },
        "stage_scope": {
            "task_mode": stage.task_mode,
            "xml_path": str(stage.xml_path),
            "scene_variant": stage.scene_variant,
            "terrain_curriculum_path": str(stage.terrain_curriculum_path),
            "terrain_stage_id": stage.terrain_stage_id,
            "controller_backend": stage.controller_backend,
        },
        "experiment_config": {
            "xml_sha256": digest(stage.xml_path),
            "terrain_curriculum_sha256": digest(stage.terrain_curriculum_path),
            "adapter_config_sha256": (
                None if stage.adapter_config_path is None else digest(stage.adapter_config_path)
            ),
            "flat_ppo_config_sha256": digest(config.flat_ppo_config_path),
        },
        "model_state_dict": {"actor.0.weight": object()},
    }


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
                "rmuc_stair_jump",
                "official_grade15_up",
                "official_grade15_down",
                "official_grade20_up",
                "official_grade20_down",
                "official_step150_up",
                "official_step150_down",
                "official_stair2x100_up",
                "official_stair2x100_down",
                "official_step200_lab",
                "official_fly17_jump",
                "official_doghole450",
            },
        )
        flat = config.stage("rmuc_flat")
        self.assertEqual(flat.residual_action_mask[-1], 0.0)
        self.assertTrue(config.gpu_task.allow_cpu_residual_warm_start)
        self.assertEqual(config.gpu_task.control_delay_steps, 1)
        validate_stage_capability(config, flat)
        dr_stage = config.stage("rmuc_flat_dr")
        validate_stage_capability(config, dr_stage)
        self.assertEqual(dr_stage.prerequisite_stage_ids, ("rmuc_flat",))
        self.assertEqual(config.stage("grades").prerequisite_stage_ids, ("rmuc_flat_dr",))

    def test_parser_requires_stage_and_accepts_warm_start(self) -> None:
        args = parse_args([
            "--curriculum", str(CONFIG), "--stage", "rmuc_flat",
            "--init-residual-checkpoint", "warm_start.pt", "--smoke",
        ])
        self.assertEqual(args.stage, "rmuc_flat")
        self.assertEqual(args.init_residual_checkpoint, Path("warm_start.pt"))
        self.assertTrue(args.smoke)

    def test_rmuc_grades_have_a_declared_gpu_parity_contract(self) -> None:
        config = load_curriculum_config(CONFIG)
        capability = validate_stage_capability(config, config.stage("grades"))
        self.assertTrue(capability.terrain)
        self.assertTrue(capability.domain_randomization)
        self.assertTrue(capability.speed_command)
        self.assertTrue(capability.runtime_gate_required)

    def test_every_formal_stage_requires_promotion_and_flat_uses_post_policy_evaluation(self) -> None:
        config = load_curriculum_config(CONFIG)
        flat = config.stage("rmuc_flat")
        self.assertTrue(_stage_requires_promotion(flat, smoke=False))
        self.assertFalse(_stage_requires_promotion(flat, smoke=True))
        expected = {"stage_id": flat.stage_id, "passed": True}
        with mock.patch("train_warp_curriculum._evaluate_flat_policy_stage", return_value=expected) as evaluate:
            result = _run_post_training_evaluation(
                task=object(),
                batch=object(),
                policy=object(),
                stage=flat,
                config=config,
                smoke=False,
            )
        self.assertEqual(result, expected)
        evaluate.assert_called_once()

    def test_conditional_gate_report_cannot_be_static_or_incomplete(self) -> None:
        config = load_curriculum_config(CONFIG)
        stage = config.stage("official_grade15_up")
        capability = validate_stage_capability(config, stage)
        report = _valid_dual_gate_report(config, stage)
        self.assertEqual(
            _validate_runtime_gate_report(report, config=config, stage=stage, capability=capability),
            report,
        )
        for field in ("passed", "finite_state", "num_worlds", "gate_config_sha256", "domain_randomization_stress"):
            invalid = dict(report)
            invalid.pop(field)
            with self.subTest(field=field):
                with self.assertRaisesRegex(WarpCurriculumConfigError, "conditional GPU gate"):
                    _validate_runtime_gate_report(invalid, config=config, stage=stage, capability=capability)
        invalid_schema = _valid_dual_gate_report(config, stage)
        invalid_schema["gate_evidence_schema"] = GATE_EVIDENCE_SCHEMA - 1
        with self.assertRaisesRegex(WarpCurriculumConfigError, "incompatible evidence schema"):
            _validate_runtime_gate_report(invalid_schema, config=config, stage=stage, capability=capability)

    def test_conditional_gate_rejects_a_passed_report_with_any_faulted_world(self) -> None:
        config = load_curriculum_config(CONFIG)
        stage = config.stage("official_grade15_up")
        capability = validate_stage_capability(config, stage)
        report = _valid_dual_gate_report(config, stage)
        report["terminated_worlds"] = 1
        report["domain_randomization_stress"]["terminated_worlds"] = 1
        with self.assertRaisesRegex(WarpCurriculumConfigError, "zero terminated"):
            _validate_runtime_gate_report(report, config=config, stage=stage, capability=capability)

    def test_conditional_gate_rejects_nonfinite_reward_evidence(self) -> None:
        config = load_curriculum_config(CONFIG)
        stage = config.stage("official_grade15_up")
        capability = validate_stage_capability(config, stage)
        report = _valid_dual_gate_report(config, stage)
        report["domain_randomization_stress"]["finite_reward"] = False
        with self.assertRaisesRegex(WarpCurriculumConfigError, "finite_reward=true"):
            _validate_runtime_gate_report(report, config=config, stage=stage, capability=capability)

        report = _valid_dual_gate_report(config, stage)
        report["deterministic_baseline"]["finite_reward_terms"] = False
        with self.assertRaisesRegex(WarpCurriculumConfigError, "finite_reward_terms=true"):
            _validate_runtime_gate_report(report, config=config, stage=stage, capability=capability)

    def test_jump_gate_requires_peak_flight_and_landing_evidence_in_both_passes(self) -> None:
        config = load_curriculum_config(CONFIG)
        stage = config.stage("official_step150_up")
        capability = validate_stage_capability(config, stage)
        report = _valid_dual_gate_report(config, stage)
        self.assertEqual(
            _validate_runtime_gate_report(report, config=config, stage=stage, capability=capability),
            report,
        )
        report["deterministic_baseline"]["jump_minimum_peak_worlds"] = 127
        with self.assertRaisesRegex(WarpCurriculumConfigError, "jump_minimum_peak_worlds"):
            _validate_runtime_gate_report(report, config=config, stage=stage, capability=capability)

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
            mock.patch("train_warp_curriculum._load_required_prerequisite_checkpoint", return_value=None),
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
        metadata = _certified_exact_stage_metadata(config, stage)
        validate_checkpoint_metadata(metadata, config=config, stage=stage)
        metadata["action_size"] = 6
        with self.assertRaisesRegex(WarpCurriculumConfigError, "observation/action"):
            validate_checkpoint_metadata(metadata, config=config, stage=stage)

    def test_legacy_flat_checkpoint_cannot_cross_the_terrain_reward_schema_boundary(self) -> None:
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

    def test_certified_declared_predecessor_is_the_only_gpu_cross_stage_warm_start(self) -> None:
        config = load_curriculum_config(CONFIG)
        target = config.stage("official_grade15_up")
        metadata = _certified_predecessor_metadata(config, target)
        validate_checkpoint_metadata(metadata, config=config, stage=target)
        metadata["artifact_status"] = "candidate"
        with self.assertRaisesRegex(WarpCurriculumConfigError, "not a certified"):
            validate_checkpoint_metadata(metadata, config=config, stage=target)

    def test_flat_predecessor_must_be_certified_before_flat_dr_resume(self) -> None:
        config = load_curriculum_config(CONFIG)
        source = config.stage("rmuc_flat")
        target = config.stage("rmuc_flat_dr")
        metadata = _certified_exact_stage_metadata(config, source)
        validate_checkpoint_metadata(metadata, config=config, stage=target)
        metadata["artifact_status"] = "candidate"
        with self.assertRaisesRegex(WarpCurriculumConfigError, "not a certified"):
            validate_checkpoint_metadata(metadata, config=config, stage=target)

    def test_gpu_warm_start_rejects_a_non_prerequisite_even_if_it_claims_certification(self) -> None:
        config = load_curriculum_config(CONFIG)
        target = config.stage("official_grade15_up")
        metadata = _certified_predecessor_metadata(config, target)
        metadata["stage_id"] = "grades"
        metadata["course_evaluation"]["stage_id"] = "grades"
        metadata["course_certificate"]["stage_id"] = "grades"
        with self.assertRaisesRegex(WarpCurriculumConfigError, "explicitly declared prerequisite"):
            validate_checkpoint_metadata(metadata, config=config, stage=target)

    def test_exact_route_checkpoint_requires_current_flat_ppo_digest(self) -> None:
        config = load_curriculum_config(CONFIG)
        stage = config.stage("grades")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {
            "format_version": CURRICULUM_CHECKPOINT_FORMAT,
            "checkpoint_backend": CURRICULUM_BACKEND,
            "algorithm": "ppo",
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "action_semantics": "seven_dimensional_fixed_gain_residual",
            "reward_schema": stage.reward_schema,
            "stage_id": stage.stage_id,
            "policy_action_mask": list(stage.residual_action_mask),
            "artifact_status": "certified",
            "course_evaluation": {"stage_id": stage.stage_id, "passed": True},
            "course_certificate": {
                "certificate_schema": 1,
                "passed": True,
                "stage_id": stage.stage_id,
                "curriculum_config_sha256": config.source_digest,
                "adapter_config_sha256": digest(stage.adapter_config_path),
            },
            "stage_scope": {
                "task_mode": stage.task_mode,
                "xml_path": str(stage.xml_path),
                "scene_variant": stage.scene_variant,
                "terrain_curriculum_path": str(stage.terrain_curriculum_path),
                "terrain_stage_id": stage.terrain_stage_id,
                "controller_backend": stage.controller_backend,
            },
            "experiment_config": {
                "xml_sha256": digest(stage.xml_path),
                "terrain_curriculum_sha256": digest(stage.terrain_curriculum_path),
                "adapter_config_sha256": digest(stage.adapter_config_path),
                "flat_ppo_config_sha256": digest(config.flat_ppo_config_path),
            },
            "model_state_dict": {"actor.0.weight": object()},
        }
        validate_checkpoint_metadata(metadata, config=config, stage=stage)
        metadata["experiment_config"]["flat_ppo_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(WarpCurriculumConfigError, "flat_ppo_config_sha256"):
            validate_checkpoint_metadata(metadata, config=config, stage=stage)

    def test_route_metadata_binds_flat_ppo_and_terrain_reward_components(self) -> None:
        from warp_env import load_warp_batch_config

        config = load_curriculum_config(CONFIG)
        stage = config.stage("grades")
        batch_config = load_warp_batch_config(config.batch_config_path)
        batch = SimpleNamespace(
            num_worlds=batch_config.num_worlds,
            device="cuda:0",
            config=SimpleNamespace(domain_randomization=batch_config.domain_randomization),
        )
        task = SimpleNamespace(
            config=SimpleNamespace(
                terrain_compensated_leg_reward=SimpleNamespace(enabled=True),
            )
        )
        metadata = build_checkpoint_metadata(
            config=config,
            stage=stage,
            timesteps=0,
            update_index=0,
            source_checkpoint=None,
            batch=batch,
            task=task,
            smoke=False,
        )
        self.assertEqual(
            metadata["experiment_config"]["flat_ppo_config_sha256"],
            hashlib.sha256(config.flat_ppo_config_path.read_bytes()).hexdigest(),
        )
        self.assertIn("terrain_compensated_leg_difference_cost", metadata["reward_terms"])
        self.assertIn("terrain_attitude_tracking", metadata["reward_terms"])
        self.assertNotIn("leg_symmetry_cost", metadata["reward_terms"])


if __name__ == "__main__":
    unittest.main()
