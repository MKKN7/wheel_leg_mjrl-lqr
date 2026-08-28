"""Contracts for the fail-closed full CUDA curriculum orchestrator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest import mock

from evaluate_official_full_course import load_full_evaluation_config
from train_full_curriculum import DEFAULT_FULL_EVALUATION_CONFIG, parse_args, run_full_curriculum
from train_warp_curriculum import WarpCurriculumConfigError, load_curriculum_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "warp_curriculum_ppo.yaml"
FULL_EVALUATION_CONFIG = ROOT / "configs" / "official_full_evaluation.yaml"


class FullCurriculumRunnerTest(unittest.TestCase):
    def test_orchestrator_runs_every_manifest_stage_in_yaml_order(self) -> None:
        config = load_curriculum_config(CONFIG)
        plan = load_full_evaluation_config(FULL_EVALUATION_CONFIG)
        expected = [stage.stage_id for stage in config.stages]
        called: list[str] = []

        def fake_run(config_value, *, stage_id, smoke, init_residual_checkpoint):
            self.assertIs(config_value, config)
            self.assertFalse(smoke)
            self.assertIsNone(init_residual_checkpoint)
            called.append(stage_id)
            return ROOT / "artifacts" / f"{stage_id}.pt"

        with (
            mock.patch("train_full_curriculum.load_curriculum_config", return_value=config),
            mock.patch("train_full_curriculum.run_curriculum_training", side_effect=fake_run),
            mock.patch(
                "train_full_curriculum.evaluate_full_course",
                return_value={"passed": True},
            ) as evaluate,
            mock.patch("train_full_curriculum._atomic_json") as write_report,
        ):
            outputs = run_full_curriculum(CONFIG, full_evaluation_config=FULL_EVALUATION_CONFIG)
        self.assertEqual(called, expected)
        self.assertEqual(len(outputs), len(expected))
        self.assertEqual(tuple(path.name for path in outputs), tuple(f"{stage_id}.pt" for stage_id in expected))
        evaluate.assert_called_once_with(
            plan=plan,
            checkpoint_path=ROOT / "artifacts" / "official_doghole450.pt",
        )
        write_report.assert_called_once_with(plan.report_path, {"passed": True})

    def test_final_official_failure_writes_report_then_blocks_completion(self) -> None:
        config = load_curriculum_config(CONFIG)
        plan = load_full_evaluation_config(FULL_EVALUATION_CONFIG)

        def fake_run(_config_value, *, stage_id, smoke, init_residual_checkpoint):
            self.assertFalse(smoke)
            return ROOT / "artifacts" / f"{stage_id}.pt"

        with (
            mock.patch("train_full_curriculum.load_curriculum_config", return_value=config),
            mock.patch("train_full_curriculum.run_curriculum_training", side_effect=fake_run),
            mock.patch(
                "train_full_curriculum.evaluate_full_course",
                return_value={"passed": False, "unsafe_rate": 1.0},
            ),
            mock.patch("train_full_curriculum._atomic_json") as write_report,
            self.assertRaisesRegex(WarpCurriculumConfigError, "official full-course certification failed"),
        ):
            run_full_curriculum(CONFIG, full_evaluation_config=FULL_EVALUATION_CONFIG)
        write_report.assert_called_once_with(plan.report_path, {"passed": False, "unsafe_rate": 1.0})

    def test_partial_sequence_without_final_checkpoint_does_not_run_full_evaluation(self) -> None:
        config = load_curriculum_config(CONFIG)
        partial = replace(
            config,
            stages=tuple(stage for stage in config.stages if stage.stage_id != "official_doghole450"),
        )

        def fake_run(_config_value, *, stage_id, smoke, init_residual_checkpoint):
            self.assertFalse(smoke)
            return ROOT / "artifacts" / f"{stage_id}.pt"

        with (
            mock.patch("train_full_curriculum.load_curriculum_config", return_value=partial),
            mock.patch("train_full_curriculum.run_curriculum_training", side_effect=fake_run),
            mock.patch("train_full_curriculum.evaluate_full_course") as evaluate,
            mock.patch("train_full_curriculum._atomic_json") as write_report,
        ):
            outputs = run_full_curriculum(CONFIG, full_evaluation_config=FULL_EVALUATION_CONFIG)
        self.assertEqual(len(outputs), len(partial.stages))
        evaluate.assert_not_called()
        write_report.assert_not_called()

    def test_cli_default_keeps_full_evaluation_yaml_owned(self) -> None:
        self.assertEqual(parse_args([]).full_evaluation_config, DEFAULT_FULL_EVALUATION_CONFIG)

    def test_resume_refuses_missing_prior_certificates(self) -> None:
        with self.assertRaisesRegex(WarpCurriculumConfigError, "certified predecessor artifact is missing"):
            run_full_curriculum(CONFIG, from_stage="official_grade15_up")


if __name__ == "__main__":
    unittest.main()
