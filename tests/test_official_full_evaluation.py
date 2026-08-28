"""Static contracts for the one-policy official full-course certifier."""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from evaluate_official_full_course import _validate_route_plan, load_full_evaluation_config
from official_course_warp import evaluate_policy_stage
from train_warp_curriculum import load_curriculum_config


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "official_full_evaluation.yaml"


class OfficialFullEvaluationStaticTest(unittest.TestCase):
    def test_plan_covers_the_exact_official_full_task_order(self) -> None:
        plan = load_full_evaluation_config(PLAN_PATH)
        curriculum = load_curriculum_config(plan.curriculum_config_path)
        checkpoint_stage, stages = _validate_route_plan(curriculum, plan)
        self.assertEqual(checkpoint_stage.stage_id, "official_doghole450")
        self.assertEqual(tuple(stage.stage_id for stage in stages), plan.official_stage_ids)
        self.assertEqual(len(stages), 11)
        self.assertTrue(all(stage.stage_id.startswith("official_") for stage in stages))

    def test_route_evaluator_accepts_the_full_stage_threshold_override(self) -> None:
        self.assertIn("threshold_terrain_stage_id", inspect.signature(evaluate_policy_stage).parameters)


if __name__ == "__main__":
    unittest.main()
