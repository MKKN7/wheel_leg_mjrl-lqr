"""Static contracts for RMUC route, turning, and direct-stair CUDA courses."""

from __future__ import annotations

from pathlib import Path
import inspect
import unittest

from rmuc_curriculum_warp import (
    GPU_CURRICULUM_CAPABILITIES,
    REWARD_SCHEMA,
    _validate_course_contract,
    evaluate_policy_stage,
    load_rmuc_course_config,
)
from train_warp_curriculum import gpu_stage_capability, load_curriculum_config, validate_stage_capability


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "warp_curriculum_ppo.yaml"
ADAPTER = ROOT / "configs" / "rmuc_course_warp.yaml"


class RmucCurriculumAdapterStaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_curriculum_config(MANIFEST)
        self.adapter = load_rmuc_course_config(ADAPTER)

    def test_every_rmuc_route_course_has_matching_manifest_and_capability(self) -> None:
        self.assertEqual(set(self.adapter.courses), set(GPU_CURRICULUM_CAPABILITIES))
        for stage_id in sorted(self.adapter.courses):
            with self.subTest(stage_id=stage_id):
                stage = self.manifest.stage(stage_id)
                course, specs = _validate_course_contract(stage, self.adapter)
                capability = validate_stage_capability(self.manifest, stage)
                self.assertTrue(specs)
                self.assertTrue(capability.runtime_gate_required)
                self.assertEqual(gpu_stage_capability(stage_id).backend, stage.controller_backend)
                self.assertEqual(stage.reward_schema, REWARD_SCHEMA)
                self.assertEqual(capability.reward_schema, REWARD_SCHEMA)

    def test_rmuc_stair_is_a_direct_jump_only_course(self) -> None:
        course = self.adapter.courses["rmuc_stair_jump"]
        stage = self.manifest.stage("rmuc_stair_jump")
        self.assertTrue(course.direct_jump)
        self.assertEqual(course.task_ids, ("stair_up",))
        self.assertTrue(stage.terrain_enabled)
        self.assertTrue(stage.steps_enabled)
        self.assertTrue(stage.jump_enabled)
        self.assertEqual(stage.residual_action_mask[-1], 0.0)
        self.assertGreaterEqual(self.adapter.jump.prelanding_seconds, 0.050)
        self.assertLessEqual(self.adapter.jump.jump_residual_fraction, 1.0)

    def test_yaml_order_requires_certified_course_progression(self) -> None:
        self.assertEqual(self.manifest.stage("rmuc_flat_dr").prerequisite_stage_ids, ("rmuc_flat",))
        self.assertEqual(self.manifest.stage("grades").prerequisite_stage_ids, ("rmuc_flat_dr",))
        self.assertEqual(self.manifest.stage("low_speed_turning").prerequisite_stage_ids, ("grades",))
        self.assertEqual(self.manifest.stage("rmuc_stair_jump").prerequisite_stage_ids, ("low_speed_turning",))
        self.assertEqual(self.manifest.stage("official_grade15_up").prerequisite_stage_ids, ("rmuc_stair_jump",))
        self.assertEqual(self.manifest.stage("official_doghole450").prerequisite_stage_ids, ("official_fly17_jump",))

    def test_rmuc_adapter_exports_a_cuda_post_training_evaluator(self) -> None:
        self.assertTrue(callable(evaluate_policy_stage))

    def test_rmuc_jump_latches_contact_speed_before_landing_confirmation(self) -> None:
        from rmuc_curriculum_warp import RmucRouteTask

        source = inspect.getsource(RmucRouteTask._update_direct_jump_supervisor)
        self.assertIn("contacted_flight", source)
        self.assertIn("_jump_landing_vertical_speed", source)
        self.assertIn("_jump_landing_angular_speed", source)
        self.assertIn("torch.maximum", source)


if __name__ == "__main__":
    unittest.main()
