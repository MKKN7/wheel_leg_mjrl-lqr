"""Static contracts for the fail-closed all-route official CUDA adapter."""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from official_course_warp import (
    GPU_CURRICULUM_CAPABILITIES,
    OfficialCourseAdapterError,
    REWARD_SCHEMA,
    load_official_course_config,
    _validate_course_contract,
)
from train_warp_curriculum import gpu_stage_capability, load_curriculum_config


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "warp_curriculum_ppo.yaml"
ADAPTER = ROOT / "configs" / "official_course_warp.yaml"


class OfficialCourseAdapterStaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_curriculum_config(MANIFEST)
        self.adapter = load_official_course_config(ADAPTER)

    def test_every_declared_official_route_has_a_matching_stage_and_capability(self) -> None:
        expected = set(self.adapter.courses)
        manifest_ids = {stage.stage_id for stage in self.manifest.stages}
        self.assertTrue(expected <= manifest_ids)
        self.assertEqual(set(GPU_CURRICULUM_CAPABILITIES), expected)
        for stage_id in sorted(expected):
            with self.subTest(stage_id=stage_id):
                stage = self.manifest.stage(stage_id)
                course, _, task, route = _validate_course_contract(stage, self.adapter)
                capability = gpu_stage_capability(stage_id)
                self.assertEqual(course.task_id, task.task_id)
                self.assertEqual(course.support_geoms, next(
                    binding.support_geoms
                    for binding in _.scene_contract.route_bindings
                    if binding.task_id == task.task_id and binding.route_id == route.route_id
                ))
                self.assertTrue(capability.runtime_gate_required)
                self.assertTrue(capability.terrain)
                self.assertTrue(capability.domain_randomization)
                self.assertEqual(stage.reward_schema, REWARD_SCHEMA)
                self.assertEqual(capability.reward_schema, REWARD_SCHEMA)

    def test_only_upward_step_exercises_use_the_direct_jump_supervisor(self) -> None:
        expected_jump = {
            "official_step150_up",
            "official_stair2x100_up",
            "official_step200_lab",
            "official_fly17_jump",
        }
        actual_jump = {name for name, course in self.adapter.courses.items() if course.direct_jump}
        self.assertEqual(actual_jump, expected_jump)
        for stage_id, course in self.adapter.courses.items():
            stage = self.manifest.stage(stage_id)
            with self.subTest(stage_id=stage_id):
                self.assertEqual(stage.jump_enabled, course.direct_jump)
                self.assertEqual(stage.residual_action_mask[-1], 0.0)
                self.assertEqual(GPU_CURRICULUM_CAPABILITIES[stage_id]["jump"], course.direct_jump)

    def test_landing_preload_is_never_less_than_fifty_ms(self) -> None:
        self.assertGreaterEqual(self.adapter.jump.prelanding_seconds, 0.050)

    def test_fall_reference_rebases_only_on_current_support_and_syncs_task_guard(self) -> None:
        from official_course_warp import OfficialCourseTask

        source = inspect.getsource(OfficialCourseTask._before_policy_step)
        self.assertIn("self._side_support_contacts().all(dim=1)", source)
        self.assertIn("self._reference_height[self._fall_guard_update_mask]", source)

    def test_doghole_requires_its_declared_obstacle_guard(self) -> None:
        stage = self.manifest.stage("official_doghole450")
        self.assertIn("doghole", stage.task_mode)
        self.assertEqual(
            self.adapter.courses[stage.stage_id].obstacle_geoms,
            (
                "obstacle_doghole_roof",
                "obstacle_doghole_left_wall",
                "obstacle_doghole_right_wall",
            ),
        )

    def test_invalid_direct_jump_contract_is_rejected_before_cuda_allocation(self) -> None:
        stage = self.manifest.stage("official_step150_up")
        stage = type(stage)(**{**stage.__dict__, "jump_enabled": False})
        with self.assertRaisesRegex(OfficialCourseAdapterError, "jump_enabled"):
            _validate_course_contract(stage, self.adapter)


if __name__ == "__main__":
    unittest.main()
