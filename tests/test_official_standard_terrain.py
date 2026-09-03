"""Focused regression checks for the synthetic RM2025/2026 terrain fixture."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from env import DomainRandomizationConfig, WheelLegResidualEnv
from evaluate_policy import terrain_route_gates
from terrain_curriculum import load_terrain_curriculum, validate_scene_contract


ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_PATH = ROOT / "configs" / "official_standard_curriculum.yaml"
SCENE_PATH = ROOT / "official_standard_ground.xml"
RMUC_CURRICULUM_PATH = ROOT / "configs" / "rmuc_terrain_curriculum.yaml"


class OfficialStandardTerrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.curriculum = load_terrain_curriculum(CURRICULUM_PATH)

    def test_scene_contract_matches_generated_mjcf(self) -> None:
        validate_scene_contract(
            self.curriculum,
            SCENE_PATH,
            curriculum_path=CURRICULUM_PATH,
        )

    def test_grade_spawn_uses_projection_plane_height(self) -> None:
        environment = WheelLegResidualEnv(
            xml_path=SCENE_PATH,
            episode_seconds=40.0,
            randomize_command=False,
            randomize_leg_length=False,
            max_forward_speed=1.0,
            command_speed_limit_mps=0.60,
            max_command_yaw_delta_rad=0.0,
            max_command_yaw_rate_rad_s=0.35,
            jump_probability=0.0,
            domain_randomization=DomainRandomizationConfig.disabled(),
            terrain_curriculum=self.curriculum,
            terrain_stage_id="grade20",
            terrain_evaluation=True,
        )
        try:
            environment.reset(
                seed=421,
                options={"terrain_task_id": "grade20_up", "terrain_route_index": 0},
            )
            route = self.curriculum.task("grade20_up").route_at(0)
            spawn_xy = np.asarray(route.spawn.xy(), dtype=np.float64)
            expected_root_z = (
                environment._stance_qpos[environment._root_qpos_address + 2]
                + environment.terrain_surface_height_m(spawn_xy)
                - environment._terrain_projection_support_height_m
            )
            actual_root_z = environment.data.qpos[environment._root_qpos_address + 2]
            self.assertAlmostEqual(float(actual_root_z), float(expected_root_z), places=8)
        finally:
            environment.close()

    def test_step200_requires_three_consecutive_route_successes(self) -> None:
        stage = self.curriculum.stage("step200_lab")
        results = []
        for task_id in stage.task_ids:
            task = self.curriculum.task(task_id)
            for route in task.routes:
                repetitions = 2 if task_id == "step200_lab" else 3
                for _ in range(repetitions):
                    results.append(
                        SimpleNamespace(
                            task=task_id,
                            terrain_route=route.route_id,
                            task_completed=True,
                            physically_safe=True,
                            task_succeeded=True,
                            speed_mae_mps=0.0,
                            yaw_mae_rad=0.0,
                        )
                    )
        gates = terrain_route_gates(results, self.curriculum, stage)
        lab_gate = next(gate for gate in gates if gate.task_id == "step200_lab")
        self.assertEqual(lab_gate.maximum_success_streak, 2)
        self.assertEqual(lab_gate.required_success_streak, 3)
        self.assertFalse(lab_gate.passed)


class RmucTurningCurriculumTest(unittest.TestCase):
    def test_turning_is_low_speed_and_high_speed_is_explicit(self) -> None:
        curriculum = load_terrain_curriculum(RMUC_CURRICULUM_PATH)
        left = curriculum.task("turn_left")
        right = curriculum.task("turn_right")
        self.assertAlmostEqual(left.command.forward_speed_mps, 0.20)
        self.assertAlmostEqual(right.command.forward_speed_mps, 0.20)
        self.assertAlmostEqual(left.command.yaw_rate_rad_s, 0.08)
        self.assertAlmostEqual(right.command.yaw_rate_rad_s, -0.08)
        self.assertAlmostEqual(left.lqr_speed_reference_scale, 0.40)
        self.assertAlmostEqual(right.lqr_speed_reference_scale, 0.40)

        turning = curriculum.stage("turning")
        left_command = turning.command_for(left)
        right_command = turning.command_for(right)
        self.assertAlmostEqual(left_command.forward_speed_mps, 0.20)
        self.assertAlmostEqual(left_command.yaw_rate_rad_s, 0.08)
        self.assertFalse(left_command.jump_request)
        self.assertAlmostEqual(right_command.forward_speed_mps, 0.20)
        self.assertAlmostEqual(right_command.yaw_rate_rad_s, -0.08)
        self.assertFalse(right_command.jump_request)
        dynamic = curriculum.stage("dynamic_locomotion")
        self.assertNotIn("accel_turn", dynamic.task_ids)
        self.assertNotIn("accel_turn_right", dynamic.task_ids)
        stress = curriculum.stage("high_speed_turn")
        self.assertIn("accel_turn", stress.task_ids)
        self.assertIn("accel_turn_right", stress.task_ids)


if __name__ == "__main__":
    unittest.main()
