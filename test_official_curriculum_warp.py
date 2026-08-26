"""Static contracts for the conditional official grade15-up CUDA adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco

from official_curriculum_warp import (
    CONTROLLER_BACKEND,
    EXPECTED_SUPPORT_GEOMS,
    GPU_CURRICULUM_CAPABILITIES,
    OfficialGrade15AdapterError,
    STAGE_ID,
    StaticBoxSupportLayout,
    TASK_MODE,
    build_curriculum_stage,
    load_official_grade15_adapter_config,
    validate_official_grade15_contract,
)
from train_warp_curriculum import gpu_stage_capability
from warp_env import load_warp_batch_config


ROOT = Path(__file__).resolve().parent
ADAPTER_CONFIG = ROOT / "configs" / "official_grade15_warp.yaml"
SCENE = ROOT / "official_standard_warp_ground.xml"
CANONICAL_SCENE = ROOT / "official_standard_ground.xml"
CURRICULUM = ROOT / "official_standard_curriculum.yaml"


def _stage() -> SimpleNamespace:
    return SimpleNamespace(
        stage_id=STAGE_ID,
        task_mode=TASK_MODE,
        xml_path=SCENE,
        terrain_curriculum_path=CURRICULUM,
        terrain_stage_id="grade15",
        controller_backend=CONTROLLER_BACKEND,
        terrain_enabled=True,
        jump_enabled=False,
        steps_enabled=False,
        domain_randomization_enabled=False,
        command_speed_mps=0.195,
        command_yaw_rate_rad_s=0.0,
        residual_action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        requires_gpu_parity=True,
        adapter_config_path=ADAPTER_CONFIG,
        scene_variant="official_warp_compat",
    )


class OfficialGrade15AdapterStaticTest(unittest.TestCase):
    def test_static_box_layout_matches_grade15_topology(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        layout = StaticBoxSupportLayout.from_model(model, EXPECTED_SUPPORT_GEOMS, mujoco)

        lead_height, lead_valid = layout.surface_height_cpu((-6.0, 6.0))
        ramp_height, ramp_valid = layout.surface_height_cpu((-4.15, 6.0))
        platform_height, platform_valid = layout.surface_height_cpu((-1.0, 6.0))
        outside_height, outside_valid = layout.surface_height_cpu((0.0, 0.0))

        self.assertTrue(lead_valid)
        self.assertTrue(ramp_valid)
        self.assertTrue(platform_valid)
        self.assertFalse(outside_valid)
        self.assertAlmostEqual(lead_height, 0.0, places=6)
        self.assertGreater(ramp_height, lead_height)
        self.assertLess(ramp_height, platform_height)
        self.assertAlmostEqual(platform_height, 0.4, places=6)
        self.assertEqual(outside_height, 0.0)

    def test_yaml_and_stage_contract_admit_only_grade15_up(self) -> None:
        adapter = load_official_grade15_adapter_config(ADAPTER_CONFIG)
        curriculum, task, route = validate_official_grade15_contract(_stage(), adapter)

        self.assertEqual(curriculum.schema_version, 4)
        self.assertEqual(task.task_id, "grade15_up")
        self.assertEqual(route.route_id, "official_grade15_up")
        self.assertEqual(adapter.support_geoms, EXPECTED_SUPPORT_GEOMS)
        self.assertEqual(len(adapter.terrain_features.lookahead_distances_m) * len(adapter.terrain_features.lateral_offsets_m), 12)
        self.assertTrue(adapter.stability_gate.require_no_terminated)
        self.assertGreater(adapter.stability_gate.minimum_progress_m, 0.0)
        self.assertGreater(adapter.stability_gate.maximum_speed_mae_mps, 0.0)
        self.assertEqual(adapter.stability_gate.maximum_unsafe_rate, 0.0)

    def test_steps_jump_and_doghole_cannot_be_relabelled_as_grade15(self) -> None:
        adapter = load_official_grade15_adapter_config(ADAPTER_CONFIG)
        for task_id in ("step150_up", "fly17_jump", "doghole450"):
            with self.subTest(task_id=task_id):
                with self.assertRaisesRegex(OfficialGrade15AdapterError, "only official grade15_up"):
                    validate_official_grade15_contract(_stage(), replace(adapter, task_id=task_id))

    def test_capability_is_terrain_and_speed_only_with_runtime_gate(self) -> None:
        capability = GPU_CURRICULUM_CAPABILITIES[STAGE_ID]
        self.assertTrue(capability["terrain"])
        self.assertTrue(capability["speed_command"])
        self.assertFalse(capability["steps"])
        self.assertFalse(capability["jump"])
        self.assertFalse(capability["domain_randomization"])
        self.assertTrue(capability["conditional_runtime_gate"])
        self.assertEqual(gpu_stage_capability(STAGE_ID).backend, CONTROLLER_BACKEND)

    def test_official_batch_is_fixed_and_safety_derated(self) -> None:
        adapter = load_official_grade15_adapter_config(ADAPTER_CONFIG)
        batch = load_warp_batch_config(adapter.batch_config_path)
        self.assertEqual(batch.xml_path, SCENE.resolve())
        self.assertFalse(batch.domain_randomization.enabled)
        self.assertLessEqual(batch.safety.torque_fraction_of_rated, 0.80)

    def test_factory_rejects_downhill_before_gpu_allocation(self) -> None:
        unsupported = _stage()
        unsupported.stage_id = "official_grade15_down"
        with self.assertRaisesRegex(OfficialGrade15AdapterError, "official_grade15_up"):
            build_curriculum_stage(unsupported, object())

    def test_canonical_scene_is_not_accepted_by_direct_factory_contract(self) -> None:
        stage = _stage()
        stage.scene_variant = "canonical"
        adapter = load_official_grade15_adapter_config(ADAPTER_CONFIG)
        with self.assertRaisesRegex(OfficialGrade15AdapterError, "official_warp_compat"):
            validate_official_grade15_contract(stage, adapter)

    def test_warp_compat_label_cannot_point_to_canonical_scene(self) -> None:
        stage = _stage()
        stage.xml_path = CANONICAL_SCENE
        adapter = load_official_grade15_adapter_config(ADAPTER_CONFIG)
        with self.assertRaisesRegex(OfficialGrade15AdapterError, "Warp scene variant validation failed"):
            validate_official_grade15_contract(stage, adapter)


if __name__ == "__main__":
    unittest.main()
