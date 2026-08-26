"""Focused contracts for the parity-gated flat GPU DR curriculum factory."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from curriculum_warp import (
    GPU_CURRICULUM_CAPABILITIES,
    STAGE_ID,
    WarpCurriculumStageError,
    _resolve_gpu_task_settings,
    _validate_batch_contract,
    _validate_stage_contract,
    build_curriculum_stage,
)
from train_warp_curriculum import load_curriculum_config
from warp_env import load_warp_batch_config


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "warp_curriculum_ppo.yaml"


class WarpCurriculumFactoryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_curriculum_config(CONFIG_PATH)
        self.stage = self.config.stage(STAGE_ID)

    def test_publishes_only_flat_dr_capability(self) -> None:
        self.assertEqual(set(GPU_CURRICULUM_CAPABILITIES), {STAGE_ID})
        capability = GPU_CURRICULUM_CAPABILITIES[STAGE_ID]
        self.assertTrue(capability["domain_randomization"])
        self.assertFalse(capability["terrain"])
        self.assertFalse(capability["steps"])
        self.assertFalse(capability["jump"])
        self.assertFalse(capability["speed_command"])
        self.assertFalse(capability["yaw_command"])

    def test_stage_contract_is_zero_command_flat_only(self) -> None:
        _validate_stage_contract(self.stage)
        unsafe_stage = SimpleNamespace(**{**self.stage.__dict__, "terrain_enabled": True})
        with self.assertRaisesRegex(WarpCurriculumStageError, "terrain_enabled"):
            _validate_stage_contract(unsafe_stage)

    def test_gpu_task_settings_are_loaded_from_manifest(self) -> None:
        settings = _resolve_gpu_task_settings(self.config)
        self.assertAlmostEqual(settings.sensor_noise_std, 0.015)
        self.assertEqual(settings.control_delay_steps, 1)
        self.assertAlmostEqual(settings.stability_gate_seconds, 8.0)
        self.assertAlmostEqual(settings.command_wheel_feedforward_limit_nm, 0.60)

    def test_dr_batch_declares_vehicle_only_parameters(self) -> None:
        batch_config = load_warp_batch_config(self.config.batch_config_path)
        _validate_batch_contract(batch_config, self.stage)
        self.assertTrue(batch_config.domain_randomization.enabled)
        self.assertFalse(batch_config.domain_randomization.terrain_geometry_randomization)
        self.assertEqual(batch_config.domain_randomization.delay.steps, 0)
        self.assertLessEqual(batch_config.safety.torque_fraction_of_rated, 0.80)
        self.assertFalse(batch_config.runtime.use_precompiled_headers)

    def test_factory_rejects_non_flat_stage_before_gpu_allocation(self) -> None:
        blocked = SimpleNamespace(
            stage_id="grades",
            task_mode="rmuc_grades",
            controller_backend="rmuc_curriculum_controller_v1",
            terrain_enabled=True,
            steps_enabled=False,
            jump_enabled=False,
            domain_randomization_enabled=True,
            command_speed_mps=0.10,
            command_yaw_rate_rad_s=0.0,
            residual_action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
            requires_gpu_parity=True,
            xml_path=ROOT / "rm_train_ground.xml",
        )
        with self.assertRaisesRegex(WarpCurriculumStageError, "cannot build stage"):
            build_curriculum_stage(blocked, self.config)


@unittest.skipUnless(
    os.environ.get("WARP_CURRICULUM_CUDA_TEST") == "1",
    "set WARP_CURRICULUM_CUDA_TEST=1 for the MuJoCo-Warp curriculum integration test",
)
class WarpCurriculumFactoryCudaTest(unittest.TestCase):
    def test_factory_builds_gates_and_closes_flat_dr_stage(self) -> None:
        bundle = build_curriculum_stage(
            load_curriculum_config(CONFIG_PATH).stage(STAGE_ID),
            load_curriculum_config(CONFIG_PATH),
        )
        try:
            self.assertTrue(bundle.task.config.domain_randomization_enabled)
            self.assertGreater(bundle.task.config.sensor_noise_std, 0.0)
            report = bundle.run_stability_gate()
            self.assertTrue(report["zero_residual"])
            self.assertTrue(report["domain_randomization_enabled"])
            self.assertEqual(report["terminated_worlds"], 0)
            self.assertEqual(report["overflowed_worlds"], 0)
            self.assertEqual(report["estopped_worlds"], 0)
            self.assertTrue(report["finite_state"])
        finally:
            bundle.close()
            bundle.close()


if __name__ == "__main__":
    unittest.main()
