"""Focused tests for the fail-closed MuJoCo-Warp batch preflight configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest

import numpy as np
import yaml

from warp_env import WarpBatchError, _signed_rated_control_limits, load_warp_batch_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "warp_batch_preflight.yaml"


class WarpBatchConfigTest(unittest.TestCase):
    def _config_mapping(self) -> dict:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def _write_config(self, content: dict) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        with temporary:
            yaml.safe_dump(content, temporary, sort_keys=False)
        return Path(temporary.name)

    def test_preflight_config_is_raw_control_only_and_not_training_enabled(self) -> None:
        config = load_warp_batch_config(CONFIG_PATH)
        self.assertEqual(config.backend, "mujoco_warp")
        self.assertEqual(config.num_worlds, 128)
        self.assertEqual(config.controller_backend, "raw_controls_only")
        self.assertFalse(config.ppo_training_enabled)
        self.assertLessEqual(config.safety.torque_fraction_of_rated, 0.80)
        self.assertTrue(config.fall_guard.enabled)
        self.assertAlmostEqual(config.fall_guard.max_attitude_error_rad, 1.0)
        self.assertTrue(config.preflight.verify_single_step_parity)
        self.assertTrue(config.preflight.verify_estop)

    def test_rejects_torque_limit_over_hard_margin(self) -> None:
        content = self._config_mapping()
        content["safety"]["torque_fraction_of_rated"] = 0.81
        path = self._write_config(content)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(WarpBatchError, "torque_fraction"):
            load_warp_batch_config(path)

    def test_rejects_premature_ppo_enablement(self) -> None:
        content = self._config_mapping()
        content["scope"]["ppo_training_enabled"] = True
        path = self._write_config(content)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(WarpBatchError, "PPO training cannot be enabled"):
            load_warp_batch_config(path)

    def test_rejects_cuda_graph_before_controller_parity(self) -> None:
        content = self._config_mapping()
        content["runtime"]["cuda_graph"] = True
        path = self._write_config(content)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(WarpBatchError, "CUDA graph capture"):
            load_warp_batch_config(path)

    def test_rejects_nonpositive_parity_tolerance(self) -> None:
        content = self._config_mapping()
        content["preflight"]["qvel_max_abs_error"] = 0.0
        path = self._write_config(content)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(WarpBatchError, "qvel_max_abs_error"):
            load_warp_batch_config(path)

    def test_signed_caps_intersect_declared_ranges_before_derating(self) -> None:
        model = SimpleNamespace(
            nu=2,
            actuator_ctrlrange=np.asarray([[-2.0, 3.0], [-4.0, 4.0]]),
            actuator_forcerange=np.asarray([[-1.0, 5.0], [-2.0, 2.0]]),
            actuator_ctrllimited=np.asarray([True, False]),
            actuator_forcelimited=np.asarray([True, True]),
        )
        lower, upper = _signed_rated_control_limits(model, 0.80)
        np.testing.assert_allclose(lower, [-0.8, -1.6])
        np.testing.assert_allclose(upper, [2.4, 1.6])

    def test_curriculum_dr_config_keeps_terrain_geometry_fixed(self) -> None:
        config = load_warp_batch_config(ROOT / "configs" / "warp_rmuc_curriculum_batch.yaml")
        dr = config.domain_randomization
        self.assertTrue(dr.enabled)
        self.assertFalse(dr.terrain_geometry_randomization)
        self.assertEqual(dr.ranges.body_mass, (-0.04, 0.04))
        self.assertEqual(dr.ranges.body_inertia, (-0.10, 0.10))
        self.assertEqual(dr.ranges.dof_damping, (-0.12, 0.12))
        self.assertEqual(dr.ranges.geom_friction, (-0.10, 0.10))
        self.assertEqual(dr.ranges.actuator_strength, (-0.07, 0.0))

    def test_rejects_terrain_geometry_randomization(self) -> None:
        content = self._config_mapping()
        content["xml_path"] = str((ROOT / "rm_train_ground.xml").resolve())
        content["domain_randomization"] = {
            "enabled": True,
            "seed": 1,
            "ranges": {},
            "noise": {"std": 0.0},
            "delay": {"steps": 0},
            "terrain_geometry_randomization": True,
        }
        path = self._write_config(content)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(WarpBatchError, "geometry randomization"):
            load_warp_batch_config(path)


@unittest.skipUnless(
    os.environ.get("WARP_QFRC_CUDA_TEST") == "1",
    "set WARP_QFRC_CUDA_TEST=1 for the MuJoCo-Warp applied-force integration test",
)
class WarpAppliedForcesCudaTest(unittest.TestCase):
    def test_latch_estop_immediately_clears_resident_actuation_buffers(self) -> None:
        import torch

        from warp_env import WarpPhysicsBatch

        batch = WarpPhysicsBatch(load_warp_batch_config(CONFIG_PATH))
        batch.reset()
        batch._safe_controls.fill_(0.25)
        batch._safe_applied_forces.fill_(0.5)
        world_mask = torch.zeros(batch.num_worlds, dtype=torch.bool, device=batch.device)
        world_mask[0] = True
        batch.latch_estop(world_mask)
        batch._warp.synchronize()
        self.assertTrue(bool(batch.estopped[0].item()))
        self.assertFalse(bool(batch.estopped[1:].any().item()))
        self.assertTrue(bool(torch.equal(batch._safe_controls[0], torch.zeros_like(batch._safe_controls[0]))))
        self.assertTrue(
            bool(torch.equal(batch._safe_applied_forces[0], torch.zeros_like(batch._safe_applied_forces[0])))
        )
        self.assertTrue(bool(torch.equal(batch.ctrl[0], torch.zeros_like(batch.ctrl[0]))))
        self.assertTrue(bool(torch.equal(batch.qfrc_applied[0], torch.zeros_like(batch.qfrc_applied[0]))))

    def test_applied_forces_are_staged_and_cleared_per_step(self) -> None:
        import torch

        from warp_env import WarpPhysicsBatch

        batch = WarpPhysicsBatch(load_warp_batch_config(CONFIG_PATH))
        batch.reset()
        controls = torch.zeros(
            (batch.num_worlds, batch.num_actuators), dtype=torch.float32, device=batch.device
        )
        applied = torch.zeros(
            (batch.num_worlds, batch.host_model.nv), dtype=torch.float32, device=batch.device
        )
        controlled_dof = int(batch._force_budget_dofs[0].item())
        applied[:, controlled_dof] = 0.01
        result = batch.step(controls, physics_substeps=1, applied_forces=applied)
        torch.cuda.synchronize()
        self.assertEqual(tuple(result.applied_forces.shape), (batch.num_worlds, batch.host_model.nv))
        torch.testing.assert_close(result.applied_forces, applied)

        # ``None`` means no external generalized force, and must clear the
        # prior buffer instead of retaining it in subsequent MuJoCo steps.
        cleared = batch.step(controls, physics_substeps=1)
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.equal(cleared.applied_forces, torch.zeros_like(applied))))

    def test_over_budget_applied_force_estops_only_its_world_and_zeroes_force(self) -> None:
        import torch

        from warp_env import WarpPhysicsBatch

        batch = WarpPhysicsBatch(load_warp_batch_config(CONFIG_PATH))
        batch.reset()
        controls = torch.zeros(
            (batch.num_worlds, batch.num_actuators), dtype=torch.float32, device=batch.device
        )
        applied = torch.zeros(
            (batch.num_worlds, batch.host_model.nv), dtype=torch.float32, device=batch.device
        )
        controlled_dof = int(batch._force_budget_dofs[0].item())
        applied[0, controlled_dof] = float(batch._generalized_force_limit[controlled_dof].item()) + 1.0
        result = batch.step(controls, physics_substeps=1, applied_forces=applied)
        torch.cuda.synchronize()
        self.assertTrue(bool(result.estopped[0].item()))
        self.assertFalse(bool(result.estopped[1:].any().item()))
        self.assertTrue(bool(torch.equal(result.applied_forces[0], torch.zeros_like(result.applied_forces[0]))))

    def test_nonfinite_applied_force_estops_only_its_world(self) -> None:
        import torch

        from warp_env import WarpPhysicsBatch

        batch = WarpPhysicsBatch(load_warp_batch_config(CONFIG_PATH))
        batch.reset()
        controls = torch.zeros(
            (batch.num_worlds, batch.num_actuators), dtype=torch.float32, device=batch.device
        )
        applied = torch.zeros(
            (batch.num_worlds, batch.host_model.nv), dtype=torch.float32, device=batch.device
        )
        applied[0, 0] = float("nan")
        result = batch.step(controls, physics_substeps=1, applied_forces=applied)
        torch.cuda.synchronize()
        self.assertTrue(bool(result.estopped[0].item()))
        self.assertFalse(bool(result.estopped[1:].any().item()))
        self.assertTrue(bool(torch.equal(result.applied_forces[0], torch.zeros(batch.host_model.nv, device=batch.device))))


if __name__ == "__main__":
    unittest.main()
