"""Focused contract tests for the fixed-gain CUDA flat controller."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import unittest

import numpy as np

from warp_flat_controller import (
    ACTION_SIZE,
    STATE_SIZE,
    FixedGainFlatController,
    WarpFlatControllerCalibration,
    WarpFlatControllerConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _calibration() -> WarpFlatControllerCalibration:
    """Small structural fixture; CUDA tests use a real CPU calibration."""

    return WarpFlatControllerCalibration(
        qpos=np.zeros(21, dtype=np.float32),
        qvel=np.zeros(20, dtype=np.float32),
        nominal_control=np.zeros(ACTION_SIZE, dtype=np.float32),
        gain=np.zeros((ACTION_SIZE, STATE_SIZE), dtype=np.float32),
        reference_qpos=np.zeros(21, dtype=np.float32),
        reference_qvel=np.zeros(20, dtype=np.float32),
        reference_hip_qpos=np.zeros(4, dtype=np.float32),
        hip_qpos_addresses=np.array((7, 8, 9, 10), dtype=np.int64),
        hip_dof_addresses=np.array((6, 7, 8, 9), dtype=np.int64),
        hip_actuator_ids=np.array((0, 1, 3, 4), dtype=np.int64),
        wheel_qpos_addresses=np.array((11, 12), dtype=np.int64),
        wheel_dof_addresses=np.array((10, 11), dtype=np.int64),
        controlled_dof_indices=np.arange(20, dtype=np.int64),
        leg_jacobian=0.1,
        leg_length_m=0.25,
        gas_spring_dofs=np.array((6, 13), dtype=np.int64),
        linearization_heading_yaw=0.0,
        state_digest="0" * 64,
    )


class WarpFlatControllerContractTest(unittest.TestCase):
    def test_default_config_preserves_twenty_percent_torque_reserve(self) -> None:
        config = WarpFlatControllerConfig()
        self.assertLessEqual(config.max_torque_fraction, 0.80)
        self.assertTrue(config.gas_spring_enabled)
        self.assertIsNone(config.command_wheel_accel_limit_nm)
        self.assertIsNone(config.command_wheel_brake_limit_nm)
        self.assertEqual(
            config.gas_spring_max_abs_generalized_force_nm,
            config.gas_spring_torque_nm,
        )

    def test_rejects_invalid_torque_fraction_and_unknown_config_key(self) -> None:
        with self.assertRaises(ValueError):
            WarpFlatControllerConfig(max_torque_fraction=0.81)
        with self.assertRaisesRegex(ValueError, "gas_spring_torque_nm"):
            WarpFlatControllerConfig(
                gas_spring_torque_nm=10.776,
                gas_spring_max_abs_generalized_force_nm=10.775,
            )
        with self.assertRaises(ValueError):
            WarpFlatControllerConfig.from_mapping({"unknown": 1})

    def test_calibration_converts_to_task_reset_payload(self) -> None:
        calibration = _calibration()
        stance = calibration.to_task_calibration()
        self.assertEqual(stance.qpos.shape, (21,))
        self.assertEqual(stance.qvel.shape, (20,))
        self.assertEqual(stance.nominal_control.shape, (ACTION_SIZE,))

    def test_calibration_rejects_wrong_gain_shape(self) -> None:
        kwargs = _calibration().__dict__.copy()
        kwargs["gain"] = np.zeros((ACTION_SIZE, STATE_SIZE - 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            WarpFlatControllerCalibration(**kwargs)


@unittest.skipUnless(
    os.environ.get("WARP_FLAT_CONTROLLER_CUDA_TEST") == "1",
    "set WARP_FLAT_CONTROLLER_CUDA_TEST=1 for the fixed-gain CUDA integration test",
)
class WarpFlatControllerCudaTest(unittest.TestCase):
    def test_control_and_gas_spring_buffers_are_cuda_resident(self) -> None:
        import torch

        from warp_env import WarpPhysicsBatch, load_warp_batch_config
        from warp_flat_controller import calibrate_flat_controller
        from warp_task import WarpFlatWalkingConfig, WarpFlatWalkingTask

        batch_config = replace(
            load_warp_batch_config(ROOT / "configs" / "warp_batch_preflight.yaml"),
            num_worlds=1,
        )
        batch = WarpPhysicsBatch(batch_config)
        calibration = calibrate_flat_controller(batch)
        task = WarpFlatWalkingTask(
            batch,
            WarpFlatWalkingConfig(),
            calibration=calibration.to_task_calibration(),
        )
        controller = FixedGainFlatController(calibration, task)
        task._controller = controller
        task.reset()
        controls = controller.compute_controls(task)
        controls[:, controller._gas_actuator_ids] = batch._control_low[controller._gas_actuator_ids]
        forces = controller.applied_generalized_forces(task, safe_controls=controls)
        self.assertEqual(tuple(controls.shape), (1, ACTION_SIZE))
        self.assertEqual(tuple(forces.shape), (1, batch.host_model.nv))
        self.assertEqual(controls.device, batch.device)
        self.assertEqual(forces.device, batch.device)
        self.assertTrue(bool(torch.isfinite(controls).all().item()))
        # At the negative derated actuator boundary, the negative gas spring
        # has no remaining signed headroom on either matching hip DOF.
        torch.testing.assert_close(
            forces[0, calibration.gas_spring_dofs],
            torch.zeros(2, dtype=torch.float32, device=batch.device),
        )
        self.assertTrue(
            bool(
                (
                    forces[0, calibration.gas_spring_dofs].abs()
                    <= controller.config.gas_spring_max_abs_generalized_force_nm
                ).all().item()
            )
        )
        combined = controls.index_select(1, controller._gas_actuator_ids) + forces.index_select(
            1, controller._gas_dofs
        )
        self.assertTrue(bool((combined >= controller._gas_control_low.unsqueeze(0) - 1.0e-6).all().item()))
        self.assertTrue(bool((combined <= controller._gas_control_high.unsqueeze(0) + 1.0e-6).all().item()))
        result = batch.step(controls, physics_substeps=1, applied_forces=forces)
        batch._warp.synchronize()
        self.assertFalse(bool(result.estopped[0].item()))
        torch.testing.assert_close(result.applied_forces, forces)

    def test_rejects_torque_fraction_mismatch_with_batch(self) -> None:
        from warp_env import WarpPhysicsBatch, load_warp_batch_config
        from warp_flat_controller import calibrate_flat_controller
        from warp_task import WarpFlatWalkingConfig, WarpFlatWalkingTask

        batch_config = replace(
            load_warp_batch_config(ROOT / "configs" / "warp_batch_preflight.yaml"),
            num_worlds=1,
        )
        batch = WarpPhysicsBatch(batch_config)
        calibration = calibrate_flat_controller(batch)
        task = WarpFlatWalkingTask(
            batch,
            WarpFlatWalkingConfig(),
            calibration=calibration.to_task_calibration(),
        )
        with self.assertRaisesRegex(ValueError, "max_torque_fraction"):
            FixedGainFlatController(
                calibration,
                task,
                WarpFlatControllerConfig(max_torque_fraction=0.79),
            )

    def test_rejects_generalized_force_cap_above_derated_hip_limit(self) -> None:
        from warp_env import WarpPhysicsBatch, load_warp_batch_config
        from warp_flat_controller import calibrate_flat_controller
        from warp_task import WarpFlatWalkingConfig, WarpFlatWalkingTask

        batch_config = replace(
            load_warp_batch_config(ROOT / "configs" / "warp_batch_preflight.yaml"),
            num_worlds=1,
        )
        batch = WarpPhysicsBatch(batch_config)
        calibration = calibrate_flat_controller(batch)
        task = WarpFlatWalkingTask(
            batch,
            WarpFlatWalkingConfig(),
            calibration=calibration.to_task_calibration(),
        )
        with self.assertRaisesRegex(ValueError, "gas_spring_max_abs_generalized_force_nm"):
            FixedGainFlatController(
                calibration,
                task,
                WarpFlatControllerConfig(
                    gas_spring_torque_nm=10.775,
                    gas_spring_max_abs_generalized_force_nm=33.0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
