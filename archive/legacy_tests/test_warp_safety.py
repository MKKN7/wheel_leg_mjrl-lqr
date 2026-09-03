"""CPU unit tests for the device-local Warp safety primitives."""

from __future__ import annotations

import unittest

import torch

from warp_safety import (
    SAFETY_REASON_ATTITUDE,
    SAFETY_REASON_CONTACT_LOSS,
    SAFETY_REASON_HEIGHT,
    SAFETY_REASON_LATCHED,
    SAFETY_REASON_LEG_LIMIT,
    SAFETY_REASON_NONFINITE_CONTROL,
    SAFETY_REASON_NONFINITE_STATE,
    SAFETY_REASON_JOINT_LIMIT,
    SAFETY_REASON_OVERFLOW,
    WarpSafetyLimits,
    WarpSafetyScratch,
    clip_controls,
    evaluate_safety,
    quat_attitude_error,
    sanitize_observation,
)


class WarpSafetyTest(unittest.TestCase):
    def _inputs(self, worlds: int = 3):
        qpos = torch.zeros((worlds, 10), dtype=torch.float32)
        qpos[:, 3] = 1.0
        qpos[:, 2] = 0.4
        qvel = torch.zeros((worlds, 6), dtype=torch.float32)
        controls = torch.zeros((worlds, 2), dtype=torch.float32)
        overflow = torch.zeros(worlds, dtype=torch.int32)
        return qpos, qvel, controls, overflow

    def test_limits_reject_more_than_twenty_percent_reserve(self) -> None:
        with self.assertRaises(ValueError):
            WarpSafetyLimits(torque_fraction_of_rated=0.81)

    def test_clip_controls_replaces_nan_and_preserves_asymmetric_caps(self) -> None:
        controls = torch.tensor([[float("nan"), 9.0], [-9.0, -2.0]], dtype=torch.float32)
        low = torch.tensor([-2.0, -1.0])
        high = torch.tensor([1.0, 3.0])
        estopped = torch.tensor([False, True])
        clipped, malformed = clip_controls(controls, low, high, estopped=estopped)
        self.assertTrue(torch.equal(malformed, torch.tensor([True, False])))
        torch.testing.assert_close(clipped, torch.tensor([[0.0, 3.0], [0.0, 0.0]]))
        aliased = controls.clone()
        _, aliased_malformed = clip_controls(aliased, low, high, out=aliased)
        self.assertTrue(bool(aliased_malformed[0]))

    def test_scratch_reuses_preallocated_safety_outputs(self) -> None:
        qpos, qvel, controls, overflow = self._inputs(2)
        scratch = WarpSafetyScratch(2, 2, "cpu", torch.float32)
        result = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            reference_root_height_m=0.4,
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(),
            scratch=scratch,
        )
        self.assertEqual(result.safe_controls.data_ptr(), scratch.safe_controls.data_ptr())
        self.assertEqual(result.terminated.data_ptr(), scratch.terminated.data_ptr())
        self.assertEqual(result.reason_code.data_ptr(), scratch.reason_code.data_ptr())
        clipped, malformed = clip_controls(
            controls,
            torch.tensor([-2.0, -1.0]),
            torch.tensor([2.0, 1.0]),
            scratch=scratch,
        )
        self.assertEqual(clipped.data_ptr(), scratch.safe_controls.data_ptr())
        self.assertEqual(malformed.data_ptr(), scratch.control_nonfinite.data_ptr())

    def test_invalid_cached_limits_fail_closed_without_host_readback(self) -> None:
        controls = torch.zeros((2, 2), dtype=torch.float32)
        clipped, malformed = clip_controls(
            controls,
            torch.tensor([-1.0, float("nan")]),
            torch.tensor([1.0, 1.0]),
        )
        self.assertTrue(bool(malformed.all()))
        # The caller will estop the world from ``malformed``; clipping still
        # produces a finite tensor and never lets invalid bounds reach Warp.
        self.assertTrue(bool(torch.isfinite(clipped).all()))
        torch.testing.assert_close(clipped, torch.zeros_like(clipped))

    def test_quaternion_error_marks_zero_norm_invalid(self) -> None:
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        error, valid = quat_attitude_error(quaternion, torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertAlmostEqual(float(error[0]), 0.0, places=6)
        self.assertAlmostEqual(float(error[1]), torch.pi, places=6)

    def test_per_world_reference_quaternions_are_supported(self) -> None:
        current = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
        )
        reference = current.clone()
        error, valid = quat_attitude_error(current, reference)
        torch.testing.assert_close(error, torch.zeros(2))
        self.assertEqual(valid.tolist(), [True, True])

    def test_evaluate_safety_latches_independent_worlds(self) -> None:
        qpos, qvel, controls, overflow = self._inputs()
        controls[0, 0] = float("nan")
        qpos[1, 2] = 0.1
        result = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            reference_root_height_m=0.4,
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(max_root_height_drop_m=0.2),
        )
        self.assertEqual(result.reason_code.tolist(), [1, SAFETY_REASON_HEIGHT, 0])
        self.assertEqual(result.terminated.tolist(), [True, True, False])
        self.assertTrue(torch.equal(result.safe_controls[0], torch.zeros(2)))
        # A prior estop remains latched even after its immediate fault clears.
        controls.zero_()
        latched = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            reference_root_height_m=0.4,
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(max_root_height_drop_m=0.2),
            previous_estopped=torch.ones(3, dtype=torch.bool),
            previous_reason_code=result.reason_code,
        )
        self.assertEqual(latched.reason_code.tolist(), [1, SAFETY_REASON_HEIGHT, SAFETY_REASON_LATCHED])

    def test_joint_leg_and_contact_guards_are_optional_but_independent(self) -> None:
        qpos, qvel, controls, overflow = self._inputs()
        result = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            reference_root_height_m=0.4,
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(
                max_leg_length_difference_m=0.05,
                min_leg_length_m=0.2,
                max_leg_length_m=0.8,
                max_contact_loss_steps=2,
            ),
            joint_positions=torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
            joint_lower=torch.tensor([-1.0, -1.0]),
            joint_upper=torch.tensor([1.0, 1.0]),
            leg_lengths=torch.tensor([[0.4, 0.4], [0.4, 0.4], [0.4, 0.4]]),
            wheel_contact=torch.tensor([[True, True], [True, True], [False, True]]),
            contact_loss_steps=torch.tensor([[0, 0], [0, 0], [2, 0]], dtype=torch.int32),
        )
        self.assertFalse(bool(result.terminated[0]))
        self.assertEqual(int(result.reason_code[1]), SAFETY_REASON_JOINT_LIMIT)
        self.assertEqual(int(result.reason_code[2]), SAFETY_REASON_CONTACT_LOSS)

    def test_attitude_guard_and_observation_sanitizer(self) -> None:
        qpos, qvel, controls, overflow = self._inputs(1)
        qpos[0, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        result = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            reference_root_height_m=0.4,
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(max_attitude_error_rad=1.0),
        )
        self.assertTrue(bool(result.attitude_limit[0]))
        self.assertEqual(int(result.reason_code[0]), SAFETY_REASON_ATTITUDE)
        obs = torch.tensor([[float("nan"), float("inf"), -float("inf"), 12.0, -12.0]])
        clean, finite = sanitize_observation(obs)
        self.assertFalse(bool(finite[0]))
        torch.testing.assert_close(clean, torch.tensor([[0.0, 10.0, -10.0, 10.0, -10.0]]))
        aliased = obs.clone()
        _, aliased_finite = sanitize_observation(aliased, out=aliased)
        self.assertFalse(bool(aliased_finite[0]))

    def test_nonfinite_sensor_data_is_a_state_failure(self) -> None:
        qpos, qvel, controls, overflow = self._inputs(1)
        sensors = torch.zeros((1, 4), dtype=torch.float32)
        sensors[0, 2] = float("nan")
        result = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            reference_root_height_m=0.4,
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(),
            sensordata=sensors,
        )
        self.assertTrue(bool(result.sensor_nonfinite[0]))
        self.assertTrue(bool(result.terminated[0]))

    def test_per_world_reference_height_rebases_after_masked_reset(self) -> None:
        qpos, qvel, controls, overflow = self._inputs(2)
        qpos[:, 2] = torch.tensor([0.30, 0.60])
        result = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            reference_root_height_m=torch.tensor([0.50, 0.60]),
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(max_root_height_drop_m=0.15),
        )
        self.assertEqual(result.height_limit.tolist(), [True, False])

    def test_nonfinite_reference_or_joint_bounds_fail_closed(self) -> None:
        qpos, qvel, controls, overflow = self._inputs(2)
        result = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            reference_root_height_m=torch.tensor([float("nan"), 0.4]),
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(),
            joint_positions=torch.zeros((2, 2)),
            joint_lower=torch.tensor([-1.0, float("nan")]),
            joint_upper=torch.tensor([1.0, 1.0]),
        )
        self.assertTrue(bool(result.height_limit[0]))
        self.assertTrue(bool(result.joint_limit.all()))
        self.assertTrue(bool(result.terminated.all()))

    def test_vectorized_p0_fault_matrix_isolated_and_zeroes_controls(self) -> None:
        """Each P0 signal terminates only the malformed world in a batch."""

        worlds = 9
        qpos, qvel, controls, overflow = self._inputs(worlds)
        sensors = torch.zeros((worlds, 4), dtype=torch.float32)
        controls[1, 0] = float("nan")
        qvel[2, 0] = float("nan")
        overflow[3] = 1
        qpos[4, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        qpos[5, 2] = 0.1
        joint_positions = torch.zeros((worlds, 2), dtype=torch.float32)
        joint_positions[6, 0] = 2.0
        leg_lengths = torch.full((worlds, 2), 0.35, dtype=torch.float32)
        leg_lengths[7, 1] = 0.50
        wheel_contact = torch.ones((worlds, 2), dtype=torch.bool)
        wheel_contact[8, 0] = False
        contact_loss_steps = torch.zeros((worlds, 2), dtype=torch.int32)
        contact_loss_steps[8, 0] = 2
        result = evaluate_safety(
            qpos,
            qvel,
            controls,
            overflow,
            root_qpos_address=0,
            reference_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            reference_root_height_m=0.4,
            control_low=torch.tensor([-2.0, -1.0]),
            control_high=torch.tensor([2.0, 1.0]),
            limits=WarpSafetyLimits(
                max_attitude_error_rad=1.0,
                max_root_height_drop_m=0.2,
                max_leg_length_difference_m=0.05,
                min_leg_length_m=0.18,
                max_leg_length_m=0.40,
                max_contact_loss_steps=2,
            ),
            sensordata=sensors,
            joint_positions=joint_positions,
            joint_lower=torch.tensor([-1.0, -1.0]),
            joint_upper=torch.tensor([1.0, 1.0]),
            leg_lengths=leg_lengths,
            wheel_contact=wheel_contact,
            contact_loss_steps=contact_loss_steps,
        )
        self.assertEqual(
            result.reason_code.tolist(),
            [
                0,
                SAFETY_REASON_NONFINITE_CONTROL,
                SAFETY_REASON_NONFINITE_STATE,
                SAFETY_REASON_OVERFLOW,
                SAFETY_REASON_ATTITUDE,
                SAFETY_REASON_HEIGHT,
                SAFETY_REASON_JOINT_LIMIT,
                SAFETY_REASON_LEG_LIMIT,
                SAFETY_REASON_CONTACT_LOSS,
            ],
        )
        self.assertEqual(result.terminated.tolist(), [False] + [True] * (worlds - 1))
        torch.testing.assert_close(result.safe_controls[0], torch.zeros(2))
        torch.testing.assert_close(result.safe_controls[1:], torch.zeros((worlds - 1, 2)))


if __name__ == "__main__":
    unittest.main()
