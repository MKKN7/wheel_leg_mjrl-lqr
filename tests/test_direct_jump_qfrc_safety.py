"""Regression coverage for official direct-jump torque and telemetry safety.

The CUDA cases are deliberately opt-in because they build the full official
MuJoCo-Warp scene.  The static contracts remain part of the normal unit suite
so a safety telemetry regression cannot be hidden by an unavailable GPU.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import unittest

from official_course_warp import OfficialCourseTask, load_official_course_config
from train_warp_curriculum import load_curriculum_config


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "warp_curriculum_ppo.yaml"
ADAPTER_PATH = ROOT / "configs" / "official_course_warp.yaml"
DIRECT_STAGE_ID = "official_step150_up"


class DirectJumpSafetyTelemetryStaticTest(unittest.TestCase):
    """Contracts that must remain testable without creating a CUDA scene."""

    def test_direct_jump_telemetry_exposes_landing_and_flight_safety_evidence(self) -> None:
        source = inspect.getsource(OfficialCourseTask.tensors)
        expected = {
            "jump_phase",
            "jump_triggered",
            "jump_landing_confirmed",
            "jump_failed",
            "jump_peak_rise_m",
            "jump_minimum_peak_met",
            "jump_landing_vertical_speed_mps",
            "jump_landing_angular_speed_rad_s",
            "jump_landing_kinematics_ok",
            "jump_flight_seconds",
        }
        for field in expected:
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', source)

    def test_jump_supervisor_consumes_all_declared_direct_jump_safety_limits(self) -> None:
        source = inspect.getsource(OfficialCourseTask)
        for setting in (
            "jump_residual_fraction",
            "minimum_peak_body_rise_m",
            "maximum_landing_vertical_speed_mps",
            "maximum_landing_angular_speed_rad_s",
            "prelanding_seconds",
        ):
            with self.subTest(setting=setting):
                self.assertIn(setting, source)
        self.assertIn("contacted_flight", source)
        self.assertIn("torch.maximum", source)

    def test_direct_jump_stage_keeps_the_leg_action_masked(self) -> None:
        manifest = load_curriculum_config(MANIFEST_PATH)
        stage = manifest.stage(DIRECT_STAGE_ID)
        self.assertTrue(stage.jump_enabled)
        self.assertEqual(stage.residual_action_mask[-1], 0.0)
        self.assertEqual(tuple(stage.residual_action_mask[:6]), (1.0,) * 6)

    def test_jump_config_preserves_landing_margin_and_bounded_residual_authority(self) -> None:
        config = load_official_course_config(ADAPTER_PATH)
        self.assertGreaterEqual(config.jump.prelanding_seconds, 0.050)
        self.assertGreaterEqual(config.jump.minimum_peak_body_rise_m, 0.0)
        self.assertGreater(config.jump.maximum_landing_vertical_speed_mps, 0.0)
        self.assertGreater(config.jump.maximum_landing_angular_speed_rad_s, 0.0)
        self.assertGreaterEqual(config.jump.jump_residual_fraction, 0.0)
        self.assertLessEqual(config.jump.jump_residual_fraction, 1.0)


@unittest.skipUnless(
    os.environ.get("WARP_DIRECT_JUMP_CUDA_TEST") == "1",
    "set WARP_DIRECT_JUMP_CUDA_TEST=1 for direct-jump MuJoCo-Warp qfrc integration tests",
)
class DirectJumpGeneralizedForceCudaTest(unittest.TestCase):
    """Full official-scene checks of the 80-percent combined force envelope."""

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from official_course_warp import build_curriculum_stage

        cls.torch = torch
        cls.config = load_curriculum_config(MANIFEST_PATH)
        cls.stage = cls.config.stage(DIRECT_STAGE_ID)
        cls.bundle = build_curriculum_stage(cls.stage, cls.config)
        cls.batch = cls.bundle.batch
        cls.task = cls.bundle.task
        if cls.batch.num_worlds < 2:
            raise AssertionError("direct-jump qfrc safety test requires at least two CUDA worlds")

    @classmethod
    def tearDownClass(cls) -> None:
        bundle = getattr(cls, "bundle", None)
        if bundle is not None:
            bundle.close()

    def setUp(self) -> None:
        self.task.reset()

    def _combined_generalized_force(self):
        """Return the resident actuator-plus-external generalized-force vector."""

        torch = self.torch
        batch = self.batch
        combined = torch.zeros_like(batch._safe_applied_forces)
        actuator_force = torch.zeros_like(combined)
        if batch._force_budget_dofs.numel() > 0:
            controls = torch.index_select(
                batch._safe_controls,
                1,
                batch._force_budget_actuator_indices,
            )
            controls.mul_(batch._force_budget_gears.unsqueeze(0))
            actuator_force.index_add_(1, batch._force_budget_dofs, controls)
        return combined.copy_(actuator_force).add_(batch._safe_applied_forces)

    def _assert_within_combined_force_budget(self, snapshots) -> None:
        torch = self.torch
        batch = self.batch
        self.assertGreater(len(snapshots), 0)
        combined = torch.stack(snapshots)
        budget_dofs = batch._force_budget_dofs
        limited = torch.index_select(combined, 2, budget_dofs)
        limits = torch.index_select(batch._generalized_force_limit, 0, budget_dofs)
        tolerance = torch.finfo(limited.dtype).eps * 64.0 + 1.0e-5
        within = torch.abs(limited) <= limits.view(1, 1, -1) + tolerance
        self.assertTrue(bool(within.all().item()))

    def test_nonzero_direct_jump_residual_never_exceeds_combined_budget_per_substep(self) -> None:
        torch = self.torch
        batch = self.batch
        task = self.task
        self.assertTrue(task._course.direct_jump)
        trigger = task._terrain_task.jump_trigger_progress_m
        self.assertIsNotNone(trigger)
        task._progress.fill_(float(trigger))
        # Seed the supervisor after filling its route-progress evidence.  The
        # trigger event is consumed by the next action transform, which must
        # flush any queued full-scale residual before the delay is read.
        task._action_delay_buffer[:, :, :6] = 1.0
        task._update_jump_supervisor()
        self.assertTrue(bool(task._jump_triggered.all().item()))

        action = torch.full((batch.num_worlds, task.action_size), 0.70, dtype=torch.float32, device=batch.device)
        action[:, -1] = 1.0
        snapshots = []
        original_stage_applied_forces = batch._stage_applied_forces

        def capture_stage_applied_forces(applied_forces) -> None:
            original_stage_applied_forces(applied_forces)
            snapshots.append(self._combined_generalized_force().clone())

        batch._stage_applied_forces = capture_stage_applied_forces
        try:
            # The configured one-step policy delay requires two intervals for
            # the nonzero residual to reach the direct-jump controller path.
            intervals = int(task.config.control_delay_steps) + 2
            for _ in range(intervals):
                task.step(action)
        finally:
            batch._stage_applied_forces = original_stage_applied_forces
            batch._warp.synchronize()

        expected_substeps = intervals * int(batch.config.physics_substeps_per_action)
        self.assertEqual(len(snapshots), expected_substeps)
        self.assertTrue(bool(task._jump_triggered.all().item()))
        self.assertLessEqual(
            float(task._action_delay_buffer[:, :, :6].abs().max().item()),
            0.70 * task._jump_settings.jump_residual_fraction + 1.0e-6,
        )
        self.assertGreater(float(task._effective_action[:, :6].abs().max().item()), 0.0)
        self.assertTrue(bool(task._effective_action[:, -1].eq(0.0).all().item()))
        self.assertTrue(bool(task._policy_action_authority()[:, :6].eq(1.0).all().item()))
        self.assertTrue(bool(task._policy_action_authority()[:, -1].eq(0.0).all().item()))
        self.assertGreater(int(torch.count_nonzero(torch.stack(snapshots)).item()), 0)
        self._assert_within_combined_force_budget(snapshots)

    def test_over_budget_external_force_estops_only_the_faulted_official_world(self) -> None:
        torch = self.torch
        batch = self.batch
        controls = torch.zeros(
            (batch.num_worlds, batch.num_actuators), dtype=torch.float32, device=batch.device
        )
        applied = torch.zeros(
            (batch.num_worlds, batch.host_model.nv), dtype=torch.float32, device=batch.device
        )
        faulted_world = 1
        dof = int(batch._force_budget_dofs[0].item())
        limit = float(batch._generalized_force_limit[dof].item())
        applied[faulted_world, dof] = limit + max(1.0, abs(limit))

        result = batch.step(controls, physics_substeps=1, applied_forces=applied)
        batch._warp.synchronize()
        self.assertTrue(bool(result.estopped[faulted_world].item()))
        healthy = torch.ones(batch.num_worlds, dtype=torch.bool, device=batch.device)
        healthy[faulted_world] = False
        self.assertFalse(bool(result.estopped[healthy].any().item()))
        self.assertTrue(
            bool(torch.equal(result.applied_forces[faulted_world], torch.zeros_like(result.applied_forces[faulted_world])))
        )


if __name__ == "__main__":
    unittest.main()
