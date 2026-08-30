import dataclasses
import torch
import warp_env
from warp_flat_controller import FixedGainFlatController

from official_course_warp import build_curriculum_stage
from train_warp_curriculum import load_curriculum_config

original_loader = warp_env.load_warp_batch_config
warp_env.load_warp_batch_config = lambda path: dataclasses.replace(
    original_loader(path), num_worlds=1, physics_substeps_per_action=10
)
config = load_curriculum_config("configs/warp_curriculum_ppo.yaml")
bundle = build_curriculum_stage(config.stage("official_step150_up"), config)
try:
    task = bundle.task
    # Diagnostic only: preserve a fraction of the nominal state-feedback
    # correction around the deterministic thrust target to test angular damping.
    original_compute = FixedGainFlatController.compute_controls
    def blended_compute(self, active_task=None):
        active_task = self.task if active_task is None else active_task
        mask = getattr(active_task, "_jump_actuator_override_enabled", None)
        if mask is None or not bool(mask.any().item()):
            return original_compute(self, active_task)
        saved = mask.clone()
        mask.zero_()
        normal = original_compute(self, active_task).clone()
        mask.copy_(saved)
        target = active_task._jump_actuator_override_target
        # Half-strength feedback correction; target remains clipped by the
        # controller's normal actuator envelope below.
        mixed = target + 0.50 * (normal - self._nominal_control)
        self._command.copy_(torch.where(mask.unsqueeze(1), mixed, normal))
        torch.nan_to_num(self._command, nan=0.0, posinf=0.0, neginf=0.0, out=self._command)
        torch.clamp(self._command, min=self._control_low, max=self._control_high, out=self._command)
        return self._command
    FixedGainFlatController.compute_controls = blended_compute
    root = task.root_qpos_address
    qpos = task._reset_qpos.clone()
    qpos[:, root] = task._route_start_xy[0] + 0.751
    bundle.batch.reset_to_state(qpos, task._reset_qvel, task._calibrated_nominal_controls)
    action = torch.zeros((1, 7), dtype=torch.float32, device=bundle.batch.device)
    for index in range(500):
        result = task.step(action)
        phase = int(task._jump_phase[0].item())
        if index < 10 or phase == 3 or bool(result.terminated.any().item()):
            print(
                "tick", index,
                "progress", float(task._progress[0].item()),
                "phase", phase,
                "phase_t", float(task._jump_phase_elapsed[0].item()),
                "z", float(task.batch.qpos[0, root + 2].item()),
                "vz", float(task.batch.qvel[0, task.root_dof_address + 2].item()),
                "contact", task._side_support_contacts().tolist(),
                "override", task._jump_actuator_override_enabled.tolist(),
                "cmd", task._controller._command.tolist(),
                "safe", task.batch._safe_controls.tolist(),
                "gas", task.batch._safe_applied_forces[:, task._controller._gas_dofs].tolist(),
                "reason", task._safety_reason_code.tolist(),
                "term", result.terminated.tolist(),
                "jumpfail", task._jump_failed.tolist(),
                "leg", task.batch.sensordata[:, task._leg_length_indices].tolist(),
                "att_err", task._safety_scratch.attitude_error_rad.tolist(),
                "angvel", task.batch.qvel[:, task.root_dof_address + 3:task.root_dof_address + 6].tolist(),
                "xy", task.batch.qpos[:, root:root + 2].tolist(),
                "peak", task._jump_peak_rise.tolist(),
            )
        if bool(result.terminated.any().item()):
            print("FINAL", index, task._safety_reason_code.tolist(), task._jump_phase.tolist())
            break
finally:
    bundle.close()
