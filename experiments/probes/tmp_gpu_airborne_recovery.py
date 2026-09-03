import dataclasses
import torch
import warp_env
from official_course_warp import build_curriculum_stage
from train_warp_curriculum import load_curriculum_config
from warp_flat_controller import FixedGainFlatController

original_loader = warp_env.load_warp_batch_config
warp_env.load_warp_batch_config = lambda path: dataclasses.replace(
    original_loader(path), num_worlds=4, physics_substeps_per_action=10
)
config = load_curriculum_config('configs/warp_curriculum_ppo.yaml')
bundle = build_curriculum_stage(config.stage('official_step150_up'), config)
try:
    task = bundle.task
    task.set_domain_randomization_active(False)
    task.reset()
    root = task.root_qpos_address
    qpos = task._reset_qpos.clone()
    qpos[:, root] = task._route_start_xy[0] + 0.751
    bundle.batch.reset_to_state(qpos, task._reset_qvel, task._calibrated_nominal_controls)
    fractions = torch.full((4,), 0.50, dtype=torch.float32, device=bundle.batch.device)
    original_update = task._update_jump_supervisor
    def tuned_update():
        original_update()
        flight = task._jump_phase == 4
        task._jump_torque_scale.copy_(torch.where(flight, fractions, task._jump_torque_scale))
        task.set_controller_torque_scale(task._jump_torque_scale)
    task._update_jump_supervisor = tuned_update
    original_compute = FixedGainFlatController.compute_controls
    def recovery_compute(self, active_task=None):
        active_task = self.task if active_task is None else active_task
        command = original_compute(self, active_task)
        flight = active_task._jump_phase == 4
        if bool(flight.any().item()):
            recovery = self._nominal_control.clone()
            pos_error = self._reference_hip_qpos.unsqueeze(0) - active_task.batch.qpos.index_select(1, self._hip_qpos)
            vel_error = -active_task.batch.qvel.index_select(1, self._hip_dof)
            hip = self._nominal_control.index_select(1, self._hip_actuator) + 20.0 * pos_error + 4.0 * vel_error
            recovery.index_copy_(1, self._hip_actuator, hip)
            recovery.index_fill_(1, self._wheel_actuator, 0.0)
            self._command.copy_(torch.where(flight.unsqueeze(1), recovery, command))
            torch.nan_to_num(self._command, nan=0.0, posinf=0.0, neginf=0.0, out=self._command)
            torch.clamp(self._command, min=self._control_low, max=self._control_high, out=self._command)
        return self._command
    FixedGainFlatController.compute_controls = recovery_compute
    action = torch.zeros((4, 7), dtype=torch.float32, device=bundle.batch.device)
    for _ in range(130):
        task.step(action)
    for i in range(4):
        print(i, 'phase', int(task._jump_phase[i].item()), 'reason', int(task._safety_reason_code[i].item()),
              'z', float(task.batch.qpos[i, root + 2].item()), 'vz', float(task.batch.qvel[i, task.root_dof_address + 2].item()),
              'ang', [round(float(x), 3) for x in task.batch.qvel[i, task.root_dof_address + 3:task.root_dof_address + 6].tolist()],
              'att', float(task._safety_scratch.attitude_error_rad[i].item()), 'peak', float(task._jump_peak_rise[i].item()),
              'target', bool(task._jump_landing_target_met[i].item()), 'progress', float(task._progress[i].item()))
finally:
    bundle.close()
