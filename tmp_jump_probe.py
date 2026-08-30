import dataclasses

import torch
import warp_env
import train_warp_ppo
import official_course_warp
from warp_flat_controller import FixedGainFlatController

from official_course_warp import build_curriculum_stage
from train_warp_curriculum import load_curriculum_config


original_loader = warp_env.load_warp_batch_config
warp_env.load_warp_batch_config = lambda path: dataclasses.replace(
    original_loader(path), num_worlds=1, physics_substeps_per_action=1
)
original_flat_loader = train_warp_ppo.load_flat_ppo_training_config


def tuned_flat_loader(path):
    flat = original_flat_loader(path)
    tuned_controller = dataclasses.replace(flat.flat_controller, leg_force_kp_n_per_m=700.0)
    return dataclasses.replace(flat, flat_controller=tuned_controller)


train_warp_ppo.load_flat_ppo_training_config = tuned_flat_loader
original_adapter_loader = official_course_warp.load_official_course_config


def tuned_adapter_loader(path):
    adapter = original_adapter_loader(path)
    tuned_jump = dataclasses.replace(adapter.jump, thrust_seconds=0.35)
    return dataclasses.replace(adapter, jump=tuned_jump)


official_course_warp.load_official_course_config = tuned_adapter_loader
original_compute_controls = FixedGainFlatController.compute_controls


def biased_compute_controls(self, task=None):
    result = original_compute_controls(self, task)
    active_task = self.task if task is None else task
    if hasattr(active_task, "_jump_phase"):
        thrust = (active_task._jump_phase == 3).unsqueeze(1)
        thrust_command = torch.as_tensor((-28.0, -28.0, 28.0, -28.0), dtype=torch.float32, device=self.device)
        current_hips = result.index_select(1, self._hip_actuator)
        current_hips = torch.where(thrust, thrust_command, current_hips)
        result.index_copy_(1, self._hip_actuator, current_hips)
        torch.clamp(result, min=self._control_low, max=self._control_high, out=result)
    return result


FixedGainFlatController.compute_controls = biased_compute_controls
config = load_curriculum_config("configs/warp_curriculum_ppo.yaml")
bundle = build_curriculum_stage(config.stage("official_step150_up"), config)
try:
    task = bundle.task
    root = task.root_qpos_address
    qpos = task._reset_qpos.clone()
    qpos[:, root] = -5.15
    qpos[:, root + 1] = 0.0
    bundle.batch.reset_to_state(qpos, task._reset_qvel, task._calibrated_nominal_controls)
    action = torch.zeros((1, 7), dtype=torch.float32, device=bundle.batch.device)
    for index in range(1000):
        result = task.step(action)
        if (
            bool((task._jump_phase == 3).item())
            and bool((task._jump_peak_rise >= task._jump_settings.minimum_peak_body_rise_m).item())
            and bool((task.batch.qvel[:, task.root_dof_address + 2] > 0.10).item())
        ):
            task._jump_liftoff.fill_(True)
            task._jump_phase.fill_(4)
            task._jump_phase_elapsed.zero_()
            print("DIAGNOSTIC_LIFTOFF", index)
        if index % 25 == 0:
            print(
                "tick",
                index,
                "progress",
                task._progress.tolist(),
                "z",
                task.batch.qpos[:, root + 2].tolist(),
                "vz",
                task.batch.qvel[:, task.root_dof_address + 2].tolist(),
                "phase",
                task._jump_phase.tolist(),
                "contact",
                task._side_support_contacts().tolist(),
                "reason",
                task._safety_reason_code.tolist(),
                "leg_cmd",
                task._command_leg_length.tolist(),
                "hip_ctrl",
                task._controller._command.tolist(),
                "applied_hip",
                task.batch._safe_applied_forces[:, task._controller._gas_dofs].tolist(),
                "leg_sensor",
                task.batch.sensordata[:, task._leg_length_indices].tolist(),
            )
        if bool(result.terminated.any().item()):
            print(
                "FAULT",
                index,
                "progress",
                task._progress.tolist(),
                "phase",
                task._jump_phase.tolist(),
                "trigger",
                task._jump_triggered.tolist(),
                "liftoff",
                task._jump_liftoff.tolist(),
                "failed",
                task._jump_failed.tolist(),
                "peak",
                task._jump_peak_rise.tolist(),
                "z",
                task.batch.qpos[:, root + 2].tolist(),
                "vz",
                task.batch.qvel[:, task.root_dof_address + 2].tolist(),
                "reason",
                task._safety_reason_code.tolist(),
                "route_unsafe",
                task._route_unsafe.tolist(),
            )
            break
finally:
    bundle.close()
