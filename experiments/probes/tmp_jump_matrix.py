import dataclasses

import torch
import warp_env

from official_course_warp import build_curriculum_stage
from train_warp_curriculum import load_curriculum_config


original_loader = warp_env.load_warp_batch_config
warp_env.load_warp_batch_config = lambda path: dataclasses.replace(
    original_loader(path), num_worlds=1, physics_substeps_per_action=1
)
config = load_curriculum_config("configs/warp_curriculum_ppo.yaml")
bundle = build_curriculum_stage(config.stage("official_step150_up"), config)
task = bundle.task
controller = task._controller
root = task.root_qpos_address
root_dof = task.root_dof_address
task._safety_limits = dataclasses.replace(task._safety_limits, max_leg_length_m=0.500)
original_compute = controller.compute_controls


def recovery_compute(bound_task):
    # Preserve a bounded vertical launch while constrained; use a CPU-matched
    # airborne attitude recovery after the contact proxy clears.
    if not bool(task._jump_phase.eq(3).any().item()):
        task._jump_actuator_override_enabled.zero_()
    controls = original_compute(bound_task)
    flight = task._jump_phase.eq(4)
    if bool(flight.any().item()):
        hip_error = -controller._position_error.index_select(1, controller._hip_dof)
        hip_rate = -bound_task.batch.qvel.index_select(1, controller._hip_dof)
        controls[:, controller._hip_actuator] += 0.50 * (20.0 * hip_error + 4.0 * hip_rate)
        attitude = controller._position_error[:, controller.root_dof_address + 3]
        pitch_rate = bound_task.batch.qvel[:, controller.root_dof_address + 3]
        wheel = torch.clamp(6.0 * attitude + 0.48 * pitch_rate, -2.4, 2.4)
        controls[:, controller._wheel_actuator] = wheel.unsqueeze(1)
        torch.clamp(controls, min=controller._control_low, max=controller._control_high, out=controls)
    return controls


controller.compute_controls = recovery_compute
saved_qpos = task._reset_qpos.clone()
saved_qvel = task._reset_qvel.clone()
task._reset_qpos[:, root] = -5.25
task._reset_qpos[:, root + 1] = 0.0
task._reset_qvel.zero_()
task._reset_qvel[:, root_dof] = 0.20
action = torch.zeros((1, 7), dtype=torch.float32, device=bundle.batch.device)
for amplitude in (20.0, 22.0, 24.0, 26.0):
    for landing_fraction in (0.0, 0.20, 0.30):
        task._jump_settings = dataclasses.replace(
            task._jump_settings,
            thrust_hip_torque_nm=amplitude,
            landing_torque_fraction=landing_fraction,
            flight_torque_fraction=0.50,
            landing_length_m=0.320,
            minimum_peak_body_rise_m=0.120,
        )
        target = task._jump_actuator_override_target
        target.zero_()
        hip_target = torch.as_tensor(
            (-amplitude, -amplitude, amplitude, -amplitude),
            dtype=torch.float32,
            device=bundle.batch.device,
        ).view(1, 4)
        target[:, controller._hip_actuator] = hip_target
        task._reset_qpos.copy_(saved_qpos)
        task._reset_qvel.copy_(saved_qvel)
        task.reset()
        task._reset_qpos[:, root] = -5.25
        task._reset_qvel[:, root_dof] = 0.20
        task.reset()
        task._progress.fill_(0.75)
        task._jump_triggered.fill_(True)
        task._jump_phase.fill_(3)
        task._jump_phase_elapsed.zero_()
        task._update_jump_supervisor()
        peak = float(task.batch.qpos[0, root + 2].item())
        for index in range(500):
            result = task.step(action)
            peak = max(peak, float(task.batch.qpos[0, root + 2].item()))
            if bool(result.terminated.any().item()):
                break
        print(
            "amp", amplitude,
            "land", landing_fraction,
            "peak", round(peak - float(task._route_reference_height[0].item()), 3),
            "phase", task._jump_phase.tolist(),
            "target", task._jump_landing_target_met.tolist(),
            "landed", task._jump_landing_confirmed.tolist(),
            "fault", index if bool(result.terminated.any().item()) else -1,
            "reason", task._safety_reason_code.tolist(),
            "leg", task.batch.sensordata[:, task._leg_length_indices].tolist(),
        )
bundle.close()
