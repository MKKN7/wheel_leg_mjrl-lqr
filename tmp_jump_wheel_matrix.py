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
saved_qpos = task._reset_qpos.clone()
saved_qvel = task._reset_qvel.clone()
action = torch.zeros((1, 7), dtype=torch.float32, device=bundle.batch.device)

for hip, wheel, rise in ((20.0, 0.0, 0.12), (22.0, 0.0, 0.12), (24.0, 0.0, 0.12),
                         (20.0, 0.15, 0.12), (22.0, 0.15, 0.12), (24.0, 0.15, 0.12),
                         (20.0, 0.30, 0.12), (22.0, 0.30, 0.12), (24.0, 0.30, 0.12)):
    task._jump_settings = dataclasses.replace(
        task._jump_settings,
        thrust_hip_torque_nm=hip,
        thrust_wheel_torque_nm=wheel,
        minimum_liftoff_body_rise_m=rise,
        landing_torque_fraction=0.0,
        flight_torque_fraction=0.5,
        landing_length_m=0.320,
    )
    target = task._jump_actuator_override_target
    target.zero_()
    hip_target = torch.as_tensor((-hip, -hip, hip, -hip), dtype=torch.float32, device=bundle.batch.device).view(1, 4)
    target[:, controller._hip_actuator] = hip_target
    target[:, controller._wheel_actuator] = -wheel
    task._reset_qpos.copy_(saved_qpos)
    task._reset_qvel.copy_(saved_qvel)
    task._reset_qpos[:, root] = -5.25
    task._reset_qvel[:, root_dof] = 0.20
    task.reset()
    task._progress.fill_(0.75)
    task._jump_triggered.fill_(True)
    task._jump_phase.fill_(3)
    task._jump_phase_elapsed.zero_()
    task._update_jump_supervisor()
    peak = float(task.batch.qpos[0, root + 2].item())
    max_x = float(task.batch.qpos[0, root].item())
    final_target = False
    for index in range(700):
        result = task.step(action)
        peak = max(peak, float(task.batch.qpos[0, root + 2].item()))
        max_x = max(max_x, float(task.batch.qpos[0, root].item()))
        final_target = final_target or bool(task._jump_landing_target_met.any().item())
        if bool(result.terminated.any().item()):
            break
    print(
        "hip", hip, "wheel", wheel, "rise", rise,
        "peak", round(peak - float(task._route_reference_height[0].item()), 3),
        "x", round(max_x, 3), "phase", task._jump_phase.tolist(),
        "target", task._jump_landing_target_met.tolist(), "landed", task._jump_landing_confirmed.tolist(),
        "fault", index if bool(result.terminated.any().item()) else -1,
        "reason", task._safety_reason_code.tolist(),
        "leg", task.batch.sensordata[:, task._leg_length_indices].tolist(),
    )
bundle.close()
