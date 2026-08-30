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
try:
    task = bundle.task
    root = task.root_qpos_address
    root_dof = task.root_dof_address
    task._safety_limits = dataclasses.replace(task._safety_limits, max_leg_length_m=0.500)
    task._jump_settings = dataclasses.replace(
        task._jump_settings,
        flight_torque_fraction=0.50,
        landing_torque_fraction=0.00,
        landing_length_m=0.320,
        thrust_seconds=0.16,
        minimum_liftoff_body_rise_m=0.120,
    )
    for initial_speed in (0.20,):
        saved_qpos = task._reset_qpos.clone()
        saved_qvel = task._reset_qvel.clone()
        task._reset_qpos[:, root] = -5.25
        task._reset_qpos[:, root + 1] = 0.0
        task._reset_qvel.zero_()
        task._reset_qvel[:, root_dof] = initial_speed
        task.reset()
        task._reset_qpos.copy_(saved_qpos)
        task._reset_qvel.copy_(saved_qvel)
        task._progress.fill_(0.75)
        task._jump_triggered.fill_(True)
        task._jump_phase.fill_(3)
        task._jump_phase_elapsed.zero_()
        task._update_jump_supervisor()
        action = torch.zeros((1, 7), dtype=torch.float32, device=bundle.batch.device)
        peak = float(task.batch.qpos[0, root + 2].item())
        max_progress = float(task._progress[0].item())
        for index in range(1200):
            result = task.step(action)
            peak = max(peak, float(task.batch.qpos[0, root + 2].item()))
            max_progress = max(max_progress, float(task._progress[0].item()))
            if index % 100 == 0:
                task._course_terrain.wheel_clearances_and_contacts()
                print(
                    "trace", index,
                    "phase", task._jump_phase.tolist(),
                    "elapsed", task._jump_phase_elapsed.tolist(),
                    "x", task.batch.qpos[:, root].tolist(),
                    "z", task.batch.qpos[:, root + 2].tolist(),
                    "vz", task.batch.qvel[:, root_dof + 2].tolist(),
                    "contact", task._side_support_contacts().tolist(),
                    "target", task._jump_landing_target_met.tolist(),
                    "wpos", task._wheel_positions.tolist(),
                    "widx", task._course_terrain._wheel_support_index.tolist(),
                    "wh", task._course_terrain._wheel_height.tolist(),
                    "confirm", task._jump_contact_confirm.tolist(),
                    "vland", task._jump_landing_vertical_speed.tolist(),
                    "aland", task._jump_landing_angular_speed.tolist(),
                    "kin", task._jump_landing_kinematics_ok.tolist(),
                    "failed", task._jump_failed.tolist(),
                    "reason", task._safety_reason_code.tolist(),
                )
            if bool(result.terminated.any().item()):
                break
        print(
            "speed", initial_speed,
            "peak_rise", peak - float(task._route_reference_height[0].item()),
            "max_progress", max_progress,
            "phase", task._jump_phase.tolist(),
            "liftoff", task._jump_liftoff.tolist(),
            "target", task._jump_landing_target_met.tolist(),
            "landed", task._jump_landing_confirmed.tolist(),
            "fault_step", index if bool(result.terminated.any().item()) else -1,
            "reason", task._safety_reason_code.tolist(),
            "leg_lengths", task.batch.sensordata[:, task._leg_length_indices].tolist(),
            "command_leg", task._command_leg_length.tolist(),
            "contact", task._side_support_contacts().tolist(),
            "attitude", task._orientation_error()[0].norm(dim=1).tolist(),
        )
finally:
    bundle.close()
