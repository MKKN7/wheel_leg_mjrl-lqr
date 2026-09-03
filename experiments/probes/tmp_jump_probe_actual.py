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
    # Diagnostic only: official jump envelope is validated separately from
    # the flat 0.400 m stance envelope; production YAML will own this value.
    task._safety_limits = dataclasses.replace(task._safety_limits, max_leg_length_m=0.500)
    task._jump_settings = dataclasses.replace(task._jump_settings, flight_torque_fraction=1.00)
    task._controller._wheel_accel_limit = 1.20
    task._controller._wheel_brake_limit = 1.20
    root = task.root_qpos_address
    qpos = task._reset_qpos.clone()
    # Keep the official route spawn so the pre-jump command can build its
    # configured forward velocity before reaching the riser.
    bundle.batch.reset_to_state(qpos, task._reset_qvel, task._calibrated_nominal_controls)
    task._progress.fill_(0.75)
    task._jump_triggered.fill_(True)
    task._jump_phase.fill_(3)
    task._jump_phase_elapsed.zero_()
    task._update_jump_supervisor()
    print(
        "route_start",
        "qpos_xy", task.batch.qpos[:, root:root + 2].tolist(),
        "quat", task.batch.qpos[:, root + 3:root + 7].tolist(),
        "forward", task.forward_direction().tolist(),
        "route_direction", task._route_direction.tolist(),
    )
    original_evaluate_safety = task._evaluate_safety
    def debug_evaluate_safety(controls):
        result = original_evaluate_safety(controls)
        if bool(result.terminated.any().item()) or bool(result.failure.any().item()):
            print(
                "SAFETY_RAW", "terminated", result.terminated.tolist(),
                "failure", result.failure.tolist(), "reason", result.reason_code.tolist(),
                "attitude", result.attitude_limit.tolist(), "height", result.height_limit.tolist(),
                "joint", result.joint_limit.tolist(), "leg", result.leg_limit.tolist(),
                "contact", result.contact_limit.tolist(),
                "leg_lengths", task.batch.sensordata[:, task._leg_length_indices].tolist(),
                "rawdiff", task._terrain_leg_raw_difference_m.tolist(),
                "known_uneven", task._terrain_leg_known_uneven_support.tolist(),
                "contact_exempt", task._contact_loss_exempt.tolist(),
                "batch_estop", task.batch.estopped.tolist(),
                "batch_step_failure", task.batch._step_failures.tolist(),
            )
        return result
    task._evaluate_safety = debug_evaluate_safety
    action = torch.zeros((1, 7), dtype=torch.float32, device=bundle.batch.device)
    for index in range(1000):
        result = task.step(action)
        if index % 25 == 0:
            print(
                "tick", index,
                "progress", task._progress.tolist(),
                "z", task.batch.qpos[:, root + 2].tolist(),
                "vz", task.batch.qvel[:, task.root_dof_address + 2].tolist(),
                "phase", task._jump_phase.tolist(),
                "contact", task._side_support_contacts().tolist(),
                "target", task._jump_landing_target_met.tolist(),
                "reason", task._safety_reason_code.tolist(),
            )
        if bool(result.terminated.any().item()):
            print(
                "FAULT", index,
                "progress", task._progress.tolist(),
                "phase", task._jump_phase.tolist(),
                "trigger", task._jump_triggered.tolist(),
                "liftoff", task._jump_liftoff.tolist(),
                "landed", task._jump_landing_confirmed.tolist(),
                "target", task._jump_landing_target_met.tolist(),
                "failed", task._jump_failed.tolist(),
                "peak", task._jump_peak_rise.tolist(),
                "reason", task._safety_reason_code.tolist(),
            )
            break
finally:
    bundle.close()
