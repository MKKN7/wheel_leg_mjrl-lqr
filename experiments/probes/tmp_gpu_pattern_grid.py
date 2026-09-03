import dataclasses
import itertools
import torch
import warp_env
from official_course_warp import build_curriculum_stage
from train_warp_curriculum import load_curriculum_config

original_loader = warp_env.load_warp_batch_config
warp_env.load_warp_batch_config = lambda path: dataclasses.replace(
    original_loader(path), num_worlds=16, physics_substeps_per_action=10
)
config = load_curriculum_config('configs/warp_curriculum_ppo.yaml')
bundle = build_curriculum_stage(config.stage('official_step150_up'), config)
try:
    task = bundle.task
    # Diagnostic only: test a higher flight retract target against the strict
    # 0.180 m lower-leg safety gate; production YAML remains unchanged here.
    task._jump_settings = dataclasses.replace(task._jump_settings, flight_retract_length_m=0.260)
    root = task.root_qpos_address
    qpos = task._reset_qpos.clone()
    qpos[:, root] = task._route_start_xy[0] + 0.751
    bundle.batch.reset_to_state(qpos, task._reset_qvel, task._calibrated_nominal_controls)
    patterns = list(itertools.product((-1.0, 1.0), repeat=4))
    pattern = torch.tensor(patterns, dtype=torch.float32, device=bundle.batch.device) * 26.0
    task._jump_actuator_override_target.index_copy_(1, task._controller._hip_actuator, pattern)
    action = torch.zeros((16, 7), dtype=torch.float32, device=bundle.batch.device)
    for index in range(70):
        result = task.step(action)
        if bool(result.terminated.any().item()):
            # Keep stepping to compare worlds that remain healthy; estopped
            # worlds stay zeroed by the independent safety latch.
            pass
    for i, p in enumerate(patterns):
        print(
            i, p,
            'phase', int(task._jump_phase[i].item()),
            'reason', int(task._safety_reason_code[i].item()),
            'z', float(task.batch.qpos[i, root + 2].item()),
            'vz', float(task.batch.qvel[i, task.root_dof_address + 2].item()),
            'ang', [round(float(x), 3) for x in task.batch.qvel[i, task.root_dof_address + 3:task.root_dof_address + 6].tolist()],
            'att', float(task._safety_scratch.attitude_error_rad[i].item()),
            'peak', float(task._jump_peak_rise[i].item()),
            'progress', float(task._progress[i].item()),
        )
finally:
    bundle.close()
