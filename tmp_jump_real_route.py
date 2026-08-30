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
root = task.root_qpos_address
root_dof = task.root_dof_address
action = torch.zeros((1, 7), dtype=torch.float32, device=bundle.batch.device)
last_phase = int(task._jump_phase[0].item())
for index in range(8000):
    result = task.step(action)
    phase = int(task._jump_phase[0].item())
    if phase != last_phase or index % 500 == 0:
        task._course_terrain.wheel_clearances_and_contacts()
        print(
            "step", index, "phase", phase, "elapsed", task._jump_phase_elapsed.tolist(),
            "progress", task._progress.tolist(), "x", task.batch.qpos[:, root].tolist(),
            "vx", task.batch.qvel[:, root_dof].tolist(), "z", task.batch.qpos[:, root + 2].tolist(),
            "vz", task.batch.qvel[:, root_dof + 2].tolist(), "contact", task._side_support_contacts().tolist(),
            "widx", task._course_terrain._wheel_support_index.tolist(), "target", task._jump_landing_target_met.tolist(),
            "landed", task._jump_landing_confirmed.tolist(), "reason", task._safety_reason_code.tolist(),
        )
        last_phase = phase
    if bool(result.terminated.any().item()):
        print("terminated", index, task._safety_reason_code.tolist())
        break
bundle.close()
