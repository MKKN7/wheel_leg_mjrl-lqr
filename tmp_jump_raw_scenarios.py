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
    scenarios = (
        ("hip26_gas-6", (-26.0, -26.0, 26.0, -26.0), -6.0),
        ("hip26_gas+6", (-26.0, -26.0, 26.0, -26.0), 6.0),
        ("hip32_gas0", (-32.0, -32.0, 32.0, -32.0), 0.0),
        ("hip28_gas-4", (-28.0, -28.0, 28.0, -28.0), -4.0),
    )
    for name, hips, gas in scenarios:
        qpos = task._reset_qpos.clone()
        qpos[:, root] = -5.25
        qpos[:, root + 1] = 0.0
        bundle.batch.reset_to_state(qpos, task._reset_qvel, task._calibrated_nominal_controls)
        controls = torch.zeros((1, 6), dtype=torch.float32, device=bundle.batch.device)
        controls[:, [0, 1, 3, 4]] = torch.as_tensor(hips, dtype=torch.float32, device=bundle.batch.device)
        forces = torch.zeros((1, int(bundle.batch.host_model.nv)), dtype=torch.float32, device=bundle.batch.device)
        forces[:, [6, 13]] = gas
        z0 = float(bundle.batch.qpos[0, root + 2].item())
        peak = z0
        for index in range(301):
            bundle.batch.step(controls, physics_substeps=1, applied_forces=forces)
            bundle.batch.forward()
            peak = max(peak, float(bundle.batch.qpos[0, root + 2].item()))
        print(
            name,
            "peak_rise", peak - z0,
            "z", float(bundle.batch.qpos[0, root + 2].item()),
            "vz", float(bundle.batch.qvel[0, task.root_dof_address + 2].item()),
            "estop", bool(bundle.batch.estopped[0].item()),
            "overflow", int(bundle.batch.overflow[0].item()),
        )
finally:
    bundle.close()
