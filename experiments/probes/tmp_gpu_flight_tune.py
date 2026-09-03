import dataclasses
import torch
import warp_env
from official_course_warp import build_curriculum_stage
from train_warp_curriculum import load_curriculum_config

original_loader = warp_env.load_warp_batch_config
warp_env.load_warp_batch_config = lambda path: dataclasses.replace(
    original_loader(path), num_worlds=4, physics_substeps_per_action=10
)
config = load_curriculum_config('configs/warp_curriculum_ppo.yaml')
bundle = build_curriculum_stage(config.stage('official_step150_up'), config)
try:
    task = bundle.task
    # Deterministic comparison: restore nominal vehicle parameters before the
    # flight-gain sweep; the formal gate runs this pass before DR.
    task.set_domain_randomization_active(False)
    task.reset()
    root = task.root_qpos_address
    qpos = task._reset_qpos.clone()
    qpos[:, root] = task._route_start_xy[0] + 0.751
    bundle.batch.reset_to_state(qpos, task._reset_qvel, task._calibrated_nominal_controls)
    fractions = torch.tensor((0.10, 0.30, 0.50, 0.80), dtype=torch.float32, device=bundle.batch.device)
    original_update = task._update_jump_supervisor
    def tuned_update():
        original_update()
        flight = task._jump_phase == 4
        task._jump_torque_scale.copy_(torch.where(flight, fractions, task._jump_torque_scale))
        task.set_controller_torque_scale(task._jump_torque_scale)
    task._update_jump_supervisor = tuned_update
    action = torch.zeros((4, 7), dtype=torch.float32, device=bundle.batch.device)
    for _ in range(130):
        task.step(action)
    for i, frac in enumerate(fractions.tolist()):
        print(i, 'flight_frac', frac,
              'phase', int(task._jump_phase[i].item()),
              'reason', int(task._safety_reason_code[i].item()),
              'z', float(task.batch.qpos[i, root + 2].item()),
              'vz', float(task.batch.qvel[i, task.root_dof_address + 2].item()),
              'ang', [round(float(x), 3) for x in task.batch.qvel[i, task.root_dof_address + 3:task.root_dof_address + 6].tolist()],
              'att', float(task._safety_scratch.attitude_error_rad[i].item()),
              'peak', float(task._jump_peak_rise[i].item()),
              'target', bool(task._jump_landing_target_met[i].item()),
              'progress', float(task._progress[i].item()),
              'contacts', task._side_support_contacts()[i].tolist(),
              'cmd', [round(float(x), 3) for x in task._controller._command[i].tolist()])
finally:
    bundle.close()
