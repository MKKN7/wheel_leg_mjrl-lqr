import itertools
import mujoco
import numpy as np
from train_warp_ppo import load_flat_ppo_training_config
from warp_env import load_warp_batch_config
from warp_flat_controller import calibrate_flat_controller

flat = load_flat_ppo_training_config('configs/warp_flat_ppo.yaml')
bc = load_warp_batch_config(flat.batch_config_path)
cal = calibrate_flat_controller(bc, flat.flat_controller)
m = mujoco.MjModel.from_xml_path('official_standard_ground.xml')
d = mujoco.MjData(m)
hips = np.asarray(cal.hip_actuator_ids, dtype=np.int64)
gas = np.asarray(cal.gas_spring_dofs, dtype=np.int64)
nom = np.asarray(cal.nominal_control, dtype=np.float64)
for amp in (14., 18., 22., 26., 30.):
    rows = []
    for signs in itertools.product((-1., 1.), repeat=4):
        d.qpos[:] = cal.qpos
        d.qvel[:] = cal.qvel
        d.ctrl[:] = nom
        d.ctrl[hips] = np.asarray(signs) * amp
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[gas] = -6.0
        mujoco.mj_forward(m, d)
        z0 = float(d.qpos[2])
        for _ in range(170):
            mujoco.mj_step(m, d)
        rows.append((float(d.qpos[2] - z0), float(d.qvel[2]), float(np.linalg.norm(d.qvel[3:6])), signs, np.round(d.qvel[3:6], 2).tolist()))
    rows.sort(reverse=True)
    print('AMP', amp)
    for row in rows[:5]:
        print(row)
