import itertools
import mujoco
import numpy as np
from train_warp_ppo import load_flat_ppo_training_config
from warp_env import load_warp_batch_config
from warp_flat_controller import calibrate_flat_controller

flat = load_flat_ppo_training_config('configs/warp_flat_ppo.yaml')
bc = load_warp_batch_config(flat.batch_config_path)
cal = calibrate_flat_controller(bc, flat.flat_controller)
m = mujoco.MjModel.from_xml_path(str(bc.xml_path))
d = mujoco.MjData(m)
hips = np.asarray(cal.hip_actuator_ids)
gas = np.asarray(cal.gas_spring_dofs)
nom = np.asarray(cal.nominal_control)
wheel_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('left_wheel_contact', 'right_wheel_contact')]
for signs in itertools.product((-1.0, 1.0), repeat=4):
    d.qpos[:] = cal.qpos
    d.qvel[:] = cal.qvel
    d.ctrl[:] = nom
    d.ctrl[hips] = np.asarray(signs) * 26.0
    d.qfrc_applied[:] = 0.0
    d.qfrc_applied[gas] = -6.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d)
    wheel_nf = []
    for k in range(d.ncon):
        contact = d.contact[k]
        if contact.geom1 in wheel_geoms or contact.geom2 in wheel_geoms:
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, k, force)
            wheel_nf.append(float(force[0]))
    print(signs, 'root_acc', np.round(d.qacc[3:6], 2).tolist(), 'root_vel', np.round(d.qvel[3:6], 3).tolist(), 'wheel_nf', np.round(wheel_nf, 2).tolist(), 'zacc', round(float(d.qacc[2]), 2))
