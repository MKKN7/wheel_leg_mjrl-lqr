import mujoco
import numpy as np

from train_warp_ppo import load_flat_ppo_training_config
from warp_env import load_warp_batch_config
from warp_flat_controller import calibrate_flat_controller

c = load_flat_ppo_training_config("configs/warp_flat_ppo.yaml")
b = load_warp_batch_config(c.batch_config_path)
cal = calibrate_flat_controller(b, c.flat_controller)
m = mujoco.MjModel.from_xml_path("official_standard_warp_ground.xml")
d = mujoco.MjData(m)
hips = np.asarray(cal.hip_actuator_ids)
gas = np.asarray(cal.gas_spring_dofs)
nom = np.asarray(cal.nominal_control)
patterns = (
    np.asarray((-20, -20, 20, -20), dtype=float),
    np.asarray((-16, -16, 16, -16), dtype=float),
    np.asarray((-20, -10, 10, -20), dtype=float),
    np.asarray((-20, -10, 10, -10), dtype=float),
    np.asarray((-24, -8, 8, -24), dtype=float),
)
for p in patterns:
    d.qpos[:] = cal.qpos
    d.qvel[:] = cal.qvel
    d.qfrc_applied[:] = 0.0
    d.qfrc_applied[gas] = -6.0
    z0 = float(d.qpos[2])
    max_z = z0
    max_ang = 0.0
    for _ in range(220):
        d.ctrl[:] = nom
        d.ctrl[hips] += p
        d.ctrl[2] = d.ctrl[5] = 0.0
        d.ctrl[:] = np.clip(d.ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
        max_z = max(max_z, float(d.qpos[2]))
        max_ang = max(max_ang, float(np.linalg.norm(d.qvel[3:6])))
    print("delta", p.tolist(), "rise", max_z - z0, "ang", max_ang, "qvel", d.qvel[3:6].tolist())
