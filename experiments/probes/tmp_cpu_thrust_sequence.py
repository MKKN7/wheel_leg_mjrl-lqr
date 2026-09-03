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
ref_hip = np.asarray(cal.reference_hip_qpos)
hip_qpos = np.asarray(cal.hip_qpos_addresses)
hip_dof = np.asarray(cal.hip_dof_addresses)

def run(amp, thrust_steps, brake_mode):
    d.qpos[:] = cal.qpos; d.qvel[:] = cal.qvel; d.qfrc_applied[:] = 0; d.qfrc_applied[gas] = -6
    z0 = float(d.qpos[2]); maxz=z0; maxang=0.; maxatt=0.
    for i in range(700):
        d.ctrl[:] = nom
        if i < thrust_steps:
            d.ctrl[hips] = np.asarray((-amp,-amp,amp,-amp))
        elif brake_mode == "air":
            d.ctrl[hips] = nom[hips] + 20*(ref_hip-d.qpos[hip_qpos]) - 4*d.qvel[hip_dof]
        elif brake_mode == "hold":
            d.ctrl[hips] = np.asarray((-amp,-amp,amp,-amp))
        d.ctrl[2] = d.ctrl[5] = 0
        d.ctrl[:] = np.clip(d.ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        mujoco.mj_step(m,d)
        maxz=max(maxz,float(d.qpos[2])); maxang=max(maxang,float(np.linalg.norm(d.qvel[3:6])))
        if float(d.qpos[2]) > z0: maxatt=max(maxatt,float(np.linalg.norm(d.qvel[3:6])))
    return maxz-z0,maxang,d.qpos[2],d.qvel[2],d.qvel[3:6]

for amp in (18,20,22,24,26):
    for steps in (40,60,80,100,120,160):
        print(amp,steps,'air',run(amp,steps,'air'))
