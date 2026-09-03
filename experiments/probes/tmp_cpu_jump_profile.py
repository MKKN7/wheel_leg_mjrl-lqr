import itertools

import mujoco
import numpy as np

from train_warp_ppo import load_flat_ppo_training_config
from warp_flat_controller import calibrate_flat_controller
from warp_env import load_warp_batch_config


flat = load_flat_ppo_training_config("configs/warp_flat_ppo.yaml")
bc = load_warp_batch_config(flat.batch_config_path)
cal = calibrate_flat_controller(bc, flat.flat_controller)
model = mujoco.MjModel.from_xml_path("official_standard_warp_ground.xml")
data = mujoco.MjData(model)
hips = np.asarray(cal.hip_actuator_ids, dtype=np.int64)
gas = np.asarray(cal.gas_spring_dofs, dtype=np.int64)
nom = np.asarray(cal.nominal_control, dtype=np.float64)
ref = np.asarray(cal.reference_qpos, dtype=np.float64)


def attitude(q):
    # Roll/pitch only relative to the initial yaw-aligned body frame.
    q = q / max(np.linalg.norm(q), 1e-9)
    refq = ref[3:7] / np.linalg.norm(ref[3:7])
    # quaternion q * conjugate(refq)
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = refq[0], -refq[1], -refq[2], -refq[3]
    rel = np.array((w1*w2-x1*x2-y1*y2-z1*z2,
                    w1*x2+x1*w2+y1*z2-z1*y2,
                    w1*y2-x1*z2+y1*w2+z1*x2,
                    w1*z2+x1*y2-y1*x2+z1*w2))
    if rel[0] < 0: rel *= -1
    n = np.linalg.norm(rel[1:])
    return np.linalg.norm(rel[1:] * (2*np.arctan2(n,max(rel[0],0))/max(n,1e-8)))


def run(u, steps=220):
    data.qpos[:] = cal.qpos
    data.qvel[:] = cal.qvel
    data.ctrl[:] = nom
    data.qfrc_applied[:] = 0
    data.qfrc_applied[gas] = -6
    z0 = float(data.qpos[2])
    maxz = z0
    maxatt = 0.0
    maxang = 0.0
    for i in range(steps):
        data.ctrl[:] = nom
        data.ctrl[hips] = u
        data.ctrl[2] = data.ctrl[5] = 0
        mujoco.mj_step(model,data)
        maxz = max(maxz,float(data.qpos[2]))
        maxatt = max(maxatt,attitude(data.qpos[3:7]))
        maxang = max(maxang,float(np.linalg.norm(data.qvel[3:6])))
    return maxz-z0,maxatt,maxang,float(data.qpos[0]),float(data.qvel[0]),float(data.qvel[2])


for amp in (16,18,20,22,24,26,28,30):
    best=[]
    for signs in itertools.product((-1.,1.),repeat=4):
        out=run(np.asarray(signs)*amp)
        best.append((out[0],out[1],out[2],signs,out[3:]))
    best.sort(key=lambda x:(x[1]>0.8,-x[0]))
    print('AMP',amp)
    for row in best[:6]: print(row)
