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
rng = np.random.default_rng(4)
samples = []
outputs = []
for _ in range(160):
    u = rng.uniform(-30.0, 30.0, size=4)
    d.qpos[:] = cal.qpos
    d.qvel[:] = cal.qvel
    d.ctrl[:] = nom
    d.ctrl[hips] = u
    d.qfrc_applied[:] = 0.0
    d.qfrc_applied[gas] = -6.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d)
    samples.append(np.r_[u, 1.0])
    outputs.append(np.r_[d.qacc[2], d.qacc[3], d.qacc[4], d.qacc[5]])
X = np.asarray(samples)
Y = np.asarray(outputs)
B = np.linalg.lstsq(X, Y, rcond=None)[0]
print('fit intercept', B[-1], 'coef rows z/rx/ry/rz', B[:4])
best = []
for _ in range(200000):
    u = rng.uniform(-32.0, 32.0, size=4)
    pred = np.r_[u, 1.0] @ B
    score = max(0.0, 20.0 - pred[0]) ** 2 + 2.0 * pred[1] ** 2 + 0.2 * (pred[2] ** 2 + pred[3] ** 2)
    best.append((score, u, pred))
best.sort(key=lambda x: x[0])
for score, u, pred in best[:20]:
    print('score', round(score, 2), 'u', np.round(u, 2), 'pred', np.round(pred, 2))
for _, u, _ in best[:10]:
    d.qpos[:] = cal.qpos
    d.qvel[:] = cal.qvel
    d.ctrl[:] = nom
    d.ctrl[hips] = u
    d.qfrc_applied[:] = 0.0
    d.qfrc_applied[gas] = -6.0
    mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2])
    for _ in range(170):
        mujoco.mj_step(m, d)
    print('VAL', np.round(u, 2), 'dz', round(float(d.qpos[2] - z0), 3), 'vz', round(float(d.qvel[2]), 3), 'ang', np.round(d.qvel[3:6], 2), 'q', np.round(d.qpos[3:7], 3))
