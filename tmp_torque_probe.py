import itertools

import mujoco
import numpy as np

from train_warp_ppo import load_flat_ppo_training_config
from warp_env import load_warp_batch_config
from warp_flat_controller import calibrate_flat_controller


flat = load_flat_ppo_training_config("configs/warp_flat_ppo.yaml")
batch_config = load_warp_batch_config(flat.batch_config_path)
calibration = calibrate_flat_controller(batch_config, flat.flat_controller)
model = mujoco.MjModel.from_xml_path(str(batch_config.xml_path))
data = mujoco.MjData(model)
data.qpos[:] = calibration.qpos
data.qvel[:] = calibration.qvel
mujoco.mj_forward(model, data)
hip_ids = np.asarray(calibration.hip_actuator_ids, dtype=np.int64)
gas_dofs = np.asarray(calibration.gas_spring_dofs, dtype=np.int64)
nominal = np.asarray(calibration.nominal_control, dtype=np.float64)
for signs in itertools.product((-1.0, 1.0), repeat=4):
    data.qpos[:] = calibration.qpos
    data.qvel[:] = calibration.qvel
    data.ctrl[:] = nominal
    data.ctrl[hip_ids] += np.asarray(signs) * 20.0
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[gas_dofs] = -10.775
    mujoco.mj_forward(model, data)
    z0 = float(data.qpos[2])
    vz0 = float(data.qvel[2])
    for _ in range(20):
        mujoco.mj_step(model, data)
    print(signs, "dz", float(data.qpos[2] - z0), "dvz", float(data.qvel[2] - vz0))
