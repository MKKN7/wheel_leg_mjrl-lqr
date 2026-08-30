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
hip_ids = np.asarray(calibration.hip_actuator_ids, dtype=np.int64)
gas_dofs = np.asarray(calibration.gas_spring_dofs, dtype=np.int64)
wheel_ids = [int(model.geom(name).id) for name in ("left_wheel_contact", "right_wheel_contact")]
nominal = np.asarray(calibration.nominal_control, dtype=np.float64)
data.ctrl[:] = nominal
data.ctrl[hip_ids] = np.asarray((-28.0, -28.0, 28.0, -28.0))
data.qfrc_applied[:] = 0.0
data.qfrc_applied[gas_dofs] = -10.775
mujoco.mj_forward(model, data)
for index in range(600):
    mujoco.mj_step(model, data)
    if index % 25 == 0:
        wheel_z = [float(data.geom_xpos[g, 2]) for g in wheel_ids]
        wheel_clearance = [wheel_z[0] - float(model.geom_size[wheel_ids[0], 0]), wheel_z[1] - float(model.geom_size[wheel_ids[1], 0])]
        print(index, "root_z", float(data.qpos[2]), "vz", float(data.qvel[2]), "wheel_z", wheel_z, "wheel_bottom", wheel_clearance)
