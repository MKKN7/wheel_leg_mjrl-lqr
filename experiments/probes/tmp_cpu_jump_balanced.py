import itertools

import mujoco
import numpy as np

from train_warp_ppo import load_flat_ppo_training_config
from warp_flat_controller import calibrate_flat_controller
from warp_env import load_warp_batch_config


flat = load_flat_ppo_training_config("configs/warp_flat_ppo.yaml")
batch_config = load_warp_batch_config(flat.batch_config_path)
cal = calibrate_flat_controller(batch_config, flat.flat_controller)
model = mujoco.MjModel.from_xml_path("official_standard_warp_ground.xml")
data = mujoco.MjData(model)
hip_ids = np.asarray(cal.hip_actuator_ids, dtype=np.int64)
gas_dofs = np.asarray(cal.gas_spring_dofs, dtype=np.int64)
nominal = np.asarray(cal.nominal_control, dtype=np.float64)
reference_qpos = np.asarray(cal.reference_qpos, dtype=np.float64)
reference_hip = np.asarray(cal.reference_hip_qpos, dtype=np.float64)
hip_qpos = np.asarray(cal.hip_qpos_addresses, dtype=np.int64)
hip_dof = np.asarray(cal.hip_dof_addresses, dtype=np.int64)
root_dof = 0


def quat_error(q):
    q = q / max(np.linalg.norm(q), 1.0e-9)
    ref = reference_qpos[3:7] / np.linalg.norm(reference_qpos[3:7])
    rel = np.empty(4)
    rel[0] = q[0] * ref[0] + q[1] * ref[1] + q[2] * ref[2] + q[3] * ref[3]
    rel[1] = q[1] * ref[0] - q[0] * ref[1] - q[3] * ref[2] + q[2] * ref[3]
    rel[2] = q[2] * ref[0] + q[3] * ref[1] - q[0] * ref[2] - q[1] * ref[3]
    rel[3] = q[3] * ref[0] - q[2] * ref[1] + q[1] * ref[2] - q[0] * ref[3]
    if rel[0] < 0.0:
        rel *= -1.0
    vnorm = np.linalg.norm(rel[1:])
    return rel[1:] * (2.0 * np.arctan2(vnorm, max(rel[0], 0.0)) / max(vnorm, 1.0e-8))


def run(hip_command, thrust_steps=350):
    data.qpos[:] = cal.qpos
    data.qvel[:] = cal.qvel
    data.ctrl[:] = nominal
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[gas_dofs] = -6.0
    start_z = float(data.qpos[2])
    for step in range(thrust_steps):
        data.ctrl[:] = nominal
        data.ctrl[hip_ids] = hip_command
        data.ctrl[2] = 0.0
        data.ctrl[5] = 0.0
        mujoco.mj_step(model, data)
    peak_z = float(data.qpos[2])
    peak_att = float(np.linalg.norm(quat_error(data.qpos[3:7])))
    peak_ang = float(np.linalg.norm(data.qvel[3:6]))
    # Let the body coast with an airborne hip PD and no wheel drive.
    for _ in range(500):
        data.ctrl[:] = nominal
        data.ctrl[hip_ids] = (
            nominal[hip_ids]
            + 20.0 * (reference_hip - data.qpos[hip_qpos])
            - 4.0 * data.qvel[hip_dof]
        )
        data.ctrl[2] = 0.0
        data.ctrl[5] = 0.0
        data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        peak_z = max(peak_z, float(data.qpos[2]))
        peak_att = max(peak_att, float(np.linalg.norm(quat_error(data.qpos[3:7]))))
        peak_ang = max(peak_ang, float(np.linalg.norm(data.qvel[3:6])))
        if float(data.qpos[2]) < 0.05:
            break
    return {
        "dz": peak_z - start_z,
        "att": peak_att,
        "ang": peak_ang,
        "z": float(data.qpos[2]),
        "vz": float(data.qvel[2]),
        "x": float(data.qpos[0]),
    }


candidates = (
    (-26.0, -0.31, -0.94, -23.22),
    (-26.0, -0.25, -1.00, -24.65),
    (-26.0, -0.07, -1.06, -26.0),
    (-26.0, 0.22, -1.07, -26.0),
    (-22.0, -0.25, -0.80, -22.0),
    (-24.0, -0.20, -0.80, -24.0),
    (-20.0, -0.20, -0.70, -20.0),
)
for candidate in candidates:
    print("candidate", candidate, run(np.asarray(candidate, dtype=np.float64)))
