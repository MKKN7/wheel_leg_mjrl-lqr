from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
XML_PATH = ROOT / "wheeled_infantry.xml"
BUILD_SCRIPT = ROOT / "build_wheeled_infantry.py"
MAX_PERTURB_FORCE_N = 60.0
MAX_PERTURB_TORQUE_NM = 6.0

# actuator name, joint name, continuous torque, peak torque, passive assist
DRIVE_SPECS = (
    ("left_hip_motor", "left_hip_pitch", 20.0, 40.0, 0.0),
    ("left_active_hip_motor", "left_active_link_pitch", 20.0, 40.0, 0.0),
    ("left_wheel_motor", "left_wheel_spin", 3.0, 3.0, 0.0),
    ("right_hip_motor", "right_hip_pitch", 20.0, 40.0, 0.0),
    ("right_active_hip_motor", "right_active_link_pitch", 20.0, 40.0, 0.0),
    ("right_wheel_motor", "right_wheel_spin", 3.0, 3.0, 0.0),
)

ANGLE_REF_NAMES = (
    ("left", "left_node_link", "left_node_long_site", "left_long_link", "left_long_node_site"),
    ("right", "right_node_link", "right_node_long_site", "right_long_link", "right_long_node_site"),
)

CONNECT_SITE_NAMES = (
    ("left_loop1", "left_node_upper_site", "left_upper_node_site"),
    ("right_loop1", "right_node_upper_site", "right_upper_node_site"),
    ("left_loop2", "left_node_long_site", "left_long_node_site"),
    ("right_loop2", "right_node_long_site", "right_long_node_site"),
)

SENSOR_NAMES = (
    "world_horizontal_position_xy",
    "world_horizontal_velocity_xy",
    "left_wheel_angle",
    "left_wheel_angular_velocity",
    "right_wheel_angle",
    "right_wheel_angular_velocity",
    "left_leg_tilt_angle",
    "left_leg_tilt_angular_velocity",
    "right_leg_tilt_angle",
    "right_leg_tilt_angular_velocity",
    "left_leg_length",
    "left_leg_length_velocity",
    "right_leg_length",
    "right_leg_length_velocity",
    "world_body_orientation_quat",
    "body_angular_velocity",
    "imu_gyroscope",
    "imu_linear_accelerometer",
    "imu_linear_velocity",
    "left_hip_motor_torque",
    "left_active_hip_motor_torque",
    "left_wheel_motor_torque",
    "right_hip_motor_torque",
    "right_active_hip_motor_torque",
    "right_wheel_motor_torque",
)

WHEEL_GEAR_RATIO = 15.7647058824


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the wheeled infantry MuJoCo model in mujoco.viewer."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate meshes_mj and wheeled_infantry.xml before opening the viewer.",
    )
    gravity_group = parser.add_mutually_exclusive_group()
    gravity_group.add_argument(
        "--gravity-on",
        dest="gravity_on",
        action="store_true",
        help="Enable gravity. This is the default for dynamic driving.",
    )
    gravity_group.add_argument(
        "--gravity-off",
        dest="gravity_on",
        action="store_false",
        help="Disable gravity for static inspection.",
    )
    parser.set_defaults(gravity_on=True)
    return parser.parse_args()


def ensure_model(rebuild: bool) -> None:
    if rebuild or not XML_PATH.exists() or not model_assets_ready():
        subprocess.run([sys.executable, str(BUILD_SCRIPT)], check=True, cwd=str(ROOT))


def model_assets_ready() -> bool:
    try:
        root = ET.parse(XML_PATH).getroot()
    except (ET.ParseError, FileNotFoundError):
        return False

    mesh_dir = ROOT / "meshes_mj"
    for mesh in root.findall("./asset/mesh"):
        file_name = mesh.attrib.get("file")
        if file_name and not (mesh_dir / file_name).exists():
            return False
    return True


def configure_camera(viewer: mujoco.viewer.Handle) -> None:
    viewer.cam.distance = 2.4
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = (0.0, 0.0, 0.35)


def enable_perturb_visuals(viewer: mujoco.viewer.Handle) -> None:
    with viewer.lock():
        if hasattr(mujoco.mjtVisFlag, "mjVIS_PERTOBJ"):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1
        if hasattr(mujoco.mjtVisFlag, "mjVIS_PERTFORCE"):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1
        viewer.opt.sitegroup[:] = 1


def clamp_external_wrenches(data: mujoco.MjData) -> None:
    for wrench in data.xfrc_applied:
        for start, limit in ((0, MAX_PERTURB_FORCE_N), (3, MAX_PERTURB_TORQUE_NM)):
            x, y, z = wrench[start : start + 3]
            magnitude = math.sqrt(float(x * x + y * y + z * z))
            if magnitude > limit:
                wrench[start : start + 3] *= limit / magnitude


def build_sensor_refs(model: mujoco.MjModel) -> dict[str, tuple[int, int]]:
    refs: dict[str, tuple[int, int]] = {}
    for sensor_name in SENSOR_NAMES:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id < 0:
            raise ValueError(f"Required sensor is missing: {sensor_name}")
        refs[sensor_name] = (int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id]))
    return refs


def sensor_values(
    data: mujoco.MjData,
    sensor_refs: dict[str, tuple[int, int]],
    sensor_name: str,
) -> tuple[float, ...]:
    address, dimension = sensor_refs[sensor_name]
    return tuple(float(value) for value in data.sensordata[address : address + dimension])


def quaternion_to_euler_zyx(quaternion: tuple[float, ...]) -> tuple[float, float, float]:
    """Return world-frame roll, pitch, yaw in radians from MuJoCo's WXYZ quaternion."""
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_argument)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def rotate_body_vector_to_world(
    quaternion: tuple[float, ...], vector: tuple[float, ...]
) -> tuple[float, float, float]:
    """Rotate a body-frame vector into the world frame using a WXYZ quaternion."""
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def build_telemetry_body_refs(model: mujoco.MjModel) -> dict[str, int]:
    body_names = ("robot", "left_upper_leg", "right_upper_leg", "left_wheel", "right_wheel")
    refs: dict[str, int] = {}
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Telemetry body is missing: {body_name}")
        refs[body_name] = body_id
    return refs


def add_viewer_label(
    viewer: mujoco.viewer.Handle,
    label: str,
    position: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> None:
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LABEL,
        np.zeros(3),
        np.asarray(position),
        np.eye(3).reshape(-1),
        np.asarray(rgba),
    )
    geom.label = label
    scene.ngeom += 1


def update_viewer_telemetry(
    viewer: mujoco.viewer.Handle,
    data: mujoco.MjData,
    sensor_refs: dict[str, tuple[int, int]],
    body_refs: dict[str, int],
) -> None:
    """Populate the MuJoCo viewer with live labels instead of console telemetry."""
    viewer.user_scn.ngeom = 0
    position = sensor_values(data, sensor_refs, "world_horizontal_position_xy")
    velocity = sensor_values(data, sensor_refs, "world_horizontal_velocity_xy")
    quaternion = sensor_values(data, sensor_refs, "world_body_orientation_quat")
    body_angular_velocity = sensor_values(data, sensor_refs, "body_angular_velocity")
    roll, pitch, yaw = quaternion_to_euler_zyx(quaternion)
    world_angular_velocity = rotate_body_vector_to_world(quaternion, body_angular_velocity)
    gyro = sensor_values(data, sensor_refs, "imu_gyroscope")
    acceleration = sensor_values(data, sensor_refs, "imu_linear_accelerometer")
    imu_velocity = sensor_values(data, sensor_refs, "imu_linear_velocity")
    left_wheel_torque = sensor_values(data, sensor_refs, "left_wheel_motor_torque")[0]
    right_wheel_torque = sensor_values(data, sensor_refs, "right_wheel_motor_torque")[0]
    left_wheel_angle = sensor_values(data, sensor_refs, "left_wheel_angle")[0]
    right_wheel_angle = sensor_values(data, sensor_refs, "right_wheel_angle")[0]
    left_wheel_velocity = sensor_values(data, sensor_refs, "left_wheel_angular_velocity")[0]
    right_wheel_velocity = sensor_values(data, sensor_refs, "right_wheel_angular_velocity")[0]
    left_leg_angle = sensor_values(data, sensor_refs, "left_leg_tilt_angle")[0]
    right_leg_angle = sensor_values(data, sensor_refs, "right_leg_tilt_angle")[0]
    left_leg_velocity = sensor_values(data, sensor_refs, "left_leg_tilt_angular_velocity")[0]
    right_leg_velocity = sensor_values(data, sensor_refs, "right_leg_tilt_angular_velocity")[0]
    left_hip_torque = sensor_values(data, sensor_refs, "left_hip_motor_torque")[0]
    left_active_hip_torque = sensor_values(data, sensor_refs, "left_active_hip_motor_torque")[0]
    right_hip_torque = sensor_values(data, sensor_refs, "right_hip_motor_torque")[0]
    right_active_hip_torque = sensor_values(data, sensor_refs, "right_active_hip_motor_torque")[0]

    robot_position = data.xpos[body_refs["robot"]]
    left_leg_position = data.xpos[body_refs["left_upper_leg"]]
    right_leg_position = data.xpos[body_refs["right_upper_leg"]]
    left_wheel_position = data.xpos[body_refs["left_wheel"]]
    right_wheel_position = data.xpos[body_refs["right_wheel"]]
    add_viewer_label(
        viewer,
        f"BASE t={data.time:.2f} p=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) "
        f"v=({velocity[0]:.2f},{velocity[1]:.2f},{velocity[2]:.2f}) "
        f"RPY=({math.degrees(roll):.1f},{math.degrees(pitch):.1f},{math.degrees(yaw):.1f})deg",
        (float(robot_position[0]), float(robot_position[1]), float(robot_position[2] + 0.20)),
        (0.2, 0.9, 1.0, 1.0),
    )
    add_viewer_label(
        viewer,
        f"IMU q={tuple(round(value, 2) for value in quaternion)} gyro={tuple(round(value, 2) for value in gyro)} "
        f"a={tuple(round(value, 2) for value in acceleration)} wW={tuple(round(value, 2) for value in world_angular_velocity)}",
        (float(robot_position[0]), float(robot_position[1]), float(robot_position[2] + 0.14)),
        (0.8, 0.9, 0.3, 1.0),
    )
    add_viewer_label(
        viewer,
        f"L LEG q={left_leg_angle:.2f} dq={left_leg_velocity:.2f} hip={left_hip_torque:.2f}Nm active_hip={left_active_hip_torque:.2f}Nm",
        (float(left_leg_position[0]), float(left_leg_position[1]), float(left_leg_position[2] + 0.08)),
        (0.3, 1.0, 0.4, 1.0),
    )
    add_viewer_label(
        viewer,
        f"R LEG q={right_leg_angle:.2f} dq={right_leg_velocity:.2f} hip={right_hip_torque:.2f}Nm active_hip={right_active_hip_torque:.2f}Nm",
        (float(right_leg_position[0]), float(right_leg_position[1]), float(right_leg_position[2] + 0.08)),
        (1.0, 0.6, 0.2, 1.0),
    )
    add_viewer_label(
        viewer,
        f"L WHEEL q={left_wheel_angle:.2f} dq={left_wheel_velocity:.2f} motor={left_wheel_torque:.2f}Nm output={left_wheel_torque * WHEEL_GEAR_RATIO:.2f}Nm",
        (float(left_wheel_position[0]), float(left_wheel_position[1]), float(left_wheel_position[2] + 0.10)),
        (0.4, 1.0, 0.9, 1.0),
    )
    add_viewer_label(
        viewer,
        f"R WHEEL q={right_wheel_angle:.2f} dq={right_wheel_velocity:.2f} motor={right_wheel_torque:.2f}Nm output={right_wheel_torque * WHEEL_GEAR_RATIO:.2f}Nm",
        (float(right_wheel_position[0]), float(right_wheel_position[1]), float(right_wheel_position[2] + 0.10)),
        (1.0, 0.5, 0.7, 1.0),
    )


def build_drive_refs(model: mujoco.MjModel) -> list[tuple[str, int, int, int, float, float, float]]:
    refs = []
    for actuator_name, joint_name, continuous_nm, peak_nm, spring_nm in DRIVE_SPECS:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        refs.append((actuator_name, actuator_id, int(model.jnt_dofadr[joint_id]), int(model.jnt_qposadr[joint_id]), continuous_nm, peak_nm, spring_nm))
    return refs


def apply_drive_commands(
    data: mujoco.MjData,
    drive_refs: list[tuple[str, int, int, int, float, float, float]],
    commands_nm: list[float],
    peak_enabled: bool,
) -> None:
    for index, (_, actuator_id, _, _, continuous_nm, peak_nm, _) in enumerate(drive_refs):
        limit_nm = peak_nm if peak_enabled else continuous_nm
        data.ctrl[actuator_id] = max(-limit_nm, min(limit_nm, commands_nm[index]))


def print_drive_state(
    drive_refs: list[tuple[str, int, int, int, float, float, float]],
    commands_nm: list[float],
    selected_drive: int,
    peak_enabled: bool,
) -> None:
    name, _, _, _, continuous_nm, peak_nm, _ = drive_refs[selected_drive]
    active_limit = peak_nm if peak_enabled else continuous_nm
    print(
        f"Drive {selected_drive + 1}:{name} command={commands_nm[selected_drive]:.2f}Nm "
        f"limit={active_limit:.2f}Nm"
    )


def build_angle_refs(
    model: mujoco.MjModel,
) -> list[tuple[str, int, int, int, int]]:
    refs: list[tuple[str, int, int, int, int]] = []
    for side, node_body, node_site, long_body, long_site in ANGLE_REF_NAMES:
        refs.append(
            (
                side,
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node_body),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, node_site),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, long_body),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, long_site),
            )
        )
    return refs


def build_connect_refs(model: mujoco.MjModel) -> list[tuple[str, int, int]]:
    refs: list[tuple[str, int, int]] = []
    for name, site1_name, site2_name in CONNECT_SITE_NAMES:
        refs.append(
            (
                name,
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site1_name),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site2_name),
            )
        )
    return refs


def link_angle_deg(
    data: mujoco.MjData,
    node_body_id: int,
    node_site_id: int,
    long_body_id: int,
    long_site_id: int,
) -> float:
    node_origin = data.xpos[node_body_id]
    node_tip = data.site_xpos[node_site_id]
    long_origin = data.xpos[long_body_id]
    long_tip = data.site_xpos[long_site_id]

    node_vec = node_tip - node_origin
    long_vec = long_tip - long_origin
    node_norm = float((node_vec @ node_vec) ** 0.5)
    long_norm = float((long_vec @ long_vec) ** 0.5)
    if node_norm < 1e-9 or long_norm < 1e-9:
        return 0.0

    cos_theta = float((node_vec @ long_vec) / (node_norm * long_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def print_leg_angles(
    data: mujoco.MjData,
    angle_refs: list[tuple[str, int, int, int, int]],
) -> None:
    parts = []
    for side, node_body_id, node_site_id, long_body_id, long_site_id in angle_refs:
        angle_deg = link_angle_deg(data, node_body_id, node_site_id, long_body_id, long_site_id)
        parts.append(f"{side}:{angle_deg:.1f}deg")
    print("Long-node angle", "  ".join(parts))


def print_connect_offsets(
    data: mujoco.MjData,
    connect_refs: list[tuple[str, int, int]],
) -> None:
    parts = []
    for name, site1_id, site2_id in connect_refs:
        delta = data.site_xpos[site1_id] - data.site_xpos[site2_id]
        offset_mm = 1000.0 * float((delta @ delta) ** 0.5)
        parts.append(f"{name}:{offset_mm:.1f}mm")
    print("Connect offset", "  ".join(parts))


def print_leg_state(
    data: mujoco.MjData,
    angle_refs: list[tuple[str, int, int, int, int]],
    connect_refs: list[tuple[str, int, int]],
) -> None:
    print_leg_angles(data, angle_refs)
    print_connect_offsets(data, connect_refs)


def main() -> None:
    args = parse_args()
    ensure_model(args.rebuild)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    gravity_enabled = args.gravity_on
    if not gravity_enabled:
        model.opt.gravity[:] = (0.0, 0.0, 0.0)
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    angle_refs = build_angle_refs(model)
    connect_refs = build_connect_refs(model)
    drive_refs = build_drive_refs(model)
    drive_commands_nm = [0.0] * len(drive_refs)
    selected_drive = 0
    peak_enabled = False
    paused = False

    def key_callback(keycode: int) -> None:
        nonlocal gravity_enabled, paused, peak_enabled, selected_drive
        # MuJoCo forwards GLFW key codes.  Normalize letter keys so Caps Lock
        # and Shift cannot change the control mapping.
        if ord("a") <= keycode <= ord("z"):
            keycode = ord(chr(keycode).upper())
        if keycode == ord(" "):
            paused = not paused
            print(f"{'Paused' if paused else 'Running'}")
        elif keycode == ord("G"):
            gravity_enabled = not gravity_enabled
            model.opt.gravity[:] = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
            print(f"Gravity {'enabled' if gravity_enabled else 'disabled'}")
        elif keycode == ord("P"):
            print_leg_state(data, angle_refs, connect_refs)
        elif keycode == ord("R"):
            mujoco.mj_resetData(model, data)
            drive_commands_nm[:] = [0.0] * len(drive_refs)
            mujoco.mj_forward(model, data)
            print("Reset to the assembled home pose")
        elif keycode == ord("O"):
            print_connect_offsets(data, connect_refs)
        elif ord("1") <= keycode <= ord("6"):
            selected_drive = keycode - ord("1")
            print_drive_state(drive_refs, drive_commands_nm, selected_drive, peak_enabled)
        elif keycode in (ord("W"), ord("S")):
            step_nm = 0.25 if drive_refs[selected_drive][0].endswith("wheel_motor") else 1.0
            direction = 1.0 if keycode == ord("W") else -1.0
            drive_commands_nm[selected_drive] += direction * step_nm
            print_drive_state(drive_refs, drive_commands_nm, selected_drive, peak_enabled)
        elif keycode == ord("Z"):
            drive_commands_nm[selected_drive] = 0.0
            print_drive_state(drive_refs, drive_commands_nm, selected_drive, peak_enabled)
        elif keycode == ord("X"):
            drive_commands_nm[:] = [0.0] * len(drive_refs)
            print("All motor commands set to 0Nm")
        elif keycode == ord("F"):
            peak_enabled = not peak_enabled
            print(f"Joint motor limit: {'40Nm peak' if peak_enabled else '20Nm continuous'}")

    print("Click the 3D model area, then use: 1-6 select, W/S torque, Z selected stop, X all stop, F peak.")
    print(f"Model: {XML_PATH}")
    print(f"Root freejoint: {root_joint_id >= 0}; gravity: {model.opt.gravity.tolist()}")
    print(f"Gravity {'enabled' if gravity_enabled else 'disabled'}")
    print(f"Interactive perturb capped at {MAX_PERTURB_FORCE_N:.0f}N and {MAX_PERTURB_TORQUE_NM:.0f}Nm")
    print_leg_state(data, angle_refs, connect_refs)

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        configure_camera(viewer)
        enable_perturb_visuals(viewer)
        viewer.sync()
        while viewer.is_running():
            step_start = time.perf_counter()
            with viewer.lock():
                clamp_external_wrenches(data)
                apply_drive_commands(data, drive_refs, drive_commands_nm, peak_enabled)
            if not paused:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
