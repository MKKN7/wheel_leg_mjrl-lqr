"""Open the RMUC training scene with a third-person camera that follows the robot."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
from threading import Lock
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from env import DomainRandomizationConfig, WheelLegResidualEnv
import lqr_deploy as lqr


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "rm_train_ground.xml"
MANUAL_VIEWER_EPISODE_SECONDS = 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class ManualCommand:
    name: str
    value: float = 0.0


class ManualCommandQueue:
    """Move viewer-thread key events onto the simulation thread."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._commands: deque[ManualCommand] = deque()

    def submit(self, name: str, value: float = 0.0) -> None:
        with self._lock:
            self._commands.append(ManualCommand(name, value))

    def drain(self) -> list[ManualCommand]:
        with self._lock:
            commands = list(self._commands)
            self._commands.clear()
        return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View the RMUC wheeled-leg RL training scene.")
    parser.add_argument("--speed", type=float, default=0.0, help="Initial forward command in m/s.")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Initial relative heading command in degrees.")
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        help="Optional residual PPO checkpoint to replay instead of zero residual actions.",
    )
    parser.add_argument("--policy-device", default="cpu", help="PyTorch device for --ppo-checkpoint.")
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample PPO actions instead of using deterministic mean actions.",
    )
    args = parser.parse_args()
    if args.ppo_checkpoint is not None:
        args.ppo_checkpoint = args.ppo_checkpoint.resolve()
        if not args.ppo_checkpoint.is_file():
            parser.error(f"--ppo-checkpoint does not exist: {args.ppo_checkpoint}")
    return args


def configure_follow_camera(viewer: mujoco.viewer.Handle, robot_body: int) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = robot_body
    viewer.cam.distance = 3.2
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0


def reset_scene(environment: WheelLegResidualEnv, speed: float, yaw_delta_rad: float) -> np.ndarray:
    observation, info = environment.reset(
        options={
            "command_speed": speed,
            "command_leg_length": 0.244,
            "command_yaw_delta_rad": yaw_delta_rad,
        }
    )
    print(
        f"RMUC reset: contacts={info['wheel_contacts']}, "
        f"command_speed={info['command_speed_mps']:.2f}m/s"
    )
    return observation


def make_key_callback(commands: ManualCommandQueue):
    def key_callback(keycode: int) -> None:
        if keycode in (ord("W"), ord("w"), lqr.GLFW_KEY_UP):
            commands.submit("speed", lqr.SPEED_INCREMENT_MPS)
        elif keycode in (ord("S"), ord("s"), lqr.GLFW_KEY_DOWN):
            commands.submit("speed", -lqr.SPEED_INCREMENT_MPS)
        elif keycode in (ord("A"), ord("a"), lqr.GLFW_KEY_LEFT):
            commands.submit("yaw", lqr.YAW_INCREMENT_RAD)
        elif keycode in (ord("D"), ord("d"), lqr.GLFW_KEY_RIGHT):
            commands.submit("yaw", -lqr.YAW_INCREMENT_RAD)
        elif keycode in (ord("X"), ord("x")):
            commands.submit("stop")
        elif keycode in (ord("C"), ord("c")):
            commands.submit("hold_yaw")
        elif keycode in (ord("R"), ord("r")):
            commands.submit("leg", 0.01)
        elif keycode in (ord("F"), ord("f")):
            commands.submit("leg", -0.01)
        elif keycode in (ord("J"), ord("j")):
            commands.submit("jump")
        elif keycode in (ord("Q"), ord("q")):
            commands.submit("quit")

    return key_callback


def apply_manual_commands(
    environment: WheelLegResidualEnv,
    commands: list[ManualCommand],
) -> bool:
    """Apply queued manual LQR commands.  Return True when the user exits."""
    controller = environment.lqr_controller
    exit_requested = False
    speed_changed = False
    for command in commands:
        if command.name == "speed":
            target = environment.adjust_command_speed(command.value)
            speed_changed = True
            print(f"Target speed: {target:.2f}m/s")
        elif command.name == "yaw":
            target = environment.adjust_command_yaw(command.value)
            print(f"Command yaw: {target:.3f}rad ({np.rad2deg(target):.1f}deg)")
        elif command.name == "stop":
            environment.set_command_speed(0.0)
            speed_changed = True
            print("Target speed: 0.00m/s")
        elif command.name == "hold_yaw":
            target = environment.hold_current_yaw()
            print(f"Command yaw aligned to current heading: {target:.3f}rad")
        elif command.name == "leg":
            target = environment.adjust_command_leg_length(command.value)
            print(f"Target leg length: {target:.3f}m")
        elif command.name == "jump":
            environment.request_lqr_jump()
            print("Jump sequence started: prepare, crouch, thrust, flight, landing")
        elif command.name == "quit":
            exit_requested = True

    if speed_changed:
        controller.print_speed_telemetry(environment.data, force=True)
    return exit_requested


def main() -> None:
    args = parse_args()
    speed = lqr.clamp_speed_command(args.speed)
    if not -180.0 <= args.yaw_deg <= 180.0:
        raise ValueError("--yaw-deg must be within -180..180")
    if not XML_PATH.is_file():
        raise FileNotFoundError(f"RMUC scene is missing: {XML_PATH}")

    environment = WheelLegResidualEnv(
        xml_path=XML_PATH,
        randomize_command=False,
        randomize_leg_length=False,
        jump_probability=0.0,
        episode_seconds=MANUAL_VIEWER_EPISODE_SECONDS,
        domain_randomization=DomainRandomizationConfig.disabled(),
    )
    try:
        observation = reset_scene(environment, speed, math.radians(args.yaw_deg))
        neutral_action = np.zeros(environment.action_space.shape, dtype=np.float32)
        policy_runtime = None
        if args.ppo_checkpoint is not None:
            from policy_runtime import environment_compatibility_warnings, load_ppo_residual_policy

            policy_runtime = load_ppo_residual_policy(
                args.ppo_checkpoint,
                environment.observation_space.shape[0],
                environment.action_space.shape[0],
                device=args.policy_device,
            )
            print(
                f"Loaded PPO residual policy: {policy_runtime.checkpoint_path} "
                f"timesteps={policy_runtime.timesteps}"
            )
            for warning in environment_compatibility_warnings(
                policy_runtime,
                xml_path=XML_PATH,
                lqr_source_path=Path(lqr.__file__),
            ):
                print(f"PPO replay warning: {warning}")
        control_period = environment.control_decimation * environment.model.opt.timestep
        commands = ManualCommandQueue()
        print(
            "Keys: W/S or Up/Down speed, A/D or Left/Right turn command yaw, "
            "X stop, C hold current heading, R/F extend/retract legs, "
            "J starts/restarts the LQR jump, Q exits."
        )
        print("Unsafe state exits this viewer without resetting the robot.")
        with mujoco.viewer.launch_passive(
            environment.model,
            environment.data,
            key_callback=make_key_callback(commands),
        ) as viewer:
            with viewer.lock():
                configure_follow_camera(viewer, environment.refs.robot_body)
            while viewer.is_running():
                start = time.perf_counter()
                if apply_manual_commands(environment, commands.drain()):
                    print("Manual RMUC viewer exited by user command.")
                    break
                action = (
                    neutral_action
                    if policy_runtime is None
                    else policy_runtime.action(
                        observation,
                        deterministic=not args.stochastic_policy,
                    )
                )
                observation, _, terminated, truncated, info = environment.step(action)
                if terminated:
                    print(f"RMUC unsafe: {info['safety_reason']}. Viewer exits without reset.")
                    break
                if truncated:
                    print("RMUC manual viewer time limit reached. Viewer exits without reset.")
                    break
                environment.lqr_controller.print_speed_telemetry(environment.data)
                viewer.sync()
                remaining = control_period - (time.perf_counter() - start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
