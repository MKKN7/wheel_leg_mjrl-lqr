"""Open the RMUC training scene with a third-person camera that follows the robot."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
from threading import Lock
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from env import (
    DEFAULT_EPISODE_SECONDS,
    DomainRandomizationConfig,
    LocomotionCommand,
    WheelLegResidualEnv,
)
import lqr_deploy as lqr
from terrain_curriculum import (
    TerrainCurriculumConfig,
    TerrainCurriculumError,
    load_terrain_curriculum,
)


ROOT = Path(__file__).resolve().parents[2]
XML_PATH = ROOT / "rm_train_ground.xml"
MANUAL_VIEWER_EPISODE_SECONDS = 24.0 * 60.0 * 60.0
DEFAULT_RMUC_MANUAL_SPEED_LIMIT_MPS = 0.08
RMUC_MANUAL_SPEED_INCREMENT_MPS = 0.02
RMUC_MANUAL_YAW_RATE_INCREMENT_RAD_S = 0.05


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
    parser.add_argument(
        "--max-speed",
        type=float,
        default=DEFAULT_RMUC_MANUAL_SPEED_LIMIT_MPS,
        help=(
            "RMUC forward-speed cap in m/s. The default 0.08 m/s is the "
            "validated manual terrain limit; raise only after route validation."
        ),
    )
    parser.add_argument(
        "--yaw-rate",
        type=float,
        default=0.0,
        help=f"Initial high-level yaw-rate command in rad/s (-{lqr.MAX_YAW_RATE_RAD_S:.2f}..{lqr.MAX_YAW_RATE_RAD_S:.2f}).",
    )
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
    parser.add_argument(
        "--terrain-curriculum",
        type=Path,
        help=(
            "Optional RMUC curriculum YAML. Together with --terrain-stage it resets the "
            "robot through the same task spawn, terrain support reference and heading "
            "stabilization path used for PPO training."
        ),
    )
    parser.add_argument(
        "--terrain-stage",
        help="Curriculum stage id for --terrain-curriculum.",
    )
    parser.add_argument(
        "--terrain-task",
        help=(
            "Optional task id enabled by --terrain-stage. Omitting it samples a task from "
            "the selected stage; supplying it selects route 0 unless --route-index is set."
        ),
    )
    parser.add_argument(
        "--route-index",
        type=int,
        help="Zero-based fixed route index for --terrain-task.",
    )
    args = parser.parse_args()
    if not 0.05 <= args.max_speed <= lqr.MAX_FORWARD_SPEED_MPS:
        parser.error(f"--max-speed must be within 0.05..{lqr.MAX_FORWARD_SPEED_MPS:.2f} m/s")
    if not -lqr.MAX_YAW_RATE_RAD_S <= args.yaw_rate <= lqr.MAX_YAW_RATE_RAD_S:
        parser.error(f"--yaw-rate must be within -{lqr.MAX_YAW_RATE_RAD_S:.2f}..{lqr.MAX_YAW_RATE_RAD_S:.2f}")
    if args.ppo_checkpoint is not None:
        args.ppo_checkpoint = args.ppo_checkpoint.resolve()
        if not args.ppo_checkpoint.is_file():
            parser.error(f"--ppo-checkpoint does not exist: {args.ppo_checkpoint}")
    resolve_terrain_replay_args(args, parser)
    return args


def resolve_terrain_replay_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Resolve an optional deterministic RMUC task replay without affecting free control."""
    if args.terrain_curriculum is None:
        if args.terrain_stage is not None:
            parser.error("--terrain-stage requires --terrain-curriculum")
        if args.terrain_task is not None:
            parser.error("--terrain-task requires --terrain-curriculum and --terrain-stage")
        if args.route_index is not None:
            parser.error("--route-index requires --terrain-task")
        args.terrain_curriculum_config = None
        args.terrain_stage_id = None
        args.terrain_route_index = None
        return

    curriculum_path = args.terrain_curriculum.expanduser().resolve()
    try:
        curriculum = load_terrain_curriculum(curriculum_path)
    except (FileNotFoundError, RuntimeError, TerrainCurriculumError) as error:
        parser.error(str(error))
    if args.terrain_stage is None:
        parser.error("--terrain-curriculum requires --terrain-stage")
    try:
        stage = curriculum.stage(args.terrain_stage)
    except KeyError:
        parser.error(
            f"unknown --terrain-stage {args.terrain_stage!r}; available stages: "
            + ", ".join(item.stage_id for item in curriculum.stages)
        )
    if curriculum.limits.max_forward_speed_mps > lqr.MAX_FORWARD_SPEED_MPS + 1e-9:
        parser.error(
            "terrain curriculum max_forward_speed_mps exceeds the physical LQR limit: "
            f"{curriculum.limits.max_forward_speed_mps:.3f} > {lqr.MAX_FORWARD_SPEED_MPS:.3f}"
        )
    if curriculum.limits.max_yaw_rate_rad_s > lqr.MAX_YAW_RATE_RAD_S + 1e-9:
        parser.error(
            "terrain curriculum max_yaw_rate_rad_s exceeds the physical LQR limit: "
            f"{curriculum.limits.max_yaw_rate_rad_s:.3f} > {lqr.MAX_YAW_RATE_RAD_S:.3f}"
        )

    route_index: int | None = None
    if args.terrain_task is None:
        if args.route_index is not None:
            parser.error("--route-index requires --terrain-task")
    else:
        try:
            task = curriculum.task(args.terrain_task)
        except KeyError:
            parser.error(
                f"unknown --terrain-task {args.terrain_task!r}; available tasks: "
                + ", ".join(curriculum.task_ids)
            )
        if task.task_id not in stage.task_ids:
            parser.error(
                f"terrain task {task.task_id!r} is not enabled by stage {stage.stage_id!r}; "
                f"enabled tasks: {', '.join(stage.task_ids)}"
            )
        route_index = 0 if args.route_index is None else args.route_index
        if route_index < 0 or route_index >= len(task.routes):
            parser.error(
                f"--route-index must be within 0..{len(task.routes) - 1} for "
                f"terrain task {task.task_id!r}"
            )

    args.terrain_curriculum = curriculum_path
    args.terrain_curriculum_config = curriculum
    args.terrain_stage_id = stage.stage_id
    args.terrain_route_index = route_index


def terrain_replay_environment_settings(
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    """Return physical, command, manual and episode limits for one viewer mode."""
    curriculum = getattr(args, "terrain_curriculum_config", None)
    if curriculum is None:
        return (
            lqr.MIN_FORWARD_SPEED_LIMIT_MPS,
            args.max_speed,
            args.max_speed,
            MANUAL_VIEWER_EPISODE_SECONDS,
        )
    if not isinstance(curriculum, TerrainCurriculumConfig):
        raise ValueError("terrain curriculum arguments were not resolved")
    stage_id = getattr(args, "terrain_stage_id", None)
    if not isinstance(stage_id, str):
        raise ValueError("terrain stage arguments were not resolved")
    stage = curriculum.stage(stage_id)
    stage_speed_ceiling = max(
        abs(stage.command_for(curriculum.task(task_id)).forward_speed_mps)
        for task_id in stage.task_ids
    )
    physical_speed_limit = lqr.validate_forward_speed_limit(max(
        lqr.MIN_FORWARD_SPEED_LIMIT_MPS,
        args.max_speed,
        float(curriculum.limits.max_forward_speed_mps),
    ))
    command_speed_limit = max(args.max_speed, float(curriculum.limits.max_forward_speed_mps))
    manual_speed_limit = max(args.max_speed, stage_speed_ceiling)
    episode_seconds = max(
        float(DEFAULT_EPISODE_SECONDS),
        float(curriculum.stage_max_episode_seconds(stage_id)),
    )
    return physical_speed_limit, command_speed_limit, manual_speed_limit, episode_seconds


def configure_follow_camera(viewer: mujoco.viewer.Handle, robot_body: int) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = robot_body
    viewer.cam.distance = 3.2
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0


def reset_scene(
    environment: WheelLegResidualEnv,
    speed: float,
    yaw_rate_rad_s: float,
    *,
    terrain_task_id: str | None,
    terrain_route_index: int | None,
) -> np.ndarray:
    """Reset either free-manual control or the trainer's terrain-task path."""
    if environment.terrain_curriculum is None:
        options = {
            "locomotion_command": LocomotionCommand(speed, yaw_rate_rad_s),
            "command_leg_length": 0.244,
        }
    else:
        # A terrain task owns its reset command, spawn yaw and support-height
        # reference.  Do not inject the viewer's manual command during this
        # reset or it would no longer reproduce the PPO training state.
        options: dict[str, str | int] = {}
        if terrain_task_id is not None:
            options["terrain_task_id"] = terrain_task_id
            if terrain_route_index is None:
                raise RuntimeError("resolved terrain task has no route index")
            options["terrain_route_index"] = terrain_route_index
    observation, info = environment.reset(options=options)
    if info["terrain_task_id"] is None:
        print(
            f"RMUC reset: contacts={info['wheel_contacts']}, "
            f"command_speed={info['command_speed_mps']:.2f}m/s "
            f"command_yaw_rate={info['command_yaw_rate_rad_s']:.2f}rad/s"
        )
    else:
        print(
            f"RMUC terrain reset: stage={info['terrain_stage_id']} "
            f"task={info['terrain_task_id']} route={info['terrain_route_id']} "
            f"support={info['terrain_support_height_m']:.3f}m "
            f"heading_stabilization={info['terrain_heading_stabilization']} "
            f"command_speed={info['command_speed_mps']:.2f}m/s "
            f"command_yaw_rate={info['command_yaw_rate_rad_s']:.2f}rad/s"
        )
    return observation


def report_terrain_replay_provenance(
    policy_runtime: object,
    args: argparse.Namespace,
    *,
    strict: bool = False,
) -> None:
    """Check a policy's recorded curriculum before a task-level terrain replay."""
    metadata = getattr(policy_runtime, "metadata", None)
    task_config = metadata.get("task_config") if isinstance(metadata, dict) else None
    saved_terrain = task_config.get("terrain_curriculum") if isinstance(task_config, dict) else None
    curriculum = getattr(args, "terrain_curriculum_config", None)
    if curriculum is None:
        if isinstance(saved_terrain, dict):
            print(
                "PPO replay warning: checkpoint carries RMUC curriculum metadata, but the viewer "
                "is in free-manual mode without a terrain support reference."
            )
        return
    if not isinstance(saved_terrain, dict):
        message = (
            "checkpoint has no RMUC curriculum metadata; it cannot be used for a strict "
            "terrain-task replay"
        )
        if strict:
            raise ValueError("PPO terrain replay compatibility mismatch: " + message)
        print("PPO replay warning: " + message + ".")
        return
    curriculum_path = getattr(args, "terrain_curriculum", None)
    if not isinstance(curriculum_path, Path):
        raise ValueError("terrain curriculum path was not resolved")
    expected = {
        "yaml_sha256": hashlib.sha256(curriculum_path.read_bytes()).hexdigest(),
        "schema_version": int(curriculum.schema_version),
        "stage_id": args.terrain_stage_id,
        "stage_task_ids": list(curriculum.stage(args.terrain_stage_id).task_ids),
    }
    mismatches = [
        f"{key}: checkpoint={saved_terrain.get(key)!r}, viewer={value!r}"
        for key, value in expected.items()
        if saved_terrain.get(key) != value
    ]
    if mismatches:
        message = "terrain curriculum metadata differs: " + "; ".join(mismatches)
        if strict:
            raise ValueError("PPO terrain replay compatibility mismatch: " + message)
        print("PPO replay warning: " + message)


def make_key_callback(commands: ManualCommandQueue):
    def key_callback(keycode: int) -> None:
        if keycode in (ord("W"), ord("w"), lqr.GLFW_KEY_UP):
            commands.submit("speed", RMUC_MANUAL_SPEED_INCREMENT_MPS)
        elif keycode in (ord("S"), ord("s"), lqr.GLFW_KEY_DOWN):
            commands.submit("speed", -RMUC_MANUAL_SPEED_INCREMENT_MPS)
        elif keycode in (ord("A"), ord("a"), lqr.GLFW_KEY_LEFT):
            commands.submit("yaw_rate", RMUC_MANUAL_YAW_RATE_INCREMENT_RAD_S)
        elif keycode in (ord("D"), ord("d"), lqr.GLFW_KEY_RIGHT):
            commands.submit("yaw_rate", -RMUC_MANUAL_YAW_RATE_INCREMENT_RAD_S)
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
    maximum_speed: float,
) -> bool:
    """Apply queued manual LQR commands.  Return True when the user exits."""
    controller = environment.lqr_controller
    exit_requested = False
    speed_changed = False
    for command in commands:
        if command.name == "speed":
            target = environment.set_locomotion_command(LocomotionCommand(
                forward_speed_mps=float(np.clip(
                    environment.command_speed + command.value,
                    -lqr.MAX_REVERSE_SPEED_MPS,
                    maximum_speed,
                )),
                yaw_rate_rad_s=environment.locomotion_command.yaw_rate_rad_s,
            ))
            speed_changed = True
            print(f"Target speed: {target.forward_speed_mps:.2f}m/s")
        elif command.name == "yaw_rate":
            target = environment.set_locomotion_command(LocomotionCommand(
                forward_speed_mps=environment.command_speed,
                yaw_rate_rad_s=float(np.clip(
                    environment.locomotion_command.yaw_rate_rad_s + command.value,
                    -lqr.MAX_YAW_RATE_RAD_S,
                    lqr.MAX_YAW_RATE_RAD_S,
                )),
            ))
            print(f"Target yaw rate: {target.yaw_rate_rad_s:.2f}rad/s")
        elif command.name == "stop":
            environment.set_locomotion_command(LocomotionCommand(
                forward_speed_mps=0.0,
                yaw_rate_rad_s=environment.locomotion_command.yaw_rate_rad_s,
            ))
            speed_changed = True
            print("Target speed: 0.00m/s")
        elif command.name == "hold_yaw":
            target = environment.hold_current_yaw()
            print(f"Command yaw aligned to current heading: {target:.3f}rad")
        elif command.name == "leg":
            target = environment.adjust_command_leg_length(command.value)
            print(f"Target leg length: {target:.3f}m")
        elif command.name == "jump":
            environment.set_locomotion_command(LocomotionCommand(
                forward_speed_mps=environment.command_speed,
                yaw_rate_rad_s=environment.locomotion_command.yaw_rate_rad_s,
                jump_request=True,
            ))
            print("Jump requested: speed is regulated into the low-speed launch window.")
        elif command.name == "quit":
            exit_requested = True

    if speed_changed:
        controller.print_speed_telemetry(environment.data, force=True)
    return exit_requested


def main() -> None:
    args = parse_args()
    speed = float(np.clip(args.speed, -lqr.MAX_REVERSE_SPEED_MPS, args.max_speed))
    if not XML_PATH.is_file():
        raise FileNotFoundError(f"RMUC scene is missing: {XML_PATH}")
    (
        physical_speed_limit,
        command_speed_limit,
        manual_speed_limit,
        episode_seconds,
    ) = terrain_replay_environment_settings(args)

    environment = WheelLegResidualEnv(
        xml_path=XML_PATH,
        randomize_command=False,
        randomize_leg_length=False,
        max_forward_speed=physical_speed_limit,
        command_speed_limit_mps=command_speed_limit,
        max_command_yaw_rate_rad_s=lqr.MAX_YAW_RATE_RAD_S,
        jump_probability=0.0,
        episode_seconds=episode_seconds,
        domain_randomization=DomainRandomizationConfig.disabled(),
        terrain_curriculum=args.terrain_curriculum_config,
        terrain_stage_id=args.terrain_stage_id,
        terrain_evaluation=False,
    )
    try:
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
            report_terrain_replay_provenance(
                policy_runtime,
                args,
                strict=args.terrain_curriculum_config is not None,
            )
        observation = reset_scene(
            environment,
            speed,
            args.yaw_rate,
            terrain_task_id=args.terrain_task,
            terrain_route_index=args.terrain_route_index,
        )
        neutral_action = np.zeros(environment.action_space.shape, dtype=np.float32)
        control_period = environment.control_decimation * environment.model.opt.timestep
        commands = ManualCommandQueue()
        print(
            "Keys: W/S or Up/Down speed, A/D or Left/Right target yaw rate, "
            "X stop, C hold current heading, R/F extend/retract legs, "
            "J queues a controlled rolling LQR jump, Q exits."
        )
        if args.terrain_curriculum_config is None:
            print(
                f"RMUC manual speed cap: {args.max_speed:.2f}m/s. "
                "Physical unsafe states exit without resetting the robot."
            )
        else:
            print(
                f"RMUC terrain replay stage={args.terrain_stage_id}; keyboard speed cap: "
                f"{manual_speed_limit:.2f}m/s. Physical unsafe states exit without reset."
            )
        recovery_was_active = False
        with mujoco.viewer.launch_passive(
            environment.model,
            environment.data,
            key_callback=make_key_callback(commands),
        ) as viewer:
            with viewer.lock():
                configure_follow_camera(viewer, environment.refs.robot_body)
            while viewer.is_running():
                start = time.perf_counter()
                if apply_manual_commands(environment, commands.drain(), manual_speed_limit):
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
                recovery_active = bool(info.get("contact_recovery_active", False))
                if recovery_active and not recovery_was_active:
                    print("Contact recovery: residual control masked; LQR is braking and holding heading.")
                elif recovery_was_active and not recovery_active:
                    print("Contact recovery complete: restoring the requested speed, heading and leg length.")
                recovery_was_active = recovery_active
                if terminated:
                    if bool(info.get("terrain_task_completed", False)) and info.get("safety_reason") is None:
                        print("RMUC terrain task completed. Viewer exits without reset.")
                        break
                    if (
                        not bool(info.get("physical_unsafe", True))
                        and environment.resume_after_nonphysical_jump_task_miss()
                    ):
                        print(
                            "RMUC jump height task miss: stable landing retained; "
                            "manual control continues."
                        )
                        continue
                    print(f"RMUC unsafe: {info['safety_reason']}. Viewer exits without reset.")
                    break
                if truncated:
                    if bool(info.get("terrain_task_failed", False)):
                        print("RMUC terrain task deadline reached. Viewer exits without reset.")
                    else:
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
