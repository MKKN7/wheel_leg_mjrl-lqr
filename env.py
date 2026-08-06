"""Gymnasium environment for safe residual control of the wheeled-leg robot.

The policy never replaces the physical LQR controller.  Its six normalized
actions are bounded torque residuals that are added to ``PhysicalLqr.command``.
The low-centre closed-chain stance is projected once and then restored with
``mj_resetData`` on subsequent episode resets, so model and data objects are
not rebuilt in the simulation loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

import lqr_deploy as lqr


DEFAULT_EPISODE_SECONDS = 8.0
DEFAULT_CONTROL_DECIMATION = 10
MAX_ATTITUDE_ERROR_RAD = 1.0
MAX_CONTACT_LOSS_STEPS = 3
RL_LEG_LENGTH_COMMAND_RATE_MPS = 0.05


class WheelLegResidualEnv(gym.Env):
    """Low-centre LQR locomotion with a bounded six-actuator residual policy.

    The first six actions are residual torques in XML actuator order. The
    seventh action is a rate-limited common leg-length command, which lets the
    policy use the LQR's closed-chain extension controller without directly
    exciting the passive links. The observation includes current physical
    state, speed and leg-length commands, contacts, and the previous action.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        render_mode: str | None = None,
        episode_seconds: float = DEFAULT_EPISODE_SECONDS,
        control_decimation: int = DEFAULT_CONTROL_DECIMATION,
        randomize_command: bool = True,
        randomize_leg_length: bool = True,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if episode_seconds <= 0.0:
            raise ValueError("episode_seconds must be positive")
        if control_decimation < 1:
            raise ValueError("control_decimation must be at least one")

        self.xml_path = Path(xml_path) if xml_path is not None else lqr.XML_PATH
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.refs = lqr.build_refs(self.model)
        self._sensor_refs = dict(self.refs.sensor_refs)
        velocity_sensor = lqr.object_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "world_horizontal_velocity_xy",
        )
        self._sensor_refs["world_horizontal_velocity_xy"] = (
            int(self.model.sensor_adr[velocity_sensor]),
            int(self.model.sensor_dim[velocity_sensor]),
        )

        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None
        self.episode_seconds = float(episode_seconds)
        self.control_decimation = int(control_decimation)
        self.randomize_command = bool(randomize_command)
        self.randomize_leg_length = bool(randomize_leg_length)
        self.max_speed = float(getattr(lqr, "MAX_WALK_SPEED_MPS", 0.25))
        self.acceleration_limit = float(getattr(lqr, "DEFAULT_ACCELERATION_MPS2", 0.35))

        self._control_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._control_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._control_scale = np.maximum(np.abs(self._control_low), np.abs(self._control_high))
        requested_residual_limits = np.array((8.0, 8.0, 0.75, 8.0, 8.0, 0.75))
        self.residual_limits = np.minimum(
            requested_residual_limits,
            0.25 * (self._control_high - self._control_low),
        )
        self.action_space = spaces.Box(
            low=-np.ones(self.model.nu + 1, dtype=np.float32),
            high=np.ones(self.model.nu + 1, dtype=np.float32),
            dtype=np.float32,
        )

        self._hip_qpos_addresses = np.array(
            [
                self.model.jnt_qposadr[
                    lqr.object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in (
                    "left_hip_pitch",
                    "left_active_link_pitch",
                    "right_hip_pitch",
                    "right_active_link_pitch",
                )
            ],
            dtype=np.int32,
        )
        self._hip_dof_addresses = np.array(
            [
                self.model.jnt_dofadr[
                    lqr.object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in (
                    "left_hip_pitch",
                    "left_active_link_pitch",
                    "right_hip_pitch",
                    "right_active_link_pitch",
                )
            ],
            dtype=np.int32,
        )
        self._robot_body = self.refs.robot_body

        # 3 attitude + 3 velocity + 3 angular velocity + 4 hip positions +
        # 4 hip velocities + 2 wheel velocities + 2 leg lengths + 2 length
        # rates + speed/leg commands + 2 contacts + 7 previous actions.
        self._observation_size = 34
        self.observation_space = spaces.Box(
            low=np.full(self._observation_size, -10.0, dtype=np.float32),
            high=np.full(self._observation_size, 10.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._controller: lqr.PhysicalLqr | None = None
        self._stance_qpos: np.ndarray | None = None
        self._stance_qvel: np.ndarray | None = None
        self._stance_ctrl: np.ndarray | None = None
        self._stance_leg_lengths: np.ndarray | None = None
        self._default_leg_length = 0.0
        self._reference_quaternion: np.ndarray | None = None
        self._reference_body_height = 0.0
        self._command_speed = 0.0
        self._command_leg_length = 0.0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self._last_lqr_control = np.zeros(self.model.nu, dtype=np.float64)
        self._last_control = np.zeros(self.model.nu, dtype=np.float64)
        self._contact_loss_steps = 0

    def _project_low_centre_stance(self) -> None:
        """Create one validated LQR working point and cache its physical state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._controller = lqr.settle_and_relinearize(
            self.model,
            self.data,
            self.refs,
            speed=0.0,
            acceleration_limit=self.acceleration_limit,
        )
        self._stance_qpos = self.data.qpos.copy()
        self._stance_qvel = self.data.qvel.copy()
        self._stance_ctrl = self._controller.control_equilibrium.copy()
        self._stance_leg_lengths = np.array(
            (
                self._sensor("left_leg_length")[0],
                self._sensor("right_leg_length")[0],
            ),
            dtype=np.float64,
        )
        self._default_leg_length = float(np.mean(self._stance_leg_lengths))

    def _restore_stance(self) -> None:
        if self._stance_qpos is None or self._stance_qvel is None or self._stance_ctrl is None:
            raise RuntimeError("low-centre stance has not been projected")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._stance_qpos
        self.data.qvel[:] = self._stance_qvel
        self.data.ctrl[:] = self._stance_ctrl
        self.data.qfrc_applied[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _reset_lqr_command(self) -> None:
        if self._controller is None:
            raise RuntimeError("LQR controller is not available")
        self._controller.reset_commands(
            self.data,
            self._command_speed,
            leg_length=self._command_leg_length,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "command_speed" in options:
            command_speed = float(options["command_speed"])
        elif self.randomize_command:
            command_speed = float(self.np_random.uniform(-0.65 * self.max_speed, 0.65 * self.max_speed))
        else:
            command_speed = 0.0
        self._command_speed = float(np.clip(command_speed, -self.max_speed, self.max_speed))
        if "command_leg_length" in options:
            command_leg_length = float(options["command_leg_length"])
        elif self.randomize_leg_length:
            command_leg_length = float(self.np_random.uniform(
                lqr.WALK_LEG_LENGTH_MIN_M + 0.01,
                lqr.WALK_LEG_LENGTH_MAX_M - 0.01,
            ))
        else:
            command_leg_length = self._default_leg_length or 0.5 * (
                lqr.WALK_LEG_LENGTH_MIN_M + lqr.WALK_LEG_LENGTH_MAX_M
            )
        if not lqr.WALK_LEG_LENGTH_MIN_M <= command_leg_length <= lqr.WALK_LEG_LENGTH_MAX_M:
            raise ValueError(
                f"command_leg_length must be within {lqr.WALK_LEG_LENGTH_MIN_M:.3f}.."
                f"{lqr.WALK_LEG_LENGTH_MAX_M:.3f}m"
            )
        self._command_leg_length = command_leg_length

        if self._controller is None:
            self._project_low_centre_stance()
        else:
            self._restore_stance()
        if not self.randomize_leg_length and "command_leg_length" not in options:
            self._command_leg_length = self._default_leg_length
        self._reset_lqr_command()

        self._reference_quaternion = self.data.xquat[self._robot_body].copy()
        self._reference_body_height = float(self.data.xpos[self._robot_body, 2])
        self._previous_action.fill(0.0)
        self._last_lqr_control[:] = self._stance_ctrl
        self._last_control[:] = self._stance_ctrl
        self._contact_loss_steps = 0
        return self._observation(), self._info("reset")

    def _sensor(self, name: str) -> np.ndarray:
        address, dimension = self._sensor_refs[name]
        return self.data.sensordata[address : address + dimension].copy()

    def _orientation_error(self) -> np.ndarray:
        if self._reference_quaternion is None:
            return np.zeros(3, dtype=np.float64)
        error = np.empty(3, dtype=np.float64)
        mujoco.mju_subQuat(
            error,
            self.data.xquat[self._robot_body],
            self._reference_quaternion,
        )
        return error

    def _wheel_contacts(self) -> tuple[int, int]:
        return tuple(
            lqr.wheel_ground_contacts(self.data, self.refs, geom_id)
            for geom_id in self.refs.wheel_geoms
        )

    def _ground_has_nonwheel_contact(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 == self.refs.ground_geom:
                other_geom = contact.geom2
            elif contact.geom2 == self.refs.ground_geom:
                other_geom = contact.geom1
            else:
                continue
            if other_geom not in self.refs.wheel_geoms:
                return True
        return False

    def _safety_reason(self) -> str | None:
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return "non_finite_state"
        if self._ground_has_nonwheel_contact():
            return "nonwheel_ground_contact"
        try:
            lqr.validate_linkage_clearance(self.data, self.refs)
        except RuntimeError:
            return "linkage_self_contact"
        try:
            lqr.validate_closed_loop_error(self.data, self.refs)
        except RuntimeError:
            return "closed_linkage_residual"
        try:
            lqr.validate_leg_length_state(self.data, self.refs)
        except RuntimeError:
            return "leg_length_limit"

        contacts = self._wheel_contacts()
        if all(contacts):
            self._contact_loss_steps = 0
        else:
            self._contact_loss_steps += 1
            if self._contact_loss_steps >= MAX_CONTACT_LOSS_STEPS:
                return "wheel_contact_lost"

        attitude_error = float(np.linalg.norm(self._orientation_error()))
        if attitude_error > MAX_ATTITUDE_ERROR_RAD:
            return "attitude_limit"
        body_height = float(self.data.xpos[self._robot_body, 2])
        if body_height < self._reference_body_height - 0.22:
            return "body_height_limit"
        return None

    def _observation(self) -> np.ndarray:
        if self._stance_leg_lengths is None:
            raise RuntimeError("low-centre stance has not been projected")
        world_velocity = self._sensor("world_horizontal_velocity_xy")
        body_angular_velocity = self._sensor("body_angular_velocity")
        wheel_velocity = np.array(
            (
                self._sensor("left_wheel_angular_velocity")[0],
                self._sensor("right_wheel_angular_velocity")[0],
            )
        )
        leg_lengths = np.array(
            (
                self._sensor("left_leg_length")[0],
                self._sensor("right_leg_length")[0],
            )
        )
        leg_length_velocity = np.array(
            (
                self._sensor("left_leg_length_velocity")[0],
                self._sensor("right_leg_length_velocity")[0],
            )
        )
        contacts = np.asarray(self._wheel_contacts(), dtype=np.float64)
        observation = np.concatenate(
            (
                self._orientation_error() / MAX_ATTITUDE_ERROR_RAD,
                world_velocity / 2.0,
                body_angular_velocity / 10.0,
                self.data.qpos[self._hip_qpos_addresses] / 2.5,
                self.data.qvel[self._hip_dof_addresses] / 15.0,
                wheel_velocity / 50.0,
                (leg_lengths - self._stance_leg_lengths) / 0.15,
                leg_length_velocity / 3.0,
                np.array((self._command_speed / self.max_speed,)),
                np.array(((self._command_leg_length - self._default_leg_length) / 0.10,)),
                contacts,
                self._previous_action,
            )
        )
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(observation, -10.0, 10.0).astype(np.float32)

    def _forward_speed(self) -> float:
        if self._controller is None:
            return 0.0
        return self._controller.forward_speed(self.data)

    def _reward(self, control: np.ndarray, action: np.ndarray, unsafe: bool) -> float:
        speed_error = self._forward_speed() - self._command_speed
        tracking = float(np.exp(-((speed_error / 0.20) ** 2)))
        leg_error = float(np.mean((
            self._sensor("left_leg_length")[0],
            self._sensor("right_leg_length")[0],
        )) - self._command_leg_length)
        leg_tracking = float(np.exp(-((leg_error / 0.05) ** 2)))
        attitude_cost = float(np.dot(self._orientation_error(), self._orientation_error()))
        energy_cost = float(np.mean((control / self._control_scale) ** 2))
        residual_cost = float(np.mean(action * action))
        contact_bonus = 0.10 if all(self._wheel_contacts()) else -0.15
        leg_diff=abs(self._sensor("left_leg_length")[0]-self._sensor("right_leg_length")[0])
        reward = 0.65 * tracking + 0.30 * leg_tracking + 0.30 * np.exp(-attitude_cost) + contact_bonus
        reward -= 0.03 * energy_cost + 0.02 * residual_cost+ 2.0 * leg_diff
        if unsafe:
            reward -= 15.0
        return float(reward)
#速度跟随最高权重，腿长误差和关节位姿次权重 额外加入双腿相对误差
    def _info(self, safety_reason: str | None = None) -> dict[str, Any]:
        return {
            "command_speed_mps": self._command_speed,
            "command_leg_length_m": self._command_leg_length,
            "forward_speed_mps": self._forward_speed(),
            "leg_lengths_m": np.array(
                (self._sensor("left_leg_length")[0], self._sensor("right_leg_length")[0]),
                dtype=np.float32,
            ),
            "wheel_contacts": self._wheel_contacts(),
            "lqr_control": self._last_lqr_control.astype(np.float32).copy(),
            "applied_control": self._last_control.astype(np.float32).copy(),
            "residual_limits": self.residual_limits.astype(np.float32).copy(),
            "safety_reason": safety_reason,
        }

    def expert_action(self, *, residual: bool = True) -> np.ndarray:
        """Return the safe residual action that preserves the LQR baseline."""
        if self._controller is None:
            raise RuntimeError("call reset before requesting an expert action")
        if not residual:
            raise ValueError(
                "WheelLegResidualEnv only accepts residual actions; "
                "absolute LQR torques cannot be used as policy actions"
            )
        return np.zeros(self.action_space.shape, dtype=np.float32)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._controller is None:
            raise RuntimeError("call reset before step")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action_array.shape}")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action contains non-finite values")
        action_array = np.clip(action_array, -1.0, 1.0)
        if self._controller.jump.active:
            leg_length_action = 0.0
        else:
            leg_length_action = float(action_array[-1])
        self._command_leg_length = self._controller.adjust_target_leg_length(
            leg_length_action * RL_LEG_LENGTH_COMMAND_RATE_MPS * self.control_decimation * self.model.opt.timestep
        )

        safety_reason: str | None = None
        for _ in range(self.control_decimation):
            lqr_control = self._controller.command(self.data)
            control = np.clip(
                lqr_control + action_array[: self.model.nu] * self.residual_limits,
                self._control_low,
                self._control_high,
            )
            self.data.ctrl[:] = control
            mujoco.mj_step(self.model, self.data)
            self._last_lqr_control[:] = lqr_control
            self._last_control[:] = control
            safety_reason = self._safety_reason()
            if safety_reason is not None:
                break

        terminated = safety_reason is not None
        truncated = bool(self.data.time >= self.episode_seconds and not terminated)
        self._previous_action[:] = action_array
        observation = self._observation()
        reward = self._reward(self._last_control, action_array, terminated)
        return observation, reward, terminated, truncated, self._info(safety_reason)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the low-centre residual Gymnasium environment.")
    parser.add_argument("--steps", type=int, default=10, help="Number of zero-residual environment steps.")
    parser.add_argument("--smoke", action="store_true", help="Run a short deterministic reset/step test.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    steps = min(args.steps, 3) if args.smoke else args.steps

    environment = WheelLegResidualEnv(randomize_command=False, randomize_leg_length=False)
    try:
        observation, info = environment.reset(
            seed=0,
            options={
                "command_speed": 0.0,
                "command_leg_length": 0.244,
            },
        )
        print(f"reset: observation={observation.shape}, action={environment.action_space.shape}, contacts={info['wheel_contacts']}")
        reward = 0.0
        for _ in range(steps):
            observation, reward, terminated, truncated, info = environment.step(
                np.zeros(environment.action_space.shape, dtype=np.float32)
            )
            if terminated or truncated:
                break
        print(
            f"smoke: t={environment.data.time:.3f}s reward={reward:.3f} "
            f"terminated={terminated} truncated={truncated} reason={info['safety_reason']}"
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
