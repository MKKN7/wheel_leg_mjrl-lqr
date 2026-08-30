"""Fixed-gain flat walking controller for MuJoCo-Warp CUDA batches.

The CPU :class:`lqr_deploy.PhysicalLqr` is used exactly once by
``calibrate_flat_controller``.  Its equilibrium, linearized gain, closed-chain
stance guard and leg-length Jacobian are copied to immutable calibration
arrays.  ``FixedGainFlatController.compute_controls`` then evaluates the
controller from resident CUDA state only; it never constructs ``MjData`` or
reads a state value back to the host.

This module intentionally covers the validated flat, zero-jump controller
scope.  Terrain and jump supervisors remain separate tasks.  The gas spring is
represented as generalized force and is written to Warp's existing
``qfrc_applied`` view before each physics action.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


ACTION_SIZE = 6
STATE_SIZE = 40
_HINGE_JOINT_TYPE = 3
_FREE_JOINT_TYPE = 0


@dataclass(frozen=True)
class WarpFlatControllerConfig:
    """Controller and one-time calibration parameters.

    Defaults mirror the validated flat branch in ``lqr_deploy.py``.  Training
    manifests can override every runtime gain through ``from_mapping``; no
    values are read from YAML inside the rollout hot path.
    """

    command_speed_mps: float = 0.0
    command_yaw_rate_rad_s: float = 0.0
    command_leg_length_m: float | None = None
    calibration_seed: int = 0
    gas_spring_enabled: bool = True
    gas_spring_torque_nm: float = 10.775
    gas_spring_max_abs_generalized_force_nm: float = 10.775
    stance_guard_kp_nm_per_rad: float = 800.0
    stance_guard_kd_nm_per_rad_per_s: float = 20.0
    leg_force_kp_n_per_m: float = 700.0
    leg_force_kd_ns_per_m: float = 45.0
    leg_force_limit_n: float = 100.0
    max_forward_feedback_mps: float = 0.25
    lqr_reference_speed_limit_mps: float = 0.10
    command_speed_gain_nm_per_mps: float = 0.0
    command_yaw_gain_nm_per_rad_s: float = 0.0
    command_wheel_feedforward_limit_nm: float = 0.0
    command_wheel_accel_limit_nm: float | None = None
    command_wheel_brake_limit_nm: float | None = None
    terrain_support_reference_max_rate_mps: float = 0.0
    max_torque_fraction: float = 0.80
    yaw_alignment_enabled: bool = True

    def __post_init__(self) -> None:
        numeric = (
            "command_speed_mps",
            "command_yaw_rate_rad_s",
            "gas_spring_torque_nm",
            "gas_spring_max_abs_generalized_force_nm",
            "stance_guard_kp_nm_per_rad",
            "stance_guard_kd_nm_per_rad_per_s",
            "leg_force_kp_n_per_m",
            "leg_force_kd_ns_per_m",
            "leg_force_limit_n",
            "max_forward_feedback_mps",
            "lqr_reference_speed_limit_mps",
            "command_speed_gain_nm_per_mps",
            "command_yaw_gain_nm_per_rad_s",
            "command_wheel_feedforward_limit_nm",
            "command_wheel_accel_limit_nm",
            "command_wheel_brake_limit_nm",
            "terrain_support_reference_max_rate_mps",
            "max_torque_fraction",
        )
        for name in numeric:
            raw_value = getattr(self, name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if isinstance(self.calibration_seed, bool) or not isinstance(self.calibration_seed, int):
            raise ValueError("calibration_seed must be an integer")
        if self.command_yaw_rate_rad_s < -0.45 or self.command_yaw_rate_rad_s > 0.45:
            raise ValueError("command_yaw_rate_rad_s must be within [-0.45, 0.45]")
        if (
            self.gas_spring_torque_nm < 0.0
            or self.gas_spring_max_abs_generalized_force_nm < 0.0
            or self.stance_guard_kp_nm_per_rad < 0.0
            or self.stance_guard_kd_nm_per_rad_per_s < 0.0
        ):
            raise ValueError("controller gains and gas spring torque must be non-negative")
        if self.gas_spring_torque_nm > self.gas_spring_max_abs_generalized_force_nm:
            raise ValueError(
                "gas_spring_torque_nm cannot exceed "
                "gas_spring_max_abs_generalized_force_nm"
            )
        if self.leg_force_kp_n_per_m < 0.0 or self.leg_force_kd_ns_per_m < 0.0 or self.leg_force_limit_n <= 0.0:
            raise ValueError("leg force gains must be non-negative and the force limit positive")
        if self.max_forward_feedback_mps <= 0.0 or self.lqr_reference_speed_limit_mps < 0.0:
            raise ValueError("velocity feedback limits are invalid")
        if (
            self.command_speed_gain_nm_per_mps < 0.0
            or self.command_yaw_gain_nm_per_rad_s < 0.0
            or self.command_wheel_feedforward_limit_nm < 0.0
            or (
                self.command_wheel_accel_limit_nm is not None
                and self.command_wheel_accel_limit_nm < 0.0
            )
            or (
                self.command_wheel_brake_limit_nm is not None
                and self.command_wheel_brake_limit_nm < 0.0
            )
            or self.terrain_support_reference_max_rate_mps < 0.0
        ):
            raise ValueError("command feedforward gains and limit must be non-negative")
        if not 0.0 < self.max_torque_fraction <= 0.80:
            raise ValueError("max_torque_fraction must be in (0, 0.80]")
        if not isinstance(self.gas_spring_enabled, bool) or not isinstance(self.yaw_alignment_enabled, bool):
            raise ValueError("gas_spring_enabled and yaw_alignment_enabled must be boolean")
        if self.command_leg_length_m is not None and not math.isfinite(float(self.command_leg_length_m)):
            raise ValueError("command_leg_length_m must be finite when supplied")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WarpFlatControllerConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("flat controller config must be a mapping")
        allowed = {
            "command_speed_mps",
            "command_yaw_rate_rad_s",
            "command_leg_length_m",
            "calibration_seed",
            "gas_spring_enabled",
            "gas_spring_torque_nm",
            "gas_spring_max_abs_generalized_force_nm",
            "stance_guard_kp_nm_per_rad",
            "stance_guard_kd_nm_per_rad_per_s",
            "leg_force_kp_n_per_m",
            "leg_force_kd_ns_per_m",
            "leg_force_limit_n",
            "max_forward_feedback_mps",
            "lqr_reference_speed_limit_mps",
            "command_speed_gain_nm_per_mps",
            "command_yaw_gain_nm_per_rad_s",
            "command_wheel_feedforward_limit_nm",
            "command_wheel_accel_limit_nm",
            "command_wheel_brake_limit_nm",
            "terrain_support_reference_max_rate_mps",
            "max_torque_fraction",
            "yaw_alignment_enabled",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown flat controller config keys: {unknown}")
        return cls(**dict(raw))


def load_warp_flat_controller_config(
    path: str | Path,
    *,
    section: str = "flat_controller",
) -> WarpFlatControllerConfig:
    """Load a strict controller section from YAML."""

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to read controller config {source}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("controller YAML root must be a mapping")
    selected = raw.get(section, raw)
    return WarpFlatControllerConfig.from_mapping(selected)


@dataclass(frozen=True)
class WarpFlatControllerCalibration:
    """Immutable CPU calibration payload uploaded by the GPU controller."""

    qpos: np.ndarray
    qvel: np.ndarray
    nominal_control: np.ndarray
    gain: np.ndarray
    reference_qpos: np.ndarray
    reference_qvel: np.ndarray
    reference_hip_qpos: np.ndarray
    hip_qpos_addresses: np.ndarray
    hip_dof_addresses: np.ndarray
    hip_actuator_ids: np.ndarray
    wheel_qpos_addresses: np.ndarray
    wheel_dof_addresses: np.ndarray
    controlled_dof_indices: np.ndarray
    leg_jacobian: float
    leg_length_m: float
    gas_spring_dofs: np.ndarray
    linearization_heading_yaw: float
    state_digest: str

    def __post_init__(self) -> None:
        vectors = (
            "qpos",
            "qvel",
            "nominal_control",
            "reference_qpos",
            "reference_qvel",
            "reference_hip_qpos",
            "hip_qpos_addresses",
            "hip_dof_addresses",
            "hip_actuator_ids",
            "wheel_qpos_addresses",
            "wheel_dof_addresses",
            "controlled_dof_indices",
            "gas_spring_dofs",
        )
        for name in vectors:
            value = np.asarray(getattr(self, name))
            if value.ndim != 1:
                raise ValueError(f"{name} must be a vector")
            if name not in {"hip_qpos_addresses", "hip_dof_addresses", "hip_actuator_ids", "wheel_qpos_addresses", "wheel_dof_addresses", "controlled_dof_indices", "gas_spring_dofs"}:
                if not np.isfinite(value).all():
                    raise ValueError(f"{name} contains non-finite values")
        gain = np.asarray(self.gain)
        if gain.shape != (ACTION_SIZE, STATE_SIZE) or not np.isfinite(gain).all():
            raise ValueError(f"gain must have shape {(ACTION_SIZE, STATE_SIZE)} and be finite")
        if not math.isfinite(float(self.leg_jacobian)) or not math.isfinite(float(self.leg_length_m)):
            raise ValueError("leg calibration scalars must be finite")
        if len(self.state_digest) != 64:
            raise ValueError("state_digest must be a SHA-256 hex digest")
        for name in vectors:
            value = np.asarray(getattr(self, name))
            dtype = np.int64 if name in {"hip_qpos_addresses", "hip_dof_addresses", "hip_actuator_ids", "wheel_qpos_addresses", "wheel_dof_addresses", "controlled_dof_indices", "gas_spring_dofs"} else np.float32
            object.__setattr__(self, name, np.ascontiguousarray(value, dtype=dtype))
        object.__setattr__(self, "gain", np.ascontiguousarray(np.asarray(self.gain, dtype=np.float32)))
        object.__setattr__(self, "leg_jacobian", float(self.leg_jacobian))
        object.__setattr__(self, "leg_length_m", float(self.leg_length_m))
        object.__setattr__(self, "linearization_heading_yaw", float(self.linearization_heading_yaw))

    def to_task_calibration(self) -> Any:
        """Return the state subset accepted by ``WarpFlatWalkingTask``."""

        from warp_task import WarpFlatStanceCalibration

        return WarpFlatStanceCalibration(
            qpos=self.qpos,
            qvel=self.qvel,
            nominal_control=self.nominal_control,
        )


def _resolve_xml_source(source: Any) -> Path:
    if hasattr(source, "config") and hasattr(source.config, "xml_path"):
        return Path(source.config.xml_path)
    if hasattr(source, "xml_path"):
        return Path(source.xml_path)
    if isinstance(source, (str, Path)):
        return Path(source)
    raise TypeError("calibration source must be a WarpPhysicsBatch/config/path with xml_path")


def calibrate_flat_controller(
    source: Any,
    config: WarpFlatControllerConfig | None = None,
) -> WarpFlatControllerCalibration:
    """Run one CPU low-centre calibration and freeze the fixed-gain payload.

    ``source`` may be a ``WarpPhysicsBatch``, a batch config, or an XML path.
    The CPU environment is closed before this function returns.
    """

    controller_config = WarpFlatControllerConfig() if config is None else config
    xml_path = _resolve_xml_source(source)
    try:
        from env import WheelLegResidualEnv
        import lqr_deploy as cpu_lqr
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("flat controller calibration requires env.WheelLegResidualEnv") from error

    environment = WheelLegResidualEnv(
        xml_path=xml_path,
        randomize_command=False,
        randomize_leg_length=False,
        max_command_yaw_rate_rad_s=0.45,
        jump_probability=0.0,
    )
    try:
        options: dict[str, Any] = {
            "command_speed": controller_config.command_speed_mps,
            "command_yaw_rate_rad_s": controller_config.command_yaw_rate_rad_s,
        }
        if controller_config.command_leg_length_m is not None:
            options["command_leg_length"] = controller_config.command_leg_length_m
        environment.reset(seed=controller_config.calibration_seed, options=options)
        cpu_controller = environment.lqr_controller
        model = environment.model
        # PhysicalLqr owns the gas spring through a module-level calibration
        # constant.  Retune it only during this one CPU setup phase, then
        # restore the module global before returning.  No CPU controller/data
        # object survives into the GPU rollout.
        standard_gas_torque = float(cpu_lqr.GAS_SPRING_TORQUE_NM)
        if (
            cpu_controller.gas_spring_enabled != controller_config.gas_spring_enabled
            or not math.isclose(
                float(controller_config.gas_spring_torque_nm),
                standard_gas_torque,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            previous_torque = cpu_lqr.GAS_SPRING_TORQUE_NM
            cpu_lqr.GAS_SPRING_TORQUE_NM = float(controller_config.gas_spring_torque_nm)
            try:
                cpu_controller.gas_spring_enabled = controller_config.gas_spring_enabled
                cpu_controller.control_equilibrium = cpu_controller.solve_equilibrium(environment.data)
                cpu_controller.gain = cpu_controller.linear_lqr(environment.data)
                cpu_controller._reference_control[:] = cpu_controller.control_equilibrium
                cpu_controller._reference_gain[:] = cpu_controller.gain
            finally:
                cpu_lqr.GAS_SPRING_TORQUE_NM = previous_torque
        qpos = np.asarray(cpu_controller.qpos_equilibrium, dtype=np.float32)
        qvel = np.asarray(cpu_controller.qvel_equilibrium, dtype=np.float32)
        nominal = np.asarray(cpu_controller.control_equilibrium, dtype=np.float32)
        gain = np.asarray(cpu_controller.gain, dtype=np.float32)
        reference_qpos = np.asarray(cpu_controller._reference_qpos, dtype=np.float32)
        reference_qvel = np.asarray(cpu_controller._reference_qvel, dtype=np.float32)
        reference_hip = np.asarray(cpu_controller._reference_hip_qpos, dtype=np.float32)
        hip_qpos = np.asarray(cpu_controller.hip_qpos_addresses, dtype=np.int64)
        hip_dof = np.asarray(cpu_controller.hip_dof_addresses, dtype=np.int64)
        hip_actuator = np.asarray(cpu_controller.hip_actuator_ids, dtype=np.int64)
        wheel_joints = np.asarray(cpu_controller.refs.wheel_joints, dtype=np.int64)
        wheel_qpos = np.asarray(model.jnt_qposadr[wheel_joints], dtype=np.int64)
        wheel_dof = np.asarray(model.jnt_dofadr[wheel_joints], dtype=np.int64)
        controlled_dofs = np.asarray(cpu_controller.controlled_dof_indices, dtype=np.int64)
        gas_dofs = np.asarray(cpu_controller.gas_spring_dofs, dtype=np.int64)
        leg_profile = cpu_controller.leg_profile
        leg_jacobian = (
            float(leg_profile.hip_length_jacobian(cpu_controller._reference_shape))
            if leg_profile is not None
            else 0.0
        )
        leg_length = float(cpu_controller._reference_leg_length)
        heading = float(cpu_controller._linearization_heading_yaw)
        arrays = (qpos, qvel, nominal, gain, reference_qpos, reference_qvel, reference_hip, controlled_dofs)
        if any(not np.isfinite(array).all() for array in arrays):
            raise RuntimeError("CPU flat calibration produced non-finite values")
        if qpos.shape != (model.nq,) or qvel.shape != (model.nv,) or nominal.shape != (model.nu,):
            raise RuntimeError("CPU flat calibration dimensions do not match the MJCF model")
        if gain.shape != (ACTION_SIZE, STATE_SIZE):
            raise RuntimeError(f"CPU LQR gain has unexpected shape {gain.shape}")
        if (
            controlled_dofs.ndim != 1
            or controlled_dofs.size * 2 != STATE_SIZE
            or np.any(controlled_dofs < 0)
            or np.any(controlled_dofs >= model.nv)
            or np.unique(controlled_dofs).size != controlled_dofs.size
        ):
            raise RuntimeError("CPU LQR controlled DOF contract does not preserve the 40-D baseline state")
        digest = hashlib.sha256(
            qpos.tobytes() + qvel.tobytes() + nominal.tobytes() + gain.tobytes() + controlled_dofs.tobytes()
        ).hexdigest()
        return WarpFlatControllerCalibration(
            qpos=qpos,
            qvel=qvel,
            nominal_control=nominal,
            gain=gain,
            reference_qpos=reference_qpos,
            reference_qvel=reference_qvel,
            reference_hip_qpos=reference_hip,
            hip_qpos_addresses=hip_qpos,
            hip_dof_addresses=hip_dof,
            hip_actuator_ids=hip_actuator,
            wheel_qpos_addresses=wheel_qpos,
            wheel_dof_addresses=wheel_dof,
            controlled_dof_indices=controlled_dofs,
            leg_jacobian=leg_jacobian,
            leg_length_m=leg_length,
            gas_spring_dofs=gas_dofs,
            linearization_heading_yaw=heading,
            state_digest=digest,
        )
    finally:
        environment.close()


def _quat_conjugate(torch: Any, quaternion: Any) -> Any:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _quat_multiply(torch: Any, lhs: Any, rhs: Any) -> Any:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_to_matrix(torch: Any, quaternion: Any) -> Any:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


class FixedGainFlatController:
    """Vectorized fixed-gain flat controller consumed by ``WarpFlatWalkingTask``."""

    def __init__(
        self,
        calibration: WarpFlatControllerCalibration,
        batch_or_task: Any,
        config: WarpFlatControllerConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.config = WarpFlatControllerConfig() if config is None else config
        self.task = batch_or_task if hasattr(batch_or_task, "batch") else None
        self.batch = batch_or_task.batch if self.task is not None else batch_or_task
        if not hasattr(self.batch, "_torch") or not hasattr(self.batch, "_warp"):
            raise TypeError("batch_or_task must expose MuJoCo-Warp torch/warp handles")
        torch = self.batch._torch
        self.torch = torch
        self.device = self.batch.device
        self.num_worlds = self.batch.num_worlds
        if self.batch.num_actuators != ACTION_SIZE:
            raise ValueError(f"fixed flat controller expects {ACTION_SIZE} actuators")
        model = self.batch.host_model
        if calibration.qpos.shape != (int(model.nq),) or calibration.qvel.shape != (int(model.nv),):
            raise ValueError("calibration state dimensions do not match the Warp model")
        if calibration.nominal_control.shape != (ACTION_SIZE,):
            raise ValueError("calibration actuator dimension must be six")
        controlled_dofs = np.asarray(calibration.controlled_dof_indices, dtype=np.int64)
        if (
            controlled_dofs.ndim != 1
            or controlled_dofs.size * 2 != STATE_SIZE
            or np.any(controlled_dofs < 0)
            or np.any(controlled_dofs >= int(model.nv))
            or np.unique(controlled_dofs).size != controlled_dofs.size
        ):
            raise ValueError("calibration controlled_dof_indices do not preserve the 40-D LQR state")
        self._qpos_reference = torch.as_tensor(calibration.reference_qpos, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_worlds, 1)
        self._qvel_reference = torch.as_tensor(calibration.reference_qvel, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_worlds, 1)
        self._nominal_control = torch.as_tensor(calibration.nominal_control, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_worlds, 1)
        self._gain = torch.as_tensor(calibration.gain, dtype=torch.float32, device=self.device)
        self._reference_hip_qpos = torch.as_tensor(calibration.reference_hip_qpos, dtype=torch.float32, device=self.device)
        self._hip_qpos = torch.as_tensor(calibration.hip_qpos_addresses, dtype=torch.long, device=self.device)
        self._hip_dof = torch.as_tensor(calibration.hip_dof_addresses, dtype=torch.long, device=self.device)
        self._hip_actuator = torch.as_tensor(calibration.hip_actuator_ids, dtype=torch.long, device=self.device)
        actuator_names = [model.actuator(index).name for index in range(int(model.nu))]
        wheel_ids = [
            index for index, name in enumerate(actuator_names)
            if name in {"left_wheel_motor", "left_wheel", "right_wheel_motor", "right_wheel"}
        ]
        if len(wheel_ids) != 2:
            # The wheeled-infantry XML has a fixed actuator order; retain a
            # strict fallback for compatible derived scenes only.
            wheel_ids = [2, 5] if int(model.nu) == ACTION_SIZE else []
        if len(wheel_ids) != 2:
            raise ValueError("command feedforward requires two identifiable wheel actuators")
        wheel_ids.sort()
        self._wheel_actuator = torch.as_tensor(wheel_ids, dtype=torch.long, device=self.device)
        self._wheel_qpos = torch.as_tensor(calibration.wheel_qpos_addresses, dtype=torch.long, device=self.device)
        self._wheel_dof = torch.as_tensor(calibration.wheel_dof_addresses, dtype=torch.long, device=self.device)
        self._controlled_dofs = torch.as_tensor(controlled_dofs, dtype=torch.long, device=self.device)
        self._gas_dofs = torch.as_tensor(calibration.gas_spring_dofs, dtype=torch.long, device=self.device)
        self._leg_jacobian = torch.as_tensor(float(calibration.leg_jacobian), dtype=torch.float32, device=self.device)
        self._leg_virtual_work_enabled = bool(self._hip_actuator.numel() == 4 and abs(calibration.leg_jacobian) > 0.0)
        self._control_low = self.batch._control_low
        self._control_high = self.batch._control_high
        batch_torque_fraction = float(self.batch.config.safety.torque_fraction_of_rated)
        if not math.isclose(
            float(self.config.max_torque_fraction),
            batch_torque_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError(
                "controller max_torque_fraction must match the batch safety "
                f"torque_fraction_of_rated ({batch_torque_fraction:.3f})"
            )
        if bool((self._nominal_control < self._control_low).any().item() or (self._nominal_control > self._control_high).any().item()):
            raise ValueError("calibrated nominal control exceeds the 80% derated actuator limits")
        gas_force_cap_by_dof: list[float] = []
        gas_actuator_ids: list[int] = []
        hip_dofs = np.asarray(calibration.hip_dof_addresses, dtype=np.int64)
        hip_actuators = np.asarray(calibration.hip_actuator_ids, dtype=np.int64)
        for gas_dof in np.asarray(calibration.gas_spring_dofs, dtype=np.int64):
            matching_hip = np.flatnonzero(hip_dofs == gas_dof)
            if matching_hip.size != 1:
                raise ValueError(
                    "each gas spring generalized-force dof must map to exactly "
                    "one hip actuator"
                )
            actuator_id = int(hip_actuators[int(matching_hip[0])])
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            if int(model.jnt_dofadr[joint_id]) != int(gas_dof):
                raise ValueError(
                    "gas spring generalized-force dof must share its hip actuator joint"
                )
            gear = float(model.actuator_gear[actuator_id, 0])
            if not math.isclose(gear, 1.0, rel_tol=0.0, abs_tol=1.0e-7):
                raise ValueError(
                    "combined actuator/generalized-force torque limiting requires unit hip gear"
                )
            lower = float(self._control_low[actuator_id].item())
            upper = float(self._control_high[actuator_id].item())
            torque_cap = min(abs(lower), abs(upper))
            if not math.isfinite(torque_cap) or torque_cap <= 0.0:
                raise ValueError("gas spring hip actuator has no finite derated torque cap")
            gas_force_cap_by_dof.append(torque_cap)
            gas_actuator_ids.append(actuator_id)
        relevant_gas_torque_cap = min(gas_force_cap_by_dof, default=math.inf)
        if float(self.config.gas_spring_max_abs_generalized_force_nm) > relevant_gas_torque_cap + 1.0e-7:
            raise ValueError(
                "gas_spring_max_abs_generalized_force_nm exceeds the 80% "
                f"derated hip actuator torque cap ({relevant_gas_torque_cap:.6g} Nm)"
            )
        self._state_error = torch.zeros((self.num_worlds, STATE_SIZE), dtype=torch.float32, device=self.device)
        self._position_error = torch.zeros((self.num_worlds, int(model.nv)), dtype=torch.float32, device=self.device)
        self._velocity_error = torch.zeros_like(self._position_error)
        self._command = torch.zeros((self.num_worlds, ACTION_SIZE), dtype=torch.float32, device=self.device)
        self._task_actuator_override = torch.empty_like(self._command)
        # Dedicated airborne recovery workspaces.  They are allocated once so
        # direct-jump flight never creates per-substep CUDA arrays.
        self._flight_command = torch.empty_like(self._command)
        self._flight_hip_position_error = torch.empty(
            (self.num_worlds, int(self._hip_dof.numel())), dtype=torch.float32, device=self.device
        )
        self._flight_hip_velocity_error = torch.empty_like(self._flight_hip_position_error)
        self._flight_hip_correction = torch.empty_like(self._flight_hip_position_error)
        self._flight_hip_target = torch.empty_like(self._flight_hip_position_error)
        self._flight_wheel_target = torch.empty((self.num_worlds, 2), dtype=torch.float32, device=self.device)
        self._flight_phase_mask = torch.empty(self.num_worlds, dtype=torch.bool, device=self.device)
        self._wheel_feedforward = torch.zeros(
            (self.num_worlds, 2), dtype=torch.float32, device=self.device
        )
        # Keep force input separate from ``data.qfrc_applied``.  The physics
        # batch clears/stages that view before every substep, so directly
        # writing it here would silently drop the gas spring when callers use
        # ``batch.step(..., applied_forces=None)``.
        self._applied_forces = torch.zeros(
            (self.num_worlds, int(model.nv)), dtype=torch.float32, device=self.device
        )
        self._gas_force_values = torch.full(
            (self.num_worlds, int(self._gas_dofs.numel())),
            -float(self.config.gas_spring_torque_nm),
            dtype=torch.float32,
            device=self.device,
        )
        self._gas_actuator_ids = torch.as_tensor(
            gas_actuator_ids, dtype=torch.long, device=self.device
        )
        self._gas_actuator_controls = torch.empty_like(self._gas_force_values)
        self._gas_remaining_low = torch.empty_like(self._gas_force_values)
        self._gas_remaining_high = torch.empty_like(self._gas_force_values)
        self._gas_applied_values = torch.empty_like(self._gas_force_values)
        self._gas_control_low = (
            self._control_low.index_select(0, self._gas_actuator_ids)
            if self._gas_actuator_ids.numel() > 0
            else torch.empty(0, dtype=torch.float32, device=self.device)
        )
        self._gas_control_high = (
            self._control_high.index_select(0, self._gas_actuator_ids)
            if self._gas_actuator_ids.numel() > 0
            else torch.empty(0, dtype=torch.float32, device=self.device)
        )
        if self._gas_dofs.numel() > 0:
            gas_force_caps = torch.as_tensor(
                gas_force_cap_by_dof,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).expand_as(self._gas_force_values)
            runtime_cap = min(
                float(self.config.gas_spring_max_abs_generalized_force_nm),
                relevant_gas_torque_cap,
            )
            torch.clamp(self._gas_force_values, min=-runtime_cap, max=runtime_cap, out=self._gas_force_values)
            torch.maximum(self._gas_force_values, -gas_force_caps, out=self._gas_force_values)
            torch.minimum(self._gas_force_values, gas_force_caps, out=self._gas_force_values)

        wheel_cap = float(torch.minimum(
            self._control_high.index_select(0, self._wheel_actuator).abs(),
            self._control_low.index_select(0, self._wheel_actuator).abs(),
        ).min().item())
        if self.config.command_wheel_feedforward_limit_nm > wheel_cap + 1.0e-7:
            raise ValueError(
                "command_wheel_feedforward_limit_nm exceeds the 80% derated wheel torque cap"
            )
        self._wheel_torque_cap = wheel_cap
        self._wheel_feedforward_limit = float(self.config.command_wheel_feedforward_limit_nm)
        self._wheel_accel_limit = float(
            self.config.command_wheel_feedforward_limit_nm
            if self.config.command_wheel_accel_limit_nm is None
            else self.config.command_wheel_accel_limit_nm
        )
        self._wheel_brake_limit = float(
            self.config.command_wheel_feedforward_limit_nm
            if self.config.command_wheel_brake_limit_nm is None
            else self.config.command_wheel_brake_limit_nm
        )
        if self._wheel_accel_limit > wheel_cap + 1.0e-7 or self._wheel_brake_limit > wheel_cap + 1.0e-7:
            raise ValueError("asymmetric wheel feedforward limits exceed the 80% derated wheel torque cap")

        free_joints = np.flatnonzero(np.asarray(model.jnt_type, dtype=np.int32) == _FREE_JOINT_TYPE)
        if free_joints.size != 1:
            raise ValueError("fixed flat controller requires exactly one free root joint")
        root = int(free_joints[0])
        self.root_qpos_address = int(model.jnt_qposadr[root])
        self.root_dof_address = int(model.jnt_dofadr[root])
        reference_quaternion = torch.as_tensor(
            calibration.reference_qpos[self.root_qpos_address + 3 : self.root_qpos_address + 7],
            dtype=torch.float32,
            device=self.device,
        )
        self._reference_quaternion = reference_quaternion.unsqueeze(0).repeat(self.num_worlds, 1)
        self._reference_root_height = torch.full(
            (self.num_worlds,),
            float(calibration.reference_qpos[self.root_qpos_address + 2]),
            dtype=torch.float32,
            device=self.device,
        )
        self._linearization_heading_yaw = torch.full(
            (self.num_worlds,),
            float(calibration.linearization_heading_yaw),
            dtype=torch.float32,
            device=self.device,
        )
        self._terrain_base_root_height = self._reference_root_height.clone()
        self._terrain_reference_target = torch.empty_like(self._reference_root_height)
        self._terrain_reference_delta = torch.empty_like(self._reference_root_height)
        self._time_step = float(model.opt.timestep) * int(self.batch.config.physics_substeps_per_action)

    def _current_quaternion(self, task: Any) -> Any:
        qpos = task.batch.qpos
        quat = qpos[:, self.root_qpos_address + 3 : self.root_qpos_address + 7]
        return quat / self.torch.linalg.vector_norm(quat, dim=1, keepdim=True).clamp_min(1.0e-7)

    def set_reference_state(
        self,
        qpos: Any,
        qvel: Any | None = None,
        world_mask: Any | None = None,
    ) -> None:
        """Rebase the fixed-gain state reference at a terrain-route reset.

        A static platform can be 0.4m above the calibration plane.  Keeping
        the old root-Z/quaternion reference there makes the LQR inject a false
        fall correction.  This reset-boundary API updates only resident CUDA
        reference tensors; it never changes a model or allocates physics data.
        """

        torch = self.torch
        expected_qpos = (self.num_worlds, int(self.batch.host_model.nq))
        if (
            not isinstance(qpos, torch.Tensor)
            or qpos.shape != expected_qpos
            or qpos.device != self.device
            or qpos.dtype != torch.float32
            or not qpos.is_contiguous()
        ):
            raise ValueError("qpos reference must be contiguous float32 CUDA [world, nq]")
        if qvel is not None:
            expected_qvel = (self.num_worlds, int(self.batch.host_model.nv))
            if (
                not isinstance(qvel, torch.Tensor)
                or qvel.shape != expected_qvel
                or qvel.device != self.device
                or qvel.dtype != torch.float32
                or not qvel.is_contiguous()
            ):
                raise ValueError("qvel reference must be contiguous float32 CUDA [world, nv]")
        if world_mask is None:
            world_mask = torch.ones(self.num_worlds, dtype=torch.bool, device=self.device)
        elif (
            not isinstance(world_mask, torch.Tensor)
            or world_mask.shape != (self.num_worlds,)
            or world_mask.device != self.device
            or world_mask.dtype != torch.bool
            or not world_mask.is_contiguous()
        ):
            raise ValueError("world_mask must be contiguous CUDA bool [world]")
        self._qpos_reference[world_mask] = qpos[world_mask]
        if qvel is not None:
            self._qvel_reference[world_mask] = qvel[world_mask]
        quaternion = qpos[:, self.root_qpos_address + 3 : self.root_qpos_address + 7]
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=1, keepdim=True).clamp_min(1.0e-7)
        self._reference_quaternion[world_mask] = quaternion[world_mask]
        self._reference_root_height[world_mask] = qpos[world_mask, self.root_qpos_address + 2]
        self._terrain_base_root_height[world_mask] = self._reference_root_height[world_mask]
        # Keep the immutable CPU-LQR linearization frame.  The route reset may
        # rotate the robot in world space, but changing this frame would make
        # a gain identified at the calibrated +Y heading act as if it had been
        # identified at the route heading, destabilizing translational feedback.

    def update_terrain_support_reference(self, support_height: Any) -> None:
        """Rate-limit the LQR root-height reference to a GPU terrain surface."""

        torch = self.torch
        expected = (self.num_worlds,)
        if (
            not isinstance(support_height, torch.Tensor)
            or support_height.shape != expected
            or support_height.device != self.device
            or support_height.dtype != torch.float32
            or not support_height.is_contiguous()
        ):
            raise ValueError("support_height must be contiguous float32 CUDA [world]")
        if self.config.terrain_support_reference_max_rate_mps <= 0.0:
            return
        # Non-finite terrain values are deliberately allowed to propagate into
        # the resident reference and then trip the independent P0 finite-state
        # estop; do not perform a host synchronization in the rollout path.
        self._terrain_reference_target.copy_(self._terrain_base_root_height)
        self._terrain_reference_target.add_(support_height)
        self._terrain_reference_delta.copy_(self._terrain_reference_target)
        self._terrain_reference_delta.sub_(self._reference_root_height)
        maximum_delta = float(self.config.terrain_support_reference_max_rate_mps) * self._time_step
        self._terrain_reference_delta.clamp_(-maximum_delta, maximum_delta)
        self._reference_root_height.add_(self._terrain_reference_delta)
        self._qpos_reference[:, self.root_qpos_address + 2].copy_(self._reference_root_height)

    def _heading_and_forward(self, quaternion: Any) -> tuple[Any, Any]:
        rotation = _quat_to_matrix(self.torch, quaternion)
        axle = rotation[..., :, 0]
        forward = self.torch.stack((-axle[..., 1], axle[..., 0]), dim=-1)
        forward = forward / self.torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1.0e-7)
        yaw = self.torch.atan2(forward[:, 1], forward[:, 0])
        return yaw, forward

    def _state_error_from_task(self, task: Any) -> Any:
        torch = self.torch
        qpos = task.batch.qpos
        qvel = task.batch.qvel
        self._position_error.zero_()
        # Scalar joints map one-for-one from qpos to qvel addresses.  The
        # free-root position/quaternion block is filled below in tangent form.
        self._position_error[:, self.root_dof_address + 6 :] = qpos[:, self.root_qpos_address + 7 :] - self._qpos_reference[:, self.root_qpos_address + 7 :]
        self._position_error[:, self.root_dof_address : self.root_dof_address + 3] = qpos[:, self.root_qpos_address : self.root_qpos_address + 3] - self._qpos_reference[:, self.root_qpos_address : self.root_qpos_address + 3]
        self._position_error[:, self.root_dof_address : self.root_dof_address + 2] = 0.0
        # Wheel angles are intentionally not regulated by the walking LQR.
        if self._wheel_dof.numel() > 0:
            self._position_error[:, self._wheel_dof] = 0.0

        current_quat = self._current_quaternion(task)
        reference_quat = self._reference_quaternion
        # ``q_current * conjugate(q_reference)`` is a world-frame delta;
        # MuJoCo's free-joint tangent is expressed in the reference body frame.
        relative = _quat_multiply(torch, current_quat, _quat_conjugate(torch, reference_quat))
        sign = torch.where(relative[:, :1] < 0.0, -1.0, 1.0)
        relative = relative * sign
        vector = relative[:, 1:]
        vector_norm = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
        angle = 2.0 * torch.atan2(vector_norm, relative[:, :1].clamp_min(0.0))
        scale = torch.where(vector_norm > 1.0e-7, angle / vector_norm, torch.full_like(vector_norm, 2.0))
        world_rotvec = vector * scale
        reference_rotation = _quat_to_matrix(torch, reference_quat)
        body_rotvec = torch.bmm(reference_rotation.transpose(1, 2), world_rotvec.unsqueeze(-1)).squeeze(-1)
        self._position_error[:, self.root_dof_address + 3 : self.root_dof_address + 6] = body_rotvec

        self._velocity_error.copy_(qvel - self._qvel_reference)
        yaw, forward = self._heading_and_forward(current_quat)
        forward_error = (self._velocity_error[:, self.root_dof_address : self.root_dof_address + 2] * forward).sum(dim=1)
        clipped_forward = torch.clamp(
            forward_error,
            -self.config.max_forward_feedback_mps,
            self.config.max_forward_feedback_mps,
        )
        self._velocity_error[:, self.root_dof_address : self.root_dof_address + 2] += (
            forward * (clipped_forward - forward_error).unsqueeze(1)
        )
        if self.config.yaw_alignment_enabled:
            delta = torch.remainder(yaw - self._linearization_heading_yaw + math.pi, 2.0 * math.pi) - math.pi
            cosine = torch.cos(delta)
            sine = torch.sin(delta)
            for error in (
                self._position_error[:, self.root_dof_address : self.root_dof_address + 2],
                self._velocity_error[:, self.root_dof_address : self.root_dof_address + 2],
            ):
                x = error[:, 0].clone()
                y = error[:, 1].clone()
                error[:, 0] = cosine * x + sine * y
                error[:, 1] = -sine * x + cosine * y

        controlled_count = self._controlled_dofs.numel()
        torch.index_select(
            self._position_error,
            1,
            self._controlled_dofs,
            out=self._state_error[:, :controlled_count],
        )
        torch.index_select(
            self._velocity_error,
            1,
            self._controlled_dofs,
            out=self._state_error[:, controlled_count:],
        )
        return self._state_error

    def _write_gas_spring(self, task: Any, safe_controls: Any | None = None) -> Any:
        forces = self._applied_forces
        forces.zero_()
        if self.config.gas_spring_enabled and self._gas_dofs.numel() > 0:
            gas_values = self._gas_force_values
            if safe_controls is not None:
                if (
                    not isinstance(safe_controls, self.torch.Tensor)
                    or safe_controls.shape != (self.num_worlds, ACTION_SIZE)
                    or safe_controls.device != self.device
                    or safe_controls.dtype != self.torch.float32
                    or not safe_controls.is_contiguous()
                ):
                    raise ValueError("safe_controls must be contiguous float32 CUDA actuator controls")
                self.torch.index_select(
                    safe_controls,
                    1,
                    self._gas_actuator_ids,
                    out=self._gas_actuator_controls,
                )
                self.torch.sub(
                    self._gas_control_low.unsqueeze(0),
                    self._gas_actuator_controls,
                    out=self._gas_remaining_low,
                )
                self.torch.sub(
                    self._gas_control_high.unsqueeze(0),
                    self._gas_actuator_controls,
                    out=self._gas_remaining_high,
                )
                self.torch.clamp(
                    gas_values,
                    min=self._gas_remaining_low,
                    max=self._gas_remaining_high,
                    out=self._gas_applied_values,
                )
                gas_values = self._gas_applied_values
            forces.index_copy_(1, self._gas_dofs, gas_values)
        torque_scale = getattr(task, "_controller_torque_scale", None)
        if torque_scale is not None:
            if (
                not isinstance(torque_scale, self.torch.Tensor)
                or torque_scale.shape != (self.num_worlds,)
                or torque_scale.device != self.device
                or torque_scale.dtype != self.torch.float32
            ):
                raise ValueError("task controller torque scale must be CUDA float32 [world]")
            forces.mul_(torque_scale.unsqueeze(1))
        return forces

    def applied_generalized_forces(self, task: Any | None = None, *, safe_controls: Any | None = None) -> Any:
        """Return the resident gas-spring force view within combined torque headroom."""

        if task is not None and task.batch is not self.batch:
            raise ValueError("controller is bound to a different Warp batch")
        return self._write_gas_spring(self.task if task is None else task, safe_controls)

    def compute_controls(self, task: Any | None = None) -> Any:
        """Compute bounded nominal actuator controls entirely on CUDA."""

        torch = self.torch
        task = self.task if task is None else task
        if task is None:
            raise ValueError("compute_controls requires the bound task")
        if task.batch is not self.batch:
            raise ValueError("controller is bound to a different Warp batch")
        # MuJoCo-Warp's step does not guarantee fresh sensor values after
        # integration.  Forward is a GPU kernel over the existing model/data,
        # not a model construction or CPU readback, and makes leg impedance
        # observe the current qpos/qvel on every physical control update.
        self.batch.forward()
        state_error = self._state_error_from_task(task)
        self._command.copy_(self._nominal_control - state_error.matmul(self._gain.transpose(0, 1)))

        hip_position_error = self._reference_hip_qpos.unsqueeze(0) - task.batch.qpos.index_select(1, self._hip_qpos)
        hip_velocity_error = -task.batch.qvel.index_select(1, self._hip_dof)
        guard = (
            float(self.config.stance_guard_kp_nm_per_rad) * hip_position_error
            + float(self.config.stance_guard_kd_nm_per_rad_per_s) * hip_velocity_error
        )
        self._command.index_add_(1, self._hip_actuator, guard)

        # The flat task exposes the commanded common leg length and measured
        # leg lengths on CUDA.  Use its cached sensor indices when available;
        # this keeps the controller independent of sensor address assumptions.
        leg_lengths = task.batch.sensordata.index_select(1, task._leg_length_indices)
        leg_rates = task.batch.sensordata.index_select(1, task._leg_velocity_indices)
        desired_length = task._command_leg_length
        desired_rate = torch.zeros_like(desired_length)
        force = (
            float(self.config.leg_force_kp_n_per_m) * (desired_length.unsqueeze(1) - leg_lengths)
            + float(self.config.leg_force_kd_ns_per_m) * (desired_rate.unsqueeze(1) - leg_rates)
        )
        force = torch.clamp(force, -float(self.config.leg_force_limit_n), float(self.config.leg_force_limit_n))
        if self._leg_virtual_work_enabled:
            virtual_work = self._leg_jacobian * torch.stack((-force[:, 0], -force[:, 0], force[:, 1], -force[:, 1]), dim=1)
            self._command.index_add_(1, self._hip_actuator, virtual_work)

        # Command-conditioned wheel feedforward is optional and zero in the
        # validated flat stance manifest.  Curriculum tasks provide their
        # per-world command tensors; the bounded term gives the residual actor
        # a moving baseline without bypassing the final derated clip.
        self._wheel_feedforward.zero_()
        if (
            max(self._wheel_accel_limit, self._wheel_brake_limit) > 0.0
            and hasattr(task, "_command_speed")
            and hasattr(task, "_command_yaw_rate")
        ):
            # Use the wheel-axis-derived rolling velocity.  A terrain route
            # may start at any yaw, so world-X is not a valid speed signal.
            forward_error = task._command_speed - task.forward_speed()
            yaw_error = task._command_yaw_rate - task.batch.qvel[:, self.root_dof_address + 5]
            # MuJoCo's wheel actuator convention is opposite the task's
            # rolling-speed sign: positive motor torque produces negative
            # world rolling. Negate the error so overspeed receives positive
            # braking torque instead of reinforcing the drift.
            common = -forward_error * float(self.config.command_speed_gain_nm_per_mps)
            # A calibrated stance can generate a small passive forward drift
            # after a route yaw rebase. Permit only the explicitly configured
            # acceleration torque, while retaining a larger independent brake
            # cap so overspeed is actively arrested before the fall guard trips.
            common = torch.clamp(common, min=-self._wheel_accel_limit, max=0.0) + torch.clamp(
                common, min=0.0, max=self._wheel_brake_limit
            )
            differential = yaw_error * float(self.config.command_yaw_gain_nm_per_rad_s)
            self._wheel_feedforward[:, 0] = common - differential
            self._wheel_feedforward[:, 1] = common + differential
            torch.clamp(
                self._wheel_feedforward,
                -self._wheel_torque_cap,
                self._wheel_torque_cap,
                out=self._wheel_feedforward,
            )
            self._command.index_add_(1, self._wheel_actuator, self._wheel_feedforward)

        # Direct-jump tasks may publish a deterministic, YAML-bounded actuator
        # target for their short thrust phase.  It is applied before the same
        # final 80-percent actuator clip used by all residual commands, and
        # the generalized-force path still reserves gas-spring headroom from
        # that clipped command.
        override_target = getattr(task, "_jump_actuator_override_target", None)
        override_enabled = getattr(task, "_jump_actuator_override_enabled", None)
        if override_target is not None or override_enabled is not None:
            if (
                not isinstance(override_target, torch.Tensor)
                or override_target.shape != self._command.shape
                or override_target.device != self.device
                or override_target.dtype != torch.float32
                or not isinstance(override_enabled, torch.Tensor)
                or override_enabled.shape != (self.num_worlds,)
                or override_enabled.device != self.device
                or override_enabled.dtype != torch.bool
            ):
                raise ValueError("jump actuator override must be CUDA [world, 6] with a CUDA [world] mask")
            torch.where(
                override_enabled.unsqueeze(1),
                override_target,
                self._command,
                out=self._task_actuator_override,
            )
            self._command.copy_(self._task_actuator_override)

        # Direct-jump flight uses the dedicated airborne recovery law from the
        # CPU controller, evaluated from the already-populated CUDA state-error
        # buffer.  The task publishes only immutable, YAML-validated gains;
        # non-jump tasks leave this path disabled.
        jump_phase = getattr(task, "_jump_phase", None)
        if getattr(task, "_jump_airborne_recovery_enabled", False):
            if (
                not isinstance(jump_phase, torch.Tensor)
                or jump_phase.shape != (self.num_worlds,)
                or jump_phase.device != self.device
                or jump_phase.dtype != torch.int64
            ):
                raise ValueError("jump phase must be a CUDA int64 [world] tensor")
            torch.eq(jump_phase, 4, out=self._flight_phase_mask)
            self._flight_command.copy_(self._command)
            torch.index_select(
                self._position_error,
                1,
                self._hip_dof,
                out=self._flight_hip_position_error,
            )
            self._flight_hip_position_error.neg_()
            torch.index_select(
                task.batch.qvel,
                1,
                self._hip_dof,
                out=self._flight_hip_velocity_error,
            )
            self._flight_hip_velocity_error.neg_()
            self._flight_hip_correction.copy_(self._flight_hip_position_error)
            self._flight_hip_correction.mul_(float(getattr(task, "_jump_airborne_hip_kp")))
            self._flight_hip_correction.add_(
                self._flight_hip_velocity_error,
                alpha=float(getattr(task, "_jump_airborne_hip_kd")),
            )
            torch.index_select(
                self._flight_command,
                1,
                self._hip_actuator,
                out=self._flight_hip_target,
            )
            self._flight_hip_target.add_(self._flight_hip_correction)
            self._flight_command.index_copy_(1, self._hip_actuator, self._flight_hip_target)
            self._flight_wheel_target.copy_(
                self._position_error[:, self.root_dof_address + 3].unsqueeze(1)
            )
            self._flight_wheel_target.mul_(float(getattr(task, "_jump_airborne_wheel_kp")))
            self._flight_wheel_target.add_(
                task.batch.qvel[:, self.root_dof_address + 3].unsqueeze(1),
                alpha=float(getattr(task, "_jump_airborne_wheel_kd")),
            )
            self._flight_wheel_target.clamp_(
                -float(getattr(task, "_jump_airborne_wheel_limit")),
                float(getattr(task, "_jump_airborne_wheel_limit")),
            )
            self._flight_command.index_copy_(1, self._wheel_actuator, self._flight_wheel_target)
            torch.where(
                self._flight_phase_mask.unsqueeze(1),
                self._flight_command,
                self._command,
                out=self._task_actuator_override,
            )
            self._command.copy_(self._task_actuator_override)

        torch.nan_to_num(self._command, nan=0.0, posinf=0.0, neginf=0.0, out=self._command)
        torch.clamp(self._command, min=self._control_low, max=self._control_high, out=self._command)
        torque_scale = getattr(task, "_controller_torque_scale", None)
        if torque_scale is not None:
            if (
                not isinstance(torque_scale, torch.Tensor)
                or torque_scale.shape != (self.num_worlds,)
                or torque_scale.device != self.device
                or torque_scale.dtype != torch.float32
            ):
                raise ValueError("task controller torque scale must be CUDA float32 [world]")
            self._command.mul_(torque_scale.unsqueeze(1))
        return self._command


Calibration = WarpFlatControllerCalibration


__all__ = [
    "ACTION_SIZE",
    "STATE_SIZE",
    "Calibration",
    "FixedGainFlatController",
    "WarpFlatControllerCalibration",
    "WarpFlatControllerConfig",
    "calibrate_flat_controller",
    "load_warp_flat_controller_config",
]
