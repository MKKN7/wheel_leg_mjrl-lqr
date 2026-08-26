"""CUDA-native safety and termination primitives for MuJoCo-Warp rollouts.

The functions in this module deliberately have no MuJoCo or NumPy dependency
on their hot path.  They operate on batched Torch tensors and can therefore be
called between ``mujoco_warp.step`` calls without transferring state to the
host.  A caller owns the persistent ``previous_estopped`` and reason tensors;
the returned masks are suitable for a vectorized PPO collector.

This module does not infer wheel contact from ``sensordata``.  MuJoCo contact
manifolds are not exposed as a stable sensor in every Warp model, so the task
adapter must provide an explicit ``wheel_contact`` mask (and optional loss
counters) when that check is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


MAX_TORQUE_FRACTION_OF_RATED = 0.80

# Integer codes are kept stable for logging and replay buffers.  Zero means
# that the world is safe at the current step.
SAFETY_REASON_NONE = 0
SAFETY_REASON_NONFINITE_CONTROL = 1
SAFETY_REASON_NONFINITE_STATE = 2
SAFETY_REASON_OVERFLOW = 3
SAFETY_REASON_ATTITUDE = 4
SAFETY_REASON_HEIGHT = 5
SAFETY_REASON_JOINT_LIMIT = 6
SAFETY_REASON_LEG_LIMIT = 7
SAFETY_REASON_CONTACT_LOSS = 8
SAFETY_REASON_LATCHED = 9


@dataclass(frozen=True)
class WarpSafetyLimits:
    """Scalar safety limits loaded by the YAML task configuration.

    Optional joint, leg, and contact checks are disabled when their values are
    ``None``.  This is intentional: a task must provide model-specific ranges
    rather than relying on guessed geometry constants.
    """

    torque_fraction_of_rated: float = MAX_TORQUE_FRACTION_OF_RATED
    estop_on_nonfinite_control: bool = True
    estop_on_nonfinite_state: bool = True
    estop_on_overflow: bool = True
    fall_guard_enabled: bool = True
    max_attitude_error_rad: float = 1.0
    max_root_height_drop_m: float = 0.22
    max_leg_length_difference_m: float | None = None
    min_leg_length_m: float | None = None
    max_leg_length_m: float | None = None
    max_contact_loss_steps: int | None = None

    def __post_init__(self) -> None:
        fraction = float(self.torque_fraction_of_rated)
        if not math.isfinite(fraction) or not 0.0 < fraction <= MAX_TORQUE_FRACTION_OF_RATED:
            raise ValueError(
                "torque_fraction_of_rated must be within "
                f"(0, {MAX_TORQUE_FRACTION_OF_RATED:.2f}]"
            )
        if (
            not math.isfinite(float(self.max_attitude_error_rad))
            or not math.isfinite(float(self.max_root_height_drop_m))
            or self.max_attitude_error_rad <= 0.0
            or self.max_root_height_drop_m <= 0.0
        ):
            raise ValueError("fall guard limits must be positive")
        for name in (
            "max_leg_length_difference_m",
            "min_leg_length_m",
            "max_leg_length_m",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when enabled")
        if self.min_leg_length_m is not None and self.max_leg_length_m is not None:
            if self.min_leg_length_m > self.max_leg_length_m:
                raise ValueError("min_leg_length_m cannot exceed max_leg_length_m")
        if self.max_contact_loss_steps is not None:
            if (
                isinstance(self.max_contact_loss_steps, bool)
                or not isinstance(self.max_contact_loss_steps, int)
                or self.max_contact_loss_steps < 1
            ):
                raise ValueError("max_contact_loss_steps must be a positive integer when enabled")


@dataclass(frozen=True)
class WarpSafetyResult:
    """Per-world safety outputs kept on the same device as the inputs."""

    safe_controls: torch.Tensor
    terminated: torch.Tensor
    failure: torch.Tensor
    reason_code: torch.Tensor
    control_nonfinite: torch.Tensor
    state_nonfinite: torch.Tensor
    sensor_nonfinite: torch.Tensor
    overflow: torch.Tensor
    attitude_limit: torch.Tensor
    height_limit: torch.Tensor
    joint_limit: torch.Tensor
    leg_limit: torch.Tensor
    contact_limit: torch.Tensor
    attitude_error_rad: torch.Tensor
    root_height_m: torch.Tensor

    @property
    def estopped(self) -> torch.Tensor:
        """Alias used by the physics batch adapter and PPO collector."""

        return self.terminated


@dataclass
class WarpSafetyScratch:
    """Persistent device buffers for :func:`evaluate_safety`.

    A scratch instance has one fixed batch shape and must be constructed during
    environment setup, not in the control loop.  When supplied to
    ``evaluate_safety``, result tensors alias these buffers and are therefore
    only valid until the next evaluation with the same scratch instance.
    """

    worlds: int
    action_dim: int
    device: torch.device | str
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.worlds < 1 or self.action_dim < 1:
            raise ValueError("worlds and action_dim must be positive")
        self.device = torch.device(self.device)
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be a floating point torch dtype")

        tensor = lambda *shape, dtype=self.dtype: torch.empty(*shape, device=self.device, dtype=dtype)
        self.safe_controls = tensor(self.worlds, self.action_dim)
        self.control_nonfinite = tensor(self.worlds, dtype=torch.bool)
        self.state_nonfinite = tensor(self.worlds, dtype=torch.bool)
        self.sensor_nonfinite = tensor(self.worlds, dtype=torch.bool)
        self.overflow = tensor(self.worlds, dtype=torch.bool)
        self.attitude_limit = tensor(self.worlds, dtype=torch.bool)
        self.height_limit = tensor(self.worlds, dtype=torch.bool)
        self.joint_limit = tensor(self.worlds, dtype=torch.bool)
        self.leg_limit = tensor(self.worlds, dtype=torch.bool)
        self.contact_limit = tensor(self.worlds, dtype=torch.bool)
        self.failure = tensor(self.worlds, dtype=torch.bool)
        self.terminated = tensor(self.worlds, dtype=torch.bool)
        self.default_previous_estopped = tensor(self.worlds, dtype=torch.bool)
        self.bool_temp = tensor(self.worlds, dtype=torch.bool)
        self.bool_temp_2 = tensor(self.worlds, dtype=torch.bool)
        self.quaternion_valid = tensor(self.worlds, dtype=torch.bool)
        self.reason_code = tensor(self.worlds, dtype=torch.int64)
        self.attitude_error_rad = tensor(self.worlds)
        self.clamp_low = tensor(self.action_dim)
        self.clamp_high = tensor(self.action_dim)

    def validate(self, *, worlds: int, action_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self.worlds != worlds
            or self.action_dim != action_dim
            or self.device != device
            or self.dtype != dtype
        ):
            raise ValueError(
                "scratch must match the current worlds, action dimension, device, and dtype"
            )


def _require_batch_tensor(value: Any, name: str, *, ndim: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point() and value.dtype not in (torch.bool, torch.int32, torch.int64):
        raise TypeError(f"{name} has unsupported dtype {value.dtype}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    return value


def _as_vector(value: Any, *, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
        if result.device != device or result.dtype != dtype:
            raise ValueError(f"{name} must use device {device} and dtype {dtype}")
    else:
        result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return result


def clip_controls(
    controls: torch.Tensor,
    control_low: torch.Tensor,
    control_high: torch.Tensor,
    *,
    estopped: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    scratch: WarpSafetyScratch | None = None,
    nonfinite_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace malformed controls, clamp to signed limits, and optionally zero estopped worlds.

    ``control_low`` and ``control_high`` must already include the XML actuator
    range intersected with the configured 80% torque derating.  The returned
    ``nonfinite`` mask is per world and is suitable for latching an estop.
    """

    controls = _require_batch_tensor(controls, "controls", ndim=2)
    if not controls.is_floating_point():
        raise TypeError("controls must be floating point")
    low = _as_vector(control_low, device=controls.device, dtype=controls.dtype, name="control_low")
    high = _as_vector(control_high, device=controls.device, dtype=controls.dtype, name="control_high")
    if low.numel() != controls.shape[1] or high.numel() != controls.shape[1]:
        raise ValueError("control limits must match the actuator dimension")
    # Limits are model metadata cached during environment construction.  Do
    # not call ``.item()`` here: a scalar readback in every policy step would
    # serialize the CUDA stream.  Non-finite or reversed limits fail closed
    # through the returned malformed-control mask, while startup validation
    # remains the responsibility of the model/config loader.
    valid_limits = torch.isfinite(low).all() & torch.isfinite(high).all() & (low < high).all()
    if scratch is not None:
        scratch.validate(
            worlds=controls.shape[0],
            action_dim=controls.shape[1],
            device=controls.device,
            dtype=controls.dtype,
        )
        if out is None:
            out = scratch.safe_controls
        if nonfinite_out is None:
            nonfinite_out = scratch.control_nonfinite
    if out is None:
        out = torch.empty_like(controls)
    if out.shape != controls.shape or out.device != controls.device or out.dtype != controls.dtype:
        raise ValueError("out must have the same shape, device, and dtype as controls")
    # Compute the malformed mask before writing into ``out``.  ``out`` may be
    # deliberately aliased to ``controls`` by a collector to avoid a buffer.
    nonfinite = ~torch.isfinite(controls).all(dim=1) | ~valid_limits.expand(controls.shape[0])
    if nonfinite_out is not None:
        if (
            nonfinite_out.shape != (controls.shape[0],)
            or nonfinite_out.device != controls.device
            or nonfinite_out.dtype != torch.bool
        ):
            raise ValueError("nonfinite_out must be a matching bool vector")
        nonfinite_out.copy_(nonfinite)
        nonfinite = nonfinite_out
    torch.nan_to_num(controls, nan=0.0, posinf=0.0, neginf=0.0, out=out)
    # ``nan * 0`` is still NaN.  Replace the whole invalid cached range with
    # a zero-width safe interval *before* clamping, then latch the malformed
    # mask above.  This keeps even the standalone helper fail-closed.
    if scratch is not None:
        scratch.clamp_low.zero_()
        scratch.clamp_high.zero_()
        torch.where(valid_limits, low, scratch.clamp_low, out=scratch.clamp_low)
        torch.where(valid_limits, high, scratch.clamp_high, out=scratch.clamp_high)
        clamp_low, clamp_high = scratch.clamp_low, scratch.clamp_high
    else:
        clamp_low = torch.where(valid_limits, low, torch.zeros_like(low))
        clamp_high = torch.where(valid_limits, high, torch.zeros_like(high))
    torch.clamp(out, min=clamp_low, max=clamp_high, out=out)
    # A corrupted cached limit must never leave NaN/Inf in the control buffer
    # before the owner observes ``nonfinite`` and latches its estop.
    out.mul_(valid_limits.to(dtype=out.dtype))
    if estopped is not None:
        estopped = _require_batch_tensor(estopped, "estopped", ndim=1)
        if estopped.dtype != torch.bool or estopped.shape[0] != controls.shape[0] or estopped.device != controls.device:
            raise ValueError("estopped must be a CUDA/CPU bool vector matching controls")
        out.masked_fill_(estopped.unsqueeze(1), 0.0)
    return out, nonfinite


def quat_attitude_error(
    quaternion: torch.Tensor,
    reference_quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shortest-angle error and a validity mask for batched quaternions."""

    quaternion = _require_batch_tensor(quaternion, "quaternion", ndim=2)
    if quaternion.shape[1] != 4 or not quaternion.is_floating_point():
        raise ValueError("quaternion must have shape [worlds, 4] and be floating point")
    reference = reference_quaternion
    if not isinstance(reference, torch.Tensor):
        reference = torch.as_tensor(reference, device=quaternion.device, dtype=quaternion.dtype)
    if reference.device != quaternion.device or reference.dtype != quaternion.dtype:
        raise ValueError("reference_quaternion must use the quaternion device and dtype")
    if reference.ndim == 1:
        if reference.numel() != 4:
            raise ValueError("reference_quaternion must have four elements")
        reference = reference.unsqueeze(0).expand_as(quaternion)
    elif reference.shape != quaternion.shape:
        raise ValueError("reference_quaternion must have shape [4] or match quaternion")
    norm = torch.linalg.vector_norm(quaternion, dim=1)
    ref_norm = torch.linalg.vector_norm(reference, dim=1)
    valid = torch.isfinite(quaternion).all(dim=1) & torch.isfinite(norm) & (norm > 1.0e-7)
    valid = valid & torch.isfinite(reference).all(dim=1) & torch.isfinite(ref_norm) & (ref_norm > 1.0e-7)
    normalized = quaternion / norm.clamp_min(1.0e-7).unsqueeze(1)
    normalized_reference = reference / ref_norm.clamp_min(1.0e-7).unsqueeze(1)
    dot = (normalized * normalized_reference).sum(dim=1).abs()
    error = 2.0 * torch.acos(dot.clamp(min=-1.0, max=1.0))
    error = torch.where(valid, error, torch.full_like(error, torch.pi))
    return error, valid


def sanitize_observation(
    observation: torch.Tensor,
    *,
    low: float = -10.0,
    high: float = 10.0,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep policy observations finite and bounded entirely on the device."""

    observation = _require_batch_tensor(observation, "observation")
    if not observation.is_floating_point() or not low < high:
        raise ValueError("observation must be floating point and low < high")
    if out is None:
        out = torch.empty_like(observation)
    if out.shape != observation.shape or out.device != observation.device or out.dtype != observation.dtype:
        raise ValueError("out must match observation")
    # Compute before writing: callers may intentionally reuse observation as
    # the output buffer to avoid an allocation in a rollout hot path.
    finite = (
        torch.isfinite(observation).all(dim=tuple(range(1, observation.ndim)))
        if observation.ndim > 1
        else torch.isfinite(observation)
    )
    torch.nan_to_num(observation, nan=0.0, posinf=float(high), neginf=float(low), out=out)
    torch.clamp(out, min=float(low), max=float(high), out=out)
    return out, finite


def evaluate_safety(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    controls: torch.Tensor,
    overflow: torch.Tensor,
    *,
    root_qpos_address: int,
    reference_quaternion: torch.Tensor,
    reference_root_height_m: float | torch.Tensor,
    control_low: torch.Tensor,
    control_high: torch.Tensor,
    limits: WarpSafetyLimits,
    previous_estopped: torch.Tensor | None = None,
    previous_reason_code: torch.Tensor | None = None,
    sensordata: torch.Tensor | None = None,
    joint_positions: torch.Tensor | None = None,
    joint_lower: torch.Tensor | None = None,
    joint_upper: torch.Tensor | None = None,
    leg_lengths: torch.Tensor | None = None,
    wheel_contact: torch.Tensor | None = None,
    contact_loss_steps: torch.Tensor | None = None,
    safe_controls_out: torch.Tensor | None = None,
    scratch: WarpSafetyScratch | None = None,
) -> WarpSafetyResult:
    """Evaluate all configured safety checks without host synchronization.

    The caller should retain ``terminated`` and feed it back as
    ``previous_estopped`` on the next control step.  This gives an explicit
    per-world estop latch while allowing masked GPU resets to clear selected
    worlds.  Optional geometry checks are only active when their tensors and
    corresponding limits are supplied.
    """

    qpos = _require_batch_tensor(qpos, "qpos", ndim=2)
    qvel = _require_batch_tensor(qvel, "qvel", ndim=2)
    controls = _require_batch_tensor(controls, "controls", ndim=2)
    overflow = _require_batch_tensor(overflow, "overflow", ndim=1)
    if not qpos.is_floating_point() or not qvel.is_floating_point() or not controls.is_floating_point():
        raise TypeError("qpos, qvel, and controls must be floating point")
    worlds = qpos.shape[0]
    if qvel.shape[0] != worlds or controls.shape[0] != worlds or overflow.shape[0] != worlds:
        raise ValueError("qpos, qvel, controls, and overflow must have the same world dimension")
    if qpos.device != qvel.device or qpos.device != controls.device or qpos.device != overflow.device:
        raise ValueError("all safety tensors must reside on the same device")
    if overflow.dtype not in (torch.bool, torch.int32, torch.int64):
        raise TypeError("overflow must be a bool or integer vector")
    device = qpos.device
    dtype = qpos.dtype
    if scratch is not None:
        scratch.validate(
            worlds=worlds,
            action_dim=controls.shape[1],
            device=device,
            dtype=controls.dtype,
        )

    if previous_estopped is None:
        if scratch is None:
            previous_estopped = torch.zeros(worlds, dtype=torch.bool, device=device)
        else:
            previous_estopped = scratch.default_previous_estopped.zero_()
    else:
        previous_estopped = _require_batch_tensor(previous_estopped, "previous_estopped", ndim=1)
        if previous_estopped.dtype != torch.bool or previous_estopped.shape[0] != worlds or previous_estopped.device != device:
            raise ValueError("previous_estopped must be a matching bool vector")
    if previous_reason_code is not None:
        previous_reason_code = _require_batch_tensor(previous_reason_code, "previous_reason_code", ndim=1)
        if previous_reason_code.dtype not in (torch.int32, torch.int64) or previous_reason_code.shape[0] != worlds or previous_reason_code.device != device:
            raise ValueError("previous_reason_code must be a matching integer vector")

    safe_controls, control_nonfinite = clip_controls(
        controls,
        control_low,
        control_high,
        out=safe_controls_out,
        scratch=scratch,
        nonfinite_out=None if scratch is None else scratch.control_nonfinite,
    )
    state_nonfinite = ~torch.isfinite(qpos).all(dim=1) | ~torch.isfinite(qvel).all(dim=1)
    if scratch is not None:
        scratch.state_nonfinite.copy_(state_nonfinite)
        state_nonfinite = scratch.state_nonfinite
    if sensordata is None:
        sensor_nonfinite = (
            torch.zeros(worlds, dtype=torch.bool, device=device)
            if scratch is None
            else scratch.sensor_nonfinite.zero_()
        )
    else:
        sensordata = _require_batch_tensor(sensordata, "sensordata", ndim=2)
        if sensordata.shape[0] != worlds or sensordata.device != device or not sensordata.is_floating_point():
            raise ValueError("sensordata must be floating point with the qpos world dimension and device")
        sensor_nonfinite = ~torch.isfinite(sensordata).all(dim=1)
        if scratch is not None:
            scratch.sensor_nonfinite.copy_(sensor_nonfinite)
            sensor_nonfinite = scratch.sensor_nonfinite
    overflow_mask = overflow.ne(0)
    if scratch is not None:
        scratch.overflow.copy_(overflow_mask)
        overflow_mask = scratch.overflow

    qstart = int(root_qpos_address)
    if qstart < 0 or qstart + 7 > qpos.shape[1]:
        raise ValueError("root_qpos_address does not leave room for position and quaternion")
    attitude_error, quaternion_valid = quat_attitude_error(
        qpos[:, qstart + 3 : qstart + 7], reference_quaternion
    )
    if scratch is not None:
        scratch.attitude_error_rad.copy_(attitude_error)
        scratch.quaternion_valid.copy_(quaternion_valid)
        attitude_error, quaternion_valid = scratch.attitude_error_rad, scratch.quaternion_valid
    root_height = qpos[:, qstart + 2]
    reference_height = (
        reference_root_height_m
        if isinstance(reference_root_height_m, torch.Tensor)
        else torch.as_tensor(reference_root_height_m, device=device, dtype=dtype)
    )
    if reference_height.device != device or reference_height.dtype != dtype:
        raise ValueError("reference_root_height_m must use the qpos device and dtype")
    if reference_height.ndim == 0:
        reference_height = reference_height.expand(worlds)
    elif reference_height.shape != (worlds,):
        raise ValueError("reference_root_height_m must be scalar or have shape [worlds]")
    attitude_limit = (
        torch.zeros(worlds, dtype=torch.bool, device=device)
        if scratch is None
        else scratch.attitude_limit.zero_()
    )
    height_limit = (
        torch.zeros(worlds, dtype=torch.bool, device=device)
        if scratch is None
        else scratch.height_limit.zero_()
    )
    if limits.fall_guard_enabled:
        attitude_limit = (~quaternion_valid) | (attitude_error > limits.max_attitude_error_rad)
        height_limit = (~torch.isfinite(reference_height)) | (
            root_height < reference_height - limits.max_root_height_drop_m
        )
        if scratch is not None:
            scratch.attitude_limit.copy_(attitude_limit)
            scratch.height_limit.copy_(height_limit)
            attitude_limit, height_limit = scratch.attitude_limit, scratch.height_limit

    joint_limit = (
        torch.zeros(worlds, dtype=torch.bool, device=device)
        if scratch is None
        else scratch.joint_limit.zero_()
    )
    if joint_positions is not None or joint_lower is not None or joint_upper is not None:
        if joint_positions is None or joint_lower is None or joint_upper is None:
            raise ValueError("joint_positions, joint_lower, and joint_upper must be supplied together")
        joint_positions = _require_batch_tensor(joint_positions, "joint_positions", ndim=2)
        if joint_positions.device != device or not joint_positions.is_floating_point():
            raise ValueError("joint_positions must be floating point on the qpos device")
        lower = _as_vector(joint_lower, device=device, dtype=joint_positions.dtype, name="joint_lower")
        upper = _as_vector(joint_upper, device=device, dtype=joint_positions.dtype, name="joint_upper")
        if joint_positions.shape[0] != worlds or lower.numel() != joint_positions.shape[1] or upper.numel() != joint_positions.shape[1]:
            raise ValueError("joint positions and limits have incompatible shapes")
        joint_bounds_valid = torch.isfinite(lower).all() & torch.isfinite(upper).all() & (lower < upper).all()
        joint_limit = (~joint_bounds_valid.expand(worlds)) | (~torch.isfinite(joint_positions).all(dim=1)) | torch.any(
            (joint_positions < lower.unsqueeze(0)) | (joint_positions > upper.unsqueeze(0)), dim=1
        )
        if scratch is not None:
            scratch.joint_limit.copy_(joint_limit)
            joint_limit = scratch.joint_limit

    leg_limit = (
        torch.zeros(worlds, dtype=torch.bool, device=device)
        if scratch is None
        else scratch.leg_limit.zero_()
    )
    if leg_lengths is not None:
        leg_lengths = _require_batch_tensor(leg_lengths, "leg_lengths", ndim=2)
        if leg_lengths.device != device or not leg_lengths.is_floating_point() or leg_lengths.shape[0] != worlds or leg_lengths.shape[1] < 2:
            raise ValueError("leg_lengths must have shape [worlds, >=2] on the qpos device")
        leg_finite = torch.isfinite(leg_lengths).all(dim=1)
        if limits.max_leg_length_difference_m is not None:
            leg_limit = leg_limit | (
                torch.abs(leg_lengths[:, 0] - leg_lengths[:, 1]) > limits.max_leg_length_difference_m
            )
        if limits.min_leg_length_m is not None:
            leg_limit = leg_limit | torch.any(leg_lengths < limits.min_leg_length_m, dim=1)
        if limits.max_leg_length_m is not None:
            leg_limit = leg_limit | torch.any(leg_lengths > limits.max_leg_length_m, dim=1)
        leg_limit = leg_limit | ~leg_finite
        if scratch is not None:
            scratch.leg_limit.copy_(leg_limit)
            leg_limit = scratch.leg_limit

    contact_limit = (
        torch.zeros(worlds, dtype=torch.bool, device=device)
        if scratch is None
        else scratch.contact_limit.zero_()
    )
    if wheel_contact is not None:
        wheel_contact = _require_batch_tensor(wheel_contact, "wheel_contact", ndim=2)
        if wheel_contact.device != device or wheel_contact.dtype != torch.bool or wheel_contact.shape[0] != worlds or wheel_contact.shape[1] < 2:
            raise ValueError("wheel_contact must be a matching bool tensor with at least two columns")
        if limits.max_contact_loss_steps is not None:
            if contact_loss_steps is None:
                raise ValueError("contact_loss_steps is required when max_contact_loss_steps is enabled")
            contact_loss_steps = _require_batch_tensor(contact_loss_steps, "contact_loss_steps", ndim=2)
            if contact_loss_steps.device != device or contact_loss_steps.shape != wheel_contact.shape or contact_loss_steps.dtype not in (torch.int32, torch.int64):
                raise ValueError("contact_loss_steps must match wheel_contact as an integer tensor")
            contact_limit = (~wheel_contact).any(dim=1) & (
                contact_loss_steps.max(dim=1).values >= limits.max_contact_loss_steps
            )
            if scratch is not None:
                scratch.contact_limit.copy_(contact_limit)
                contact_limit = scratch.contact_limit

    # The individual checks are gated by the corresponding P0 switches.
    failure = (
        torch.zeros(worlds, dtype=torch.bool, device=device)
        if scratch is None
        else scratch.failure.zero_()
    )
    if limits.estop_on_nonfinite_control:
        failure = failure | control_nonfinite
    if limits.estop_on_nonfinite_state:
        failure = failure | state_nonfinite | sensor_nonfinite
    if limits.estop_on_overflow:
        failure = failure | overflow_mask
    failure = failure | attitude_limit | height_limit | joint_limit | leg_limit | contact_limit
    terminated = previous_estopped | failure
    if scratch is not None:
        scratch.terminated.copy_(terminated)
        terminated = scratch.terminated

    reason_code = (
        torch.zeros(worlds, dtype=torch.int64, device=device)
        if scratch is None
        else scratch.reason_code.zero_()
    )
    # Priority is deterministic so metrics remain comparable across workers.
    for mask, code in (
        (contact_limit, SAFETY_REASON_CONTACT_LOSS),
        (leg_limit, SAFETY_REASON_LEG_LIMIT),
        (joint_limit, SAFETY_REASON_JOINT_LIMIT),
        (height_limit, SAFETY_REASON_HEIGHT),
        (attitude_limit, SAFETY_REASON_ATTITUDE),
        (overflow_mask, SAFETY_REASON_OVERFLOW),
        (state_nonfinite | sensor_nonfinite, SAFETY_REASON_NONFINITE_STATE),
        (control_nonfinite, SAFETY_REASON_NONFINITE_CONTROL),
    ):
        reason_code = torch.where(mask, torch.as_tensor(code, device=device, dtype=torch.int64), reason_code)
    if previous_reason_code is not None:
        reason_code = torch.where(
            previous_estopped & ~failure,
            previous_reason_code.to(dtype=torch.int64),
            reason_code,
        )
    reason_code = torch.where(
        previous_estopped & (reason_code == SAFETY_REASON_NONE),
        torch.full_like(reason_code, SAFETY_REASON_LATCHED),
        reason_code,
    )
    if scratch is not None:
        scratch.reason_code.copy_(reason_code)
        reason_code = scratch.reason_code
    safe_controls.masked_fill_(terminated.unsqueeze(1), 0.0)
    return WarpSafetyResult(
        safe_controls=safe_controls,
        terminated=terminated,
        failure=failure,
        reason_code=reason_code,
        control_nonfinite=control_nonfinite,
        state_nonfinite=state_nonfinite,
        sensor_nonfinite=sensor_nonfinite,
        overflow=overflow_mask,
        attitude_limit=attitude_limit,
        height_limit=height_limit,
        joint_limit=joint_limit,
        leg_limit=leg_limit,
        contact_limit=contact_limit,
        attitude_error_rad=attitude_error,
        root_height_m=root_height,
    )


__all__ = [
    "MAX_TORQUE_FRACTION_OF_RATED",
    "SAFETY_REASON_NONE",
    "SAFETY_REASON_NONFINITE_CONTROL",
    "SAFETY_REASON_NONFINITE_STATE",
    "SAFETY_REASON_OVERFLOW",
    "SAFETY_REASON_ATTITUDE",
    "SAFETY_REASON_HEIGHT",
    "SAFETY_REASON_JOINT_LIMIT",
    "SAFETY_REASON_LEG_LIMIT",
    "SAFETY_REASON_CONTACT_LOSS",
    "SAFETY_REASON_LATCHED",
    "WarpSafetyLimits",
    "WarpSafetyResult",
    "WarpSafetyScratch",
    "clip_controls",
    "quat_attitude_error",
    "sanitize_observation",
    "evaluate_safety",
]
