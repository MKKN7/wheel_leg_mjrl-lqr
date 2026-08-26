"""Physical-model LQR deployment for the MuJoCo wheeled-leg robot.

The controller linearises the assembled MJCF model at its two-wheel contact
working point.  It commands only the six actuators declared in the XML.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
from scipy.optimize import least_squares

from guide_wheel_mjcf import guide_wheel_runtime_contract


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "wheeled_infantry.xml"
LINEARIZATION_EPSILON = 1e-6
# The 3 m/s value is a validated physical ceiling.  A lower run-time cap can
# be selected with --max-speed when a shorter acceleration window is needed.
MAX_FORWARD_SPEED_MPS = 3.00
MIN_FORWARD_SPEED_LIMIT_MPS = 1.00
DEFAULT_FORWARD_SPEED_LIMIT_MPS = MAX_FORWARD_SPEED_MPS
MAX_REVERSE_SPEED_MPS = 0.35
# Compatibility alias for callers that only need the largest supported magnitude.
MAX_WALK_SPEED_MPS = MAX_FORWARD_SPEED_MPS
DEFAULT_ACCELERATION_MPS2 = 0.60
SPEED_INCREMENT_MPS = 0.25
SPEED_TRACKING_TOLERANCE_MPS = 0.025
HIGH_SPEED_TRACKING_FRACTION = 0.03
SPEED_TELEMETRY_PERIOD_S = 0.25
MAX_COMMAND_YAW_RATE_RAD_S = 0.45
YAW_INCREMENT_RAD = np.deg2rad(15.0)
YAW_TRACKING_TOLERANCE_RAD = np.deg2rad(8.0)
MAX_YAW_RATE_RAD_S = 0.45
YAW_HEADING_KP_RAD_S_PER_RAD = 1.20
YAW_RATE_KP_MOTOR_NM_PER_RAD_S = 0.55
YAW_GOVERNOR_LIMIT_MOTOR_NM = 0.45
# A height field can alternately unload one wheel while a grade crosses the
# contact patch.  Keep this stronger heading loop opt-in so the validated
# flat-scene controller remains unchanged.  The terrain environment enables it
# only after placing the robot on a declared RMUC terrain task.
TERRAIN_YAW_HEADING_KP_RAD_S_PER_RAD = 3.00
TERRAIN_YAW_RATE_KP_MOTOR_NM_PER_RAD_S = 3.00
TERRAIN_YAW_GOVERNOR_LIMIT_MOTOR_NM = 2.50
# High-speed terrain commands carry an explicit yaw-rate request.  Tracking it
# directly avoids relying on a large heading lag to synthesize the desired
# rate, which made the two turning directions load the closed chains
# differently on the RMUC hfield.
TERRAIN_RATE_COMMAND_HEADING_KP_RAD_S_PER_RAD = 0.30
TERRAIN_RATE_COMMAND_KP_MOTOR_NM_PER_RAD_S = 1.50
TERRAIN_RATE_COMMAND_LIMIT_MOTOR_NM = 1.20
YAW_RATE_FILTER_TIME_CONSTANT_S = 0.02
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT = 263
GLFW_KEY_DOWN = 264
GLFW_KEY_UP = 265
WALK_STANCE_HIP_TARGETS_RAD = np.array((0.35, 0.35, -0.35, 0.35))
WALK_STANCE_CLOSURE_TOLERANCE_M = 1e-7
WALK_STANCE_WHEEL_HEIGHT_TOLERANCE_M = 2e-4
MAX_RUNTIME_CLOSURE_ERROR_M = 0.003
WALK_STANCE_HIP_LQR_WEIGHT = 1800.0
WALK_STANCE_HIP_VELOCITY_LQR_WEIGHT = 180.0
WALK_STANCE_GUARD_KP_NM_PER_RAD = 800.0
WALK_STANCE_GUARD_KD_NM_PER_RAD_PER_S = 20.0
WALK_SPEED_KP_MOTOR_NM_PER_MPS = 4.0
WALK_SPEED_KI_MOTOR_NM_PER_M = 3.0
WALK_SPEED_INTEGRAL_LIMIT_M = 2.00
WALK_SPEED_GOVERNOR_LIMIT_MOTOR_NM = 2.90
CONTACT_RECOVERY_WHEEL_BRAKE_LIMIT_MOTOR_NM = 4.50
LQR_FORWARD_SPEED_REFERENCE_LIMIT_MPS = 0.10
LQR_FORWARD_SPEED_FEEDBACK_LIMIT_MPS = 0.25
FORWARD_SPEED_YAW_OVERRIDE_START_MPS = 0.50
LEG_LENGTH_LQR_WEIGHT = 5000.0
LEG_LENGTH_VELOCITY_LQR_WEIGHT = 500.0
LEG_LENGTH_PROFILE_SHAPES_RAD = np.array((-0.20, -0.10, 0.00, 0.05, 0.12, 0.20, 0.28, 0.35, 0.42, 0.50))
WALK_LEG_LENGTH_MIN_M = 0.205
WALK_LEG_LENGTH_MAX_M = 0.300
WALK_LEG_LENGTH_RATE_MPS = 0.10
LEG_LENGTH_SHAPE_KP_RAD_PER_M = 5.0
LEG_LENGTH_SHAPE_KI_RAD_PER_M_S = 1.5
LEG_LENGTH_SHAPE_INTEGRAL_LIMIT_M_S = 0.03
LEG_LENGTH_SHAPE_MAX_CORRECTION_RAD = 0.08
LEG_LENGTH_FORCE_KP_N_PER_M = 700.0
LEG_LENGTH_FORCE_KD_NS_PER_M = 45.0
LEG_LENGTH_FORCE_LIMIT_N = 100.0
MAX_LEG_LENGTH_DIFFERENCE_M = 0.015
HARD_LEG_LENGTH_MIN_M = 0.180
HARD_LEG_LENGTH_MAX_M = 0.400
GAS_SPRING_TORQUE_NM = 10.775
JUMP_PREPARE_LENGTH_M = 0.315
JUMP_CROUCH_LENGTH_M = 0.205
# Leave a few millimetres of clearance below the hard 0.400 m safety limit;
# compliant four-bar motion can overshoot the commanded extension under
# randomized mass and actuator strength.
JUMP_THRUST_LENGTH_M = 0.385
JUMP_PREPARE_RATE_MPS = 0.12
JUMP_CROUCH_RATE_MPS = 0.15
# The jump profile is deliberately bounded by the physical motor limits in the
# MJCF.  The flight phase folds both legs to raise the wheel bottoms, then
# preloads them before contact so touchdown does not drive a link below its
# hard length limit.
# A 1.35 m/s extension produces enough vertical height on the nominal model,
# but the resulting touchdown impulse saturates one hip branch under the RMUC
# jump vehicle-randomization profile.  1.225 m/s keeps the required 20--25 cm
# mean clearance while preserving landing torque headroom for both closed
# chains across the jump vehicle-randomization regression.
JUMP_THRUST_RATE_MPS = 1.225
JUMP_FLIGHT_RETRACT_LENGTH_M = 0.205
JUMP_FLIGHT_RETRACT_RATE_MPS = 2.00
JUMP_FLIGHT_RETRACT_STEPS = 150
JUMP_FLIGHT_RETRACT_FORCE_LIMIT_N = 170.0
# Do not re-extend to the full thrust length before touchdown.  The compliant
# closed chains otherwise arrive asymmetric and can cross the 15 mm leg-diff
# hard safety limit on the RMUC height field.  A 0.320 m preload leaves travel
# for impact absorption while reducing the left/right compression-rate split
# observed at first contact under the vehicle-only randomization profile.
JUMP_FLIGHT_PRELOAD_LENGTH_M = 0.320
JUMP_FLIGHT_PRELOAD_RATE_MPS = 0.50
JUMP_FLIGHT_PRELOAD_FORCE_LIMIT_N = 90.0
JUMP_LANDING_LENGTH_M = 0.285
JUMP_LANDING_RATE_MPS = 0.40
JUMP_LANDING_FORCE_LIMIT_N = 80.0
# Do not wait for the two-wheel touchdown debounce before cancelling flight
# preloading.  An asymmetric first wheel contact can otherwise keep extending
# the unloaded chain for up to eight simulation ticks and exceed the hard
# leg-difference protection.  This overlay is compression-only and remains
# active only until the existing two-wheel landing confirmation completes.
JUMP_IMPACT_MIN_DESCENT_SPEED_MPS = 0.30
JUMP_IMPACT_FORCE_LIMIT_N = JUMP_LANDING_FORCE_LIMIT_N
JUMP_LANDING_STANCE_GUARD_SCALE = 0.30
# The two closed chains see slightly different hfield contact impulses.  A
# bounded differential force keeps their lengths synchronized during impact;
# it does not alter the common jump trajectory or any hard safety limit.
JUMP_LEG_SYNC_KP_N_PER_M = 1200.0
JUMP_LEG_SYNC_KD_NS_PER_M = 70.0
JUMP_LEG_SYNC_FORCE_LIMIT_N = 35.0
# Keep the jump controller active while it transitions from the landing
# impedance target back to the requested walking length.  This prevents the
# former one-tick handoff to full wheel-speed control from re-launching the
# robot after a high landing.
JUMP_RECOVERY_LEG_RATE_MPS = 0.05
JUMP_RECOVERY_STABLE_SECONDS = 0.30
JUMP_RECOVERY_TIMEOUT_S = 2.50
JUMP_RECOVERY_STANCE_GUARD_INITIAL_SCALE = 0.50
JUMP_RECOVERY_STANCE_GUARD_RAMP_SECONDS = 0.15
JUMP_RECOVERY_WHEEL_BRAKE_KP_NM_PER_MPS = 6.0
JUMP_RECOVERY_WHEEL_BRAKE_LIMIT_NM = 1.20
JUMP_RECOVERY_MAX_PITCH_ERROR_RAD = 0.20
JUMP_RECOVERY_MAX_PITCH_RATE_RAD_S = 0.60
JUMP_RECOVERY_MAX_VERTICAL_SPEED_MPS = 0.30
JUMP_RECOVERY_MAX_FORWARD_SPEED_MPS = 0.12
JUMP_RECOVERY_MAX_LEG_DIFFERENCE_M = 0.012
JUMP_PREPARE_TIMEOUT_S = 3.0
JUMP_CROUCH_TIMEOUT_S = 3.0
JUMP_THRUST_TIMEOUT_S = 0.60
JUMP_FLIGHT_TIMEOUT_S = 1.20
JUMP_LANDING_TIMEOUT_S = 2.0
JUMP_LENGTH_TOLERANCE_M = 0.012
JUMP_SETTLE_STEPS = 100
# A local down-step supervisor is intentionally separate from the jump state
# machine.  It only arms when both wheel tracks see the same 14--25 cm terrain
# drop through the hfield preview supplied by the environment.  The controller
# pre-extends before the edge, permits a short expected loss of support, then
# reuses the landing impedance/recovery branch without weakening any hard
# contact, closure, attitude, or leg-length validation.
TERRAIN_DROP_MIN_HEIGHT_M = 0.140
TERRAIN_DROP_MAX_HEIGHT_M = 0.250
TERRAIN_DROP_LOOKAHEAD_M = 0.280
TERRAIN_DROP_FULL_WIDTH_TOLERANCE_M = 0.025
TERRAIN_DROP_ARM_MIN_FORWARD_SPEED_MPS = 0.040
DROP_PRELOAD_LENGTH_M = 0.285
DROP_PRELOAD_RATE_MPS = 0.08
DROP_PRELOAD_FORCE_LIMIT_N = 90.0
DROP_PRELOAD_TIMEOUT_S = 2.00
DROP_FLIGHT_ENTRY_CONFIRM_STEPS = 4
DROP_FLIGHT_EXTENSION_LENGTH_M = 0.350
DROP_FLIGHT_EXTENSION_RATE_MPS = 1.20
DROP_FLIGHT_FORCE_LIMIT_N = 130.0
DROP_FLIGHT_TIMEOUT_S = 0.240
DROP_LANDING_LENGTH_M = 0.310
DROP_LANDING_RATE_MPS = 0.45
DROP_LANDING_FORCE_LIMIT_N = 95.0
DROP_LANDING_CONTACT_CONFIRM_STEPS = 8
DROP_LANDING_SETTLE_STEPS = 40
DROP_LANDING_TIMEOUT_S = 0.80
DROP_RECOVERY_LEG_RATE_MPS = 0.05
DROP_RECOVERY_STABLE_SECONDS = 0.30
DROP_RECOVERY_TIMEOUT_S = 3.00
DROP_RECOVERY_MAX_PITCH_ERROR_RAD = JUMP_RECOVERY_MAX_PITCH_ERROR_RAD
DROP_RECOVERY_MAX_PITCH_RATE_RAD_S = JUMP_RECOVERY_MAX_PITCH_RATE_RAD_S
DROP_RECOVERY_MAX_VERTICAL_SPEED_MPS = JUMP_RECOVERY_MAX_VERTICAL_SPEED_MPS
DROP_RECOVERY_MAX_FORWARD_SPEED_MPS = 0.55
DROP_RECOVERY_MAX_LEG_DIFFERENCE_M = JUMP_RECOVERY_MAX_LEG_DIFFERENCE_M
# Hfield wheel manifolds can omit one 1 kHz contact sample while the wheel is
# still physically supported.  This debounce is only for the post-drop
# recovery handoff; a sustained loss remains visible to the normal safety
# recovery as soon as the drop sequence finishes.
DROP_RECOVERY_CONTACT_GRACE_STEPS = 4
# A requested jump first uses the normal two-wheel controller to regulate into
# a measured low-speed rolling window.  The RMUC hfield has a small persistent
# creep at a zero wheel reference, so this deliberately allows a controlled
# rolling launch rather than requiring a stationary robot.
JUMP_MAX_LAUNCH_SPEED_MPS = 0.13
JUMP_LAUNCH_SPEED_TOLERANCE_MPS = 0.0
# The hfield's nominal downhill creep is about +0.10 m/s at this tiny
# counter-command, which yields a controlled rolling rather than static launch.
JUMP_LAUNCH_REFERENCE_SPEED_MPS = -0.01
JUMP_ROLLING_LAUNCH_MAX_SPEED_MPS = 0.55
JUMP_ROLLING_LAUNCH_TOLERANCE_MPS = 0.08
JUMP_THRUST_WHEEL_KP_NM_PER_MPS = 1.50
JUMP_THRUST_WHEEL_LIMIT_NM = 0.80
JUMP_BRAKE_STABLE_SECONDS = 0.10
JUMP_BRAKE_MAX_VERTICAL_SPEED_MPS = 0.12
JUMP_BRAKE_MAX_ANGULAR_SPEED_RAD_S = 0.80
# Hfield contact manifolds can intermittently report one empty wheel-contact
# frame even when both wheels remain supported.  This only debounces the
# state-machine contact predicate; physical jump safety validation remains
# immediate and independent.
JUMP_GROUND_CONTACT_GRACE_STEPS = 4
# Flight must not reuse the launch contact hysteresis: transitioning into
# ``flight`` resets that counter, so a zero-contact frame would otherwise look
# grounded for several ticks and immediately skip the retract trajectory.
# Wait for a short, real two-wheel contact manifold before enabling landing
# impedance.  Three 1 kHz samples still includes the hfield's impact pulse;
# Eight samples remains only 8 ms, but covers the full asymmetric impact
# manifold under the vehicle-only RMUC domain randomization profile.
JUMP_LANDING_CONTACT_CONFIRM_STEPS = 8
JUMP_LIFTOFF_STEPS = 3
JUMP_THRUST_FORCE_LIMIT_N = 240.0
AIRBORNE_HIP_KP_NM_PER_RAD = 20.0
AIRBORNE_HIP_KD_NM_PER_RAD_PER_S = 4.0
AIRBORNE_WHEEL_ATTITUDE_KP_NM_PER_RAD = 6.0
AIRBORNE_WHEEL_ATTITUDE_KD_NM_PER_RAD_S = 0.48
# Terrain support is opt-in.  The values below filter externally supplied
# terrain heights before they become an LQR root-height reference, avoiding a
# discontinuous vertical position target at an hfield cell boundary.
TERRAIN_SUPPORT_REFERENCE_FILTER_TIME_CONSTANT_S = 0.08
TERRAIN_SUPPORT_REFERENCE_MAX_VERTICAL_RATE_MPS = 0.75
WORLD_UP = np.array((0.0, 0.0, 1.0))

# These exact lower-guide names are loaded from the MJCF configuration rather
# than prefix-matched. A partial or miswired package must not silently alter
# collision safety or the LQR state.
_GUIDE_WHEEL_CONTRACT = guide_wheel_runtime_contract()
GUIDE_WHEEL_CONTACT_NAMES = _GUIDE_WHEEL_CONTRACT.contact_names
GUIDE_WHEEL_JOINT_NAMES = _GUIDE_WHEEL_CONTRACT.joint_names


def jump_controller_config() -> dict[str, float | int]:
    """Return every trajectory parameter that changes jump behavior."""
    return {
        "jump_prepare_length_m": JUMP_PREPARE_LENGTH_M,
        "jump_crouch_length_m": JUMP_CROUCH_LENGTH_M,
        "jump_thrust_length_m": JUMP_THRUST_LENGTH_M,
        "jump_thrust_rate_mps": JUMP_THRUST_RATE_MPS,
        "jump_thrust_force_limit_n": JUMP_THRUST_FORCE_LIMIT_N,
        "jump_flight_retract_length_m": JUMP_FLIGHT_RETRACT_LENGTH_M,
        "jump_flight_retract_rate_mps": JUMP_FLIGHT_RETRACT_RATE_MPS,
        "jump_flight_retract_steps": JUMP_FLIGHT_RETRACT_STEPS,
        "jump_flight_retract_force_limit_n": JUMP_FLIGHT_RETRACT_FORCE_LIMIT_N,
        "jump_flight_preload_length_m": JUMP_FLIGHT_PRELOAD_LENGTH_M,
        "jump_flight_preload_rate_mps": JUMP_FLIGHT_PRELOAD_RATE_MPS,
        "jump_flight_preload_force_limit_n": JUMP_FLIGHT_PRELOAD_FORCE_LIMIT_N,
        "jump_landing_length_m": JUMP_LANDING_LENGTH_M,
        "jump_landing_rate_mps": JUMP_LANDING_RATE_MPS,
        "jump_landing_force_limit_n": JUMP_LANDING_FORCE_LIMIT_N,
        "jump_impact_min_descent_speed_mps": JUMP_IMPACT_MIN_DESCENT_SPEED_MPS,
        "jump_impact_force_limit_n": JUMP_IMPACT_FORCE_LIMIT_N,
        "jump_landing_stance_guard_scale": JUMP_LANDING_STANCE_GUARD_SCALE,
        "jump_leg_sync_kp_n_per_m": JUMP_LEG_SYNC_KP_N_PER_M,
        "jump_leg_sync_kd_ns_per_m": JUMP_LEG_SYNC_KD_NS_PER_M,
        "jump_leg_sync_force_limit_n": JUMP_LEG_SYNC_FORCE_LIMIT_N,
        "jump_recovery_leg_rate_mps": JUMP_RECOVERY_LEG_RATE_MPS,
        "jump_recovery_stable_seconds": JUMP_RECOVERY_STABLE_SECONDS,
        "jump_recovery_timeout_s": JUMP_RECOVERY_TIMEOUT_S,
        "jump_recovery_stance_guard_initial_scale": JUMP_RECOVERY_STANCE_GUARD_INITIAL_SCALE,
        "jump_recovery_stance_guard_ramp_seconds": JUMP_RECOVERY_STANCE_GUARD_RAMP_SECONDS,
        "jump_recovery_wheel_brake_kp_nm_per_mps": JUMP_RECOVERY_WHEEL_BRAKE_KP_NM_PER_MPS,
        "jump_recovery_wheel_brake_limit_nm": JUMP_RECOVERY_WHEEL_BRAKE_LIMIT_NM,
        "jump_recovery_max_pitch_error_rad": JUMP_RECOVERY_MAX_PITCH_ERROR_RAD,
        "jump_recovery_max_pitch_rate_rad_s": JUMP_RECOVERY_MAX_PITCH_RATE_RAD_S,
        "jump_recovery_max_vertical_speed_mps": JUMP_RECOVERY_MAX_VERTICAL_SPEED_MPS,
        "jump_recovery_max_forward_speed_mps": JUMP_RECOVERY_MAX_FORWARD_SPEED_MPS,
        "jump_recovery_max_leg_difference_m": JUMP_RECOVERY_MAX_LEG_DIFFERENCE_M,
        "jump_max_launch_speed_mps": JUMP_MAX_LAUNCH_SPEED_MPS,
        "jump_launch_speed_tolerance_mps": JUMP_LAUNCH_SPEED_TOLERANCE_MPS,
        "jump_launch_reference_speed_mps": JUMP_LAUNCH_REFERENCE_SPEED_MPS,
        "jump_rolling_launch_max_speed_mps": JUMP_ROLLING_LAUNCH_MAX_SPEED_MPS,
        "jump_rolling_launch_tolerance_mps": JUMP_ROLLING_LAUNCH_TOLERANCE_MPS,
        "jump_thrust_wheel_kp_nm_per_mps": JUMP_THRUST_WHEEL_KP_NM_PER_MPS,
        "jump_thrust_wheel_limit_nm": JUMP_THRUST_WHEEL_LIMIT_NM,
        "jump_brake_stable_seconds": JUMP_BRAKE_STABLE_SECONDS,
        "jump_brake_max_vertical_speed_mps": JUMP_BRAKE_MAX_VERTICAL_SPEED_MPS,
        "jump_brake_max_angular_speed_rad_s": JUMP_BRAKE_MAX_ANGULAR_SPEED_RAD_S,
        "jump_ground_contact_grace_steps": JUMP_GROUND_CONTACT_GRACE_STEPS,
        "jump_landing_contact_confirm_steps": JUMP_LANDING_CONTACT_CONFIRM_STEPS,
        "jump_settle_steps": JUMP_SETTLE_STEPS,
        "airborne_hip_kp_nm_per_rad": AIRBORNE_HIP_KP_NM_PER_RAD,
        "airborne_hip_kd_nm_per_rad_per_s": AIRBORNE_HIP_KD_NM_PER_RAD_PER_S,
        "airborne_wheel_attitude_kp_nm_per_rad": AIRBORNE_WHEEL_ATTITUDE_KP_NM_PER_RAD,
        "airborne_wheel_attitude_kd_nm_per_rad_s": AIRBORNE_WHEEL_ATTITUDE_KD_NM_PER_RAD_S,
    }


def terrain_controller_config() -> dict[str, float]:
    """Return terrain-specific feedback parameters used for checkpoint identity."""
    return {
        "support_reference_filter_time_constant_s": TERRAIN_SUPPORT_REFERENCE_FILTER_TIME_CONSTANT_S,
        "support_reference_max_vertical_rate_mps": TERRAIN_SUPPORT_REFERENCE_MAX_VERTICAL_RATE_MPS,
        "yaw_heading_kp_rad_s_per_rad": TERRAIN_YAW_HEADING_KP_RAD_S_PER_RAD,
        "yaw_rate_kp_motor_nm_per_rad_s": TERRAIN_YAW_RATE_KP_MOTOR_NM_PER_RAD_S,
        "yaw_governor_limit_motor_nm": TERRAIN_YAW_GOVERNOR_LIMIT_MOTOR_NM,
        "rate_command_heading_kp_rad_s_per_rad": TERRAIN_RATE_COMMAND_HEADING_KP_RAD_S_PER_RAD,
        "rate_command_kp_motor_nm_per_rad_s": TERRAIN_RATE_COMMAND_KP_MOTOR_NM_PER_RAD_S,
        "rate_command_limit_motor_nm": TERRAIN_RATE_COMMAND_LIMIT_MOTOR_NM,
        "drop_min_height_m": TERRAIN_DROP_MIN_HEIGHT_M,
        "drop_max_height_m": TERRAIN_DROP_MAX_HEIGHT_M,
        "drop_lookahead_m": TERRAIN_DROP_LOOKAHEAD_M,
        "drop_full_width_tolerance_m": TERRAIN_DROP_FULL_WIDTH_TOLERANCE_M,
        "drop_arm_min_forward_speed_mps": TERRAIN_DROP_ARM_MIN_FORWARD_SPEED_MPS,
        "drop_preload_length_m": DROP_PRELOAD_LENGTH_M,
        "drop_preload_rate_mps": DROP_PRELOAD_RATE_MPS,
        "drop_preload_force_limit_n": DROP_PRELOAD_FORCE_LIMIT_N,
        "drop_preload_timeout_s": DROP_PRELOAD_TIMEOUT_S,
        "drop_flight_entry_confirm_steps": DROP_FLIGHT_ENTRY_CONFIRM_STEPS,
        "drop_flight_extension_length_m": DROP_FLIGHT_EXTENSION_LENGTH_M,
        "drop_flight_extension_rate_mps": DROP_FLIGHT_EXTENSION_RATE_MPS,
        "drop_flight_force_limit_n": DROP_FLIGHT_FORCE_LIMIT_N,
        "drop_flight_timeout_s": DROP_FLIGHT_TIMEOUT_S,
        "drop_landing_length_m": DROP_LANDING_LENGTH_M,
        "drop_landing_rate_mps": DROP_LANDING_RATE_MPS,
        "drop_landing_force_limit_n": DROP_LANDING_FORCE_LIMIT_N,
        "drop_landing_contact_confirm_steps": DROP_LANDING_CONTACT_CONFIRM_STEPS,
        "drop_landing_settle_steps": DROP_LANDING_SETTLE_STEPS,
        "drop_landing_timeout_s": DROP_LANDING_TIMEOUT_S,
        "drop_recovery_leg_rate_mps": DROP_RECOVERY_LEG_RATE_MPS,
        "drop_recovery_stable_seconds": DROP_RECOVERY_STABLE_SECONDS,
        "drop_recovery_timeout_s": DROP_RECOVERY_TIMEOUT_S,
        "drop_recovery_max_forward_speed_mps": DROP_RECOVERY_MAX_FORWARD_SPEED_MPS,
        "drop_recovery_contact_grace_steps": DROP_RECOVERY_CONTACT_GRACE_STEPS,
    }


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def speed_tracking_tolerance(target_speed: float) -> float:
    return max(SPEED_TRACKING_TOLERANCE_MPS, HIGH_SPEED_TRACKING_FRACTION * abs(target_speed))


def validate_forward_speed_limit(limit: float) -> float:
    if not np.isfinite(limit) or not MIN_FORWARD_SPEED_LIMIT_MPS <= limit <= MAX_FORWARD_SPEED_MPS:
        raise ValueError(
            f"forward speed limit must be within {MIN_FORWARD_SPEED_LIMIT_MPS:.2f}.."
            f"{MAX_FORWARD_SPEED_MPS:.2f}m/s"
        )
    return float(limit)


def validate_reverse_speed_limit(limit: float) -> float:
    if not np.isfinite(limit) or not 0.0 <= limit <= MAX_REVERSE_SPEED_MPS:
        raise ValueError(f"reverse speed limit must be within 0..{MAX_REVERSE_SPEED_MPS:.2f}m/s")
    return float(limit)


def clamp_speed_command(
    speed: float,
    *,
    forward_limit: float = MAX_FORWARD_SPEED_MPS,
    reverse_limit: float = MAX_REVERSE_SPEED_MPS,
) -> float:
    """Clamp a longitudinal command to the validated directional range."""
    if not np.isfinite(speed):
        raise ValueError("speed command must be finite")
    return float(np.clip(
        speed,
        -validate_reverse_speed_limit(reverse_limit),
        validate_forward_speed_limit(forward_limit),
    ))


def speed_tracking_status(
    target_speed: float,
    ramped_command: float,
    measured_speed: float,
    *,
    jump_active: bool = False,
) -> str:
    tolerance = speed_tracking_tolerance(target_speed)
    if jump_active:
        return "JUMP"
    if abs(target_speed - ramped_command) > tolerance:
        return "RAMPING"
    direction = np.sign(target_speed) if abs(target_speed) > 1e-9 else np.sign(measured_speed)
    signed_error = (measured_speed - target_speed) * (direction if direction != 0.0 else 1.0)
    if abs(signed_error) <= tolerance:
        return "TRACKING"
    return "UNDERSPEED" if signed_error < 0.0 else "OVERSPEED"


@dataclass(frozen=True)
class ModelRefs:
    root_joint: int
    robot_body: int
    ground_geom: int
    ground_geoms: tuple[int, ...]
    obstacle_geoms: tuple[int, ...]
    wheel_geoms: tuple[int, int]
    guide_wheel_geoms: tuple[int, ...]
    base_geoms: tuple[int, int]
    linkage_collision_geoms: tuple[int, ...]
    actuator_ids: tuple[int, int, int, int, int, int]
    wheel_joints: tuple[int, int]
    guide_wheel_joints: tuple[int, ...]
    stance_passive_joints: tuple[int, int, int, int, int, int, int, int]
    stance_closure_sites: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    sensor_refs: dict[str, tuple[int, int]]


@dataclass
class MotionCommand:
    """Rate-limited forward velocity command and integrated position reference."""

    target_speed: float
    acceleration_limit: float
    current_speed: float = 0.0
    max_forward_speed: float = MAX_FORWARD_SPEED_MPS
    max_reverse_speed: float = MAX_REVERSE_SPEED_MPS

    def __post_init__(self) -> None:
        self.max_forward_speed = validate_forward_speed_limit(self.max_forward_speed)
        self.max_reverse_speed = validate_reverse_speed_limit(self.max_reverse_speed)
        self.target_speed = clamp_speed_command(
            self.target_speed,
            forward_limit=self.max_forward_speed,
            reverse_limit=self.max_reverse_speed,
        )
        if not np.isfinite(self.acceleration_limit) or self.acceleration_limit <= 0.0:
            raise ValueError("acceleration_limit must be positive and finite")

    def set_target_speed(self, speed: float) -> None:
        self.target_speed = clamp_speed_command(
            speed,
            forward_limit=self.max_forward_speed,
            reverse_limit=self.max_reverse_speed,
        )

    def advance(self, elapsed: float) -> None:
        if elapsed <= 0.0:
            return
        maximum_change = self.acceleration_limit * elapsed
        self.current_speed += float(np.clip(self.target_speed - self.current_speed, -maximum_change, maximum_change))


@dataclass
class HeadingCommand:
    """World-frame desired heading with a bounded reference slew rate."""

    command_yaw: float
    reference_yaw: float
    maximum_rate: float = MAX_COMMAND_YAW_RATE_RAD_S

    def set_command(self, yaw: float) -> float:
        if not np.isfinite(yaw):
            raise ValueError("command yaw must be finite")
        self.command_yaw = wrap_to_pi(float(yaw))
        return self.command_yaw

    def advance(self, elapsed: float) -> None:
        if elapsed <= 0.0:
            return
        yaw_error = wrap_to_pi(self.command_yaw - self.reference_yaw)
        yaw_step = float(np.clip(yaw_error, -self.maximum_rate * elapsed, self.maximum_rate * elapsed))
        self.reference_yaw = wrap_to_pi(self.reference_yaw + yaw_step)


@dataclass
class LegLengthCommand:
    """Rate-limited common leg-length reference for the symmetric wheel stance."""

    target_length: float
    current_length: float
    maximum_rate: float = WALK_LEG_LENGTH_RATE_MPS
    current_rate: float = 0.0

    def set_target(self, length: float, lower: float, upper: float) -> float:
        self.target_length = float(np.clip(length, lower, upper))
        return self.target_length

    def advance(self, elapsed: float, maximum_rate: float | None = None) -> None:
        if elapsed <= 0.0:
            self.current_rate = 0.0
            return
        rate = self.maximum_rate if maximum_rate is None else maximum_rate
        delta = float(np.clip(self.target_length - self.current_length, -rate * elapsed, rate * elapsed))
        self.current_length += delta
        self.current_rate = delta / elapsed


@dataclass
class JumpSequence:
    """Unconditionally-started, contact-aware leg-length jump state machine."""

    phase_name: str | None = None
    phase_start_time: float | None = None
    resume_length: float | None = None
    resume_speed: float | None = None
    launch_speed: float = 0.0
    airborne_steps: int = 0
    flight_steps: int = 0
    settled_steps: int = 0
    recovering: bool = False
    recovery_start_time: float | None = None
    recovery_stable_steps: int = 0
    wheel_contact_loss_steps: tuple[int, int] = (0, 0)
    impact_active: bool = False
    impact_start_time: float | None = None
    impact_first_contact: tuple[int, int] = (0, 0)
    impact_max_leg_difference_m: float = 0.0
    abort_reason: str = ""

    @property
    def active(self) -> bool:
        return self.phase_name is not None

    def start(
        self,
        sim_time: float,
        resume_length: float,
        resume_speed: float | None = None,
        launch_speed: float = 0.0,
    ) -> None:
        self.phase_name = "prepare"
        self.phase_start_time = sim_time
        self.resume_length = resume_length
        self.resume_speed = resume_speed
        self.launch_speed = float(launch_speed)
        self.airborne_steps = 0
        self.flight_steps = 0
        self.settled_steps = 0
        self.recovering = False
        self.recovery_start_time = None
        self.recovery_stable_steps = 0
        self.wheel_contact_loss_steps = (0, 0)
        self.impact_active = False
        self.impact_start_time = None
        self.impact_first_contact = (0, 0)
        self.impact_max_leg_difference_m = 0.0
        self.abort_reason = ""

    def transition(self, phase_name: str, sim_time: float) -> None:
        self.phase_name = phase_name
        self.phase_start_time = sim_time
        self.airborne_steps = 0
        self.flight_steps = 0
        self.settled_steps = 0
        self.recovering = False
        self.recovery_start_time = None
        self.recovery_stable_steps = 0
        self.wheel_contact_loss_steps = (0, 0)
        if phase_name != "flight":
            self.impact_active = False

    def begin_impact(self, sim_time: float, contacts: tuple[int, int]) -> None:
        """Enter the bounded first-touchdown overlay during flight."""
        if self.impact_active:
            return
        self.impact_active = True
        self.impact_start_time = sim_time
        self.impact_first_contact = contacts

    def update_impact_leg_difference(self, difference_m: float) -> None:
        self.impact_max_leg_difference_m = max(
            self.impact_max_leg_difference_m,
            abs(float(difference_m)),
        )

    def begin_recovery(self, sim_time: float) -> None:
        self.recovering = True
        self.recovery_start_time = sim_time
        self.recovery_stable_steps = 0

    def grounded_with_hysteresis(self, contacts: tuple[int, int]) -> bool:
        """Debounce independent wheel-contact reports for phase transitions."""
        self.wheel_contact_loss_steps = tuple(
            0 if contact_count > 0 else loss_steps + 1
            for contact_count, loss_steps in zip(contacts, self.wheel_contact_loss_steps)
        )
        return max(self.wheel_contact_loss_steps) < JUMP_GROUND_CONTACT_GRACE_STEPS

    def elapsed(self, sim_time: float) -> float:
        if self.phase_start_time is None:
            return 0.0
        return max(0.0, sim_time - self.phase_start_time)

    def finish(self) -> None:
        self.phase_name = None
        self.phase_start_time = None
        self.resume_length = None
        self.resume_speed = None
        self.launch_speed = 0.0
        self.airborne_steps = 0
        self.flight_steps = 0
        self.settled_steps = 0
        self.recovering = False
        self.recovery_start_time = None
        self.recovery_stable_steps = 0
        self.wheel_contact_loss_steps = (0, 0)
        self.impact_active = False
        self.impact_start_time = None
        self.impact_first_contact = (0, 0)
        self.impact_max_leg_difference_m = 0.0


@dataclass
class TerrainDropSequence:
    """Bounded local down-step transition; deliberately independent of jumps."""

    phase_name: str | None = None
    phase_start_time: float | None = None
    resume_length: float | None = None
    resume_speed: float | None = None
    drop_height_m: float = 0.0
    unloaded_steps: int = 0
    flight_steps: int = 0
    settled_steps: int = 0
    recovering: bool = False
    recovery_start_time: float | None = None
    recovery_stable_steps: int = 0
    wheel_contact_loss_steps: tuple[int, int] = (0, 0)
    abort_reason: str = ""

    @property
    def active(self) -> bool:
        return self.phase_name is not None

    def start(
        self,
        sim_time: float,
        resume_length: float,
        resume_speed: float | None,
        drop_height_m: float,
    ) -> None:
        self.phase_name = "preload"
        self.phase_start_time = sim_time
        self.resume_length = resume_length
        self.resume_speed = resume_speed
        self.drop_height_m = drop_height_m
        self.unloaded_steps = 0
        self.flight_steps = 0
        self.settled_steps = 0
        self.recovering = False
        self.recovery_start_time = None
        self.recovery_stable_steps = 0
        self.wheel_contact_loss_steps = (0, 0)
        self.abort_reason = ""

    def transition(self, phase_name: str, sim_time: float) -> None:
        self.phase_name = phase_name
        self.phase_start_time = sim_time
        self.unloaded_steps = 0
        self.flight_steps = 0
        self.settled_steps = 0
        self.recovering = False
        self.recovery_start_time = None
        self.recovery_stable_steps = 0
        self.wheel_contact_loss_steps = (0, 0)

    def begin_recovery(self, sim_time: float) -> None:
        self.phase_name = "recovery"
        self.phase_start_time = sim_time
        self.recovering = True
        self.recovery_start_time = sim_time
        self.recovery_stable_steps = 0
        self.wheel_contact_loss_steps = (0, 0)

    def grounded_with_hysteresis(self, contacts: tuple[int, int]) -> bool:
        """Debounce isolated hfield wheel-contact omissions during recovery."""
        self.wheel_contact_loss_steps = tuple(
            0 if contact_count > 0 else loss_steps + 1
            for contact_count, loss_steps in zip(contacts, self.wheel_contact_loss_steps)
        )
        return max(self.wheel_contact_loss_steps) < DROP_RECOVERY_CONTACT_GRACE_STEPS

    def elapsed(self, sim_time: float) -> float:
        if self.phase_start_time is None:
            return 0.0
        return max(0.0, sim_time - self.phase_start_time)

    def finish(self) -> None:
        self.phase_name = None
        self.phase_start_time = None
        self.resume_length = None
        self.resume_speed = None
        self.drop_height_m = 0.0
        self.unloaded_steps = 0
        self.flight_steps = 0
        self.settled_steps = 0
        self.recovering = False
        self.recovery_start_time = None
        self.recovery_stable_steps = 0
        self.wheel_contact_loss_steps = (0, 0)


@dataclass(frozen=True)
class LqrTrim:
    """One physically projected and linearised closed-chain operating point."""

    shape: float
    leg_length: float
    qpos: np.ndarray
    qvel: np.ndarray
    control: np.ndarray
    gain: np.ndarray
    hip_qpos: np.ndarray


@dataclass(frozen=True)
class SpeedTrim:
    """Forward rolling LQR trim used to extend the controller to 3 m/s."""

    speed: float
    leg_length: float
    qvel: np.ndarray
    control: np.ndarray
    gain: np.ndarray


@dataclass(frozen=True)
class LegLengthProfile:
    """Monotonic low-centre shape-to-length table with scheduled LQR trims."""

    trims: tuple[LqrTrim, ...]

    def __post_init__(self) -> None:
        shapes = self.shapes
        lengths = self.lengths
        if len(self.trims) < 2 or not np.all(np.diff(shapes) > 0.0):
            raise ValueError("leg-length profile must contain ascending shape samples")
        if not np.all(np.diff(lengths) < 0.0):
            raise ValueError("leg-length profile must be monotonic on the selected closed-chain branch")

    @property
    def shapes(self) -> np.ndarray:
        return np.asarray([trim.shape for trim in self.trims], dtype=np.float64)

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([trim.leg_length for trim in self.trims], dtype=np.float64)

    @property
    def minimum_length(self) -> float:
        return float(self.lengths[-1])

    @property
    def maximum_length(self) -> float:
        return float(self.lengths[0])

    def clamp_length(self, length: float) -> float:
        return float(np.clip(length, self.minimum_length, self.maximum_length))

    def shape_for_length(self, length: float) -> float:
        clamped_length = self.clamp_length(length)
        return float(np.interp(clamped_length, self.lengths[::-1], self.shapes[::-1]))

    def length_for_shape(self, shape: float) -> float:
        clamped_shape = float(np.clip(shape, self.shapes[0], self.shapes[-1]))
        return float(np.interp(clamped_shape, self.shapes, self.lengths))

    def bracket(self, shape: float) -> tuple[LqrTrim, LqrTrim, float]:
        shapes = self.shapes
        clamped_shape = float(np.clip(shape, shapes[0], shapes[-1]))
        upper_index = int(np.searchsorted(shapes, clamped_shape, side="right"))
        upper_index = int(np.clip(upper_index, 1, len(self.trims) - 1))
        lower_index = upper_index - 1
        lower = self.trims[lower_index]
        upper = self.trims[upper_index]
        interpolation = (clamped_shape - lower.shape) / (upper.shape - lower.shape)
        return lower, upper, float(interpolation)

    def hip_length_jacobian(self, shape: float) -> float:
        lower, upper, _ = self.bracket(shape)
        length_slope = (upper.leg_length - lower.leg_length) / (upper.shape - lower.shape)
        return 0.5 * abs(float(length_slope))


def object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise ValueError(f"MJCF object is missing: {name}")
    return result


def optional_guide_wheel_refs(model: mujoco.MjModel) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve a complete passive-guide package or leave a legacy MJCF unchanged.

    The baseline robot intentionally has no guide joints.  Once any guide
    object is present, however, every named contact geom and passive hinge must
    exist on the same body and no guide hinge may be actuated.
    """

    geom_ids = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in GUIDE_WHEEL_CONTACT_NAMES
    )
    joint_ids = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in GUIDE_WHEEL_JOINT_NAMES
    )
    any_present = any(identifier >= 0 for identifier in (*geom_ids, *joint_ids))
    if not any_present:
        return (), ()
    missing = [
        name
        for name, identifier in zip(GUIDE_WHEEL_CONTACT_NAMES, geom_ids)
        if identifier < 0
    ] + [
        name
        for name, identifier in zip(GUIDE_WHEEL_JOINT_NAMES, joint_ids)
        if identifier < 0
    ]
    if missing:
        raise ValueError(
            "guide-wheel MJCF package is incomplete; missing " + ", ".join(missing)
        )
    if len(set(geom_ids)) != len(geom_ids) or len(set(joint_ids)) != len(joint_ids):
        raise ValueError("guide-wheel MJCF package contains duplicate contact or joint ids")

    joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    actuator_types = np.asarray(model.actuator_trntype, dtype=np.int32)
    actuator_targets = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    for contact_name, joint_name, geom_id, joint_id in zip(
        GUIDE_WHEEL_CONTACT_NAMES,
        GUIDE_WHEEL_JOINT_NAMES,
        geom_ids,
        joint_ids,
    ):
        if int(model.jnt_type[joint_id]) != hinge_type:
            raise ValueError(f"guide joint {joint_name!r} must be a scalar hinge")
        if int(model.geom_bodyid[geom_id]) != int(model.jnt_bodyid[joint_id]):
            raise ValueError(
                f"guide contact {contact_name!r} must share a body with {joint_name!r}"
            )
        if np.any((actuator_types == joint_transmission) & (actuator_targets == joint_id)):
            raise ValueError(f"guide joint {joint_name!r} must remain passive and unactuated")
    return geom_ids, joint_ids


def build_refs(model: mujoco.MjModel) -> ModelRefs:
    sensor_names = (
        "world_horizontal_position_xy", "world_horizontal_velocity_xy",
        "left_wheel_angle", "left_wheel_angular_velocity",
        "right_wheel_angle", "right_wheel_angular_velocity",
        "left_leg_length", "left_leg_length_velocity",
        "right_leg_length", "right_leg_length_velocity",
        "world_body_orientation_quat", "body_angular_velocity",
        "imu_gyroscope", "imu_linear_accelerometer", "imu_linear_velocity",
    )
    sensor_refs: dict[str, tuple[int, int]] = {}
    for name in sensor_names:
        sensor_id = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        sensor_refs[name] = (int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id]))
    actuator_names = (
        "left_hip_motor", "left_active_hip_motor", "left_wheel_motor",
        "right_hip_motor", "right_active_hip_motor", "right_wheel_motor",
    )
    ground_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    ground_geoms = [ground_geom]
    # ``ground`` remains the mandatory flat LQR trim surface.  Existing
    # hfield scenes use their historical names, while static terrain scenes
    # register only collision supports through the explicit ``support_``
    # prefix.  Visual-only geoms therefore never become wheel support.
    for terrain_name in ("terrain", "rmuc_terrain"):
        terrain_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, terrain_name)
        if terrain_geom >= 0:
            ground_geoms.append(int(terrain_geom))
    obstacle_geoms: list[int] = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is None:
            continue
        if geom_name.startswith("support_"):
            ground_geoms.append(int(geom_id))
        elif geom_name.startswith("obstacle_"):
            obstacle_geoms.append(int(geom_id))
    ground_geoms = list(dict.fromkeys(ground_geoms))
    guide_wheel_geoms, guide_wheel_joints = optional_guide_wheel_refs(model)
    return ModelRefs(
        root_joint=object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free"),
        robot_body=object_id(model, mujoco.mjtObj.mjOBJ_BODY, "robot"),
        ground_geom=ground_geom,
        ground_geoms=tuple(ground_geoms),
        obstacle_geoms=tuple(obstacle_geoms),
        wheel_geoms=(
            object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_contact"),
            object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_contact"),
        ),
        guide_wheel_geoms=guide_wheel_geoms,
        base_geoms=(
            object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_contact"),
            object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_collision"),
        ),
        linkage_collision_geoms=tuple(
            object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "left_active_link_collision", "left_long_link_collision",
                "left_node_link_collision", "left_parallel_link_collision",
                "left_upper_leg_collision", "left_lower_leg_collision",
                "right_active_link_collision", "right_long_link_collision",
                "right_node_link_collision", "right_parallel_link_collision",
                "right_upper_leg_collision", "right_lower_leg_collision",
            )
        ),
        actuator_ids=tuple(object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names),
        wheel_joints=(
            object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_spin"),
            object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_spin"),
        ),
        guide_wheel_joints=guide_wheel_joints,
        stance_passive_joints=tuple(
            object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in (
                "left_long_link_pitch", "left_node_link_pitch", "left_parallel_link_pitch", "left_knee_pitch",
                "right_long_link_pitch", "right_node_link_pitch", "right_parallel_link_pitch", "right_knee_pitch",
            )
        ),
        stance_closure_sites=tuple(
            (
                object_id(model, mujoco.mjtObj.mjOBJ_SITE, first),
                object_id(model, mujoco.mjtObj.mjOBJ_SITE, second),
            )
            for first, second in (
                ("left_node_site", "left_upper_site"),
                ("left_parallel_site", "left_lower_site"),
                ("right_node_site", "right_upper_site"),
                ("right_parallel_site", "right_lower_site"),
            )
        ),
        sensor_refs=sensor_refs,
    )


def sensor(data: mujoco.MjData, refs: ModelRefs, name: str) -> np.ndarray:
    address, dimension = refs.sensor_refs[name]
    return data.sensordata[address : address + dimension].copy()


def contacts_for_geom(data: mujoco.MjData, geom_id: int) -> int:
    return sum(
        contact.geom1 == geom_id or contact.geom2 == geom_id
        for contact in data.contact[: data.ncon]
    )


def wheel_ground_contacts(data: mujoco.MjData, refs: ModelRefs, wheel_geom: int) -> int:
    """Count physical support contacts between one wheel and any declared ground geom."""
    return sum(
        (contact.geom1 in refs.ground_geoms and contact.geom2 == wheel_geom)
        or (contact.geom2 in refs.ground_geoms and contact.geom1 == wheel_geom)
        for contact in data.contact[: data.ncon]
    )


def nonwheel_static_contact_counts(
    data: mujoco.MjData,
    refs: ModelRefs,
) -> tuple[int, int]:
    """Count unsafe contacts while allowing passive rollers only on supports."""
    support_geoms = set(refs.ground_geoms)
    obstacle_geoms = set(refs.obstacle_geoms)
    driven_wheel_geoms = set(refs.wheel_geoms)
    support_allowed_geoms = driven_wheel_geoms | set(refs.guide_wheel_geoms)
    nonwheel_support_contacts = 0
    nonwheel_obstacle_contacts = 0
    for contact in data.contact[: data.ncon]:
        if contact.geom1 in support_geoms:
            nonwheel_support_contacts += contact.geom2 not in support_allowed_geoms
        elif contact.geom2 in support_geoms:
            nonwheel_support_contacts += contact.geom1 not in support_allowed_geoms
        if contact.geom1 in obstacle_geoms:
            nonwheel_obstacle_contacts += contact.geom2 not in driven_wheel_geoms
        elif contact.geom2 in obstacle_geoms:
            nonwheel_obstacle_contacts += contact.geom1 not in driven_wheel_geoms
    return nonwheel_support_contacts, nonwheel_obstacle_contacts


def passive_guide_dof_addresses(model: mujoco.MjModel, refs: ModelRefs) -> np.ndarray:
    """Return the scalar DOFs of validated passive guide hinges."""

    if not refs.guide_wheel_joints:
        return np.empty(0, dtype=np.int64)
    addresses = np.asarray(
        [int(model.jnt_dofadr[joint_id]) for joint_id in refs.guide_wheel_joints],
        dtype=np.int64,
    )
    if (
        np.any(addresses < 0)
        or np.any(addresses >= model.nv)
        or np.unique(addresses).size != addresses.size
    ):
        raise ValueError("guide-wheel passive hinge DOF addresses are invalid")
    return addresses


def lqr_state_indices(
    model: mujoco.MjModel,
    excluded_dof_addresses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select LQR states while leaving passive guide dynamics in physics.

    A passive roller angle is cyclic and unactuated.  It belongs in MuJoCo's
    state but not in the DARE system, where its uncontrollable unit mode makes
    the Riccati solve ill-conditioned.
    """

    excluded = np.asarray(excluded_dof_addresses, dtype=np.int64)
    if excluded.ndim != 1:
        raise ValueError("excluded LQR DOF addresses must be one-dimensional")
    if (
        np.any(excluded < 0)
        or np.any(excluded >= model.nv)
        or np.unique(excluded).size != excluded.size
    ):
        raise ValueError("excluded LQR DOF addresses are invalid")
    retained_dofs = np.setdiff1d(
        np.arange(model.nv, dtype=np.int64),
        excluded,
        assume_unique=True,
    )
    state_indices = np.concatenate(
        (
            retained_dofs,
            model.nv + retained_dofs,
            2 * model.nv + np.arange(model.na, dtype=np.int64),
        )
    )
    return retained_dofs, state_indices


class PhysicalLqr:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        refs: ModelRefs,
        speed: float,
        acceleration_limit: float,
        gas_spring_enabled: bool = True,
        max_forward_speed: float = MAX_FORWARD_SPEED_MPS,
        max_reverse_speed: float = MAX_REVERSE_SPEED_MPS,
    ) -> None:
        self.model = model
        self.refs = refs
        self.dt = float(model.opt.timestep)
        self.root_qpos = int(model.jnt_qposadr[refs.root_joint])
        self.root_dof = int(model.jnt_dofadr[refs.root_joint])
        self.gas_spring_enabled = gas_spring_enabled
        self.gas_spring_dofs = (
            int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_active_link_pitch")]),
            int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_active_link_pitch")]),
        )
        self.hip_actuator_ids = (refs.actuator_ids[0], refs.actuator_ids[1], refs.actuator_ids[3], refs.actuator_ids[4])
        self.hip_qpos_addresses = np.array((
            int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch")]),
            int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_active_link_pitch")]),
            int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch")]),
            int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_active_link_pitch")]),
        ))
        self.hip_dof_addresses = np.array((
            int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch")]),
            int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_active_link_pitch")]),
            int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch")]),
            int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_active_link_pitch")]),
        ))
        self.guide_wheel_dof_addresses = passive_guide_dof_addresses(model, refs)
        self.lqr_dof_addresses, self._lqr_state_indices = lqr_state_indices(
            model,
            self.guide_wheel_dof_addresses,
        )
        # Public immutable index contracts are carried into the CUDA fixed-gain
        # controller. Physics retains every passive guide state; the LQR gain
        # receives only the baseline controlled tangent state.
        self.controlled_dof_indices = self.lqr_dof_addresses.copy()
        self.controlled_state_indices = self._lqr_state_indices.copy()
        self._lqr_position_state_indices = np.full(model.nv, -1, dtype=np.int64)
        self._lqr_position_state_indices[self.lqr_dof_addresses] = np.arange(
            self.lqr_dof_addresses.size,
            dtype=np.int64,
        )
        self._lqr_velocity_state_indices = np.full(model.nv, -1, dtype=np.int64)
        self._lqr_velocity_state_indices[self.lqr_dof_addresses] = (
            self.lqr_dof_addresses.size + np.arange(self.lqr_dof_addresses.size, dtype=np.int64)
        )
        required_lqr_dofs = np.concatenate(
            (
                np.arange(self.root_dof, self.root_dof + 6, dtype=np.int64),
                self.hip_dof_addresses,
            )
        )
        if np.any(self._lqr_position_state_indices[required_lqr_dofs] < 0):
            raise ValueError("passive guide-wheel state exclusion overlaps a required LQR DOF")
        self.qpos_equilibrium = data.qpos.copy()
        self.qvel_equilibrium = np.zeros(model.nv)
        self.hip_qpos_equilibrium = self.qpos_equilibrium[self.hip_qpos_addresses].copy()
        self.control_equilibrium = self.solve_equilibrium(data)
        self.gain = self.linear_lqr(data)
        self._reference_qpos = self.qpos_equilibrium.copy()
        self._reference_qvel = self.qvel_equilibrium.copy()
        self._reference_control = self.control_equilibrium.copy()
        self._reference_gain = self.gain.copy()
        self._reference_hip_qpos = self.hip_qpos_equilibrium.copy()
        self._reference_shape = float(self.hip_qpos_equilibrium[0])
        self._reference_leg_length = self.average_leg_length(data)
        self.leg_profile: LegLengthProfile | None = None
        self.forward_speed_trim: SpeedTrim | None = None
        self.leg_command = LegLengthCommand(self._reference_leg_length, self._reference_leg_length)
        self._leg_shape_integral = 0.0
        # The framequat sensor follows the CAD inertial frame.  Use the actual
        # body kinematics and wheel axle to define the ground-plane heading.
        self._trim_root_quaternion = self.qpos_equilibrium[self.root_qpos + 3 : self.root_qpos + 7].copy()
        self._trim_heading_yaw = self.heading_yaw(data)
        # ``gain`` is identified in this world-frame heading.  Heading
        # commands may later rotate the free root, but the state supplied to
        # that fixed gain must remain expressed in this linearization frame.
        # Keep this immutable across reset_heading_command calls.
        self._linearization_heading_yaw = self._trim_heading_yaw
        self.heading_command = HeadingCommand(self._trim_heading_yaw, self._trim_heading_yaw)
        self._last_yaw_measurement = self._trim_heading_yaw
        self._last_yaw_measurement_time = float(data.time)
        self._measured_yaw_rate = 0.0
        self.max_forward_speed = validate_forward_speed_limit(max_forward_speed)
        self.max_reverse_speed = validate_reverse_speed_limit(max_reverse_speed)
        self.motion = MotionCommand(
            speed,
            acceleration_limit,
            max_forward_speed=self.max_forward_speed,
            max_reverse_speed=self.max_reverse_speed,
        )
        self._last_speed_report_time = -np.inf
        self._speed_report_pending = True
        self.speed_error_integral = 0.0
        self.jump = JumpSequence()
        self.drop = TerrainDropSequence()
        self._jump_pending = False
        self._jump_pending_since_s = -np.inf
        self._jump_pending_resume_length: float | None = None
        self._jump_pending_resume_speed: float | None = None
        self._jump_pending_launch_speed = 0.0
        self._jump_pending_stable_steps = 0
        self._contact_recovery_braking = False
        self.jump_rejection_reason = ""
        self._last_reference_time = float(data.time)
        # The LQR trims are intentionally built on a flat temporary support
        # plane.  A terrain caller can opt into an independent, filtered root
        # Z reference after it has placed the same physical stance on an
        # hfield.  No physical qpos is changed by this controller state.
        self._terrain_support_reference_enabled = False
        self._terrain_support_height_m = 0.0
        self._terrain_support_frozen_height_m = 0.0
        self._terrain_support_was_active = False
        self._terrain_support_root_z_offset_m = 0.0
        self._terrain_support_reference_z_m = float(self._reference_qpos[self.root_qpos + 2])
        self._terrain_support_filter_time_constant_s = TERRAIN_SUPPORT_REFERENCE_FILTER_TIME_CONSTANT_S
        self._terrain_support_max_vertical_rate_mps = TERRAIN_SUPPORT_REFERENCE_MAX_VERTICAL_RATE_MPS
        self._terrain_support_last_update_time = float(data.time)
        self._terrain_heading_stabilization_enabled = False
        # A high-speed terrain command is expressed as a yaw rate by the
        # locomotion interface.  Keep that explicit request separate from the
        # ordinary heading-only terrain loop so low-speed walking behavior is
        # unchanged.
        self._terrain_yaw_rate_command_active = False
        self._terrain_yaw_rate_command_rad_s = 0.0
        # Allocate once so delay compensation can predict the queued actuator
        # command without constructing MuJoCo state inside the control loop.
        self._delay_prediction_data = mujoco.MjData(self.model)

    def leg_lengths(self, data: mujoco.MjData) -> np.ndarray:
        return np.array((
            float(sensor(data, self.refs, "left_leg_length")[0]),
            float(sensor(data, self.refs, "right_leg_length")[0]),
        ))

    def leg_length_velocities(self, data: mujoco.MjData) -> np.ndarray:
        return np.array((
            float(sensor(data, self.refs, "left_leg_length_velocity")[0]),
            float(sensor(data, self.refs, "right_leg_length_velocity")[0]),
        ))

    def average_leg_length(self, data: mujoco.MjData) -> float:
        return float(np.mean(self.leg_lengths(data)))

    def forward_direction(self, data: mujoco.MjData) -> np.ndarray:
        """Return the current horizontal forward direction from the wheel axle."""
        body_rotation = data.xmat[self.refs.robot_body].reshape(3, 3)
        axle = body_rotation[:, 0]
        forward = np.cross(WORLD_UP, axle)
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-8:
            raise RuntimeError("Wheel axle is vertical; forward walking direction is singular.")
        return forward / forward_norm

    def heading_yaw(self, data: mujoco.MjData) -> float:
        """World yaw of the wheel rolling direction, measured counter-clockwise about +Z."""
        forward = self.forward_direction(data)
        return wrap_to_pi(float(np.arctan2(forward[1], forward[0])))

    def forward_speed(self, data: mujoco.MjData) -> float:
        world_velocity = sensor(data, self.refs, "world_horizontal_velocity_xy")
        forward = self.forward_direction(data)
        return float(np.dot(world_velocity, forward[: world_velocity.size]))

    def request_speed_report(self) -> None:
        """Request an immediate telemetry line on the next simulation tick."""
        self._speed_report_pending = True

    def speed_telemetry(self, data: mujoco.MjData) -> tuple[float, float, float, float, str]:
        target = float(self.motion.target_speed)
        ramped_command = float(self.motion.current_speed)
        measured = self.forward_speed(data)
        tracking_error = measured - ramped_command
        target_error = measured - target
        status = (
            "JUMP_BRAKING"
            if self.jump_pending
            else speed_tracking_status(
                target,
                ramped_command,
                measured,
                jump_active=self.jump.active,
            )
        )
        return target, ramped_command, measured, target_error, status

    def print_speed_telemetry(self, data: mujoco.MjData, *, force: bool = False) -> None:
        sim_time = float(data.time)
        if not force and not self._speed_report_pending and (
            sim_time - self._last_speed_report_time < SPEED_TELEMETRY_PERIOD_S
        ):
            return
        target, ramped_command, measured, target_error, status = self.speed_telemetry(data)
        tracking_error = measured - ramped_command
        print(
            f"Speed telemetry: t={sim_time:.2f}s command={target:.3f}m/s "
            f"ramp={ramped_command:.3f}m/s measured={measured:.3f}m/s "
            f"error={tracking_error:+.3f}m/s target_error={target_error:+.3f}m/s "
            f"status={status}",
            flush=True,
        )
        self._last_speed_report_time = sim_time
        self._speed_report_pending = False

    def configure_leg_profile(self, profile: LegLengthProfile, initial_length: float | None = None) -> None:
        self.leg_profile = profile
        requested_length = self._reference_leg_length if initial_length is None else initial_length
        current_length = profile.clamp_length(float(requested_length))
        self.leg_command = LegLengthCommand(current_length, current_length)
        self._leg_shape_integral = 0.0
        self._set_active_profile_reference(profile.shape_for_length(current_length))

    def configure_forward_speed_trim(self, trim: SpeedTrim) -> None:
        if trim.speed <= 0.0:
            raise ValueError("forward speed trim must have positive speed")
        self.forward_speed_trim = trim

    def configure_terrain_support_reference(
        self,
        data: mujoco.MjData,
        support_height_m: float,
        *,
        filter_time_constant_s: float = TERRAIN_SUPPORT_REFERENCE_FILTER_TIME_CONSTANT_S,
        maximum_vertical_rate_mps: float = TERRAIN_SUPPORT_REFERENCE_MAX_VERTICAL_RATE_MPS,
    ) -> float:
        """Enable terrain-following root-height reference from a spawned stance.

        ``support_height_m`` is the world Z height beneath the supporting
        wheels.  Call this after the environment has placed and forwarded the
        physical robot on terrain and selected its reset command/profile.  The
        current physical root Z is retained as the initial reference, while
        later profile changes retain their calibrated relative root-height
        motion.
        """
        support_height = float(support_height_m)
        filter_time_constant = float(filter_time_constant_s)
        maximum_vertical_rate = float(maximum_vertical_rate_mps)
        if not np.isfinite(support_height):
            raise ValueError("terrain support height must be finite")
        if not np.isfinite(filter_time_constant) or filter_time_constant <= 0.0:
            raise ValueError("terrain support filter time constant must be positive and finite")
        if not np.isfinite(maximum_vertical_rate) or maximum_vertical_rate <= 0.0:
            raise ValueError("terrain support maximum vertical rate must be positive and finite")

        root_z = float(data.qpos[self.root_qpos + 2])
        profile_root_z = float(self._reference_qpos[self.root_qpos + 2])
        self._terrain_support_reference_enabled = True
        self._terrain_support_height_m = support_height
        self._terrain_support_frozen_height_m = support_height
        self._terrain_support_was_active = True
        self._terrain_support_root_z_offset_m = root_z - profile_root_z - support_height
        self._terrain_support_reference_z_m = root_z
        self._terrain_support_filter_time_constant_s = filter_time_constant
        self._terrain_support_max_vertical_rate_mps = maximum_vertical_rate
        self._terrain_support_last_update_time = float(data.time)
        return root_z

    def update_terrain_support_reference(self, support_height_m: float) -> float:
        """Set the latest world-Z support height for an enabled terrain reference."""
        if not self._terrain_support_reference_enabled:
            raise RuntimeError(
                "terrain support reference is disabled; call configure_terrain_support_reference first"
            )
        support_height = float(support_height_m)
        if not np.isfinite(support_height):
            raise ValueError("terrain support height must be finite")
        self._terrain_support_height_m = support_height
        return support_height

    def rebase_terrain_support_reference(
        self,
        data: mujoco.MjData,
        support_height_m: float,
    ) -> float:
        """Rebase the filtered terrain reference after a validated step landing."""
        if not self._terrain_support_reference_enabled:
            raise RuntimeError(
                "terrain support reference is disabled; call configure_terrain_support_reference first"
            )
        support_height = float(support_height_m)
        if not np.isfinite(support_height):
            raise ValueError("terrain support height must be finite")
        root_z = float(data.qpos[self.root_qpos + 2])
        profile_root_z = float(self._reference_qpos[self.root_qpos + 2])
        self._terrain_support_height_m = support_height
        self._terrain_support_frozen_height_m = support_height
        self._terrain_support_root_z_offset_m = root_z - profile_root_z - support_height
        self._terrain_support_reference_z_m = root_z
        self._terrain_support_was_active = True
        self._terrain_support_last_update_time = float(data.time)
        return root_z

    def rebase_locomotion_reference(self, data: mujoco.MjData) -> None:
        """Remove the one-time horizontal LQR error at a validated step handoff."""
        self._reference_qpos[self.root_qpos : self.root_qpos + 2] = data.qpos[
            self.root_qpos : self.root_qpos + 2
        ]
        self._reference_qvel[self.root_dof : self.root_dof + 2] = data.qvel[
            self.root_dof : self.root_dof + 2
        ]

    def disable_terrain_support_reference(self) -> None:
        """Return to the original flat-trim root-height reference."""
        self._terrain_support_reference_enabled = False
        self._terrain_support_was_active = False

    def configure_terrain_heading_stabilization(self, enabled: bool) -> None:
        """Select the bounded terrain heading loop without changing yaw commands."""
        self._terrain_heading_stabilization_enabled = bool(enabled)
        if not self._terrain_heading_stabilization_enabled:
            self.set_terrain_yaw_rate_command(None)

    def set_terrain_yaw_rate_command(self, yaw_rate_rad_s: float | None) -> float:
        """Enable or clear bounded direct yaw-rate tracking on terrain.

        The caller supplies a high-level rate command only when terrain
        heading stabilization is active.  Heading error remains a small
        correction around that feedforward rate, rather than generating the
        entire turn through heading lag.
        """
        if yaw_rate_rad_s is None:
            self._terrain_yaw_rate_command_active = False
            self._terrain_yaw_rate_command_rad_s = 0.0
            return 0.0
        yaw_rate = float(yaw_rate_rad_s)
        if not np.isfinite(yaw_rate):
            raise ValueError("terrain yaw-rate command must be finite")
        self._terrain_yaw_rate_command_active = True
        self._terrain_yaw_rate_command_rad_s = float(np.clip(
            yaw_rate,
            -MAX_YAW_RATE_RAD_S,
            MAX_YAW_RATE_RAD_S,
        ))
        return self._terrain_yaw_rate_command_rad_s

    @property
    def terrain_support_reference_enabled(self) -> bool:
        return self._terrain_support_reference_enabled

    @property
    def terrain_heading_stabilization_enabled(self) -> bool:
        return self._terrain_heading_stabilization_enabled

    @property
    def terrain_yaw_rate_command_active(self) -> bool:
        return self._terrain_yaw_rate_command_active

    def _terrain_support_reference_active(self) -> bool:
        """Freeze terrain support only during airborne protected transitions."""
        return bool(
            self._terrain_support_reference_enabled
            and not self.jump.active
            and not self._jump_pending
            and not self._contact_recovery_braking
            and (not self.drop.active or self.drop.phase_name != "flight")
        )

    def _terrain_support_root_z_reference(
        self,
        data: mujoco.MjData,
        profile_root_z: float,
    ) -> float:
        """Return a rate-limited root Z target without changing physical state."""
        sim_time = float(data.time)
        if not self._terrain_support_reference_active():
            if not self._terrain_support_reference_enabled:
                return profile_root_z
            # Freeze the terrain elevation at the transition into a protected
            # phase, but retain the current leg-profile root offset.  Freezing
            # an absolute Z target here used to suppress the crouch/thrust/
            # flight root motion on RMUC, which reduced liftoff and amplified
            # asymmetric landing impulses.  The physical support is not
            # sampled while airborne; only the calibrated profile is allowed
            # to move this reference until normal walking resumes.
            if self._terrain_support_was_active:
                self._terrain_support_frozen_height_m = self._terrain_support_height_m
                self._terrain_support_was_active = False
            target_root_z = (
                float(profile_root_z)
                + self._terrain_support_frozen_height_m
                + self._terrain_support_root_z_offset_m
            )
            self._terrain_support_reference_z_m = target_root_z
            self._terrain_support_last_update_time = sim_time
            return target_root_z

        self._terrain_support_was_active = True

        target_root_z = (
            float(profile_root_z)
            + self._terrain_support_height_m
            + self._terrain_support_root_z_offset_m
        )
        elapsed = max(0.0, sim_time - self._terrain_support_last_update_time)
        if elapsed <= 0.0:
            return self._terrain_support_reference_z_m
        low_pass_gain = 1.0 - np.exp(-elapsed / self._terrain_support_filter_time_constant_s)
        filtered_root_z = self._terrain_support_reference_z_m + low_pass_gain * (
            target_root_z - self._terrain_support_reference_z_m
        )
        maximum_delta = self._terrain_support_max_vertical_rate_mps * elapsed
        self._terrain_support_reference_z_m += float(np.clip(
            filtered_root_z - self._terrain_support_reference_z_m,
            -maximum_delta,
            maximum_delta,
        ))
        self._terrain_support_last_update_time = sim_time
        return self._terrain_support_reference_z_m

    def _forward_speed_schedule_weight(self) -> float:
        if self.forward_speed_trim is None or self.motion.current_speed <= 0.0:
            return 0.0
        return float(np.clip(self.motion.current_speed / self.forward_speed_trim.speed, 0.0, 1.0))

    def _forward_speed_schedule_active(self) -> bool:
        return self._forward_speed_schedule_weight() > 0.0

    def _scheduled_control_and_gain(self) -> tuple[np.ndarray, np.ndarray]:
        weight = self._forward_speed_schedule_weight()
        if weight <= 0.0 or self.forward_speed_trim is None:
            return self._reference_control, self._reference_gain
        inverse = 1.0 - weight
        return (
            inverse * self._reference_control + weight * self.forward_speed_trim.control,
            inverse * self._reference_gain + weight * self.forward_speed_trim.gain,
        )

    def leg_length_limits(self, *, jump: bool = False) -> tuple[float, float]:
        if self.leg_profile is None:
            return WALK_LEG_LENGTH_MIN_M, WALK_LEG_LENGTH_MAX_M
        lower = max(WALK_LEG_LENGTH_MIN_M, self.leg_profile.minimum_length)
        upper = min(WALK_LEG_LENGTH_MAX_M, self.leg_profile.maximum_length)
        if jump:
            lower = max(JUMP_CROUCH_LENGTH_M, self.leg_profile.minimum_length)
            upper = min(JUMP_THRUST_LENGTH_M, self.leg_profile.maximum_length)
        return float(lower), float(upper)

    def set_target_leg_length(self, length: float, *, jump: bool = False) -> float:
        lower, upper = self.leg_length_limits(jump=jump)
        return self.leg_command.set_target(length, lower, upper)

    def adjust_target_leg_length(self, delta: float) -> float:
        return self.set_target_leg_length(self.leg_command.target_length + delta)

    def set_target_speed(self, speed: float) -> float:
        previous = self.motion.target_speed
        self.motion.set_target_speed(speed)
        self.request_speed_report()
        if previous * self.motion.target_speed < 0.0 or abs(self.motion.target_speed) < 1e-9:
            self.speed_error_integral = 0.0
        return self.motion.target_speed

    def set_jump_resume_speed(self, speed: float) -> float:
        """Update the requested post-jump speed without disturbing launch control.

        A high-level locomotion command is normally streamed at the policy or
        deployment rate.  While a jump is pending, blindly forwarding that
        speed to ``motion.target_speed`` would overwrite the temporary launch
        reference and prevent the braking gate from ever settling.  Keep the
        requested speed as the pending/active sequence's handoff target
        instead; normal walking commands still take effect immediately.
        """
        target = clamp_speed_command(
            speed,
            forward_limit=self.max_forward_speed,
            reverse_limit=self.max_reverse_speed,
        )
        if self._jump_pending:
            self._jump_pending_resume_speed = target
            return target
        if self.jump.active:
            self.jump.resume_speed = target
            return target
        if self.drop.active:
            self.drop.resume_speed = target
            return target
        return self.set_target_speed(target)

    def adjust_target_speed(self, delta: float) -> float:
        return self.set_target_speed(self.motion.target_speed + delta)

    @property
    def command_yaw(self) -> float:
        return self.heading_command.command_yaw

    def set_command_yaw(self, yaw: float) -> float:
        return self.heading_command.set_command(yaw)

    def adjust_command_yaw(self, delta: float) -> float:
        return self.set_command_yaw(self.heading_command.command_yaw + delta)

    def hold_current_yaw(self, data: mujoco.MjData) -> float:
        yaw = self.heading_yaw(data)
        self.heading_command = HeadingCommand(yaw, yaw, self.heading_command.maximum_rate)
        self.set_terrain_yaw_rate_command(None)
        return yaw

    def begin_contact_recovery(self, data: mujoco.MjData, leg_length: float) -> float:
        """Immediately switch from locomotion tracking to a two-wheel recovery trim."""
        self._contact_recovery_braking = True
        self.set_target_speed(0.0)
        self.motion.current_speed = 0.0
        self.speed_error_integral = 0.0
        self.hold_current_yaw(data)
        return self.set_target_leg_length(leg_length)

    def end_contact_recovery(self) -> None:
        """Return wheel-speed control to its regular full-support behavior."""
        self._contact_recovery_braking = False

    @property
    def jump_pending(self) -> bool:
        """Whether a requested jump is braking and waiting for a stable stance."""
        return self._jump_pending

    def jump_pending_elapsed(self, sim_time: float) -> float:
        if not self._jump_pending:
            return 0.0
        return max(0.0, float(sim_time) - self._jump_pending_since_s)

    def reset_heading_command(self, data: mujoco.MjData, yaw: float | None = None) -> float:
        measured_yaw = self.heading_yaw(data)
        self._trim_root_quaternion = data.qpos[self.root_qpos + 3 : self.root_qpos + 7].copy()
        self._trim_heading_yaw = measured_yaw
        target_yaw = measured_yaw if yaw is None else wrap_to_pi(yaw)
        self.heading_command = HeadingCommand(target_yaw, measured_yaw)
        self._last_yaw_measurement = measured_yaw
        self._last_yaw_measurement_time = float(data.time)
        self._measured_yaw_rate = 0.0
        self.set_terrain_yaw_rate_command(None)
        return target_yaw

    def reset_commands(
        self,
        data: mujoco.MjData,
        speed: float,
        leg_length: float | None = None,
        yaw: float | None = None,
    ) -> None:
        self.motion.current_speed = 0.0
        self.set_target_speed(speed)
        self.speed_error_integral = 0.0
        if leg_length is not None:
            self.set_target_leg_length(leg_length)
        self.leg_command.current_length = self.average_leg_length(data)
        if self.leg_profile is not None:
            self._set_active_profile_reference(
                self.leg_profile.shape_for_length(self.leg_command.current_length)
            )
        self._leg_shape_integral = 0.0
        self.jump = JumpSequence()
        self.drop = TerrainDropSequence()
        self._jump_pending = False
        self._jump_pending_since_s = -np.inf
        self._jump_pending_resume_length = None
        self._jump_pending_resume_speed = None
        self._jump_pending_launch_speed = 0.0
        self._jump_pending_stable_steps = 0
        self._contact_recovery_braking = False
        self.reset_heading_command(data, yaw)
        self._last_reference_time = float(data.time)
        self.request_speed_report()

    def apply_gas_spring_assist(self, data: mujoco.MjData) -> None:
        """Apply the extension-oriented gas-spring torque to both active links."""
        data.qfrc_applied[:] = 0.0
        if not self.gas_spring_enabled:
            return
        for dof_address in self.gas_spring_dofs:
            data.qfrc_applied[dof_address] = -GAS_SPRING_TORQUE_NM

    def request_jump(
        self,
        data: mujoco.MjData,
        *,
        launch_speed_mps: float | None = None,
    ) -> bool:
        """Queue a jump and regulate into the low-speed launch window.

        This is deliberately not a rejection path: a request stays pending until
        the robot can launch at a controlled low speed without inheriting a
        high-speed rolling transient.
        """
        if self.drop.active:
            self.jump_rejection_reason = "terrain drop supervisor active"
            return False
        if self.jump.active:
            # Do not interrupt airborne or landing stabilization with a second
            # key press.  The accepted request remains the active sequence.
            self.jump_rejection_reason = ""
            return True
        resume_speed = float(self.motion.target_speed)
        # Keep a small signed counter-command during the braking/settling
        # phase.  RMUC has a reproducible forward creep at a zero reference;
        # this produces a roughly 0.10 m/s rolling launch while preserving the
        # original high-level speed request for post-landing recovery.
        if launch_speed_mps is None:
            launch_speed = JUMP_LAUNCH_REFERENCE_SPEED_MPS
        else:
            launch_speed = float(launch_speed_mps)
            if not np.isfinite(launch_speed):
                raise ValueError("launch_speed_mps must be finite")
            launch_speed = float(np.clip(
                launch_speed,
                -JUMP_ROLLING_LAUNCH_MAX_SPEED_MPS,
                JUMP_ROLLING_LAUNCH_MAX_SPEED_MPS,
            ))
        # Phase remains None while the normal wheel PI is still active.  A
        # high-speed request is therefore braked into this low-speed launch
        # window, while an already slow robot continues to roll under control.
        self.set_target_speed(launch_speed)
        self.speed_error_integral = 0.0
        self.hold_current_yaw(data)
        self.jump.finish()
        self._jump_pending = True
        self._jump_pending_since_s = float(data.time)
        self._jump_pending_resume_length = float(self.leg_command.target_length)
        self._jump_pending_resume_speed = resume_speed
        self._jump_pending_launch_speed = launch_speed
        self._jump_pending_stable_steps = 0
        self.jump_rejection_reason = ""
        return True

    def request_terrain_drop(self, data: mujoco.MjData, drop_height_m: float) -> bool:
        """Arm a bounded local down-step transition from hfield-only evidence.

        The caller supplies a height difference derived from both wheel tracks,
        never a route, waypoint, or global map.  This accepts only the RMUC
        riser envelope and preserves the normal travel command until recovery.
        """
        drop_height = float(drop_height_m)
        if not np.isfinite(drop_height):
            raise ValueError("drop_height_m must be finite")
        if not TERRAIN_DROP_MIN_HEIGHT_M <= drop_height <= TERRAIN_DROP_MAX_HEIGHT_M:
            raise ValueError(
                "drop_height_m must be within the validated local terrain-drop envelope"
            )
        if self.drop.active:
            return True
        if self.jump.active or self._jump_pending:
            return False
        self.drop.start(
            float(data.time),
            float(self.leg_command.target_length),
            float(self.motion.target_speed),
            drop_height,
        )
        self.set_target_leg_length(DROP_PRELOAD_LENGTH_M, jump=True)
        self.speed_error_integral = 0.0
        self.jump_rejection_reason = ""
        return True

    def _jump_brake_ready(self, data: mujoco.MjData) -> bool:
        """Return whether the normal stance controller has removed launch momentum."""
        contacts = tuple(
            wheel_ground_contacts(data, self.refs, geom_id)
            for geom_id in self.refs.wheel_geoms
        )
        if not self.jump.grounded_with_hysteresis(contacts):
            return False
        forward_speed = self.forward_speed(data)
        vertical_speed = abs(float(data.qvel[self.root_dof + 2]))
        angular_speed = float(np.linalg.norm(data.qvel[self.root_dof + 3 : self.root_dof + 6]))
        launch_speed = self._jump_pending_launch_speed
        if abs(launch_speed) > JUMP_MAX_LAUNCH_SPEED_MPS:
            speed_ready = abs(forward_speed - launch_speed) <= JUMP_ROLLING_LAUNCH_TOLERANCE_MPS
        else:
            speed_ready = abs(forward_speed) <= JUMP_MAX_LAUNCH_SPEED_MPS + JUMP_LAUNCH_SPEED_TOLERANCE_MPS
        return bool(
            speed_ready
            and vertical_speed <= JUMP_BRAKE_MAX_VERTICAL_SPEED_MPS
            and angular_speed <= JUMP_BRAKE_MAX_ANGULAR_SPEED_RAD_S
        )

    def _update_pending_jump(self, data: mujoco.MjData) -> None:
        """Promote a queued jump only after normal rolling control has stopped it."""
        if not self._jump_pending:
            return
        if self._jump_brake_ready(data):
            self._jump_pending_stable_steps += 1
        else:
            self._jump_pending_stable_steps = 0
        required_steps = max(1, int(np.ceil(JUMP_BRAKE_STABLE_SECONDS / self.dt)))
        if self._jump_pending_stable_steps < required_steps:
            return
        resume_length = self._jump_pending_resume_length
        if resume_length is None:
            resume_length = self.leg_command.target_length
        resume_speed = self._jump_pending_resume_speed
        self.motion.current_speed = self._jump_pending_launch_speed
        self.speed_error_integral = 0.0
        self.jump.start(
            float(data.time),
            float(resume_length),
            resume_speed,
            self._jump_pending_launch_speed,
        )
        self.set_target_leg_length(JUMP_PREPARE_LENGTH_M, jump=True)
        self._jump_pending = False
        self._jump_pending_since_s = -np.inf
        self._jump_pending_resume_length = None
        self._jump_pending_resume_speed = None
        self._jump_pending_launch_speed = 0.0
        self._jump_pending_stable_steps = 0
        print(f"Jump brake settled: t={float(data.time):.3f}s", flush=True)

    def _set_active_profile_reference(self, shape: float) -> None:
        if self.leg_profile is None:
            self._reference_shape = shape
            return
        lower, upper, interpolation = self.leg_profile.bracket(shape)
        inverse = 1.0 - interpolation
        self._reference_qpos[:] = inverse * lower.qpos + interpolation * upper.qpos
        self._reference_qvel[:] = inverse * lower.qvel + interpolation * upper.qvel
        self._reference_control[:] = inverse * lower.control + interpolation * upper.control
        self._reference_gain[:] = inverse * lower.gain + interpolation * upper.gain
        self._reference_hip_qpos[:] = inverse * lower.hip_qpos + interpolation * upper.hip_qpos
        self._reference_leg_length = inverse * lower.leg_length + interpolation * upper.leg_length
        self._reference_shape = float(shape)

    def _advance_leg_reference(self, data: mujoco.MjData, maximum_rate: float) -> None:
        self.leg_command.advance(self.dt, maximum_rate)
        if self.leg_profile is None:
            return
        measured_length = self.average_leg_length(data)
        length_error = self.leg_command.current_length - measured_length
        self._leg_shape_integral = float(np.clip(
            self._leg_shape_integral + length_error * self.dt,
            -LEG_LENGTH_SHAPE_INTEGRAL_LIMIT_M_S,
            LEG_LENGTH_SHAPE_INTEGRAL_LIMIT_M_S,
        ))
        nominal_shape = self.leg_profile.shape_for_length(self.leg_command.current_length)
        correction = float(np.clip(
            -LEG_LENGTH_SHAPE_KP_RAD_PER_M * length_error
            -LEG_LENGTH_SHAPE_KI_RAD_PER_M_S * self._leg_shape_integral,
            -LEG_LENGTH_SHAPE_MAX_CORRECTION_RAD,
            LEG_LENGTH_SHAPE_MAX_CORRECTION_RAD,
        ))
        self._set_active_profile_reference(nominal_shape + correction)

    def _jump_rate_limit(self) -> float:
        phase = self.jump.phase_name
        if phase == "prepare":
            return JUMP_PREPARE_RATE_MPS
        if phase == "crouch":
            return JUMP_CROUCH_RATE_MPS
        if phase == "thrust":
            return JUMP_THRUST_RATE_MPS
        if phase == "flight":
            if self.jump.impact_active:
                return JUMP_LANDING_RATE_MPS
            return (
                JUMP_FLIGHT_RETRACT_RATE_MPS
                if self.jump.flight_steps < JUMP_FLIGHT_RETRACT_STEPS
                else JUMP_FLIGHT_PRELOAD_RATE_MPS
            )
        if phase == "landing":
            return (
                JUMP_RECOVERY_LEG_RATE_MPS
                if self.jump.recovering
                else JUMP_LANDING_RATE_MPS
            )
        return WALK_LEG_LENGTH_RATE_MPS

    def _drop_rate_limit(self, phase: str | None) -> float:
        if phase == "preload":
            return DROP_PRELOAD_RATE_MPS
        if phase == "flight":
            return DROP_FLIGHT_EXTENSION_RATE_MPS
        if phase == "landing":
            return DROP_LANDING_RATE_MPS
        if phase == "recovery":
            return DROP_RECOVERY_LEG_RATE_MPS
        return WALK_LEG_LENGTH_RATE_MPS

    def _legs_near_target(self, data: mujoco.MjData, target: float) -> bool:
        lengths = self.leg_lengths(data)
        return bool(np.max(np.abs(lengths - target)) <= JUMP_LENGTH_TOLERANCE_M)

    def _jump_recovery_body_is_stable(
        self,
        data: mujoco.MjData,
        contacts: tuple[int, int],
    ) -> bool:
        """Gate the handoff back to walking on a quiet, two-wheel stance."""
        if not all(contact_count > 0 for contact_count in contacts):
            return False
        attitude_error = self.airborne_attitude_error(data)
        pitch_error = abs(float(attitude_error[0]))
        pitch_rate = abs(float(data.qvel[self.root_dof + 3]))
        vertical_speed = abs(float(data.qvel[self.root_dof + 2]))
        forward_speed = abs(self.forward_speed(data))
        leg_difference = float(np.ptp(self.leg_lengths(data)))
        return bool(
            pitch_error <= JUMP_RECOVERY_MAX_PITCH_ERROR_RAD
            and pitch_rate <= JUMP_RECOVERY_MAX_PITCH_RATE_RAD_S
            and vertical_speed <= JUMP_RECOVERY_MAX_VERTICAL_SPEED_MPS
            and forward_speed <= JUMP_RECOVERY_MAX_FORWARD_SPEED_MPS
            and leg_difference <= JUMP_RECOVERY_MAX_LEG_DIFFERENCE_M
        )

    def _drop_recovery_body_is_stable(
        self,
        data: mujoco.MjData,
        grounded: bool,
    ) -> bool:
        """Use the jump landing checks with a bounded rolling speed allowance."""
        if not grounded:
            return False
        attitude_error = self.airborne_attitude_error(data)
        pitch_error = abs(float(attitude_error[0]))
        pitch_rate = abs(float(data.qvel[self.root_dof + 3]))
        vertical_speed = abs(float(data.qvel[self.root_dof + 2]))
        forward_speed = abs(self.forward_speed(data))
        leg_difference = float(np.ptp(self.leg_lengths(data)))
        return bool(
            pitch_error <= DROP_RECOVERY_MAX_PITCH_ERROR_RAD
            and pitch_rate <= DROP_RECOVERY_MAX_PITCH_RATE_RAD_S
            and vertical_speed <= DROP_RECOVERY_MAX_VERTICAL_SPEED_MPS
            and forward_speed <= DROP_RECOVERY_MAX_FORWARD_SPEED_MPS
            and leg_difference <= DROP_RECOVERY_MAX_LEG_DIFFERENCE_M
        )

    def _abort_jump(self, data: mujoco.MjData, reason: str) -> None:
        self.jump.abort_reason = reason
        self.jump.transition("landing", float(data.time))
        self.set_target_leg_length(JUMP_LANDING_LENGTH_M, jump=True)
        print(f"Jump recovery: {reason}")

    def _update_jump_state(self, data: mujoco.MjData) -> str | None:
        phase = self.jump.phase_name
        if phase is None:
            return None
        sim_time = float(data.time)
        contacts = tuple(
            wheel_ground_contacts(data, self.refs, geom_id)
            for geom_id in self.refs.wheel_geoms
        )
        grounded = self.jump.grounded_with_hysteresis(contacts)
        if phase == "prepare":
            self.set_target_leg_length(JUMP_PREPARE_LENGTH_M, jump=True)
            self.jump.settled_steps = self.jump.settled_steps + 1 if grounded and self._legs_near_target(data, JUMP_PREPARE_LENGTH_M) else 0
            if self.jump.settled_steps >= JUMP_SETTLE_STEPS:
                self.jump.transition("crouch", sim_time)
                self.set_target_leg_length(JUMP_CROUCH_LENGTH_M, jump=True)
            elif self.jump.elapsed(sim_time) > JUMP_PREPARE_TIMEOUT_S:
                self._abort_jump(data, "prepare timeout")
        elif phase == "crouch":
            self.set_target_leg_length(JUMP_CROUCH_LENGTH_M, jump=True)
            self.jump.settled_steps = self.jump.settled_steps + 1 if grounded and self._legs_near_target(data, JUMP_CROUCH_LENGTH_M) else 0
            if self.jump.settled_steps >= JUMP_SETTLE_STEPS:
                self.jump.transition("thrust", sim_time)
                self.set_target_leg_length(JUMP_THRUST_LENGTH_M, jump=True)
                self.speed_error_integral = 0.0
            elif self.jump.elapsed(sim_time) > JUMP_CROUCH_TIMEOUT_S:
                self._abort_jump(data, "crouch timeout")
        elif phase == "thrust":
            self.set_target_leg_length(JUMP_THRUST_LENGTH_M, jump=True)
            self.jump.airborne_steps = self.jump.airborne_steps + 1 if not grounded else 0
            if self.jump.airborne_steps >= JUMP_LIFTOFF_STEPS:
                self.jump.transition("flight", sim_time)
                self.set_target_leg_length(JUMP_FLIGHT_RETRACT_LENGTH_M, jump=True)
                print(f"Jump liftoff: t={sim_time:.3f}s")
            elif self.jump.elapsed(sim_time) > JUMP_THRUST_TIMEOUT_S:
                self._abort_jump(data, "thrust timeout without liftoff")
        elif phase == "flight":
            self.jump.flight_steps += 1
            target_length = JUMP_LANDING_LENGTH_M if self.jump.impact_active else (
                JUMP_FLIGHT_RETRACT_LENGTH_M
                if self.jump.flight_steps < JUMP_FLIGHT_RETRACT_STEPS
                else JUMP_FLIGHT_PRELOAD_LENGTH_M
            )
            self.set_target_leg_length(target_length, jump=True)
            vertical_speed = float(data.qvel[self.root_dof + 2])
            if (
                not self.jump.impact_active
                and self.jump.flight_steps >= JUMP_FLIGHT_RETRACT_STEPS
                and any(contact_count > 0 for contact_count in contacts)
                and vertical_speed <= -JUMP_IMPACT_MIN_DESCENT_SPEED_MPS
            ):
                self.jump.begin_impact(sim_time, contacts)
                self.set_target_leg_length(JUMP_LANDING_LENGTH_M, jump=True)
            if self.jump.impact_active:
                self.jump.update_impact_leg_difference(
                    float(np.ptp(self.leg_lengths(data)))
                )
            # Require actual, consecutive two-wheel contacts before moving to
            # landing. ``grounded`` is intentionally hysteretic for liftoff,
            # but it cannot be reused here because the counter was reset when
            # the flight phase began.
            landed = all(contact_count > 0 for contact_count in contacts)
            self.jump.settled_steps = self.jump.settled_steps + 1 if landed else 0
            if self.jump.settled_steps >= JUMP_LANDING_CONTACT_CONFIRM_STEPS:
                self.jump.transition("landing", sim_time)
                self.set_target_leg_length(JUMP_LANDING_LENGTH_M, jump=True)
            elif self.jump.elapsed(sim_time) > JUMP_FLIGHT_TIMEOUT_S:
                self._abort_jump(data, "flight timeout")
        elif phase == "landing":
            resume_length = self.jump.resume_length
            if self.jump.recovering:
                # Keep the jump controller in charge while the compliant
                # landing target returns to the requested walking length.  A
                # direct finish here used to restore the full walking trim in
                # one tick, causing a second bounce after an otherwise stable
                # high jump.
                target_length = (
                    JUMP_LANDING_LENGTH_M
                    if resume_length is None
                    else float(resume_length)
                )
                recovery_body_stable = self._jump_recovery_body_is_stable(data, contacts)
                if recovery_body_stable:
                    self.set_target_leg_length(target_length)
                    self.jump.recovery_stable_steps = (
                        self.jump.recovery_stable_steps + 1
                        if self._legs_near_target(data, target_length)
                        else 0
                    )
                else:
                    # Arrest the handoff on the impact target whenever the
                    # body is still pitching, bouncing or unloading a wheel.
                    # This keeps the closed-chain links synchronized instead
                    # of forcing them toward the walking target mid-impact.
                    self.set_target_leg_length(JUMP_LANDING_LENGTH_M, jump=True)
                    self.jump.recovery_stable_steps = 0
                required_steps = max(1, int(np.ceil(JUMP_RECOVERY_STABLE_SECONDS / self.dt)))
                if self.jump.recovery_stable_steps >= required_steps:
                    resume_speed = self.jump.resume_speed
                    self.jump.finish()
                    if resume_speed is not None:
                        self.set_target_speed(resume_speed)
                    self.speed_error_integral = 0.0
                elif (
                    self.jump.recovery_start_time is not None
                    and sim_time - self.jump.recovery_start_time > JUMP_RECOVERY_TIMEOUT_S
                ):
                    self.jump.abort_reason = "landing recovery timeout without stable contact"
                    resume_speed = self.jump.resume_speed
                    self.jump.finish()
                    if resume_speed is not None:
                        self.set_target_speed(resume_speed)
                    self.speed_error_integral = 0.0
            else:
                self.set_target_leg_length(JUMP_LANDING_LENGTH_M, jump=True)
                self.jump.settled_steps = (
                    self.jump.settled_steps + 1
                    if grounded and self._legs_near_target(data, JUMP_LANDING_LENGTH_M)
                    else 0
                )
                if self.jump.settled_steps >= JUMP_SETTLE_STEPS:
                    self.jump.begin_recovery(sim_time)
                    self.motion.current_speed = 0.0
                    self.set_target_speed(0.0)
                    self.speed_error_integral = 0.0
                    self.hold_current_yaw(data)
                    if resume_length is not None:
                        self.set_target_leg_length(float(resume_length))
                elif self.jump.elapsed(sim_time) > JUMP_LANDING_TIMEOUT_S:
                    self.jump.abort_reason = "landing timeout without stable contact"
                    resume_speed = self.jump.resume_speed
                    self.jump.finish()
                    if resume_length is not None:
                        self.set_target_leg_length(float(resume_length))
                    if resume_speed is not None:
                        self.set_target_speed(resume_speed)
                    self.speed_error_integral = 0.0
        return self.jump.phase_name

    def _abort_drop(self, data: mujoco.MjData, reason: str) -> None:
        self.drop.abort_reason = reason
        self.drop.transition("landing", float(data.time))
        self.set_target_leg_length(DROP_LANDING_LENGTH_M, jump=True)
        print(f"Terrain drop recovery: {reason}", flush=True)

    def _finish_drop(self, data: mujoco.MjData) -> None:
        """Return the nominal locomotion command after a stable down-step."""
        resume_length = self.drop.resume_length
        resume_speed = self.drop.resume_speed
        if self._terrain_support_reference_enabled:
            # During flight the hfield sample can already switch to the lower
            # surface, making the old height-delta condition appear false by
            # recovery completion.  Rebase unconditionally from the settled
            # physical state so the walking LQR never retains a stale root-Z
            # target after a real step-down.
            self.rebase_terrain_support_reference(data, self._terrain_support_height_m)
            self.rebase_locomotion_reference(data)
        self.drop.finish()
        if resume_length is not None:
            self.set_target_leg_length(float(resume_length))
        if resume_speed is not None:
            self.set_target_speed(float(resume_speed))
        self.speed_error_integral = 0.0

    def _update_drop_state(self, data: mujoco.MjData) -> str | None:
        """Advance the local down-step state without invoking jump semantics."""
        phase = self.drop.phase_name
        if phase is None:
            return None
        sim_time = float(data.time)
        contacts = tuple(
            wheel_ground_contacts(data, self.refs, geom_id)
            for geom_id in self.refs.wheel_geoms
        )
        full_support = all(contact_count > 0 for contact_count in contacts)
        grounded = self.drop.grounded_with_hysteresis(contacts)
        if phase == "preload":
            self.set_target_leg_length(DROP_PRELOAD_LENGTH_M, jump=True)
            self.drop.unloaded_steps = (
                self.drop.unloaded_steps + 1 if not any(contacts) else 0
            )
            if self.drop.unloaded_steps >= DROP_FLIGHT_ENTRY_CONFIRM_STEPS:
                self.drop.transition("flight", sim_time)
                self.set_target_leg_length(DROP_FLIGHT_EXTENSION_LENGTH_M, jump=True)
                print(
                    f"Terrain drop flight: t={sim_time:.3f}s height={self.drop.drop_height_m:.3f}m",
                    flush=True,
                )
            elif self.drop.elapsed(sim_time) > DROP_PRELOAD_TIMEOUT_S:
                # The local preview can see a feature on a parallel lane that
                # the wheels never cross.  Restore walking rather than holding
                # a widened stance indefinitely.
                self._finish_drop(data)
        elif phase == "flight":
            self.drop.flight_steps += 1
            self.set_target_leg_length(DROP_FLIGHT_EXTENSION_LENGTH_M, jump=True)
            self.drop.settled_steps = self.drop.settled_steps + 1 if full_support else 0
            if self.drop.settled_steps >= DROP_LANDING_CONTACT_CONFIRM_STEPS:
                self.drop.transition("landing", sim_time)
                self.set_target_leg_length(DROP_LANDING_LENGTH_M, jump=True)
            elif self.drop.elapsed(sim_time) > DROP_FLIGHT_TIMEOUT_S:
                self._abort_drop(data, "flight timeout without two-wheel landing")
        elif phase == "landing":
            self.set_target_leg_length(DROP_LANDING_LENGTH_M, jump=True)
            self.drop.settled_steps = (
                self.drop.settled_steps + 1
                if full_support and self._legs_near_target(data, DROP_LANDING_LENGTH_M)
                else 0
            )
            if self.drop.settled_steps >= DROP_LANDING_SETTLE_STEPS:
                self.drop.begin_recovery(sim_time)
                self.motion.current_speed = 0.0
                self.set_target_speed(0.0)
                self.speed_error_integral = 0.0
                self.hold_current_yaw(data)
            elif self.drop.elapsed(sim_time) > DROP_LANDING_TIMEOUT_S:
                self._abort_drop(data, "landing timeout without stable contact")
        elif phase == "recovery":
            resume_length = self.drop.resume_length
            target_length = (
                DROP_LANDING_LENGTH_M if resume_length is None else float(resume_length)
            )
            if self._drop_recovery_body_is_stable(data, grounded):
                self.set_target_leg_length(target_length)
                self.drop.recovery_stable_steps = (
                    self.drop.recovery_stable_steps + 1
                    if self._legs_near_target(data, target_length)
                    else 0
                )
            else:
                self.set_target_leg_length(DROP_LANDING_LENGTH_M, jump=True)
                self.drop.recovery_stable_steps = 0
            required_steps = max(1, int(np.ceil(DROP_RECOVERY_STABLE_SECONDS / self.dt)))
            if self.drop.recovery_stable_steps >= required_steps:
                self._finish_drop(data)
            elif (
                self.drop.recovery_start_time is not None
                and sim_time - self.drop.recovery_start_time > DROP_RECOVERY_TIMEOUT_S
            ):
                self._abort_drop(data, "landing recovery timeout without stable contact")
        return self.drop.phase_name

    def solve_equilibrium(self, data: mujoco.MjData) -> np.ndarray:
        """Find the real motor torques that balance the assembled model."""
        data.qpos[:] = self.qpos_equilibrium
        data.qvel[:] = self.qvel_equilibrium
        data.ctrl[:] = 0.0
        self.apply_gas_spring_assist(data)
        mujoco.mj_forward(self.model, data)
        data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, data)
        required_force = data.qfrc_inverse.copy() - data.qfrc_applied
        drive_matrix = self.actuator_drive_matrix(data)
        command, _, _, _ = np.linalg.lstsq(drive_matrix, required_force, rcond=None)
        command = np.clip(command, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        data.ctrl[:] = command
        mujoco.mj_forward(self.model, data)
        return command.copy()

    def actuator_drive_matrix(self, data: mujoco.MjData) -> np.ndarray:
        """Measure each real actuator's generalized force at the trim state."""
        original_control = data.ctrl.copy()
        data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, data)
        zero_control_force = data.qfrc_actuator.copy()
        drive_matrix = np.zeros((self.model.nv, self.model.nu))
        for actuator_id in range(self.model.nu):
            data.ctrl[:] = 0.0
            data.ctrl[actuator_id] = 1.0
            mujoco.mj_forward(self.model, data)
            drive_matrix[:, actuator_id] = data.qfrc_actuator - zero_control_force
        data.ctrl[:] = original_control
        mujoco.mj_forward(self.model, data)
        return drive_matrix

    def linear_lqr(self, data: mujoco.MjData) -> np.ndarray:
        full_state_size = 2 * self.model.nv + self.model.na
        a_matrix_full = np.zeros((full_state_size, full_state_size))
        b_matrix_full = np.zeros((full_state_size, self.model.nu))
        c_matrix_full = np.zeros((self.model.nsensordata, full_state_size))
        d_matrix = np.zeros((self.model.nsensordata, self.model.nu))
        mujoco.mjd_transitionFD(
            self.model,
            data,
            LINEARIZATION_EPSILON,
            1,
            a_matrix_full,
            b_matrix_full,
            c_matrix_full,
            d_matrix,
        )
        state_indices = self._lqr_state_indices
        a_matrix = a_matrix_full[np.ix_(state_indices, state_indices)]
        b_matrix = b_matrix_full[state_indices, :]
        c_matrix = c_matrix_full[:, state_indices]
        q_matrix = np.eye(state_indices.size)
        root_position_indices = self._lqr_position_state_indices[
            self.root_dof : self.root_dof + 6
        ]
        root_velocity_indices = self._lqr_velocity_state_indices[
            self.root_dof : self.root_dof + 6
        ]
        q_matrix[root_position_indices[:3], root_position_indices[:3]] *= 40.0
        q_matrix[root_position_indices[3:], root_position_indices[3:]] *= 900.0
        q_matrix[root_velocity_indices[:3], root_velocity_indices[:3]] *= 80.0
        q_matrix[root_velocity_indices[3:], root_velocity_indices[3:]] *= 200.0
        for sensor_name, weight in (
            ("left_leg_length", LEG_LENGTH_LQR_WEIGHT),
            ("right_leg_length", LEG_LENGTH_LQR_WEIGHT),
            ("left_leg_length_velocity", LEG_LENGTH_VELOCITY_LQR_WEIGHT),
            ("right_leg_length_velocity", LEG_LENGTH_VELOCITY_LQR_WEIGHT),
        ):
            sensor_address, _ = self.refs.sensor_refs[sensor_name]
            output_row = c_matrix[sensor_address]
            q_matrix += weight * np.outer(output_row, output_row)
        for dof_address in self.hip_dof_addresses:
            position_index = self._lqr_position_state_indices[dof_address]
            velocity_index = self._lqr_velocity_state_indices[dof_address]
            q_matrix[position_index, position_index] += WALK_STANCE_HIP_LQR_WEIGHT
            q_matrix[velocity_index, velocity_index] += WALK_STANCE_HIP_VELOCITY_LQR_WEIGHT
        r_matrix = np.diag((12.0, 12.0, 0.08, 12.0, 12.0, 0.08))
        try:
            solution = scipy.linalg.solve_discrete_are(a_matrix, b_matrix, q_matrix, r_matrix)
        except scipy.linalg.LinAlgError as error:
            rank = np.linalg.matrix_rank(b_matrix)
            raise RuntimeError(
                f"LQR Riccati solve failed at the two-wheel contact working point "
                f"(input rank={rank}/{self.model.nu}). Check actuator limits, contact geometry, and mass/inertia calibration."
            ) from error
        self.sensor_state_jacobian = c_matrix.copy()
        return np.linalg.solve(r_matrix + b_matrix.T @ solution @ b_matrix, b_matrix.T @ solution @ a_matrix)

    def _advance_motion_reference(self, data: mujoco.MjData) -> None:
        sim_time = float(data.time)
        elapsed = max(0.0, sim_time - self._last_reference_time)
        self.motion.advance(elapsed)
        self.heading_command.advance(elapsed)
        self._last_reference_time = sim_time

    def _update_measured_yaw_rate(self, data: mujoco.MjData) -> None:
        current_yaw = self.heading_yaw(data)
        current_time = float(data.time)
        elapsed = current_time - self._last_yaw_measurement_time
        if elapsed > 0.0:
            raw_rate = wrap_to_pi(current_yaw - self._last_yaw_measurement) / elapsed
            filter_gain = elapsed / (YAW_RATE_FILTER_TIME_CONSTANT_S + elapsed)
            self._measured_yaw_rate += filter_gain * (raw_rate - self._measured_yaw_rate)
        self._last_yaw_measurement = current_yaw
        self._last_yaw_measurement_time = current_time

    def _heading_reference_quaternion(self, yaw: float) -> np.ndarray:
        """Keep local LQR attitude feedback aligned with the current heading."""
        yaw_delta = wrap_to_pi(yaw - self._trim_heading_yaw)
        yaw_rotation = np.array((np.cos(0.5 * yaw_delta), 0.0, 0.0, np.sin(0.5 * yaw_delta)))
        reference_quaternion = np.empty(4)
        mujoco.mju_mulQuat(reference_quaternion, yaw_rotation, self._trim_root_quaternion)
        return reference_quaternion

    def reference_state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        qpos_reference = self._reference_qpos.copy()
        qvel_reference = self._reference_qvel.copy()
        # Do not impose an absolute horizontal position or wheel angle while driving.
        qpos_reference[self.root_qpos : self.root_qpos + 2] = data.qpos[self.root_qpos : self.root_qpos + 2]
        if self._terrain_support_reference_enabled:
            qpos_reference[self.root_qpos + 2] = self._terrain_support_root_z_reference(
                data,
                float(qpos_reference[self.root_qpos + 2]),
            )
        qpos_reference[self.root_qpos + 3 : self.root_qpos + 7] = self._heading_reference_quaternion(
            self.heading_yaw(data)
        )
        for joint_id in self.refs.wheel_joints:
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            qpos_reference[qpos_address] = data.qpos[qpos_address]
        speed_weight = self._forward_speed_schedule_weight()
        if speed_weight > 0.0 and self.forward_speed_trim is not None:
            qvel_reference[self.root_dof : self.root_dof + 3] = (
                self.forward_direction(data) * speed_weight * self.forward_speed_trim.speed
            )
            for joint_id in self.refs.wheel_joints:
                dof_address = int(self.model.jnt_dofadr[joint_id])
                qvel_reference[dof_address] = speed_weight * self.forward_speed_trim.qvel[dof_address]
        elif self.jump.phase_name != "flight":
            # Contact LQR has one stable velocity branch.  The outer PI chooses
            # physical travel sign using the measured world-frame velocity.
            # Keep the linearised reference near its validated trim; the wheel
            # PI governs all speed above this small-signal branch.
            lqr_speed_reference = min(
                abs(self.motion.current_speed),
                LQR_FORWARD_SPEED_REFERENCE_LIMIT_MPS,
            )
            qvel_reference[self.root_dof : self.root_dof + 3] = (
                -self.forward_direction(data) * lqr_speed_reference
            )
        return qpos_reference, qvel_reference

    def state_error(self, data: mujoco.MjData, sim_time: float | None = None) -> np.ndarray:
        del sim_time
        qpos_reference, qvel_reference = self.reference_state(data)
        position_error = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, position_error, 1.0, qpos_reference, data.qpos)
        velocity_error = data.qvel - qvel_reference
        if not self._forward_speed_schedule_active():
            # The contact LQR is linearised at walking trim.  Let the outer wheel
            # PI own high forward speed instead of feeding a multi-m/s error back
            # through a small-signal gain designed around zero velocity.
            forward = self.forward_direction(data)
            root_velocity_error = velocity_error[self.root_dof : self.root_dof + 3]
            forward_error = float(np.dot(root_velocity_error, forward))
            root_velocity_error += forward * (
                float(np.clip(
                    forward_error,
                    -LQR_FORWARD_SPEED_FEEDBACK_LIMIT_MPS,
                    LQR_FORWARD_SPEED_FEEDBACK_LIMIT_MPS,
                ))
                - forward_error
            )

        # The root's horizontal free-joint coordinates are world-frame, while
        # the fixed LQR gain was obtained at ``_linearization_heading_yaw``.
        # Express only those translational error channels in the original
        # linearization frame.  Joint, vertical and angular channels remain in
        # their native coordinates.  This preserves yaw-equivariance without
        # relinearizing on a non-smooth hfield contact at every terrain reset.
        heading_delta = wrap_to_pi(self.heading_yaw(data) - self._linearization_heading_yaw)
        cosine = float(np.cos(heading_delta))
        sine = float(np.sin(heading_delta))
        for horizontal_error in (
            position_error[self.root_dof : self.root_dof + 2],
            velocity_error[self.root_dof : self.root_dof + 2],
        ):
            world_x, world_y = float(horizontal_error[0]), float(horizontal_error[1])
            horizontal_error[:] = (
                cosine * world_x + sine * world_y,
                -sine * world_x + cosine * world_y,
            )
        full_error = np.concatenate((position_error, velocity_error))
        return full_error[self._lqr_state_indices]

    def apply_walking_stance_guard(self, data: mujoco.MjData, command: np.ndarray) -> None:
        """Track the scheduled closed-chain branch without changing MJCF topology."""
        position_error = self._reference_hip_qpos - data.qpos[self.hip_qpos_addresses]
        velocity_error = -data.qvel[self.hip_dof_addresses]
        command[list(self.hip_actuator_ids)] += (
            WALK_STANCE_GUARD_KP_NM_PER_RAD * position_error
            + WALK_STANCE_GUARD_KD_NM_PER_RAD_PER_S * velocity_error
        )

    def apply_leg_length_force(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
        force_limit: float,
    ) -> None:
        """Apply constrained virtual-work leg forces through the four hip motors."""
        if self.leg_profile is None:
            return
        lengths = self.leg_lengths(data)
        velocities = self.leg_length_velocities(data)
        desired_rate = self.leg_command.current_rate
        forces = np.clip(
            LEG_LENGTH_FORCE_KP_N_PER_M * (self.leg_command.current_length - lengths)
            + LEG_LENGTH_FORCE_KD_NS_PER_M * (desired_rate - velocities),
            -force_limit,
            force_limit,
        )
        synchronize_drop_legs = self.drop.phase_name in ("flight", "landing", "recovery")
        if self.jump.phase_name in ("thrust", "flight", "landing") or synchronize_drop_legs:
            leg_difference = float(lengths[0] - lengths[1])
            differential_velocity = float(velocities[0] - velocities[1])
            synchronization_force = float(np.clip(
                JUMP_LEG_SYNC_KP_N_PER_M * leg_difference
                + JUMP_LEG_SYNC_KD_NS_PER_M * differential_velocity,
                -JUMP_LEG_SYNC_FORCE_LIMIT_N,
                JUMP_LEG_SYNC_FORCE_LIMIT_N,
            ))
            # A positive difference means the left chain is longer; reduce its
            # extension force and add the same force to the right chain.
            forces[0] -= synchronization_force
            forces[1] += synchronization_force
            forces = np.clip(forces, -force_limit, force_limit)
        jacobian = self.leg_profile.hip_length_jacobian(self._reference_shape)
        command[list(self.hip_actuator_ids)] += jacobian * np.array(
            (-forces[0], -forces[0], forces[1], -forces[1])
        )

    def apply_jump_impact_leg_force(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
    ) -> None:
        """Apply compression-only leg impedance after the first touchdown.

        Flight preload is useful before contact, but it becomes unsafe when
        one chain touches down a few milliseconds before the other.  The
        existing landing debounce remains in charge of the phase transition;
        this short overlay merely prevents either chain from receiving a new
        extension force while that debounce is collecting two-wheel contact.
        """
        if self.leg_profile is None:
            return
        lengths = self.leg_lengths(data)
        velocities = self.leg_length_velocities(data)
        force_limit = JUMP_IMPACT_FORCE_LIMIT_N
        forces = np.clip(
            LEG_LENGTH_FORCE_KP_N_PER_M * (JUMP_LANDING_LENGTH_M - lengths)
            - LEG_LENGTH_FORCE_KD_NS_PER_M * velocities,
            -force_limit,
            0.0,
        )
        leg_difference = float(lengths[0] - lengths[1])
        differential_velocity = float(velocities[0] - velocities[1])
        synchronization_force = float(np.clip(
            JUMP_LEG_SYNC_KP_N_PER_M * leg_difference
            + JUMP_LEG_SYNC_KD_NS_PER_M * differential_velocity,
            -JUMP_LEG_SYNC_FORCE_LIMIT_N,
            JUMP_LEG_SYNC_FORCE_LIMIT_N,
        ))
        forces[0] = float(np.clip(
            forces[0] - synchronization_force,
            -force_limit,
            0.0,
        ))
        forces[1] = float(np.clip(
            forces[1] + synchronization_force,
            -force_limit,
            0.0,
        ))
        jacobian = self.leg_profile.hip_length_jacobian(self._reference_shape)
        command[list(self.hip_actuator_ids)] += jacobian * np.array(
            (-forces[0], -forces[0], forces[1], -forces[1])
        )
        self.jump.update_impact_leg_difference(float(np.ptp(lengths)))

    def apply_wheel_speed_governor(self, data: mujoco.MjData, command: np.ndarray) -> None:
        """World-frame wheel PI with conditional integration and saturation protection."""
        contacts = tuple(
            wheel_ground_contacts(data, self.refs, geom_id) > 0
            for geom_id in self.refs.wheel_geoms
        )
        full_support = all(contacts)
        if not any(contacts):
            self.speed_error_integral = 0.0
            return
        speed_error = self.motion.current_speed - self.forward_speed(data)
        # With one supporting wheel, regular locomotion must not accelerate the
        # robot.  During an explicit recovery, however, retain a bounded P-only
        # braking torque through the wheel that still has ground contact.
        if not full_support:
            self.speed_error_integral = 0.0
            if not self._contact_recovery_braking:
                return
            wheel_command = float(np.clip(
                WALK_SPEED_KP_MOTOR_NM_PER_MPS * speed_error,
                -CONTACT_RECOVERY_WHEEL_BRAKE_LIMIT_MOTOR_NM,
                CONTACT_RECOVERY_WHEEL_BRAKE_LIMIT_MOTOR_NM,
            ))
            command[self.refs.actuator_ids[2]] += wheel_command
            command[self.refs.actuator_ids[5]] += wheel_command
            return
        proposed_integral = float(np.clip(
            self.speed_error_integral + speed_error * self.dt,
            -WALK_SPEED_INTEGRAL_LIMIT_M,
            WALK_SPEED_INTEGRAL_LIMIT_M,
        ))
        unsaturated = WALK_SPEED_KP_MOTOR_NM_PER_MPS * speed_error + WALK_SPEED_KI_MOTOR_NM_PER_M * proposed_integral
        if abs(unsaturated) <= WALK_SPEED_GOVERNOR_LIMIT_MOTOR_NM or speed_error * unsaturated < 0.0:
            self.speed_error_integral = proposed_integral
        wheel_command = float(np.clip(
            WALK_SPEED_KP_MOTOR_NM_PER_MPS * speed_error + WALK_SPEED_KI_MOTOR_NM_PER_M * self.speed_error_integral,
            -WALK_SPEED_GOVERNOR_LIMIT_MOTOR_NM,
            WALK_SPEED_GOVERNOR_LIMIT_MOTOR_NM,
        ))
        command[self.refs.actuator_ids[2]] += wheel_command
        command[self.refs.actuator_ids[5]] += wheel_command

    def apply_jump_recovery_wheel_brake(self, data: mujoco.MjData, command: np.ndarray) -> None:
        """Use bounded common wheel torque to stabilize the post-landing body.

        Differential wheel torque and the normal speed integrator remain off
        until the recovery gate completes.  The airborne pitch feedback is
        retained for the first impact window; without it the landing trim's
        contact impulse can slowly drive the body past the guard threshold.
        """
        if not any(
            wheel_ground_contacts(data, self.refs, geom_id) > 0
            for geom_id in self.refs.wheel_geoms
        ):
            return
        attitude_error = self.airborne_attitude_error(data)
        pitch_rate = float(data.qvel[self.root_dof + 3])
        wheel_command = float(np.clip(
            AIRBORNE_WHEEL_ATTITUDE_KP_NM_PER_RAD * float(attitude_error[0])
            + AIRBORNE_WHEEL_ATTITUDE_KD_NM_PER_RAD_S * pitch_rate,
            -JUMP_RECOVERY_WHEEL_BRAKE_LIMIT_NM,
            JUMP_RECOVERY_WHEEL_BRAKE_LIMIT_NM,
        ))
        # Replace the landing trim's differential wheel terms rather than
        # adding to them; yaw torque during a partially loaded touchdown can
        # unload one wheel and defeat the recovery gate.
        command[self.refs.actuator_ids[2]] = wheel_command
        command[self.refs.actuator_ids[5]] = wheel_command

    def yaw_rate_target(self, data: mujoco.MjData) -> float:
        """Return the bounded yaw-rate target for telemetry and wheel control."""
        if (
            self._terrain_heading_stabilization_enabled
            and self._terrain_yaw_rate_command_active
        ):
            heading_kp = TERRAIN_RATE_COMMAND_HEADING_KP_RAD_S_PER_RAD
            target_yaw_rate = self._terrain_yaw_rate_command_rad_s
        elif self._terrain_heading_stabilization_enabled:
            heading_kp = TERRAIN_YAW_HEADING_KP_RAD_S_PER_RAD
            target_yaw_rate = 0.0
        else:
            heading_kp = YAW_HEADING_KP_RAD_S_PER_RAD
            target_yaw_rate = 0.0
        yaw_error = wrap_to_pi(self.heading_command.reference_yaw - self.heading_yaw(data))
        return float(np.clip(
            target_yaw_rate + heading_kp * yaw_error,
            -MAX_YAW_RATE_RAD_S,
            MAX_YAW_RATE_RAD_S,
        ))

    def apply_yaw_heading_governor(self, data: mujoco.MjData, command: np.ndarray) -> None:
        """Track command_yaw using bounded differential wheel torque."""
        if not all(wheel_ground_contacts(data, self.refs, geom_id) for geom_id in self.refs.wheel_geoms):
            # Preserve the IMU-derived filtered yaw estimate through a brief
            # hfield manifold gap. Resetting it here created a false yaw-rate
            # measurement and an unnecessary recovery during fast terrain
            # turns. Torque is still withheld until both raw contacts return.
            return
        if (
            self._terrain_heading_stabilization_enabled
            and self._terrain_yaw_rate_command_active
        ):
            yaw_rate_kp = TERRAIN_RATE_COMMAND_KP_MOTOR_NM_PER_RAD_S
            command_limit = TERRAIN_RATE_COMMAND_LIMIT_MOTOR_NM
        elif self._terrain_heading_stabilization_enabled:
            yaw_rate_kp = TERRAIN_YAW_RATE_KP_MOTOR_NM_PER_RAD_S
            command_limit = TERRAIN_YAW_GOVERNOR_LIMIT_MOTOR_NM
        else:
            yaw_rate_kp = YAW_RATE_KP_MOTOR_NM_PER_RAD_S
            command_limit = YAW_GOVERNOR_LIMIT_MOTOR_NM
        target_yaw_rate = self.yaw_rate_target(data)
        differential_command = float(np.clip(
            yaw_rate_kp * (target_yaw_rate - self._measured_yaw_rate),
            -command_limit,
            command_limit,
        ))
        # Positive yaw is counter-clockwise about +Z.  The measured drivetrain
        # calibration maps a left-minus-right wheel command to negative yaw.
        left_wheel = self.refs.actuator_ids[2]
        right_wheel = self.refs.actuator_ids[5]
        if self.motion.current_speed >= FORWARD_SPEED_YAW_OVERRIDE_START_MPS:
            # At high speed, the speed-trim LQR may contain a differential
            # feedback term that opposes the outer heading loop.  Preserve its
            # common balancing torque while assigning the wheel difference to
            # the explicit command_yaw controller.
            common_command = 0.5 * (command[left_wheel] + command[right_wheel])
            command[left_wheel] = common_command - differential_command
            command[right_wheel] = common_command + differential_command
        else:
            command[left_wheel] -= differential_command
            command[right_wheel] += differential_command

    def apply_airborne_recovery(self, data: mujoco.MjData, command: np.ndarray) -> None:
        position_error = self._reference_hip_qpos - data.qpos[self.hip_qpos_addresses]
        velocity_error = -data.qvel[self.hip_dof_addresses]
        command[list(self.hip_actuator_ids)] = (
            self._reference_control[list(self.hip_actuator_ids)]
            + AIRBORNE_HIP_KP_NM_PER_RAD * position_error
            + AIRBORNE_HIP_KD_NM_PER_RAD_PER_S * velocity_error
        )
        command[self.refs.actuator_ids[2]] = 0.0
        command[self.refs.actuator_ids[5]] = 0.0

    def apply_jump_thrust_wheel_control(self, data: mujoco.MjData, command: np.ndarray) -> None:
        """Preserve a bounded rolling launch impulse while both wheels support."""
        # Keep the validated near-static jump trajectory unchanged. This
        # compensation is only meaningful for an explicit rolling launch
        # request, where the contact phase must retain forward momentum to
        # clear a terrain feature.
        if abs(self.jump.launch_speed) <= JUMP_MAX_LAUNCH_SPEED_MPS:
            return
        if not all(
            wheel_ground_contacts(data, self.refs, geom_id) > 0
            for geom_id in self.refs.wheel_geoms
        ):
            return
        speed_error = self.jump.launch_speed - self.forward_speed(data)
        wheel_command = float(np.clip(
            JUMP_THRUST_WHEEL_KP_NM_PER_MPS * speed_error,
            -JUMP_THRUST_WHEEL_LIMIT_NM,
            JUMP_THRUST_WHEEL_LIMIT_NM,
        ))
        command[self.refs.actuator_ids[2]] += wheel_command
        command[self.refs.actuator_ids[5]] += wheel_command

    def airborne_attitude_error(self, data: mujoco.MjData) -> np.ndarray:
        """Return body attitude error relative to the yaw-aligned trim frame."""
        reference_qpos, _ = self.reference_state(data)
        position_error = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, position_error, 1.0, reference_qpos, data.qpos)
        return position_error[self.root_dof + 3 : self.root_dof + 6]

    def apply_airborne_wheel_attitude_control(self, data: mujoco.MjData, command: np.ndarray) -> None:
        """Use bounded common wheel torque as a reaction wheel while airborne."""
        attitude_error = self.airborne_attitude_error(data)
        pitch_rate = float(data.qvel[self.root_dof + 3])
        wheel_torque = float(np.clip(
            AIRBORNE_WHEEL_ATTITUDE_KP_NM_PER_RAD * attitude_error[0]
            + AIRBORNE_WHEEL_ATTITUDE_KD_NM_PER_RAD_S * pitch_rate,
            self.model.actuator_ctrlrange[self.refs.actuator_ids[2], 0],
            self.model.actuator_ctrlrange[self.refs.actuator_ids[2], 1],
        ))
        command[self.refs.actuator_ids[2]] = wheel_torque
        command[self.refs.actuator_ids[5]] = wheel_torque

    def _jump_leg_force_limit(self, phase: str | None) -> float:
        if phase == "thrust":
            return JUMP_THRUST_FORCE_LIMIT_N
        if phase == "flight":
            return (
                JUMP_FLIGHT_RETRACT_FORCE_LIMIT_N
                if self.jump.flight_steps < JUMP_FLIGHT_RETRACT_STEPS
                else JUMP_FLIGHT_PRELOAD_FORCE_LIMIT_N
            )
        if phase == "landing":
            return JUMP_LANDING_FORCE_LIMIT_N
        return LEG_LENGTH_FORCE_LIMIT_N

    def _drop_leg_force_limit(self, phase: str | None) -> float:
        if phase == "preload":
            return DROP_PRELOAD_FORCE_LIMIT_N
        if phase == "flight":
            return DROP_FLIGHT_FORCE_LIMIT_N
        if phase in ("landing", "recovery"):
            return DROP_LANDING_FORCE_LIMIT_N
        return LEG_LENGTH_FORCE_LIMIT_N

    def _command_after_gas_spring(self, data: mujoco.MjData) -> np.ndarray:
        """Build a command after the gas-spring generalized force is present."""
        self._advance_motion_reference(data)
        self._update_measured_yaw_rate(data)
        self._update_pending_jump(data)
        phase = self._update_jump_state(data)
        drop_phase = None if phase is not None else self._update_drop_state(data)
        leg_rate_limit = (
            self._jump_rate_limit()
            if phase is not None
            else self._drop_rate_limit(drop_phase)
        )
        self._advance_leg_reference(data, leg_rate_limit)
        if phase == "flight" or drop_phase == "flight":
            command = np.zeros(self.model.nu)
            self.apply_airborne_recovery(data, command)
            if phase == "flight" and self.jump.impact_active:
                self.apply_jump_impact_leg_force(data, command)
                self.apply_jump_recovery_wheel_brake(data, command)
            else:
                force_limit = (
                    self._jump_leg_force_limit(phase)
                    if phase is not None
                    else self._drop_leg_force_limit(drop_phase)
                )
                self.apply_leg_length_force(data, command, force_limit)
                self.apply_airborne_wheel_attitude_control(data, command)
            return np.clip(command, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

        reference_control, reference_gain = self._scheduled_control_and_gain()
        command = reference_control - reference_gain @ self.state_error(data)
        landing_managed = phase == "landing" or drop_phase in ("landing", "recovery")
        if landing_managed:
            stance_guard_scale = JUMP_LANDING_STANCE_GUARD_SCALE
            recovery_start_time = (
                self.jump.recovery_start_time
                if phase == "landing" and self.jump.recovering
                else self.drop.recovery_start_time if drop_phase == "recovery" else None
            )
            if recovery_start_time is not None:
                recovery_progress = float(np.clip(
                    (float(data.time) - recovery_start_time)
                    / JUMP_RECOVERY_STANCE_GUARD_RAMP_SECONDS,
                    0.0,
                    1.0,
                ))
                stance_guard_scale = (
                    JUMP_RECOVERY_STANCE_GUARD_INITIAL_SCALE
                    + (1.0 - JUMP_RECOVERY_STANCE_GUARD_INITIAL_SCALE) * recovery_progress
                )
            position_error = self._reference_hip_qpos - data.qpos[self.hip_qpos_addresses]
            velocity_error = -data.qvel[self.hip_dof_addresses]
            command[list(self.hip_actuator_ids)] += stance_guard_scale * (
                WALK_STANCE_GUARD_KP_NM_PER_RAD * position_error
                + WALK_STANCE_GUARD_KD_NM_PER_RAD_PER_S * velocity_error
            )
        else:
            self.apply_walking_stance_guard(data, command)
        self.apply_leg_length_force(
            data,
            command,
            self._jump_leg_force_limit(phase)
            if phase is not None
            else self._drop_leg_force_limit(drop_phase),
        )
        if phase is None or phase in ("prepare", "crouch"):
            if drop_phase in ("landing", "recovery"):
                self.apply_jump_recovery_wheel_brake(data, command)
            else:
                self.apply_wheel_speed_governor(data, command)
                self.apply_yaw_heading_governor(data, command)
        elif phase == "thrust":
            command[self.refs.actuator_ids[2]] = 0.0
            command[self.refs.actuator_ids[5]] = 0.0
            self.apply_jump_thrust_wheel_control(data, command)
        elif phase == "landing" and self.jump.recovering:
            self.apply_jump_recovery_wheel_brake(data, command)
        return np.clip(command, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

    def command(self, data: mujoco.MjData) -> np.ndarray:
        """Build the nominal command from the current measured state."""
        self.apply_gas_spring_assist(data)
        return self._command_after_gas_spring(data)

    def command_with_delay_prediction(
        self,
        data: mujoco.MjData,
        delayed_controls: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """Predict through queued controls before computing a delayed command.

        The actuator FIFO is the only source of delay.  This method advances a
        preallocated shadow state through the controls already waiting in that
        FIFO, then evaluates the LQR once at the state where its new command
        will actually take effect.  It is intentionally for normal walking
        only; jump and contact-recovery phases keep their protected direct
        control path.
        """
        if not delayed_controls:
            return self.command(data)
        if any(np.asarray(control, dtype=np.float64).shape != (self.model.nu,) for control in delayed_controls):
            raise ValueError("each delayed control must have shape (nu,)")

        self.apply_gas_spring_assist(data)
        prediction = self._delay_prediction_data
        mujoco.mj_copyData(prediction, self.model, data)
        for delayed_control in delayed_controls:
            prediction.ctrl[:] = np.clip(
                np.asarray(delayed_control, dtype=np.float64),
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
            # qfrc_applied is a per-step force field.  Reapply it for every
            # predicted physics tick just as command() does on the real data.
            self.apply_gas_spring_assist(prediction)
            mujoco.mj_step(self.model, prediction)
        return self._command_after_gas_spring(prediction)


def validate_standing_contact(data: mujoco.MjData, refs: ModelRefs) -> None:
    wheel_contacts = tuple(wheel_ground_contacts(data, refs, geom_id) for geom_id in refs.wheel_geoms)
    protected_contacts = sum(contacts_for_geom(data, geom_id) for geom_id in refs.base_geoms)
    nonwheel_ground_contacts, nonwheel_obstacle_contacts = nonwheel_static_contact_counts(
        data, refs
    )
    if (
        not all(wheel_contacts)
        or protected_contacts
        or nonwheel_ground_contacts
        or nonwheel_obstacle_contacts
    ):
        raise RuntimeError(
            f"Invalid standing contact: wheels={wheel_contacts}, "
            f"protected_contacts={protected_contacts}, "
            f"nonwheel_ground_contacts={nonwheel_ground_contacts}, "
            f"nonwheel_obstacle_contacts={nonwheel_obstacle_contacts}. "
            "Check the physical MJCF stance before enabling LQR."
        )


def validate_linkage_clearance(data: mujoco.MjData, refs: ModelRefs) -> None:
    """Reject a trim or running state with a non-adjacent linkage collision."""
    linkage_geoms = set(refs.linkage_collision_geoms)
    for contact in data.contact[: data.ncon]:
        if contact.geom1 in linkage_geoms and contact.geom2 in linkage_geoms:
            raise RuntimeError(
                "Unsafe linkage self-contact in walking stance: "
                f"geom ids {contact.geom1} / {contact.geom2}"
            )


def validate_closed_loop_error(data: mujoco.MjData, refs: ModelRefs) -> None:
    """Keep compliant connect constraints within the calibrated linkage error."""
    maximum_error = max(
        float(np.linalg.norm(data.site_xpos[first] - data.site_xpos[second]))
        for first, second in refs.stance_closure_sites
    )
    if maximum_error > MAX_RUNTIME_CLOSURE_ERROR_M:
        raise RuntimeError(
            "Closed linkage constraint exceeded safe residual: "
            f"{maximum_error:.6f}m > {MAX_RUNTIME_CLOSURE_ERROR_M:.6f}m"
        )


def validate_leg_length_state(data: mujoco.MjData, refs: ModelRefs) -> None:
    """Reject singular/extreme leg configurations before the links can interfere."""
    lengths = np.array((
        float(sensor(data, refs, "left_leg_length")[0]),
        float(sensor(data, refs, "right_leg_length")[0]),
    ))
    if (
        float(np.min(lengths)) < HARD_LEG_LENGTH_MIN_M
        or float(np.max(lengths)) > HARD_LEG_LENGTH_MAX_M
        or float(np.ptp(lengths)) > MAX_LEG_LENGTH_DIFFERENCE_M
    ):
        raise RuntimeError(
            "Unsafe leg-length state: "
            f"left={lengths[0]:.4f}m, right={lengths[1]:.4f}m, "
            f"allowed=({HARD_LEG_LENGTH_MIN_M:.3f}, {HARD_LEG_LENGTH_MAX_M:.3f})m"
        )


def validate_jump_contacts(data: mujoco.MjData, refs: ModelRefs) -> None:
    """Allow wheel liftoff during a jump, but reject unsafe body/leg impacts."""
    protected_contacts = sum(contacts_for_geom(data, geom_id) for geom_id in refs.base_geoms)
    nonwheel_ground_contacts, nonwheel_obstacle_contacts = nonwheel_static_contact_counts(
        data, refs
    )
    if nonwheel_ground_contacts or nonwheel_obstacle_contacts:
        raise RuntimeError(
            "Unsafe jump contact: a body other than a wheel touched a terrain support or obstacle."
        )
    if protected_contacts:
        raise RuntimeError("Unsafe jump contact: protected chassis collision geometry is in contact.")
    validate_linkage_clearance(data, refs)
    validate_closed_loop_error(data, refs)
    validate_leg_length_state(data, refs)


def print_sensor_state(data: mujoco.MjData, refs: ModelRefs) -> None:
    leg_lengths = (sensor(data, refs, "left_leg_length")[0], sensor(data, refs, "right_leg_length")[0])
    gyro = sensor(data, refs, "imu_gyroscope")
    print(
        f"LQR ready: leg_length=({leg_lengths[0]:.4f}, {leg_lengths[1]:.4f})m, "
        f"gyro=({gyro[0]:.3f}, {gyro[1]:.3f}, {gyro[2]:.3f})rad/s"
    )


def project_walking_stance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: ModelRefs,
    hip_targets: np.ndarray | None = None,
) -> None:
    """Solve the existing four-bar closures at the folded walking target.

    The CAD topology already contains the correct two branch chains and four
    connect constraints.  Projecting the passive joints before stepping avoids
    using motor torque to pull an initially open loop through the ground.
    """
    hip_qpos_addresses = np.array((
        int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch")]),
        int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_active_link_pitch")]),
        int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch")]),
        int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_active_link_pitch")]),
    ))
    passive_qpos_addresses = np.array(
        [int(model.jnt_qposadr[joint_id]) for joint_id in refs.stance_passive_joints]
    )
    passive_lower = np.array(
        [float(model.jnt_range[joint_id, 0]) for joint_id in refs.stance_passive_joints]
    )
    passive_upper = np.array(
        [float(model.jnt_range[joint_id, 1]) for joint_id in refs.stance_passive_joints]
    )
    root_qpos_address = int(model.jnt_qposadr[refs.root_joint])
    wheel_radius = float(np.mean([model.geom_size[geom_id, 0] for geom_id in refs.wheel_geoms]))

    targets = WALK_STANCE_HIP_TARGETS_RAD if hip_targets is None else np.asarray(hip_targets, dtype=np.float64)
    if targets.shape != (4,):
        raise ValueError(f"expected four active hip targets, got shape {targets.shape}")
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[hip_qpos_addresses] = targets

    def closure_error(passive_qpos: np.ndarray) -> np.ndarray:
        data.qpos[passive_qpos_addresses] = passive_qpos
        mujoco.mj_forward(model, data)
        return np.concatenate(
            [data.site_xpos[first] - data.site_xpos[second] for first, second in refs.stance_closure_sites]
        )

    projection = least_squares(
        closure_error,
        np.clip(data.qpos[passive_qpos_addresses], passive_lower, passive_upper),
        bounds=(passive_lower, passive_upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=3000,
    )
    closure_norm = float(np.linalg.norm(closure_error(projection.x)))
    if not projection.success or closure_norm > WALK_STANCE_CLOSURE_TOLERANCE_M:
        raise RuntimeError(
            "Walking-stance closure projection failed: "
            f"success={projection.success}, residual={closure_norm:.3e}m"
        )

    wheel_heights = np.array([data.geom_xpos[geom_id, 2] for geom_id in refs.wheel_geoms])
    data.qpos[root_qpos_address + 2] += wheel_radius - float(np.mean(wheel_heights))
    mujoco.mj_forward(model, data)
    wheel_heights = np.array([data.geom_xpos[geom_id, 2] for geom_id in refs.wheel_geoms])
    if float(np.ptp(wheel_heights)) > WALK_STANCE_WHEEL_HEIGHT_TOLERANCE_M:
        raise RuntimeError(
            "Walking stance cannot establish symmetric wheel contact: "
            f"wheel height difference={np.ptp(wheel_heights):.6f}m"
        )
    validate_standing_contact(data, refs)
    validate_linkage_clearance(data, refs)
    validate_closed_loop_error(data, refs)
    validate_leg_length_state(data, refs)


def hip_targets_from_shape(shape: float) -> np.ndarray:
    """Mirror the validated low-centre branch on both closed-chain legs."""
    return np.array((shape, shape, -shape, shape), dtype=np.float64)


def build_leg_length_profile(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: ModelRefs,
) -> LegLengthProfile:
    """Precompute safe closed-chain trims once; the control loop only interpolates."""
    trims: list[LqrTrim] = []
    for shape in LEG_LENGTH_PROFILE_SHAPES_RAD:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        project_walking_stance(model, data, refs, hip_targets_from_shape(float(shape)))
        trim_controller = PhysicalLqr(model, data, refs, speed=0.0, acceleration_limit=DEFAULT_ACCELERATION_MPS2)
        trims.append(LqrTrim(
            shape=float(shape),
            leg_length=trim_controller.average_leg_length(data),
            qpos=trim_controller.qpos_equilibrium.copy(),
            qvel=trim_controller.qvel_equilibrium.copy(),
            control=trim_controller.control_equilibrium.copy(),
            gain=trim_controller.gain.copy(),
            hip_qpos=trim_controller.hip_qpos_equilibrium.copy(),
        ))
    profile = LegLengthProfile(tuple(trims))
    lower, upper = profile.minimum_length, profile.maximum_length
    if not lower <= WALK_LEG_LENGTH_MIN_M < WALK_LEG_LENGTH_MAX_M <= upper:
        raise RuntimeError(
            "leg-length profile does not cover the configured walking interval: "
            f"profile=({lower:.3f}, {upper:.3f})m, walking=({WALK_LEG_LENGTH_MIN_M:.3f}, {WALK_LEG_LENGTH_MAX_M:.3f})m"
        )
    return profile


def build_forward_speed_trim(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: ModelRefs,
    shape: float,
    speed_limit: float = MAX_FORWARD_SPEED_MPS,
) -> SpeedTrim:
    """Precompute the selected forward rolling working point without changing the MJCF."""
    speed_limit = validate_forward_speed_limit(speed_limit)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    project_walking_stance(model, data, refs, hip_targets_from_shape(shape))
    trim_controller = PhysicalLqr(model, data, refs, speed=0.0, acceleration_limit=DEFAULT_ACCELERATION_MPS2)
    wheel_radius = float(np.mean([model.geom_size[geom_id, 0] for geom_id in refs.wheel_geoms]))
    qvel = np.zeros(model.nv)
    qvel[trim_controller.root_dof : trim_controller.root_dof + 3] = (
        speed_limit * trim_controller.forward_direction(data)
    )
    for joint_id in refs.wheel_joints:
        qvel[int(model.jnt_dofadr[joint_id])] = -speed_limit / wheel_radius

    trim_controller.qvel_equilibrium = qvel.copy()
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    control = trim_controller.solve_equilibrium(data)
    gain = trim_controller.linear_lqr(data)
    return SpeedTrim(
        speed=speed_limit,
        leg_length=trim_controller.average_leg_length(data),
        qvel=qvel,
        control=control,
        gain=gain,
    )


def settle_and_relinearize(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    refs: ModelRefs,
    speed: float,
    acceleration_limit: float,
    leg_length: float | None = None,
    *,
    max_forward_speed: float = MAX_FORWARD_SPEED_MPS,
    max_reverse_speed: float = MAX_REVERSE_SPEED_MPS,
) -> PhysicalLqr:
    """Build a scheduled low-centre LQR and select its initial leg length."""
    max_forward_speed = validate_forward_speed_limit(max_forward_speed)
    max_reverse_speed = validate_reverse_speed_limit(max_reverse_speed)
    profile = build_leg_length_profile(model, data, refs)
    default_length = profile.length_for_shape(float(WALK_STANCE_HIP_TARGETS_RAD[0]))
    requested_length = default_length if leg_length is None else float(leg_length)
    if not WALK_LEG_LENGTH_MIN_M <= requested_length <= WALK_LEG_LENGTH_MAX_M:
        raise ValueError(
            f"walking leg length must be within [{WALK_LEG_LENGTH_MIN_M:.3f}, {WALK_LEG_LENGTH_MAX_M:.3f}]m"
        )
    selected_shape = profile.shape_for_length(requested_length)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    project_walking_stance(model, data, refs, hip_targets_from_shape(selected_shape))
    controller = PhysicalLqr(
        model,
        data,
        refs,
        speed,
        acceleration_limit,
        max_forward_speed=max_forward_speed,
        max_reverse_speed=max_reverse_speed,
    )
    forward_speed_trim = build_forward_speed_trim(
        model,
        data,
        refs,
        selected_shape,
        speed_limit=max_forward_speed,
    )
    data.qpos[:] = controller.qpos_equilibrium
    data.qvel[:] = controller.qvel_equilibrium
    data.ctrl[:] = controller.control_equilibrium
    mujoco.mj_forward(model, data)
    controller.configure_leg_profile(profile, requested_length)
    controller.configure_forward_speed_trim(forward_speed_trim)
    return controller


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: PhysicalLqr,
    refs: ModelRefs,
    seconds: float,
    jump_at: float | None,
) -> None:
    jump_started = False
    steady_speeds: list[float] = []
    controller.print_speed_telemetry(data, force=True)
    while data.time < seconds:
        if jump_at is not None and not jump_started and data.time >= jump_at:
            controller.request_jump(data)
            jump_started = True
        data.ctrl[:] = controller.command(data)
        mujoco.mj_step(model, data)
        if jump_started and controller.jump.active:
            validate_jump_contacts(data, refs)
        else:
            validate_standing_contact(data, refs)
            validate_linkage_clearance(data, refs)
            validate_closed_loop_error(data, refs)
            validate_leg_length_state(data, refs)
        controller.print_speed_telemetry(data)
        if jump_at is None and data.time >= max(2.0, seconds - 2.0):
            steady_speeds.append(controller.forward_speed(data))
    measured_speed = float(np.mean(steady_speeds)) if steady_speeds else controller.forward_speed(data)
    speed_error = measured_speed - controller.motion.target_speed
    measured_yaw = controller.heading_yaw(data)
    yaw_error = wrap_to_pi(controller.command_yaw - measured_yaw)
    allowed_speed_error = speed_tracking_tolerance(controller.motion.target_speed)
    if jump_at is None and seconds >= 8.0 and abs(speed_error) > allowed_speed_error:
        raise RuntimeError(
            f"Speed tracking failed: target={controller.motion.target_speed:.3f}m/s, "
            f"measured={measured_speed:.3f}m/s, error={speed_error:.3f}m/s, "
            f"allowed={allowed_speed_error:.3f}m/s"
        )
    if jump_at is None and seconds >= 8.0 and abs(yaw_error) > YAW_TRACKING_TOLERANCE_RAD:
        raise RuntimeError(
            f"Yaw tracking failed: target={controller.command_yaw:.3f}rad, "
            f"measured={measured_yaw:.3f}rad, error={yaw_error:.3f}rad"
        )
    tracking_status = speed_tracking_status(
        controller.motion.target_speed,
        controller.motion.current_speed,
        measured_speed,
        jump_active=jump_at is not None,
    )
    if jump_at is not None and tracking_status == "JUMP":
        tracking_status = "JUMP_SEQUENCE"
    if jump_at is not None:
        completion_label = "Physical LQR jump run complete"
    elif seconds >= 8.0:
        completion_label = "Physical LQR check passed"
    else:
        completion_label = "Physical LQR warm-up complete (steady-speed check requires >=8s)"
    print(
        f"{completion_label}: t={data.time:.3f}s, "
        f"leg_length={controller.average_leg_length(data):.4f}m, "
        f"target_speed={controller.motion.target_speed:.3f}m/s, "
        f"speed={measured_speed:.3f}m/s, error={speed_error:+.3f}m/s, "
        f"tolerance={allowed_speed_error:.3f}m/s, status={tracking_status}, "
        f"yaw={measured_yaw:.3f}rad, command_yaw={controller.command_yaw:.3f}rad"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run physical-model LQR on the wheeled-leg MuJoCo model.")
    parser.add_argument("--headless", action="store_true", help="Run a two-wheel-contact LQR validation without a viewer.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Validation duration in seconds.")
    parser.add_argument(
        "--max-speed",
        "--speed-limit",
        dest="max_speed",
        type=float,
        default=DEFAULT_FORWARD_SPEED_LIMIT_MPS,
        help=(
            "Run-time forward speed limit in m/s, configurable within "
            f"{MIN_FORWARD_SPEED_LIMIT_MPS:.1f}..{MAX_FORWARD_SPEED_MPS:.1f}."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help=(
            "Body-forward reference speed in m/s, limited to "
            f"-{MAX_REVERSE_SPEED_MPS:.2f}..+{MAX_FORWARD_SPEED_MPS:.2f}."
        ),
    )
    parser.add_argument(
        "--yaw",
        type=float,
        help="World-frame desired heading in radians; 0 points along +X and positive is counter-clockwise about +Z.",
    )
    parser.add_argument(
        "--leg-length",
        type=float,
        help=(
            "Common left/right walking leg-length target in metres, limited to "
            f"{WALK_LEG_LENGTH_MIN_M:.3f}..{WALK_LEG_LENGTH_MAX_M:.3f}."
        ),
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=DEFAULT_ACCELERATION_MPS2,
        help="Maximum forward acceleration/deceleration in m/s^2.",
    )
    parser.add_argument(
        "--jump-at",
        type=float,
        help="Start one crouch-and-jump sequence at this simulation time in seconds.",
    )
    args = parser.parse_args()
    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")
    try:
        args.max_speed = validate_forward_speed_limit(args.max_speed)
    except ValueError as error:
        parser.error(str(error))
    if not -MAX_REVERSE_SPEED_MPS <= args.speed <= args.max_speed:
        parser.error(
            f"--speed must be within -{MAX_REVERSE_SPEED_MPS:.2f}..+{args.max_speed:.2f}m/s"
        )
    if args.yaw is not None and not np.isfinite(args.yaw):
        parser.error("--yaw must be finite")
    if args.leg_length is not None and not WALK_LEG_LENGTH_MIN_M <= args.leg_length <= WALK_LEG_LENGTH_MAX_M:
        parser.error(
            f"--leg-length must be within {WALK_LEG_LENGTH_MIN_M:.3f}..{WALK_LEG_LENGTH_MAX_M:.3f}m"
        )
    if args.acceleration <= 0.0:
        parser.error("--acceleration must be positive")
    if args.jump_at is not None and args.jump_at < 0.0:
        parser.error("--jump-at must be non-negative")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    refs = build_refs(model)
    controller = settle_and_relinearize(
        model,
        data,
        refs,
        args.speed,
        args.acceleration,
        leg_length=args.leg_length,
        max_forward_speed=args.max_speed,
        max_reverse_speed=MAX_REVERSE_SPEED_MPS,
    )
    if args.yaw is not None:
        controller.set_command_yaw(args.yaw)
    print_sensor_state(data, refs)
    print(
        f"Heading: {controller.heading_yaw(data):.3f}rad, "
        f"command_yaw={controller.command_yaw:.3f}rad"
    )
    print(
        f"Speed limits: forward={controller.max_forward_speed:.2f}m/s, "
        f"reverse={controller.max_reverse_speed:.2f}m/s, "
        f"acceleration={controller.motion.acceleration_limit:.2f}m/s^2"
    )
    if args.headless:
        run_headless(model, data, controller, refs, args.seconds, args.jump_at)
        return

    def key_callback(keycode: int) -> None:
        if keycode in (ord("W"), ord("w"), GLFW_KEY_UP):
            target = controller.adjust_target_speed(SPEED_INCREMENT_MPS)
            print(f"Target speed: {target:.2f}m/s")
        elif keycode in (ord("S"), ord("s"), GLFW_KEY_DOWN):
            target = controller.adjust_target_speed(-SPEED_INCREMENT_MPS)
            print(f"Target speed: {target:.2f}m/s")
        elif keycode in (ord("A"), ord("a"), GLFW_KEY_LEFT):
            target = controller.adjust_command_yaw(YAW_INCREMENT_RAD)
            print(f"Command yaw: {target:.3f}rad ({np.rad2deg(target):.1f}deg)")
        elif keycode in (ord("D"), ord("d"), GLFW_KEY_RIGHT):
            target = controller.adjust_command_yaw(-YAW_INCREMENT_RAD)
            print(f"Command yaw: {target:.3f}rad ({np.rad2deg(target):.1f}deg)")
        elif keycode in (ord("X"), ord("x")):
            controller.set_target_speed(0.0)
            print("Target speed: 0.00m/s")
        elif keycode in (ord("C"), ord("c")):
            target = controller.hold_current_yaw(data)
            print(f"Command yaw aligned to current heading: {target:.3f}rad")
        elif keycode in (ord("R"), ord("r")):
            target = controller.adjust_target_leg_length(0.01)
            print(f"Target leg length: {target:.3f}m")
        elif keycode in (ord("F"), ord("f")):
            target = controller.adjust_target_leg_length(-0.01)
            print(f"Target leg length: {target:.3f}m")
        elif keycode in (ord("J"), ord("j")):
            controller.request_jump(data)
            print("Jump sequence started: prepare, crouch, thrust, flight, landing")

    print(
        "Keys: W/S or Up/Down speed, A/D or Left/Right turn command yaw, "
        "X stop, C hold current heading, R/F extend/retract legs, J starts or restarts the jump sequence."
    )
    scheduled_jump_attempted = False
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            with viewer.lock():
                if args.jump_at is not None and not scheduled_jump_attempted and data.time >= args.jump_at:
                    scheduled_jump_attempted = True
                    controller.request_jump(data)
                    print("Scheduled jump sequence started")
                data.ctrl[:] = controller.command(data)
                controller.print_speed_telemetry(data)
            mujoco.mj_step(model, data)
            if controller.jump.active:
                validate_jump_contacts(data, refs)
            else:
                validate_standing_contact(data, refs)
                validate_linkage_clearance(data, refs)
                validate_closed_loop_error(data, refs)
                validate_leg_length_state(data, refs)
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - start)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
