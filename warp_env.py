"""Fail-closed MuJoCo-Warp batch physics harness.

This module intentionally exposes GPU-native *physics* only.  It does not
pretend that the CPU ``PhysicalLqr`` or terrain/jump safety state machine has
already been ported to CUDA, so it cannot be used to produce PPO checkpoints.
It is the validated first layer for that port: all worlds share one immutable
MuJoCo model, controls and state stay on CUDA, and each world has an
independent overflow/non-finite estop latch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import yaml


WARP_BATCH_CONFIG_SCHEMA = 1
MAX_TORQUE_FRACTION_OF_RATED = 0.80
DOMAIN_RANDOMIZATION_FIELDS = (
    "body_mass",
    "body_inertia",
    "dof_damping",
    "geom_friction",
    "actuator_strength",
)


class WarpBatchError(RuntimeError):
    """Raised when the GPU batch harness cannot safely start."""


@dataclass(frozen=True)
class WarpRuntimeConfig:
    device: str
    cache_path: Path
    temporary_path: Path
    cuda_graph: bool
    use_precompiled_headers: bool


@dataclass(frozen=True)
class WarpDataCapacity:
    nconmax: int
    nccdmax: int
    njmax: int
    njmax_nnz: int
    naconmax: int
    naccdmax: int
    nvmax: int


@dataclass(frozen=True)
class WarpSafetyConfig:
    torque_fraction_of_rated: float
    estop_on_nonfinite_control: bool
    estop_on_nonfinite_state: bool
    estop_on_overflow: bool


@dataclass(frozen=True)
class WarpFallGuardConfig:
    enabled: bool
    max_attitude_error_rad: float
    max_root_height_drop_m: float


@dataclass(frozen=True)
class WarpPreflightConfig:
    verify_single_step_parity: bool
    verify_estop: bool
    qpos_max_abs_error: float
    qvel_max_abs_error: float
    sensordata_max_abs_error: float


@dataclass(frozen=True)
class WarpDomainRandomizationRanges:
    """Multiplicative perturbation ranges for GPU model parameters.

    Values are relative deltas: ``0.10`` means +10 percent and ``-0.10``
    means -10 percent.  Actuator strength is constrained to at most zero so
    the configured 80-percent torque reserve cannot be bypassed.
    """

    body_mass: tuple[float, float] = (0.0, 0.0)
    body_inertia: tuple[float, float] = (0.0, 0.0)
    dof_damping: tuple[float, float] = (0.0, 0.0)
    geom_friction: tuple[float, float] = (0.0, 0.0)
    actuator_strength: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class WarpDomainRandomizationNoise:
    """Gaussian noise standard deviation for sampled relative deltas."""

    std: float = 0.0


@dataclass(frozen=True)
class WarpDomainRandomizationDelay:
    """Physical substep delay before sampled parameters take effect."""

    steps: int = 0


@dataclass(frozen=True)
class WarpDomainRandomizationConfig:
    """Strict, optional GPU domain-randomization configuration."""

    enabled: bool = False
    seed: int = 0
    ranges: WarpDomainRandomizationRanges = WarpDomainRandomizationRanges()
    noise: WarpDomainRandomizationNoise = WarpDomainRandomizationNoise()
    delay: WarpDomainRandomizationDelay = WarpDomainRandomizationDelay()
    terrain_geometry_randomization: bool = False


@dataclass(frozen=True)
class WarpBatchConfig:
    source_path: Path
    backend: str
    xml_path: Path
    num_worlds: int
    physics_substeps_per_action: int
    smoke_actions: int
    runtime: WarpRuntimeConfig
    capacity: WarpDataCapacity
    safety: WarpSafetyConfig
    fall_guard: WarpFallGuardConfig
    preflight: WarpPreflightConfig
    controller_backend: str
    ppo_training_enabled: bool
    domain_randomization: WarpDomainRandomizationConfig = WarpDomainRandomizationConfig()


@dataclass(frozen=True)
class WarpBatchStep:
    """Views of GPU-resident post-step buffers.

    ``qpos``, ``qvel``, ``sensordata``, ``terminated`` and ``estopped`` stay
    on CUDA.  Callers should keep them there during a future vector rollout.
    """

    qpos: Any
    qvel: Any
    sensordata: Any
    time: Any
    terminated: Any
    estopped: Any
    overflow: Any
    applied_forces: Any | None = None


@dataclass(frozen=True)
class WarpPreflightReport:
    device: str
    num_worlds: int
    physics_steps: int
    elapsed_seconds: float
    aggregate_steps_per_second: float
    terminated_worlds: int
    overflowed_worlds: int
    finite_state: bool
    parity_qpos_max_abs_error: float
    parity_qvel_max_abs_error: float
    parity_sensordata_max_abs_error: float
    estop_probe_passed: bool
    mujoco_version: str
    mujoco_warp_version: str
    warp_version: str


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WarpBatchError(f"{name} must be a YAML mapping")
    return value


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise WarpBatchError(f"missing required configuration key: {name}")
    return mapping[name]


def _resolve_config_path(source_path: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WarpBatchError(f"{name} must be a non-empty path string")
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (source_path.parent / candidate).resolve()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WarpBatchError(f"{name} must be a positive integer")
    return int(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise WarpBatchError(f"{name} must be boolean")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarpBatchError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise WarpBatchError(f"{name} must be finite and positive")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WarpBatchError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarpBatchError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise WarpBatchError(f"{name} must be finite")
    return result


def _relative_range(value: object, name: str, *, upper_bound: float | None = None) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise WarpBatchError(f"{name} must be a two-element [low, high] range")
    low = _finite_number(value[0], f"{name}[0]")
    high = _finite_number(value[1], f"{name}[1]")
    if low > high or low <= -1.0:
        raise WarpBatchError(f"{name} must satisfy -1 < low <= high")
    if upper_bound is not None and high > upper_bound:
        raise WarpBatchError(f"{name} high must be <= {upper_bound:g}")
    return low, high


def _domain_randomization_config(value: object) -> WarpDomainRandomizationConfig:
    """Parse the optional strict DR block; omission means all-zero/no-op."""

    if value is None:
        return WarpDomainRandomizationConfig()
    root = _mapping(value, "domain_randomization")
    expected = {"enabled", "seed", "ranges", "noise", "delay", "terrain_geometry_randomization"}
    unknown = set(root) - expected
    if unknown:
        raise WarpBatchError(f"domain_randomization has unknown keys: {sorted(unknown)}")
    enabled = _boolean(root.get("enabled", False), "domain_randomization.enabled")
    seed = _nonnegative_int(root.get("seed", 0), "domain_randomization.seed")

    ranges_raw = _mapping(root.get("ranges", {}), "domain_randomization.ranges")
    range_expected = set(DOMAIN_RANDOMIZATION_FIELDS)
    range_unknown = set(ranges_raw) - range_expected
    if range_unknown:
        raise WarpBatchError(f"domain_randomization.ranges has unknown keys: {sorted(range_unknown)}")
    ranges = WarpDomainRandomizationRanges(
        body_mass=_relative_range(ranges_raw.get("body_mass", (0.0, 0.0)), "domain_randomization.ranges.body_mass"),
        body_inertia=_relative_range(ranges_raw.get("body_inertia", (0.0, 0.0)), "domain_randomization.ranges.body_inertia"),
        dof_damping=_relative_range(ranges_raw.get("dof_damping", (0.0, 0.0)), "domain_randomization.ranges.dof_damping"),
        geom_friction=_relative_range(ranges_raw.get("geom_friction", (0.0, 0.0)), "domain_randomization.ranges.geom_friction"),
        actuator_strength=_relative_range(
            ranges_raw.get("actuator_strength", (0.0, 0.0)),
            "domain_randomization.ranges.actuator_strength",
            upper_bound=0.0,
        ),
    )

    noise_raw = root.get("noise", {})
    if isinstance(noise_raw, (int, float)) and not isinstance(noise_raw, bool):
        noise = WarpDomainRandomizationNoise(std=_finite_number(noise_raw, "domain_randomization.noise"))
    else:
        noise_map = _mapping(noise_raw, "domain_randomization.noise")
        unknown_noise = set(noise_map) - {"std"}
        if unknown_noise:
            raise WarpBatchError(f"domain_randomization.noise has unknown keys: {sorted(unknown_noise)}")
        noise = WarpDomainRandomizationNoise(
            std=_finite_number(noise_map.get("std", 0.0), "domain_randomization.noise.std")
        )
    if noise.std < 0.0:
        raise WarpBatchError("domain_randomization.noise.std must be non-negative")

    delay_raw = root.get("delay", {})
    if isinstance(delay_raw, int) and not isinstance(delay_raw, bool):
        delay = WarpDomainRandomizationDelay(steps=_nonnegative_int(delay_raw, "domain_randomization.delay"))
    else:
        delay_map = _mapping(delay_raw, "domain_randomization.delay")
        unknown_delay = set(delay_map) - {"steps"}
        if unknown_delay:
            raise WarpBatchError(f"domain_randomization.delay has unknown keys: {sorted(unknown_delay)}")
        delay = WarpDomainRandomizationDelay(
            steps=_nonnegative_int(delay_map.get("steps", 0), "domain_randomization.delay.steps")
        )

    terrain_geometry_randomization = _boolean(
        root.get("terrain_geometry_randomization", False),
        "domain_randomization.terrain_geometry_randomization",
    )
    if terrain_geometry_randomization:
        raise WarpBatchError("terrain/hfield geometry randomization is unsupported and must remain false")
    if not enabled and (noise.std != 0.0 or delay.steps != 0 or any(getattr(ranges, field) != (0.0, 0.0) for field in DOMAIN_RANDOMIZATION_FIELDS)):
        raise WarpBatchError("domain_randomization.enabled must be true when non-zero DR parameters are configured")
    return WarpDomainRandomizationConfig(
        enabled=enabled,
        seed=seed,
        ranges=ranges,
        noise=noise,
        delay=delay,
        terrain_geometry_randomization=terrain_geometry_randomization,
    )


def _signed_rated_control_limits(model: Any, torque_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return signed GPU control caps from the model's explicit ranges.

    MuJoCo permits asymmetric control and force ranges.  Intersect every
    declared range before applying the 80% safety derating; assuming a
    symmetric magnitude here could command outside an actuator's valid side.
    """

    ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    force_range = np.asarray(model.actuator_forcerange, dtype=np.float64)
    ctrl_limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    force_limited = np.asarray(model.actuator_forcelimited, dtype=bool)
    count = int(model.nu)
    if ctrl_range.shape != (count, 2) or force_range.shape != (count, 2):
        raise WarpBatchError("actuator control/force ranges have an unexpected shape")
    if ctrl_limited.shape != (count,) or force_limited.shape != (count,):
        raise WarpBatchError("actuator range-limit flags have an unexpected shape")
    if np.any(~ctrl_limited & ~force_limited):
        raise WarpBatchError(
            "every actuator needs an explicit control or force range to establish a rated torque cap"
        )

    lower = np.full(count, -np.inf, dtype=np.float64)
    upper = np.full(count, np.inf, dtype=np.float64)
    if np.any(ctrl_limited):
        lower = np.maximum(lower, np.where(ctrl_limited, ctrl_range[:, 0], -np.inf))
        upper = np.minimum(upper, np.where(ctrl_limited, ctrl_range[:, 1], np.inf))
    if np.any(force_limited):
        lower = np.maximum(lower, np.where(force_limited, force_range[:, 0], -np.inf))
        upper = np.minimum(upper, np.where(force_limited, force_range[:, 1], np.inf))
    if (
        np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(lower >= upper)
        or np.any(lower > 0.0)
        or np.any(upper < 0.0)
    ):
        raise WarpBatchError("actuator ranges must be finite, ordered, and include zero")
    return lower * torque_fraction, upper * torque_fraction


def load_warp_batch_config(path: str | Path) -> WarpBatchConfig:
    """Load the strictly-scoped GPU batch preflight configuration from YAML."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise WarpBatchError(f"unable to read Warp batch configuration {source_path}: {error}") from error
    root = _mapping(raw, "Warp batch configuration")
    schema_version = _required(root, "schema_version")
    if schema_version != WARP_BATCH_CONFIG_SCHEMA:
        raise WarpBatchError(
            f"unsupported Warp batch configuration schema {schema_version!r}; "
            f"expected {WARP_BATCH_CONFIG_SCHEMA}"
        )
    backend = _required(root, "backend")
    if backend != "mujoco_warp":
        raise WarpBatchError("backend must be 'mujoco_warp'")

    runtime_raw = _mapping(_required(root, "runtime"), "runtime")
    device = _required(runtime_raw, "device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise WarpBatchError("runtime.device must name a CUDA device such as 'cuda:0'")
    runtime = WarpRuntimeConfig(
        device=device,
        cache_path=_resolve_config_path(source_path, _required(runtime_raw, "cache_path"), "runtime.cache_path"),
        temporary_path=_resolve_config_path(
            source_path, _required(runtime_raw, "temporary_path"), "runtime.temporary_path"
        ),
        cuda_graph=_boolean(_required(runtime_raw, "cuda_graph"), "runtime.cuda_graph"),
        use_precompiled_headers=_boolean(
            runtime_raw.get("use_precompiled_headers", True),
            "runtime.use_precompiled_headers",
        ),
    )
    if runtime.cuda_graph:
        raise WarpBatchError(
            "CUDA graph capture is disabled until the GPU controller, per-world estop, and reset paths "
            "have been parity-validated"
        )

    capacity_raw = _mapping(_required(root, "data_capacity"), "data_capacity")
    capacity = WarpDataCapacity(
        nconmax=_positive_int(_required(capacity_raw, "nconmax"), "data_capacity.nconmax"),
        nccdmax=_positive_int(_required(capacity_raw, "nccdmax"), "data_capacity.nccdmax"),
        njmax=_positive_int(_required(capacity_raw, "njmax"), "data_capacity.njmax"),
        njmax_nnz=_positive_int(_required(capacity_raw, "njmax_nnz"), "data_capacity.njmax_nnz"),
        naconmax=_positive_int(_required(capacity_raw, "naconmax"), "data_capacity.naconmax"),
        naccdmax=_positive_int(_required(capacity_raw, "naccdmax"), "data_capacity.naccdmax"),
        nvmax=_positive_int(_required(capacity_raw, "nvmax"), "data_capacity.nvmax"),
    )
    if capacity.nccdmax > capacity.nconmax:
        raise WarpBatchError("data_capacity.nccdmax cannot exceed nconmax")
    if capacity.naccdmax > capacity.naconmax:
        raise WarpBatchError("data_capacity.naccdmax cannot exceed naconmax")

    safety_raw = _mapping(_required(root, "safety"), "safety")
    torque_fraction = _required(safety_raw, "torque_fraction_of_rated")
    if isinstance(torque_fraction, bool) or not isinstance(torque_fraction, (int, float)):
        raise WarpBatchError("safety.torque_fraction_of_rated must be numeric")
    torque_fraction = float(torque_fraction)
    if not np.isfinite(torque_fraction) or not 0.0 < torque_fraction <= MAX_TORQUE_FRACTION_OF_RATED:
        raise WarpBatchError(
            "safety.torque_fraction_of_rated must be finite and within "
            f"(0, {MAX_TORQUE_FRACTION_OF_RATED:.2f}]"
        )
    safety = WarpSafetyConfig(
        torque_fraction_of_rated=torque_fraction,
        estop_on_nonfinite_control=_boolean(
            _required(safety_raw, "estop_on_nonfinite_control"), "safety.estop_on_nonfinite_control"
        ),
        estop_on_nonfinite_state=_boolean(
            _required(safety_raw, "estop_on_nonfinite_state"), "safety.estop_on_nonfinite_state"
        ),
        estop_on_overflow=_boolean(_required(safety_raw, "estop_on_overflow"), "safety.estop_on_overflow"),
    )

    fall_guard_raw = _mapping(_required(root, "fall_guard"), "fall_guard")
    fall_guard = WarpFallGuardConfig(
        enabled=_boolean(_required(fall_guard_raw, "enabled"), "fall_guard.enabled"),
        max_attitude_error_rad=_positive_float(
            _required(fall_guard_raw, "max_attitude_error_rad"), "fall_guard.max_attitude_error_rad"
        ),
        max_root_height_drop_m=_positive_float(
            _required(fall_guard_raw, "max_root_height_drop_m"), "fall_guard.max_root_height_drop_m"
        ),
    )

    preflight_raw = _mapping(_required(root, "preflight"), "preflight")
    preflight = WarpPreflightConfig(
        verify_single_step_parity=_boolean(
            _required(preflight_raw, "verify_single_step_parity"), "preflight.verify_single_step_parity"
        ),
        verify_estop=_boolean(_required(preflight_raw, "verify_estop"), "preflight.verify_estop"),
        qpos_max_abs_error=_positive_float(
            _required(preflight_raw, "qpos_max_abs_error"), "preflight.qpos_max_abs_error"
        ),
        qvel_max_abs_error=_positive_float(
            _required(preflight_raw, "qvel_max_abs_error"), "preflight.qvel_max_abs_error"
        ),
        sensordata_max_abs_error=_positive_float(
            _required(preflight_raw, "sensordata_max_abs_error"), "preflight.sensordata_max_abs_error"
        ),
    )

    scope = _mapping(_required(root, "scope"), "scope")
    controller_backend = _required(scope, "controller_backend")
    if controller_backend != "raw_controls_only":
        raise WarpBatchError("scope.controller_backend must be 'raw_controls_only' for this harness")
    ppo_training_enabled = _boolean(_required(scope, "ppo_training_enabled"), "scope.ppo_training_enabled")
    if ppo_training_enabled:
        raise WarpBatchError(
            "PPO training cannot be enabled: the CPU PhysicalLqr and complete safety state machine are not yet "
            "GPU-parity validated"
        )

    xml_path = _resolve_config_path(source_path, _required(root, "xml_path"), "xml_path")
    if not xml_path.is_file():
        raise WarpBatchError(f"MJCF XML does not exist: {xml_path}")
    domain_randomization = _domain_randomization_config(root.get("domain_randomization"))
    return WarpBatchConfig(
        source_path=source_path,
        backend=backend,
        xml_path=xml_path,
        num_worlds=_positive_int(_required(root, "num_worlds"), "num_worlds"),
        physics_substeps_per_action=_positive_int(
            _required(root, "physics_substeps_per_action"), "physics_substeps_per_action"
        ),
        smoke_actions=_positive_int(_required(root, "smoke_actions"), "smoke_actions"),
        runtime=runtime,
        capacity=capacity,
        safety=safety,
        fall_guard=fall_guard,
        preflight=preflight,
        controller_backend=controller_backend,
        ppo_training_enabled=ppo_training_enabled,
        domain_randomization=domain_randomization,
    )


def _runtime_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import mujoco
        import mujoco_warp
        import torch
        import warp
    except ImportError as error:
        raise WarpBatchError(
            "MuJoCo-Warp batch physics requires mujoco, mujoco_warp, warp-lang, and CUDA PyTorch"
        ) from error
    return mujoco, mujoco_warp, torch, warp


def _configure_warp_runtime(config: WarpRuntimeConfig, torch: Any, warp: Any) -> None:
    """Configure writable cache paths before the first Warp initialization."""

    def canonical_cache_path(value: str | Path) -> str:
        """Compare Windows Warp cache paths despite its ``\\\\?\\`` prefix."""

        raw = str(value)
        if raw.startswith("\\\\?\\"):
            raw = raw[4:]
        return os.path.normcase(os.path.normpath(os.path.abspath(raw)))

    config.cache_path.mkdir(parents=True, exist_ok=True)
    config.temporary_path.mkdir(parents=True, exist_ok=True)
    cache_path = str(config.cache_path)
    temporary_path = str(config.temporary_path)
    existing_cache = warp.config.kernel_cache_dir
    if existing_cache is not None:
        expected_versioned_path = config.cache_path / warp.config.version
        if canonical_cache_path(existing_cache) != canonical_cache_path(expected_versioned_path):
            raise WarpBatchError(
                "Warp was already initialized with a different kernel cache directory; start a new process with "
                f"WARP_CACHE_PATH={cache_path!r}"
            )
    else:
        os.environ["WARP_CACHE_PATH"] = cache_path
    os.environ["TEMP"] = temporary_path
    os.environ["TMP"] = temporary_path
    os.environ["TMPDIR"] = temporary_path

    # Warp 1.16's NVRTC precompiled-header path is optional.  It remains on
    # by default, but a YAML runtime can disable it for a Windows CUDA setup
    # where the driver owns the temporary PCH directory with restrictive ACLs.
    # This runs before ``warp.init`` and has no physics or control effect.
    warp.config.use_precompiled_headers = config.use_precompiled_headers

    if not torch.cuda.is_available():
        raise WarpBatchError("CUDA PyTorch is unavailable; install a CUDA-enabled torch build before using this backend")
    warp.init()
    warp.set_device(config.device)
    device = warp.get_device(config.device)
    if not getattr(device, "is_cuda", False):
        raise WarpBatchError(f"Warp device {config.device!r} is not a CUDA device")


class WarpPhysicsBatch:
    """GPU-resident batched MuJoCo physics with fail-closed raw-control guards.

    The public control input is physical actuator control in MuJoCo XML units,
    not the repository's 7-D residual policy action.  That distinction prevents
    a raw GPU batch from silently bypassing the CPU nominal controller.
    """

    def __init__(self, config: WarpBatchConfig) -> None:
        self.config = config
        self._mujoco, self._mujoco_warp, self._torch, self._warp = _runtime_modules()
        _configure_warp_runtime(config.runtime, self._torch, self._warp)

        self.host_model = self._mujoco.MjModel.from_xml_path(str(config.xml_path))
        host_data = self._mujoco.MjData(self.host_model)
        self._mujoco.mj_forward(self.host_model, host_data)
        if config.capacity.nvmax > self.host_model.nv:
            raise WarpBatchError(
                f"data_capacity.nvmax={config.capacity.nvmax} exceeds model.nv={self.host_model.nv}"
            )

        # Keep mutable DR fields batched per world from the first allocation.
        # The immutable terrain/collision topology remains shared model data;
        # no geom type, size, pose, mesh, or hfield field is ever randomized.
        dr_batch_sizes = {
            "body_mass": config.num_worlds,
            "body_inertia": config.num_worlds,
            "body_subtreemass": config.num_worlds,
            "body_invweight0": config.num_worlds,
            "dof_damping": config.num_worlds,
            "dof_invweight0": config.num_worlds,
            "geom_friction": config.num_worlds,
            "actuator_gainprm": config.num_worlds,
            "actuator_biasprm": config.num_worlds,
            "actuator_acc0": config.num_worlds,
        }
        with self._warp.ScopedDevice(config.runtime.device):
            self.model = self._mujoco_warp.put_model(self.host_model, batch_sizes=dr_batch_sizes)
            self.data = self._mujoco_warp.put_data(
                self.host_model,
                host_data,
                nworld=config.num_worlds,
                nconmax=config.capacity.nconmax,
                nccdmax=config.capacity.nccdmax,
                njmax=config.capacity.njmax,
                njmax_nnz=config.capacity.njmax_nnz,
                naconmax=config.capacity.naconmax,
                naccdmax=config.capacity.naccdmax,
                nvmax=config.capacity.nvmax,
            )

        self.device = self._torch.device(config.runtime.device)
        self.num_worlds = config.num_worlds
        self.num_actuators = int(self.host_model.nu)
        self.state_size = int(
            self._mujoco.mj_stateSize(self.host_model, self._mujoco.mjtState.mjSTATE_INTEGRATION)
        )
        self.qpos = self._warp.to_torch(self.data.qpos)
        self.qvel = self._warp.to_torch(self.data.qvel)
        self.ctrl = self._warp.to_torch(self.data.ctrl)
        self.qfrc_applied = self._warp.to_torch(self.data.qfrc_applied)
        self.sensordata = self._warp.to_torch(self.data.sensordata)
        self.time = self._warp.to_torch(self.data.time)
        self.overflow = self._warp.to_torch(self.data.overflow)
        if (
            self.qpos.device != self.device
            or self.qvel.device != self.device
            or self.qfrc_applied.device != self.device
            or self.qfrc_applied.shape != (self.num_worlds, int(self.host_model.nv))
        ):
            raise WarpBatchError("MuJoCo-Warp state was not allocated on the configured CUDA device")

        # All mutable model fields are zero-copy CUDA views.  Their baseline
        # copies and sampling workspaces are made once here; DR APIs only
        # mutate these resident buffers and never replace ``self.model``.
        self._dr_model_body_mass = self._warp.to_torch(self.model.body_mass)
        self._dr_model_body_inertia = self._warp.to_torch(self.model.body_inertia)
        self._dr_model_dof_damping = self._warp.to_torch(self.model.dof_damping)
        self._dr_model_geom_friction = self._warp.to_torch(self.model.geom_friction)
        self._dr_model_actuator_gainprm = self._warp.to_torch(self.model.actuator_gainprm)
        expected_model_shapes = {
            "body_mass": (self.num_worlds, int(self.host_model.nbody)),
            "body_inertia": (self.num_worlds, int(self.host_model.nbody), 3),
            "dof_damping": (self.num_worlds, int(self.host_model.nv)),
            "geom_friction": (self.num_worlds, int(self.host_model.ngeom), 3),
            "actuator_gainprm": (self.num_worlds, self.num_actuators, 10),
        }
        for name, value in (
            ("body_mass", self._dr_model_body_mass),
            ("body_inertia", self._dr_model_body_inertia),
            ("dof_damping", self._dr_model_dof_damping),
            ("geom_friction", self._dr_model_geom_friction),
            ("actuator_gainprm", self._dr_model_actuator_gainprm),
        ):
            if value.shape != expected_model_shapes[name] or value.device != self.device:
                raise WarpBatchError(f"MuJoCo-Warp DR model field {name} was not allocated per CUDA world")
        self._initialize_domain_randomization_buffers()

        free_joints = np.flatnonzero(np.asarray(self.host_model.jnt_type, dtype=np.int32) == 0)
        if free_joints.size != 1:
            raise WarpBatchError("the GPU fall guard requires exactly one free-root joint")
        root_joint = int(free_joints[0])
        self._root_qpos_address = int(self.host_model.jnt_qposadr[root_joint])
        if self._root_qpos_address + 7 > self.host_model.nq:
            raise WarpBatchError("free-root joint quaternion is outside the model qpos range")
        reference_state = np.asarray(host_data.qpos, dtype=np.float32)
        reference_quaternion = reference_state[self._root_qpos_address + 3 : self._root_qpos_address + 7]
        reference_norm = float(np.linalg.norm(reference_quaternion))
        if not np.isfinite(reference_norm) or reference_norm <= 1.0e-6:
            raise WarpBatchError("free-root reference quaternion is invalid")
        self._reference_root_quaternion = self._torch.as_tensor(
            reference_quaternion / reference_norm, dtype=self._torch.float32, device=self.device
        )
        self._reference_root_height = float(reference_state[self._root_qpos_address + 2])
        if not np.isfinite(self._reference_root_height):
            raise WarpBatchError("free-root reference height is invalid")

        control_low, control_high = self._rated_control_limits()
        self._control_low = self._torch.as_tensor(control_low, dtype=self._torch.float32, device=self.device)
        self._control_high = self._torch.as_tensor(control_high, dtype=self._torch.float32, device=self.device)
        self._safe_controls = self._torch.zeros(
            (self.num_worlds, self.num_actuators), dtype=self._torch.float32, device=self.device
        )
        self._safe_controls_warp = self._warp.from_torch(self._safe_controls, dtype=self._warp.float32)
        # Generalized-force producers must reserve any actuator headroom on
        # the same DOF before reaching this generic batch interface.  The
        # batch still fails closed on malformed values.  These resident
        # buffers/views ensure ``step`` never constructs Warp arrays or
        # performs a per-substep allocation.
        self._safe_applied_forces = self._torch.zeros_like(self.qfrc_applied)
        self._safe_applied_forces_warp = self._warp.from_torch(
            self._safe_applied_forces, dtype=self._warp.float32
        )
        self._estopped = self._torch.zeros(self.num_worlds, dtype=self._torch.bool, device=self.device)
        self._step_failures = self._torch.zeros(self.num_worlds, dtype=self._torch.bool, device=self.device)
        self._all_worlds = self._torch.ones(self.num_worlds, dtype=self._torch.bool, device=self.device)
        self._all_worlds_warp = self._warp.from_torch(self._all_worlds, dtype=self._warp.bool)
        # A caller normally provides a freshly computed CUDA done mask.  Copy
        # it to this permanent view rather than creating a new ``from_torch``
        # Warp wrapper at every masked reset.
        self._masked_worlds = self._torch.zeros(self.num_worlds, dtype=self._torch.bool, device=self.device)
        self._masked_worlds_warp = self._warp.from_torch(self._masked_worlds, dtype=self._warp.bool)

    @property
    def estopped(self) -> Any:
        return self._estopped

    @property
    def domain_randomization_enabled(self) -> bool:
        """Whether this batch is configured to sample per-world DR profiles."""

        return self.config.domain_randomization.enabled

    @property
    def domain_randomization_factors(self) -> Any:
        """Resident ``[world, field]`` active multiplicative DR factors.

        Field order is ``body_mass``, ``body_inertia``, ``dof_damping``,
        ``geom_friction``, then ``actuator_strength``.  This is a CUDA view
        intended for diagnostics; callers must not mutate it directly.
        """

        return self._dr_active_factors

    @property
    def domain_randomization_pending(self) -> Any:
        """Resident CUDA mask for profiles waiting for their configured delay."""

        return self._dr_pending

    def _initialize_domain_randomization_buffers(self) -> None:
        """Allocate immutable baselines and all GPU DR workspaces once."""

        torch = self._torch
        config = self.config.domain_randomization
        model = self.host_model
        ranges = config.ranges
        relative_ranges = (
            ranges.body_mass,
            ranges.body_inertia,
            ranges.dof_damping,
            ranges.geom_friction,
            ranges.actuator_strength,
        )
        self._dr_field_count = len(DOMAIN_RANDOMIZATION_FIELDS)
        self._dr_base_body_mass = self._dr_model_body_mass.clone()
        self._dr_base_body_inertia = self._dr_model_body_inertia.clone()
        self._dr_base_dof_damping = self._dr_model_dof_damping.clone()
        self._dr_base_geom_friction = self._dr_model_geom_friction.clone()
        self._dr_base_actuator_gainprm = self._dr_model_actuator_gainprm.clone()
        self._dr_body_mass_candidate = torch.empty_like(self._dr_model_body_mass)
        self._dr_body_inertia_candidate = torch.empty_like(self._dr_model_body_inertia)
        self._dr_dof_damping_candidate = torch.empty_like(self._dr_model_dof_damping)
        self._dr_geom_friction_candidate = torch.empty_like(self._dr_model_geom_friction)

        # Static scenery and hfields are deliberately excluded even from the
        # friction update.  In particular, this API never writes geom type,
        # pose, size, mesh, hfield size, or hfield sample data.
        body_roots = np.asarray(model.body_rootid, dtype=np.int32)
        geom_bodies = np.asarray(model.geom_bodyid, dtype=np.int32)
        geom_types = np.asarray(model.geom_type, dtype=np.int32)
        hfield_type = int(self._mujoco.mjtGeom.mjGEOM_HFIELD)
        mutable_bodies = body_roots != 0
        mutable_geoms = mutable_bodies[geom_bodies] & (geom_types != hfield_type)
        self._dr_mutable_body_mask = torch.as_tensor(
            mutable_bodies.reshape(1, -1), dtype=torch.bool, device=self.device
        )
        self._dr_mutable_geom_mask = torch.as_tensor(
            mutable_geoms.reshape(1, -1, 1), dtype=torch.bool, device=self.device
        )

        strength_requested = config.enabled and (
            ranges.actuator_strength != (0.0, 0.0) or config.noise.std != 0.0
        )
        fixed_gain_type = int(self._mujoco.mjtGain.mjGAIN_FIXED)
        if strength_requested and np.any(np.asarray(model.actuator_gaintype, dtype=np.int32) != fixed_gain_type):
            raise WarpBatchError(
                "actuator_strength DR requires fixed-gain actuators; refusing to alter a non-fixed gain law"
            )

        self._dr_range_low = torch.as_tensor(
            [entry[0] for entry in relative_ranges], dtype=torch.float32, device=self.device
        )
        self._dr_range_high = torch.as_tensor(
            [entry[1] for entry in relative_ranges], dtype=torch.float32, device=self.device
        )
        self._dr_uniform = torch.empty(
            (self.num_worlds, self._dr_field_count), dtype=torch.float32, device=self.device
        )
        self._dr_normal = torch.empty_like(self._dr_uniform)
        self._dr_relative_candidate = torch.empty_like(self._dr_uniform)
        self._dr_sampled_factors = torch.ones_like(self._dr_uniform)
        self._dr_active_factors = torch.ones_like(self._dr_uniform)
        self._dr_pending = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self._dr_due = torch.zeros_like(self._dr_pending)
        self._dr_delay_remaining = torch.zeros(self.num_worlds, dtype=torch.int32, device=self.device)
        self._dr_generator = torch.Generator(device=self.device)
        self._dr_generator.manual_seed(config.seed)
        self._dr_requires_set_const = config.enabled and (
            ranges.body_mass != (0.0, 0.0)
            or ranges.body_inertia != (0.0, 0.0)
            or config.noise.std != 0.0
        )

    def _domain_randomization_mask(self, world_mask: Any | None) -> Any:
        if world_mask is None:
            return self._all_worlds
        return self._require_cuda_tensor(
            world_mask, (self.num_worlds,), "world_mask", dtype=self._torch.bool
        )

    def _write_domain_randomization_model(self) -> None:
        """Materialize active factors into the resident per-world model views."""

        torch = self._torch
        factors = self._dr_active_factors
        torch.mul(
            self._dr_base_body_mass,
            factors[:, 0].reshape(self.num_worlds, 1),
            out=self._dr_body_mass_candidate,
        )
        torch.where(
            self._dr_mutable_body_mask,
            self._dr_body_mass_candidate,
            self._dr_base_body_mass,
            out=self._dr_model_body_mass,
        )
        torch.mul(
            self._dr_base_body_inertia,
            factors[:, 1].reshape(self.num_worlds, 1, 1),
            out=self._dr_body_inertia_candidate,
        )
        torch.where(
            self._dr_mutable_body_mask.unsqueeze(-1),
            self._dr_body_inertia_candidate,
            self._dr_base_body_inertia,
            out=self._dr_model_body_inertia,
        )
        torch.mul(
            self._dr_base_dof_damping,
            factors[:, 2].reshape(self.num_worlds, 1),
            out=self._dr_dof_damping_candidate,
        )
        self._dr_model_dof_damping.copy_(self._dr_dof_damping_candidate)
        torch.mul(
            self._dr_base_geom_friction,
            factors[:, 3].reshape(self.num_worlds, 1, 1),
            out=self._dr_geom_friction_candidate,
        )
        torch.where(
            self._dr_mutable_geom_mask,
            self._dr_geom_friction_candidate,
            self._dr_base_geom_friction,
            out=self._dr_model_geom_friction,
        )
        self._dr_model_actuator_gainprm.copy_(self._dr_base_actuator_gainprm)
        torch.mul(
            self._dr_base_actuator_gainprm[:, :, 0],
            factors[:, 4].reshape(self.num_worlds, 1),
            out=self._dr_model_actuator_gainprm[:, :, 0],
        )

    def sample_domain_randomization(self, world_mask: Any | None = None) -> Any:
        """Sample deterministic CUDA DR profiles without changing physics yet.

        A zero configured delay commits the selected worlds immediately.  For
        a positive delay, call :meth:`advance_domain_randomization` at an
        episode/reset boundary and pass its CUDA result to
        :meth:`apply_domain_randomization`.  This design keeps model mutation
        and ``set_const`` out of the per-physics-substep hot path.
        """

        mask = self._domain_randomization_mask(world_mask)
        config = self.config.domain_randomization
        if not config.enabled:
            return self._dr_sampled_factors
        self._dr_uniform.uniform_(0.0, 1.0, generator=self._dr_generator)
        torch = self._torch
        torch.lerp(
            self._dr_range_low.unsqueeze(0),
            self._dr_range_high.unsqueeze(0),
            self._dr_uniform,
            out=self._dr_relative_candidate,
        )
        if config.noise.std > 0.0:
            self._dr_normal.normal_(0.0, config.noise.std, generator=self._dr_generator)
            self._dr_relative_candidate.add_(self._dr_normal)
        self._dr_relative_candidate.clamp_(min=-0.95)
        # Strength must never exceed its calibrated nominal value.  Combined
        # with the existing control clipping this retains the 80% torque cap.
        self._dr_relative_candidate[:, 4].clamp_(max=0.0)
        torch.add(self._dr_relative_candidate, 1.0, out=self._dr_relative_candidate)
        self._dr_sampled_factors[mask] = self._dr_relative_candidate[mask]
        self._dr_pending[mask] = True
        self._dr_delay_remaining[mask] = config.delay.steps
        if config.delay.steps == 0:
            self.apply_domain_randomization(mask)
        return self._dr_sampled_factors

    def advance_domain_randomization(self, *, steps: int = 1) -> Any:
        """Advance the explicit DR delay scheduler and return due CUDA worlds.

        This method intentionally does not mutate the model.  It is safe to
        call from a scheduler, while committing due profiles remains an
        explicit reset-boundary operation through ``apply_domain_randomization``.
        """

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        self._dr_due.zero_()
        if not self.config.domain_randomization.enabled:
            return self._dr_due
        self._dr_delay_remaining.sub_(steps).clamp_(min=0)
        self._dr_due.copy_(self._dr_pending)
        self._dr_due.logical_and_(self._dr_delay_remaining.eq(0))
        self._dr_pending.logical_and_(~self._dr_due)
        return self._dr_due

    def apply_domain_randomization(self, world_mask: Any | None = None) -> Any:
        """Commit sampled profiles to the existing GPU model at a safe boundary.

        ``mujoco_warp.set_const`` is used only when mass/inertia can change so
        derived inertial quantities remain coherent.  This API must not be
        called from ``step`` or a controller substep; call it after a masked
        reset, before the next rollout segment.
        """

        mask = self._domain_randomization_mask(world_mask)
        if not self.config.domain_randomization.enabled:
            return self._dr_active_factors
        self._dr_active_factors[mask] = self._dr_sampled_factors[mask]
        self._dr_pending[mask] = False
        self._dr_delay_remaining[mask] = 0
        self._write_domain_randomization_model()
        if self._dr_requires_set_const:
            self._mujoco_warp.set_const(self.model, self.data)
        self._mujoco_warp.forward(self.model, self.data)
        return self._dr_active_factors

    def reset_domain_randomization(self, world_mask: Any | None = None) -> Any:
        """Restore selected worlds to nominal model parameters on CUDA."""

        mask = self._domain_randomization_mask(world_mask)
        self._dr_active_factors[mask] = 1.0
        self._dr_sampled_factors[mask] = 1.0
        self._dr_pending[mask] = False
        self._dr_delay_remaining[mask] = 0
        self._write_domain_randomization_model()
        if self._dr_requires_set_const:
            self._mujoco_warp.set_const(self.model, self.data)
        self._mujoco_warp.forward(self.model, self.data)
        return self._dr_active_factors

    def forward(self) -> None:
        """Refresh derived GPU state without rebuilding model or data objects.

        MuJoCo-Warp does not guarantee that sensors and geometry transforms are
        refreshed after every integration step.  Task-level contact and leg
        safety checks therefore call this before consuming those views.
        """

        self._mujoco_warp.forward(self.model, self.data)

    def latch_estop(self, world_mask: Any) -> None:
        """Immediately make selected unsafe worlds torque-free on CUDA.

        This is intentionally independent of the policy path.  It writes only
        resident Torch/Warp views, so a post-step task safety failure cannot
        leave a non-zero command buffered for the next physical substep.
        """

        world_mask = self._require_cuda_tensor(
            world_mask, (self.num_worlds,), "world_mask", dtype=self._torch.bool
        )
        self._step_failures.logical_or_(world_mask)
        self._estopped.logical_or_(world_mask)
        self._safe_controls.masked_fill_(self._estopped.unsqueeze(1), 0.0)
        self._safe_applied_forces.masked_fill_(self._estopped.unsqueeze(1), 0.0)
        self._warp.copy(self.data.ctrl, self._safe_controls_warp)
        self._warp.copy(self.data.qfrc_applied, self._safe_applied_forces_warp)

    def set_fall_guard_reference(self, quaternion: Any, root_height_m: float) -> None:
        """Set a validated fixed reference used by the independent fall guard."""

        quaternion = self._require_cuda_tensor(
            quaternion, (4,), "reference quaternion", dtype=self._torch.float32
        )
        if not np.isfinite(float(root_height_m)):
            raise WarpBatchError("reference root height must be finite")
        norm = self._torch.linalg.vector_norm(quaternion)
        if bool((~self._torch.isfinite(norm) | (norm <= 1.0e-7)).item()):
            raise WarpBatchError("reference quaternion must have a finite non-zero norm")
        self._reference_root_quaternion.copy_(quaternion / norm)
        self._reference_root_height = float(root_height_m)

    def _rated_control_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return _signed_rated_control_limits(self.host_model, self.config.safety.torque_fraction_of_rated)

    def _require_cuda_tensor(self, value: Any, shape: tuple[int, ...], name: str, *, dtype: Any | None = None) -> Any:
        if not isinstance(value, self._torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if value.device != self.device:
            raise ValueError(f"{name} must reside on {self.device}, got {value.device}")
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {value.dtype}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous to preserve zero-copy Warp interop")
        return value

    def reset(self, world_mask: Any | None = None) -> None:
        """Reset selected worlds on GPU; no model/data object is recreated."""

        if world_mask is None:
            world_mask = self._all_worlds
            world_mask_warp = self._all_worlds_warp
        else:
            world_mask = self._require_cuda_tensor(
                world_mask, (self.num_worlds,), "world_mask", dtype=self._torch.bool
            )
            self._masked_worlds.copy_(world_mask)
            world_mask = self._masked_worlds
            world_mask_warp = self._masked_worlds_warp
        self._mujoco_warp.reset_data(self.model, self.data, reset=world_mask_warp)
        self._mujoco_warp.forward(self.model, self.data)
        self._estopped.masked_fill_(world_mask, False)
        self._step_failures.masked_fill_(world_mask, False)
        self._safe_controls.masked_fill_(world_mask.unsqueeze(1), 0.0)
        self._safe_applied_forces.masked_fill_(world_mask.unsqueeze(1), 0.0)
        self._warp.copy(self.data.ctrl, self._safe_controls_warp)
        self._warp.copy(self.data.qfrc_applied, self._safe_applied_forces_warp)
        self._record_step_health()

    def reset_to_state(
        self,
        qpos: Any,
        qvel: Any,
        controls: Any,
        world_mask: Any | None = None,
    ) -> None:
        """Reset selected worlds to preallocated GPU qpos/qvel/control views.

        This path avoids the first-use ``set_state`` kernel, which is fragile
        on Windows installations with restricted NVRTC temporary directories.
        It is also the normal masked-reset path for a calibrated flat task:
        MuJoCo-Warp owns the model/data allocation and only existing CUDA
        buffers are written here.
        """

        qpos = self._require_cuda_tensor(
            qpos, (self.num_worlds, int(self.host_model.nq)), "qpos", dtype=self._torch.float32
        )
        qvel = self._require_cuda_tensor(
            qvel, (self.num_worlds, int(self.host_model.nv)), "qvel", dtype=self._torch.float32
        )
        controls = self._require_cuda_tensor(
            controls, (self.num_worlds, self.num_actuators), "controls", dtype=self._torch.float32
        )
        if world_mask is None:
            world_mask = self._all_worlds
            world_mask_warp = self._all_worlds_warp
        else:
            world_mask = self._require_cuda_tensor(
                world_mask, (self.num_worlds,), "world_mask", dtype=self._torch.bool
            )
            self._masked_worlds.copy_(world_mask)
            world_mask = self._masked_worlds
            world_mask_warp = self._masked_worlds_warp
        self._mujoco_warp.reset_data(self.model, self.data, reset=world_mask_warp)
        self.qpos.masked_scatter_(world_mask.unsqueeze(1), qpos.masked_select(world_mask.unsqueeze(1)))
        self.qvel.masked_scatter_(world_mask.unsqueeze(1), qvel.masked_select(world_mask.unsqueeze(1)))
        self.time.masked_fill_(world_mask, 0.0)
        self.ctrl.masked_scatter_(world_mask.unsqueeze(1), controls.masked_select(world_mask.unsqueeze(1)))

        self._mujoco_warp.forward(self.model, self.data)
        self._estopped.masked_fill_(world_mask, False)
        self._step_failures.masked_fill_(world_mask, False)
        self._safe_controls[world_mask] = self._torch.clamp(
            controls[world_mask], min=self._control_low, max=self._control_high
        )
        self._safe_applied_forces.masked_fill_(world_mask.unsqueeze(1), 0.0)
        self._warp.copy(self.data.ctrl, self._safe_controls_warp)
        self._warp.copy(self.data.qfrc_applied, self._safe_applied_forces_warp)
        self._record_step_health()

    def set_integration_state(self, state: Any, world_mask: Any | None = None) -> None:
        """Set a GPU integration state for selected worlds, then refresh outputs."""

        state = self._require_cuda_tensor(
            state,
            (self.num_worlds, self.state_size),
            "state",
            dtype=self._torch.float32,
        )
        if world_mask is None:
            world_mask = self._all_worlds
            world_mask_warp = self._all_worlds_warp
        else:
            world_mask = self._require_cuda_tensor(
                world_mask, (self.num_worlds,), "world_mask", dtype=self._torch.bool
            )
            self._masked_worlds.copy_(world_mask)
            world_mask = self._masked_worlds
            world_mask_warp = self._masked_worlds_warp
        self._mujoco_warp.set_state(
            self.model,
            self.data,
            self._warp.from_torch(state, dtype=self._warp.float32),
            self._mujoco_warp.State.INTEGRATION,
            active=world_mask_warp,
        )
        self._mujoco_warp.forward(self.model, self.data)
        self._estopped.masked_fill_(world_mask, False)
        self._step_failures.masked_fill_(world_mask, False)
        self._safe_applied_forces.masked_fill_(world_mask.unsqueeze(1), 0.0)
        self._warp.copy(self.data.qfrc_applied, self._safe_applied_forces_warp)
        self._record_step_health()

    def _stage_controls(self, controls: Any) -> None:
        controls = self._require_cuda_tensor(
            controls,
            (self.num_worlds, self.num_actuators),
            "controls",
            dtype=self._torch.float32,
        )
        if self.config.safety.estop_on_nonfinite_control:
            self._step_failures.logical_or_(~self._torch.isfinite(controls).all(dim=1))
        self._safe_controls.copy_(self._torch.nan_to_num(controls, nan=0.0, posinf=0.0, neginf=0.0))
        self._torch.clamp(self._safe_controls, min=self._control_low, max=self._control_high, out=self._safe_controls)
        self._estopped.logical_or_(self._step_failures)
        self._safe_controls.masked_fill_(self._estopped.unsqueeze(1), 0.0)
        self._warp.copy(self.data.ctrl, self._safe_controls_warp)

    def _stage_applied_forces(self, applied_forces: Any | None) -> None:
        """Stage finite generalized forces without retaining stale inputs."""

        if applied_forces is None:
            self._safe_applied_forces.zero_()
        else:
            applied_forces = self._require_cuda_tensor(
                applied_forces,
                (self.num_worlds, int(self.host_model.nv)),
                "applied_forces",
                dtype=self._torch.float32,
            )
            finite = self._torch.isfinite(applied_forces).all(dim=1)
            self._step_failures.logical_or_(~finite)
            self._safe_applied_forces.copy_(
                self._torch.nan_to_num(applied_forces, nan=0.0, posinf=0.0, neginf=0.0)
            )
        self._estopped.logical_or_(self._step_failures)
        self._safe_applied_forces.masked_fill_(self._estopped.unsqueeze(1), 0.0)
        self._warp.copy(self.data.qfrc_applied, self._safe_applied_forces_warp)

    def _record_step_health(self) -> None:
        if self.config.safety.estop_on_overflow:
            self._step_failures.logical_or_(self.overflow.ne(0))
        if self.config.safety.estop_on_nonfinite_state:
            state_finite = self._torch.isfinite(self.qpos).all(dim=1)
            state_finite.logical_and_(self._torch.isfinite(self.qvel).all(dim=1))
            self._step_failures.logical_or_(~state_finite)
        if self.config.fall_guard.enabled:
            root_quaternion = self.qpos[:, self._root_qpos_address + 3 : self._root_qpos_address + 7]
            quaternion_norm = self._torch.linalg.vector_norm(root_quaternion, dim=1)
            normalized_quaternion = root_quaternion / quaternion_norm.clamp_min(1.0e-7).unsqueeze(1)
            dot = (normalized_quaternion * self._reference_root_quaternion.unsqueeze(0)).sum(dim=1).abs()
            attitude_error = 2.0 * self._torch.acos(dot.clamp(min=-1.0, max=1.0))
            root_height = self.qpos[:, self._root_qpos_address + 2]
            fall_guard_failed = (attitude_error > self.config.fall_guard.max_attitude_error_rad) | (
                root_height < self._reference_root_height - self.config.fall_guard.max_root_height_drop_m
            )
            self._step_failures.logical_or_(fall_guard_failed)
        self._estopped.logical_or_(self._step_failures)

    def step(
        self,
        controls: Any,
        *,
        physics_substeps: int | None = None,
        applied_forces: Any | None = None,
    ) -> WarpBatchStep:
        """Advance all worlds without CPU state transfer.

        A newly unsafe world has its controls set to zero before the next
        physical substep and stays terminal until ``reset`` is called.
        """

        substeps = self.config.physics_substeps_per_action if physics_substeps is None else physics_substeps
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("physics_substeps must be a positive integer")
        self._step_failures.zero_()
        for _ in range(substeps):
            self._stage_controls(controls)
            self._stage_applied_forces(applied_forces)
            self._mujoco_warp.step(self.model, self.data)
            self._record_step_health()
        self._safe_controls.masked_fill_(self._estopped.unsqueeze(1), 0.0)
        self._safe_applied_forces.masked_fill_(self._estopped.unsqueeze(1), 0.0)
        self._warp.copy(self.data.ctrl, self._safe_controls_warp)
        self._warp.copy(self.data.qfrc_applied, self._safe_applied_forces_warp)
        return WarpBatchStep(
            qpos=self.qpos,
            qvel=self.qvel,
            sensordata=self.sensordata,
            time=self.time,
            terminated=self._estopped,
            estopped=self._estopped,
            overflow=self.overflow,
            applied_forces=self._safe_applied_forces,
        )


def run_warp_preflight(config: WarpBatchConfig) -> WarpPreflightReport:
    """Run genuine GPU steps with safe zero controls and return scalar diagnostics."""

    batch = WarpPhysicsBatch(config)
    parity_qpos_error = 0.0
    parity_qvel_error = 0.0
    parity_sensor_error = 0.0
    if config.preflight.verify_single_step_parity:
        parity_qpos_error, parity_qvel_error, parity_sensor_error = _single_step_parity(batch)
        tolerances = config.preflight
        if (
            parity_qpos_error > tolerances.qpos_max_abs_error
            or parity_qvel_error > tolerances.qvel_max_abs_error
            or parity_sensor_error > tolerances.sensordata_max_abs_error
        ):
            raise WarpBatchError(
                "CPU/GPU one-step parity exceeded configured tolerance: "
                f"qpos={parity_qpos_error:.3e}/{tolerances.qpos_max_abs_error:.3e}, "
                f"qvel={parity_qvel_error:.3e}/{tolerances.qvel_max_abs_error:.3e}, "
                f"sensordata={parity_sensor_error:.3e}/{tolerances.sensordata_max_abs_error:.3e}"
            )

    estop_probe_passed = True
    if config.preflight.verify_estop:
        estop_probe_passed = _estop_probe(batch)
        if not estop_probe_passed:
            raise WarpBatchError("per-world non-finite-control estop probe failed")

    controls = batch._torch.zeros(
        (batch.num_worlds, batch.num_actuators), dtype=batch._torch.float32, device=batch.device
    )
    started = perf_counter()
    result: WarpBatchStep | None = None
    for _ in range(config.smoke_actions):
        result = batch.step(controls)
    batch._warp.synchronize()
    elapsed = perf_counter() - started
    if result is None:
        raise AssertionError("smoke_actions must be positive")
    physics_steps = config.smoke_actions * config.physics_substeps_per_action
    terminated_worlds = int(result.terminated.sum().item())
    overflowed_worlds = int(result.overflow.ne(0).sum().item())
    finite_state = bool(batch._torch.isfinite(result.qpos).all().item() and batch._torch.isfinite(result.qvel).all().item())
    return WarpPreflightReport(
        device=str(batch.device),
        num_worlds=batch.num_worlds,
        physics_steps=physics_steps,
        elapsed_seconds=elapsed,
        aggregate_steps_per_second=(physics_steps * batch.num_worlds / elapsed) if elapsed > 0.0 else float("inf"),
        terminated_worlds=terminated_worlds,
        overflowed_worlds=overflowed_worlds,
        finite_state=finite_state,
        parity_qpos_max_abs_error=parity_qpos_error,
        parity_qvel_max_abs_error=parity_qvel_error,
        parity_sensordata_max_abs_error=parity_sensor_error,
        estop_probe_passed=estop_probe_passed,
        mujoco_version=batch._mujoco.__version__,
        mujoco_warp_version=batch._mujoco_warp.__version__,
        warp_version=batch._warp.__version__,
    )


def _single_step_parity(batch: WarpPhysicsBatch) -> tuple[float, float, float]:
    """Compare one unactuated raw CPU/GPU MuJoCo step from the same model state."""

    cpu_data = batch._mujoco.MjData(batch.host_model)
    batch._mujoco.mj_forward(batch.host_model, cpu_data)
    cpu_data.ctrl[:] = 0.0
    batch._mujoco.mj_step(batch.host_model, cpu_data)

    batch.reset()
    controls = batch._torch.zeros(
        (batch.num_worlds, batch.num_actuators), dtype=batch._torch.float32, device=batch.device
    )
    gpu_result = batch.step(controls, physics_substeps=1)
    batch._warp.synchronize()
    gpu_qpos = gpu_result.qpos[0].detach().cpu().numpy()
    gpu_qvel = gpu_result.qvel[0].detach().cpu().numpy()
    gpu_sensor = gpu_result.sensordata[0].detach().cpu().numpy()
    qpos_error = float(np.max(np.abs(cpu_data.qpos - gpu_qpos)))
    qvel_error = float(np.max(np.abs(cpu_data.qvel - gpu_qvel)))
    sensor_error = float(np.max(np.abs(cpu_data.sensordata - gpu_sensor)))
    batch.reset()
    return qpos_error, qvel_error, sensor_error


def _estop_probe(batch: WarpPhysicsBatch) -> bool:
    """Verify that one malformed CUDA control terminates only its own world."""

    if batch.num_worlds < 2:
        raise WarpBatchError("the estop probe requires at least two worlds")
    controls = batch._torch.zeros(
        (batch.num_worlds, batch.num_actuators), dtype=batch._torch.float32, device=batch.device
    )
    controls[0, 0] = float("nan")
    result = batch.step(controls, physics_substeps=1)
    batch._warp.synchronize()
    first_world_estopped = bool(result.estopped[0].item())
    other_worlds_safe = not bool(result.estopped[1:].any().item())
    controls_zeroed = bool(batch._torch.eq(batch._safe_controls[0], 0.0).all().item())
    reset_mask = batch._torch.zeros(batch.num_worlds, dtype=batch._torch.bool, device=batch.device)
    reset_mask[0] = True
    batch.reset(reset_mask)
    reset_cleared = not bool(batch.estopped[0].item())
    batch.reset()
    finite_fall_estopped = True
    if batch.config.fall_guard.enabled:
        overturned_quaternion = batch._torch.tensor((0.0, 1.0, 0.0, 0.0), dtype=batch._torch.float32, device=batch.device)
        batch.qpos[0, batch._root_qpos_address + 3 : batch._root_qpos_address + 7] = overturned_quaternion
        batch._mujoco_warp.forward(batch.model, batch.data)
        result = batch.step(controls, physics_substeps=1)
        batch._warp.synchronize()
        finite_fall_estopped = bool(result.estopped[0].item()) and not bool(result.estopped[1:].any().item())
        batch.reset()
    return first_world_estopped and other_worlds_safe and controls_zeroed and reset_cleared and finite_fall_estopped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fail-closed MuJoCo-Warp GPU batch physics preflight.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/warp_batch_preflight.yaml"),
        help="YAML configuration for the GPU batch physics harness.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_warp_preflight(load_warp_batch_config(args.config))
    print(
        "MuJoCo-Warp batch preflight: "
        f"device={report.device} worlds={report.num_worlds} physics_steps={report.physics_steps} "
        f"aggregate_steps_per_second={report.aggregate_steps_per_second:.1f} "
        f"terminated={report.terminated_worlds} overflowed={report.overflowed_worlds} "
        f"finite={report.finite_state} parity=(qpos:{report.parity_qpos_max_abs_error:.3e},"
        f"qvel:{report.parity_qvel_max_abs_error:.3e},sensor:{report.parity_sensordata_max_abs_error:.3e}) "
        f"estop_probe={report.estop_probe_passed} versions="
        f"mujoco:{report.mujoco_version}/mujoco_warp:{report.mujoco_warp_version}/warp:{report.warp_version}"
    )
    if report.terminated_worlds or report.overflowed_worlds or not report.finite_state:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
