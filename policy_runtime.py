"""Runtime loading for residual PPO policies used by viewers and evaluators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

import env as environment_definition
from train_ppo import ActorCritic, CHECKPOINT_FORMAT_VERSION, REWARD_SCHEMA


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PpoPolicyRuntime:
    """A checkpointed residual PPO actor in inference mode."""

    checkpoint_path: Path
    policy: ActorCritic
    device: torch.device
    observation_size: int
    action_size: int
    metadata: dict[str, Any]

    @property
    def timesteps(self) -> int:
        return int(self.metadata.get("timesteps", 0))

    def action(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self.observation_size,):
            raise ValueError(
                f"policy expects observation shape ({self.observation_size},), got {values.shape}"
            )
        observation_tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, _, _ = self.policy.sample_action(
                observation_tensor,
                deterministic=deterministic,
            )
        return action_tensor.squeeze(0).cpu().numpy().astype(np.float32)


def load_ppo_residual_policy(
    checkpoint_path: Path | str,
    observation_size: int,
    action_size: int,
    *,
    device: str | torch.device = "cpu",
) -> PpoPolicyRuntime:
    """Load a PPO residual policy after validating its runtime interface."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PPO checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"PPO checkpoint is not a dictionary: {path}")
    if checkpoint.get("algorithm") != "ppo":
        raise ValueError(f"checkpoint is not PPO: algorithm={checkpoint.get('algorithm')!r}")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "checkpoint uses an incompatible locomotion-controller interface; "
            "start a new RMUC training run with the current trainer."
        )
    if checkpoint.get("action_semantics") != "residual":
        raise ValueError(
            "checkpoint action semantics are incompatible; expected residual actions, "
            f"got {checkpoint.get('action_semantics')!r}"
        )
    task_config = checkpoint.get("task_config")
    if not isinstance(task_config, dict):
        raise ValueError("PPO checkpoint is missing locomotion-command metadata")
    if task_config.get("locomotion_command_schema") != environment_definition.LOCOMOTION_COMMAND_SCHEMA:
        raise ValueError("PPO checkpoint uses an incompatible locomotion-command schema")
    if task_config.get("residual_authority_schema") != environment_definition.RESIDUAL_AUTHORITY_SCHEMA:
        raise ValueError("PPO checkpoint uses an incompatible residual-action authority schema")
    if task_config.get("reward_schema") != REWARD_SCHEMA:
        raise ValueError(
            "PPO checkpoint uses an incompatible locomotion reward schema; "
            "load a checkpoint trained with the current RMUC terrain contract."
        )
    saved_observation_size = checkpoint.get("observation_size")
    saved_action_size = checkpoint.get("action_size")
    if saved_observation_size != observation_size or saved_action_size != action_size:
        raise ValueError(
            "checkpoint interface does not match the current environment: "
            f"checkpoint obs/action=({saved_observation_size}, {saved_action_size}), "
            f"environment=({observation_size}, {action_size})"
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("PPO checkpoint is missing model_state_dict")
    runtime_device = torch.device(device)
    policy = ActorCritic(observation_size, action_size).to(runtime_device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return PpoPolicyRuntime(
        checkpoint_path=path,
        policy=policy,
        device=runtime_device,
        observation_size=observation_size,
        action_size=action_size,
        metadata=checkpoint,
    )


def environment_compatibility_warnings(
    runtime: PpoPolicyRuntime,
    *,
    xml_path: Path,
    lqr_source_path: Path,
) -> list[str]:
    """Report transfer differences without blocking a valid same-interface replay."""
    task_config = runtime.metadata.get("task_config")
    environment_config = task_config.get("environment_config") if isinstance(task_config, dict) else None
    if not isinstance(environment_config, dict):
        return ["checkpoint has no environment fingerprint; this replay is not provenance-verified"]

    warnings: list[str] = []
    expected_xml_hash = environment_config.get("mjcf_sha256")
    actual_xml_hash = file_sha256(xml_path)
    if expected_xml_hash != actual_xml_hash:
        warnings.append(
            "MJCF differs from the checkpoint; results are a cross-scene transfer evaluation"
        )
    expected_lqr_hash = environment_config.get("lqr_source_sha256")
    actual_lqr_hash = file_sha256(lqr_source_path)
    if expected_lqr_hash is not None and expected_lqr_hash != actual_lqr_hash:
        warnings.append(
            "LQR source differs from the checkpoint; results are not an exact training replay"
        )
    return warnings
