"""Behaviour-cloning trainer for the safe PhysicalLqr residual interface.

The environment always executes ``PhysicalLqr.command`` and interprets policy
outputs as bounded residual corrections.  BC therefore learns the neutral
residual action, producing a checkpoint that is directly compatible with
``train_ppo.py --bc-checkpoint``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

import env as environment_definition
from env import DEFAULT_EPISODE_SECONDS, DomainRandomizationConfig, WheelLegResidualEnv
import lqr_deploy as lqr


class BehaviorCloningPolicy(nn.Module):
    """Standalone tanh actor with the same actor layout as the PPO policy."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return torch.tanh(self.actor(observations))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone low-centre PhysicalLqr demonstrations with PyTorch.")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-speed",
        type=float,
        default=None,
        help=(
            "Optional forward speed limit in m/s; valid range is "
            f"{lqr.MIN_FORWARD_SPEED_LIMIT_MPS:.1f}..{lqr.MAX_FORWARD_SPEED_MPS:.1f}."
        ),
    )
    parser.add_argument(
        "--yaw-range-deg",
        type=float,
        default=45.0,
        help="Sample turn commands uniformly within +/- this yaw delta in degrees.",
    )
    parser.add_argument(
        "--jump-probability",
        type=float,
        default=0.5,
        help="Fraction of episodes that execute the LQR jump sequence.",
    )
    parser.add_argument(
        "--jump-at",
        type=float,
        default=0.80,
        help="Simulation time in seconds at which a scheduled jump begins.",
    )
    parser.add_argument(
        "--domain-randomization",
        dest="domain_randomization",
        action="store_true",
        default=True,
        help="Randomize physical dynamics, sensors and control delay at every episode reset (default).",
    )
    parser.add_argument(
        "--no-domain-randomization",
        dest="domain_randomization",
        action="store_false",
        help="Disable dynamics/random-sensor randomization for a nominal-model run.",
    )
    parser.add_argument(
        "--target",
        choices=("residual",),
        default="residual",
        help="Only residual targets are valid for WheelLegResidualEnv.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "bc_residual.pt")
    parser.add_argument("--smoke", action="store_true", help="Collect a small dataset and run one optimization epoch.")
    args = parser.parse_args()
    if args.max_speed is not None:
        try:
            args.max_speed = lqr.validate_forward_speed_limit(args.max_speed)
        except ValueError as error:
            parser.error(str(error))
    if not 0.0 <= args.yaw_range_deg <= 180.0:
        parser.error("--yaw-range-deg must be within 0..180")
    if not 0.0 <= args.jump_probability <= 1.0:
        parser.error("--jump-probability must be within 0..1")
    if not 0.0 <= args.jump_at < DEFAULT_EPISODE_SECONDS:
        parser.error(f"--jump-at must be within 0..{DEFAULT_EPISODE_SECONDS:g} seconds")
    return args


def collect_demonstrations(
    environment: WheelLegResidualEnv,
    samples: int,
    target_kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if target_kind != "residual":
        raise ValueError("only residual demonstrations are valid for this environment")
    rng = np.random.default_rng(seed)
    observation, _ = environment.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    zero_residual = np.zeros(environment.action_space.shape, dtype=np.float32)
    for _ in range(samples):
        observations.append(observation.copy())
        actions.append(environment.expert_action())
        observation, _, terminated, truncated, _ = environment.step(zero_residual)
        if terminated or truncated:
            speed = environment.sample_command_speed(rng)
            observation, _ = environment.reset(options={"command_speed": speed})
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def train_bc(
    observations: np.ndarray,
    expert_actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> BehaviorCloningPolicy:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    policy = BehaviorCloningPolicy(observations.shape[1], expert_actions.shape[1]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(expert_actions, dtype=torch.float32, device=device)
    sample_count = observation_tensor.shape[0]

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(sample_count, device=device)
        losses: list[float] = []
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size]
            prediction = policy(observation_tensor[indices])
            loss = torch.nn.functional.mse_loss(prediction, target_tensor[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} bc_mse={np.mean(losses):.6f}")
    return policy


def save_policy(
    path: Path,
    policy: BehaviorCloningPolicy,
    observation_size: int,
    action_size: int,
    target_kind: str,
    task_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "algorithm": "behavior_cloning",
            "target_kind": target_kind,
            "observation_size": observation_size,
            "action_size": action_size,
            "task_config": task_config,
            "actor_state_dict": policy.actor.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.samples = min(args.samples, 32)
        args.epochs = 1
        args.batch_size = min(args.batch_size, args.samples)
        args.output = Path("artifacts") / f"bc_{args.target}_smoke.pt"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    task_config = {
        "max_speed_mps": float(
            args.max_speed if args.max_speed is not None else lqr.DEFAULT_FORWARD_SPEED_LIMIT_MPS
        ),
        "yaw_range_deg": float(args.yaw_range_deg),
        "jump_probability": float(args.jump_probability),
        "jump_at_s": float(args.jump_at),
        "environment_config": {
            "episode_seconds": float(DEFAULT_EPISODE_SECONDS),
            "control_decimation": int(environment_definition.DEFAULT_CONTROL_DECIMATION),
            "mjcf_sha256": hashlib.sha256(lqr.XML_PATH.read_bytes()).hexdigest(),
            "jump_thrust_length_m": float(lqr.JUMP_THRUST_LENGTH_M),
            "jump_thrust_rate_mps": float(lqr.JUMP_THRUST_RATE_MPS),
            "domain_randomization": (
                DomainRandomizationConfig.training_defaults().as_dict()
                if args.domain_randomization
                else DomainRandomizationConfig.disabled().as_dict()
            ),
        },
        "domain_randomization": (
            DomainRandomizationConfig.training_defaults().as_dict()
            if args.domain_randomization
            else DomainRandomizationConfig.disabled().as_dict()
        ),
    }
    environment = WheelLegResidualEnv(
        max_forward_speed=getattr(args, "max_speed", None),
        max_command_yaw_delta_rad=np.deg2rad(args.yaw_range_deg),
        jump_probability=args.jump_probability,
        jump_at=args.jump_at,
        domain_randomization=(
            DomainRandomizationConfig.training_defaults()
            if args.domain_randomization
            else DomainRandomizationConfig.disabled()
        ),
    )
    try:
        observations, expert_actions = collect_demonstrations(
            environment,
            args.samples,
            args.target,
            args.seed,
        )
        policy = train_bc(
            observations,
            expert_actions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=torch.device(args.device),
        )
        save_policy(
            args.output,
            policy,
            observations.shape[1],
            expert_actions.shape[1],
            args.target,
            task_config,
        )
        print(f"saved BC {args.target} policy: {args.output}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
