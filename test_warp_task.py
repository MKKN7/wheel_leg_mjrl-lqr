"""Contract tests for the GPU flat walking task layer.

The CUDA integration test is opt-in through ``WARP_TASK_CUDA_TEST=1`` because
CI and static checks may not have a MuJoCo-Warp-capable GPU.  The pure tests
still protect the 67/7 checkpoint contract and configuration safety bounds.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

from warp_task import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    OBS_LAYOUT,
    WarpFlatWalkingConfig,
    load_flat_walking_config,
)


class WarpFlatWalkingContractTest(unittest.TestCase):
    def test_observation_layout_is_contiguous_67(self) -> None:
        slices = (
            OBS_LAYOUT.orientation,
            OBS_LAYOUT.world_velocity,
            OBS_LAYOUT.body_angular_velocity,
            OBS_LAYOUT.hip_position,
            OBS_LAYOUT.hip_velocity,
            OBS_LAYOUT.wheel_velocity,
            OBS_LAYOUT.leg_length,
            OBS_LAYOUT.leg_length_velocity,
            OBS_LAYOUT.command_speed,
            OBS_LAYOUT.command_leg_length,
            OBS_LAYOUT.command_yaw_rate,
            OBS_LAYOUT.jump_request,
            OBS_LAYOUT.yaw_state,
            OBS_LAYOUT.jump_phase,
            OBS_LAYOUT.jump_height,
            OBS_LAYOUT.terrain,
            OBS_LAYOUT.contacts,
            OBS_LAYOUT.previous_action,
        )
        self.assertEqual(slices[0].start, 0)
        for previous, current in zip(slices, slices[1:]):
            self.assertEqual(previous.stop, current.start)
        self.assertEqual(slices[-1].stop, OBSERVATION_SIZE)
        self.assertEqual(ACTION_SIZE, 7)

    def test_default_config_is_flat_and_safe(self) -> None:
        config = WarpFlatWalkingConfig()
        self.assertFalse(config.direct_control_mode)
        self.assertFalse(config.leg_action_enabled)
        self.assertLessEqual(abs(config.command_yaw_rate_rad_s), config.command_yaw_rate_limit_rad_s)
        self.assertLessEqual(config.leg_length_min_m, config.leg_length_max_m)
        self.assertAlmostEqual(config.max_leg_length_difference_m, 0.015)

    def test_rejects_invalid_leg_command(self) -> None:
        with self.assertRaises(ValueError):
            WarpFlatWalkingConfig(command_leg_length_m=0.01)

    def test_rejects_unknown_yaml_task_key(self) -> None:
        with self.assertRaises(ValueError):
            WarpFlatWalkingConfig.from_mapping({"command_speed_mps": 0.1, "typo": 1})

    def test_validates_gpu_sensor_noise_and_action_delay(self) -> None:
        config = WarpFlatWalkingConfig(
            domain_randomization_enabled=True,
            sensor_noise_std=0.015,
            control_delay_steps=1,
            domain_randomization_seed=17,
        )
        self.assertTrue(config.domain_randomization_enabled)
        self.assertEqual(config.control_delay_steps, 1)
        with self.assertRaises(ValueError):
            WarpFlatWalkingConfig(sensor_noise_std=-0.001)
        with self.assertRaises(ValueError):
            WarpFlatWalkingConfig(control_delay_steps=1.5)

    def test_loads_yaml_task_parameters(self) -> None:
        config = load_flat_walking_config(Path(__file__).resolve().parent / "configs" / "warp_flat_walking.yaml")
        self.assertAlmostEqual(config.command_speed_limit_mps, 3.0)
        self.assertFalse(config.direct_control_mode)
        self.assertFalse(config.leg_action_enabled)
        self.assertAlmostEqual(config.safety_leg_length_min_m, 0.180)


@unittest.skipUnless(
    os.environ.get("WARP_TASK_CUDA_TEST") == "1",
    "set WARP_TASK_CUDA_TEST=1 for the MuJoCo-Warp integration test",
)
class WarpFlatWalkingCudaTest(unittest.TestCase):
    def test_gpu_task_shapes_and_finite_step(self) -> None:
        import torch

        from warp_env import WarpPhysicsBatch, load_warp_batch_config
        from warp_task import WarpFlatWalkingConfig, WarpFlatWalkingTask

        config_path = Path(__file__).resolve().parent / "configs" / "warp_batch_preflight.yaml"
        batch = WarpPhysicsBatch(load_warp_batch_config(config_path))
        # Generic task plumbing can be exercised with the deliberately
        # explicit diagnostic direct-control mode. Residual PPO itself is
        # covered by the fixed-gain controller integration test.
        task = WarpFlatWalkingTask(batch, WarpFlatWalkingConfig(direct_control_mode=True))
        observations = task.reset()
        self.assertEqual(tuple(observations.shape), (batch.num_worlds, OBSERVATION_SIZE))
        action = torch.zeros((batch.num_worlds, ACTION_SIZE), device=batch.device, dtype=torch.float32)
        result = task.step_policy(action)
        self.assertEqual(tuple(result.observations.shape), (batch.num_worlds, OBSERVATION_SIZE))
        self.assertEqual(tuple(result.rewards.shape), (batch.num_worlds,))
        self.assertTrue(torch.isfinite(result.observations).all().item())
        self.assertTrue(torch.isfinite(result.rewards).all().item())


if __name__ == "__main__":
    unittest.main()
