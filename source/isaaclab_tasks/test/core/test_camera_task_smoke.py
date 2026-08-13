# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal task-owned smoke coverage for the camera environment boundary."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci


def test_cartpole_camera_task_produces_an_observation() -> None:
    """The registered task still wires its camera into Gym; pixels are renderer-owned coverage."""
    cfg = parse_env_cfg("Isaac-Cartpole-Camera-Direct", num_envs=1)
    env = gym.make("Isaac-Cartpole-Camera-Direct", cfg=cfg)
    try:
        observation, _ = env.reset()
        assert observation["policy"].numel() > 0
        action = torch.zeros((1, env.action_space.shape[-1]), device=env.unwrapped.device)
        observation, *_ = env.step(action)
        assert observation["policy"].numel() > 0
    finally:
        env.close()
