# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Handover goal and episode lifecycle coverage through real environment transitions."""

import gymnasium as gym
import pytest
import torch

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_handover_alternates_after_cumulative_dwell_and_resets_one_episode(device):
    """A completed transfer changes the observed goal; drops reset only the affected episode."""
    cfg = parse_env_cfg("Isaac-Shadow-Handover-Direct", device=device, num_envs=2)
    cfg.success_dwell_steps = 3
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.sim.gravity = (0.0, 0.0, 0.0)

    with (
        launch_simulation(cfg, {"device": device, "headless": True}),
        gym.make("Isaac-Shadow-Handover-Direct", cfg=cfg) as wrapped,
    ):
        env = wrapped.unwrapped
        observations, _ = env.reset(seed=42)
        # These points independently specify the task geometry in the environment frame.
        left = torch.tensor([0.0, -0.64, 0.54], device=device)
        right = torch.tensor([0.0, -0.36, 0.54], device=device)
        targets = left.repeat(2, 1)
        torch.testing.assert_close(observations["right_hand"][:, 146:149], targets)
        actions = {agent: torch.zeros((2, 20), device=device) for agent in env.possible_agents}

        def step_at(positions):
            pose = env.object.data.default_root_pose.torch.clone()
            pose[:, :3] = positions + env.scene.env_origins
            env.object.write_root_pose_to_sim_index(root_pose=pose)
            env.object.write_root_velocity_to_sim_index(root_velocity=torch.zeros((2, 6), device=device))
            return env.step(actions)

        # A brief exit keeps the earned dwell, but does not itself earn another step.
        step_at(targets)
        outside = targets.clone()
        outside[:, 0] += 0.2
        step_at(outside)
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations["left_hand"][:, 146:149], targets)
        observations, _, terminated, truncated, _ = step_at(targets)
        assert not any(value.any() for value in (*terminated.values(), *truncated.values()))
        torch.testing.assert_close(observations["left_hand"][:, 146:149], right.repeat(2, 1))

        # Complete the return trip, then drop only environment 0.
        targets[:] = right
        for _ in range(3):
            observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations["right_hand"][:, 146:149], left.repeat(2, 1))
        targets[0] = torch.tensor([0.0, -0.5, 0.1], device=device)
        targets[1] = left
        observations, _, terminated, _, extras = step_at(targets)
        assert terminated["right_hand"].tolist() == [True, False]
        assert extras["log"]["Metrics/consecutive_success"].item() == 2.0
        assert extras["log"]["Metrics/success_rate"].item() == pytest.approx(2.0 / 3.0)

        # Autoreset clears env 0's count without clearing env 1's partial dwell.
        targets[:] = left
        step_at(targets)
        observations, _, _, _, _ = step_at(targets)
        expected = torch.stack((left, right))
        torch.testing.assert_close(observations["right_hand"][:, 146:149], expected)
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations["right_hand"][:, 146:149], right.repeat(2, 1))

        targets[0] = torch.tensor([0.0, -0.5, 0.1], device=device)
        targets[1] = right
        _, _, _, _, extras = step_at(targets)
        assert extras["log"]["Metrics/consecutive_success"].item() == 1.0
        assert extras["log"]["Metrics/success_rate"].item() == 0.5

        # With a one-step dwell, the first full physics step after autoreset must count.
        cfg.success_dwell_steps = 1
        env.reset()
        targets[0] = torch.tensor([0.0, -0.5, 0.1], device=device)
        step_at(targets)
        targets[0] = left
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations["right_hand"][0, 146:149], right)
