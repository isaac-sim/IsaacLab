# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Handover goal state and episode lifecycle coverage, including real environment transitions."""

import gymnasium as gym
import pytest
import torch

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.handover.handover_common import HandoverGoal
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _two_hand_goal(device: str) -> HandoverGoal:
    """Right hand at the origin and left hand 1 m along x, both unrotated, for two environments."""
    poses = torch.zeros((2, 2, 7), device=device)
    poses[1, :, 0] = 1.0
    poses[..., 6] = 1.0
    return HandoverGoal(poses, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("dwell_steps", (0, -1))
def test_handover_goal_rejects_non_positive_dwell(dwell_steps):
    """A non-positive dwell would switch goals every step without a transfer."""
    goal = _two_hand_goal("cpu")
    with pytest.raises(ValueError, match="success_dwell_steps"):
        goal.update(torch.zeros(2, dtype=torch.bool), dwell_steps)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_handover_goal_update_does_not_synchronize(device):
    """The per-step update switches only completed goals without a host synchronization."""
    goal = _two_hand_goal(device)
    succeeded = torch.tensor([True, False], device=device)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        goal.update(succeeded, 2)
        switched = goal.update(succeeded, 2)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert switched.tolist() == [True, False]
    torch.testing.assert_close(goal.position.cpu(), torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert goal.dwell.tolist() == [0, 0]
    assert goal.success.snapshot(slice(None)).tolist() == [1.0, 0.0]


@pytest.mark.parametrize("task_name", ("Isaac-Shadow-Handover-Direct", "Isaac-Shadow-Handover"))
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_handover_alternates_after_cumulative_dwell_and_resets_one_episode(task_name, device):
    """A completed transfer changes the observed goal; drops reset only the affected episode."""
    direct = task_name.endswith("-Direct")
    cfg = parse_env_cfg(task_name, device=device, num_envs=2)
    goal_cfg = cfg if direct else cfg.commands.object_pose
    goal_cfg.success_dwell_steps = 3
    if direct:
        cfg.reset_position_noise = 0.0
        cfg.reset_dof_pos_noise = 0.0
    else:
        cfg.events.reset_object.params["pose_range"] = {}
        cfg.events.reset_right_hand.params["joint_position_noise"] = 0.0
        cfg.events.reset_left_hand.params["joint_position_noise"] = 0.0
    cfg.sim.gravity = (0.0, 0.0, 0.0)

    with (
        launch_simulation(cfg, {"device": device, "headless": True}),
        gym.make(task_name, cfg=cfg) as wrapped,
    ):
        env = wrapped.unwrapped
        goal_cfg = env.cfg if direct else env.command_manager.get_term("object_pose").cfg
        observations, _ = env.reset(seed=42)
        observations = observations["right_hand" if direct else "policy"]
        object_asset = env.scene["object"]
        # These points independently specify the task geometry in the environment frame.
        left = torch.tensor([0.0, -0.64, 0.54], device=device)
        right = torch.tensor([0.0, -0.36, 0.54], device=device)
        targets = left.repeat(2, 1)
        torch.testing.assert_close(observations[:, 146:149], targets)
        actions = (
            {agent: torch.zeros((2, 20), device=device) for agent in env.possible_agents}
            if direct
            else torch.zeros((2, 40), device=device)
        )

        def step_at(positions):
            pose = object_asset.data.default_root_pose.torch.clone()
            pose[:, :3] = positions + env.scene.env_origins
            object_asset.write_root_pose_to_sim_index(root_pose=pose)
            object_asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((2, 6), device=device))
            obs, rewards, terminated, truncated, extras = env.step(actions)
            if direct:
                return obs["right_hand"], rewards, terminated["right_hand"], truncated["right_hand"], extras
            return obs["policy"], rewards, terminated, truncated, extras

        # A brief exit keeps the earned dwell, but does not itself earn another step.
        step_at(targets)
        outside = targets.clone()
        outside[:, 0] += 0.2
        step_at(outside)
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations[:, 146:149], targets)
        observations, _, terminated, truncated, _ = step_at(targets)
        assert not terminated.any() and not truncated.any()
        torch.testing.assert_close(observations[:, 146:149], right.repeat(2, 1))

        # Complete the return trip, then drop only environment 0.
        targets[:] = right
        for _ in range(3):
            observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations[:, 146:149], left.repeat(2, 1))
        targets[0] = torch.tensor([0.0, -0.5, 0.1], device=device)
        targets[1] = left
        observations, _, terminated, _, extras = step_at(targets)
        assert terminated.tolist() == [True, False]
        assert extras["log"]["Metrics/consecutive_success"].item() == 2.0
        assert extras["log"]["Metrics/success_rate"].item() == pytest.approx(2.0 / 3.0)

        # Autoreset clears env 0's count without clearing env 1's partial dwell.
        targets[:] = left
        step_at(targets)
        observations, _, _, _, _ = step_at(targets)
        expected = torch.stack((left, right))
        torch.testing.assert_close(observations[:, 146:149], expected)
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations[:, 146:149], right.repeat(2, 1))

        targets[0] = torch.tensor([0.0, -0.5, 0.1], device=device)
        targets[1] = right
        _, _, _, _, extras = step_at(targets)
        assert extras["log"]["Metrics/consecutive_success"].item() == 1.0
        assert extras["log"]["Metrics/success_rate"].item() == 0.5

        # With a one-step dwell, the first full physics step after autoreset must count.
        goal_cfg.success_dwell_steps = 1
        goal_cfg.success_distance_threshold = 10.0
        env.reset()
        targets[0] = torch.tensor([0.0, -0.5, 0.1], device=device)
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations[0, 146:149], left)
        targets[0] = left
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations[0, 146:149], right)
