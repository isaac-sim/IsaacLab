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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_handover_goal_update_does_not_synchronize(device):
    """The per-step goal update switches completed goals without a host synchronization."""
    # Right hand at the origin and left hand 1 m along x, both unrotated.
    poses = torch.zeros((2, 2, 7), device=device)
    poses[1, :, 0] = 1.0
    poses[..., 6] = 1.0
    goal = HandoverGoal(poses, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="success_steps_required"):
        goal.update(torch.zeros(2, dtype=torch.bool, device=device), 0)

    succeeded = torch.tensor([True, False], device=device)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        switched = goal.update(succeeded, 1)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert switched.tolist() == [True, False]
    torch.testing.assert_close(goal.position.cpu(), torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_handover_goal_random_start_places_first_goal_at_either_hand(device):
    """With a random start, reset draws each episode's first goal side and the goal position follows it."""
    num_envs = 64
    # Right hand at the origin and left hand 1 m along x, both unrotated.
    poses = torch.zeros((2, num_envs, 7), device=device)
    poses[1, :, 0] = 1.0
    poses[..., 6] = 1.0
    goal = HandoverGoal(poses, (0.0, 0.0, 0.0), randomize_start=True)
    torch.manual_seed(0)
    goal.reset(slice(None))
    assert 0 < int(goal.left.sum()) < num_envs
    torch.testing.assert_close(goal.position[:, 0], goal.left.float())

    fixed = HandoverGoal(poses, (0.0, 0.0, 0.0))
    fixed.reset(slice(None))
    assert bool(fixed.left.all())


@pytest.mark.parametrize("task_name", ("Isaac-Shadow-Handover-Direct", "Isaac-Shadow-Handover"))
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_handover_random_start_places_object_opposite_first_goal(task_name, device):
    """With the default random start, each episode's object starts at the hold point of the hand opposite its goal."""
    direct = task_name.endswith("-Direct")
    num_envs = 16
    cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    goal_cfg = cfg if direct else cfg.commands.object_pose
    assert goal_cfg.randomize_start_hand
    goal_cfg.reset_position_noise = 0.0

    with (
        launch_simulation(cfg, {"device": device, "headless": True}),
        gym.make(task_name, cfg=cfg) as wrapped,
    ):
        env = wrapped.unwrapped
        observations, _ = env.reset(seed=42)
        goals = observations["right_hand" if direct else "policy"][:, 146:149]
        starts = env.scene["object"].data.root_pos_w.torch - env.scene.env_origins
        # Each hand's hold point in the environment frame.
        left = torch.tensor([0.0, -0.64, 0.54], device=device)
        right = torch.tensor([0.0, -0.36, 0.54], device=device)
        at_left = torch.isclose(goals, left.expand_as(goals), atol=1e-4).all(dim=-1)
        at_right = torch.isclose(goals, right.expand_as(goals), atol=1e-4).all(dim=-1)
        assert bool((at_left ^ at_right).all())
        assert 0 < int(at_left.sum()) < num_envs
        torch.testing.assert_close(starts, torch.where(at_left.unsqueeze(-1), right, left))


@pytest.mark.parametrize("task_name", ("Isaac-Shadow-Handover-Direct", "Isaac-Shadow-Handover"))
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_handover_alternates_after_cumulative_success_steps_and_resets_one_episode(task_name, device):
    """A completed transfer changes the observed goal; drops reset only the affected episode."""
    direct = task_name.endswith("-Direct")
    cfg = parse_env_cfg(task_name, device=device, num_envs=2)
    goal_cfg = cfg if direct else cfg.commands.object_pose
    goal_cfg.success_steps_required = 3
    # the assertions below follow the fixed right-to-left first transfer
    goal_cfg.randomize_start_hand = False
    goal_cfg.reset_position_noise = 0.0
    if direct:
        cfg.reset_dof_pos_noise = 0.0
    else:
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
        # Each hand's hold point (its goal, or the start when it holds the object) in the environment frame.
        left = torch.tensor([0.0, -0.64, 0.54], device=device)
        right = torch.tensor([0.0, -0.36, 0.54], device=device)
        targets = left.repeat(2, 1)
        torch.testing.assert_close(observations[:, 146:149], targets)
        # the object starts at the right hand's hold point, where that hand's goal is, opposite the first goal
        torch.testing.assert_close(object_asset.data.root_pos_w.torch - env.scene.env_origins, right.repeat(2, 1))
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

        # A brief exit keeps earned success steps, but does not itself earn another step.
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

        # Autoreset clears env 0's count without clearing env 1's partial success steps.
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

        # With one required success step, the first full physics step after autoreset must count.
        goal_cfg.success_steps_required = 1
        goal_cfg.success_distance_threshold = 10.0
        env.reset()
        targets[0] = torch.tensor([0.0, -0.5, 0.1], device=device)
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations[0, 146:149], left)
        targets[0] = left
        observations, _, _, _, _ = step_at(targets)
        torch.testing.assert_close(observations[0, 146:149], right)
