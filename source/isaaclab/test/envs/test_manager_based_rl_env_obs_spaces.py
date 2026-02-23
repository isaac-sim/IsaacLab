# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test texture randomization in the cartpole scene using pytest."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_camera_env_cfg import (
    CartpoleDepthCameraEnvCfg,
    CartpoleRGBCameraEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_concatenated_obs_groups_contain_all_terms(device):
    """Test that non-concatenated observation groups contain all defined terms (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict. This test ensures all terms are correctly included.
    """
    from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import (
        FrankaCubeStackEnvCfg,
    )

    # new USD stage
    sim_utils.create_new_stage()

    # configure the stack env - it has multiple non-concatenated observation groups
    env_cfg = FrankaCubeStackEnvCfg()
    env_cfg.scene.num_envs = 2  # keep num_envs small for testing
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Verify that observation space is properly structured
    assert isinstance(env.observation_space, gym.spaces.Dict), "Top-level observation space should be Dict"

    # Test 'policy' group - should have 9 terms (not just the last one due to the bug)
    assert "policy" in env.observation_space.spaces, "Policy group missing from observation space"
    policy_space = env.observation_space.spaces["policy"]
    assert isinstance(policy_space, gym.spaces.Dict), "Policy group should be Dict space"

    expected_policy_terms = [
        "actions",
        "joint_pos",
        "joint_vel",
        "object",
        "cube_positions",
        "cube_orientations",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
    ]

    # This is the key test - before the fix, only "gripper_pos" (last term) would be present
    assert len(policy_space.spaces) == len(expected_policy_terms), (
        f"Policy group should have {len(expected_policy_terms)} terms, got {len(policy_space.spaces)}:"
        f" {list(policy_space.spaces.keys())}"
    )

    for term_name in expected_policy_terms:
        assert term_name in policy_space.spaces, f"Term '{term_name}' missing from policy group"
        assert isinstance(policy_space.spaces[term_name], gym.spaces.Box), f"Term '{term_name}' should be Box space"

    # Test 'subtask_terms' group - should have 3 terms (not just the last one)
    assert "subtask_terms" in env.observation_space.spaces, "Subtask_terms group missing from observation space"
    subtask_space = env.observation_space.spaces["subtask_terms"]
    assert isinstance(subtask_space, gym.spaces.Dict), "Subtask_terms group should be Dict space"

    expected_subtask_terms = ["grasp_1", "stack_1", "grasp_2"]

    # Before the fix, only "grasp_2" (last term) would be present
    assert len(subtask_space.spaces) == len(expected_subtask_terms), (
        f"Subtask_terms group should have {len(expected_subtask_terms)} terms, got {len(subtask_space.spaces)}:"
        f" {list(subtask_space.spaces.keys())}"
    )

    for term_name in expected_subtask_terms:
        assert term_name in subtask_space.spaces, f"Term '{term_name}' missing from subtask_terms group"
        assert isinstance(subtask_space.spaces[term_name], gym.spaces.Box), f"Term '{term_name}' should be Box space"

    # Test that we can get observations and they match the space structure
    env.reset()
    action = torch.tensor(env.action_space.sample(), device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)

    # Verify all terms are present in actual observations
    for term_name in expected_policy_terms:
        assert term_name in obs["policy"], f"Term '{term_name}' missing from policy observation"

    for term_name in expected_subtask_terms:
        assert term_name in obs["subtask_terms"], f"Term '{term_name}' missing from subtask_terms observation"

    env.close()


@pytest.mark.parametrize(
    "env_cfg_cls",
    [CartpoleRGBCameraEnvCfg, CartpoleDepthCameraEnvCfg, AnymalCRoughEnvCfg],
    ids=["RGB", "Depth", "RayCaster"],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_obs_space_follows_clip_contraint(env_cfg_cls, device):
    """Ensure curriculum terms apply correctly after the fallback and replacement."""
    # new USD stage
    sim_utils.create_new_stage()

    # configure the cartpole env
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 2  # keep num_envs small for testing
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    for group_name, group_space in env.observation_space.spaces.items():
        for term_name, term_space in group_space.spaces.items():
            term_cfg = getattr(getattr(env_cfg.observations, group_name), term_name)
            low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
            high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
            assert isinstance(term_space, gym.spaces.Box), (
                f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
            )
            assert np.all(term_space.low == low)
            assert np.all(term_space.high == high)

    env.close()
