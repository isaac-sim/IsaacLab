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
from isaaclab.managers import ObservationGroupCfg

from isaaclab_tasks.contrib.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import CartpoleCameraEnvCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_concatenated_obs_groups_contain_all_terms(device):
    """Test that non-concatenated observation groups contain all defined terms (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict. This test ensures all terms are correctly included.
    """
    # new USD stage
    sim_utils.create_new_stage()

    # configure the policy group to return its terms separately
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = 2  # keep num_envs small for testing
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        assert isinstance(env.observation_space, gym.spaces.Dict)
        policy_space = env.observation_space.spaces["policy"]
        assert isinstance(policy_space, gym.spaces.Dict)

        expected_policy_terms = ["joint_pos_rel", "joint_vel_rel"]

        assert list(policy_space.spaces) == expected_policy_terms
        for term_name in expected_policy_terms:
            assert isinstance(policy_space.spaces[term_name], gym.spaces.Box)

        # Test that observations match the space structure.
        env.reset()
        action = torch.tensor(env.action_space.sample(), device=env.device)
        obs, _, _, _, _ = env.step(action)
        assert list(obs["policy"]) == expected_policy_terms
    finally:
        env.close()


@pytest.mark.parametrize(
    ("env_cfg_cls", "presets"),
    [
        (CartpoleCameraEnvCfg, ("rgb",)),
        (CartpoleCameraEnvCfg, ("depth",)),
        (AnymalCRoughEnvCfg, ()),
    ],
    ids=["RGB", "Depth", "RayCaster"],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_obs_space_follows_clip_contraint(env_cfg_cls, presets, device):
    """Ensure observation space bounds reflect the clip constraint on each term."""
    # new USD stage
    sim_utils.create_new_stage()

    # configure the env -- resolve Hydra presets so _Preset fields become plain values
    from isaaclab_tasks.utils.hydra import resolve_presets

    env_cfg = resolve_presets(env_cfg_cls(), presets)
    env_cfg.scene.num_envs = 2  # keep num_envs small for testing
    for group_cfg in vars(env_cfg.observations).values():
        if isinstance(group_cfg, ObservationGroupCfg):
            group_cfg.concatenate_terms = False
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        for group_name, group_space in env.observation_space.spaces.items():
            assert isinstance(group_space, gym.spaces.Dict)
            for term_name, term_space in group_space.spaces.items():
                term_cfg = getattr(getattr(env_cfg.observations, group_name), term_name)
                low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                assert isinstance(term_space, gym.spaces.Box), (
                    f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
                )
                assert np.all(term_space.low == low)
                assert np.all(term_space.high == high)
    finally:
        env.close()
