# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the ``ManagerBasedEnv.close`` memory leak fixed in this PR."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab_tasks  # noqa: F401 -- registers Isaac-* env IDs with gymnasium
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_close_clears_obs_buf_and_releases_spaces(device):
    """``ManagerBasedEnv.close`` should release the cached observation buffer and the
    gym space attributes attached to the env.

    Without the release, gymnasium's wrapper chain (e.g. ``OrderEnforcing`` plus the
    ``make()`` registry) keeps the env reachable past ``close``, so ``self.obs_buf``
    (the dict of cached observation tensors — tens of MB per image sensor) and the
    ``observation_space`` / ``action_space`` ``gym.spaces.Box`` attributes (~110 MB
    of numpy bounds-array memory per image-observation Box at ``(num_envs, H, W, C)``
    float32) survive the call and accumulate on each construct/teardown cycle.
    """
    env_cfg = parse_env_cfg("Isaac-Cartpole-v0", device=device, num_envs=2)
    env = gym.make("Isaac-Cartpole-v0", cfg=env_cfg)
    env.reset()
    # Step once so the ObservationManager populates ``obs_buf``.
    action = torch.zeros((2, env.action_space.shape[-1]), device=env.unwrapped.device)
    env.step(action)
    assert env.unwrapped.obs_buf, "precondition: obs_buf should be populated after step"
    assert env.unwrapped.observation_space is not None, "precondition: observation_space should be set"
    assert env.unwrapped.action_space is not None, "precondition: action_space should be set"

    env.close()

    assert not env.unwrapped.obs_buf, "obs_buf was not cleared on close"
    assert env.unwrapped.observation_space is None, "observation_space was not released on close"
    assert env.unwrapped.action_space is None, "action_space was not released on close"
