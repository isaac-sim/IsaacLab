# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the ``close`` memory leak fixed in this PR.

``ManagerBasedEnv``, ``DirectRLEnv`` and ``DirectMARLEnv`` all cached observation
tensors and held ``gym.spaces`` objects that were never released by ``close``. Since
gymnasium's wrapper chain (e.g. ``OrderEnforcing`` plus the ``make()`` registry) keeps
the env reachable past ``close``, those buffers (tens of MB per image sensor) and the
``gym.spaces.Box`` bounds arrays (~110 MB of numpy memory per image-observation Box at
``(num_envs, H, W, C)`` float32) survived the call and accumulated on each
construct/teardown cycle. ``close`` should drop them.
"""

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
def test_manager_based_env_close_clears_obs_buf_and_releases_spaces(device):
    """``ManagerBasedEnv.close`` should release the cached observation buffer and the
    batched/single gym space attributes attached to the env."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", device=device, num_envs=2)
    env = gym.make("Isaac-Cartpole", cfg=env_cfg)
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_direct_rl_env_close_clears_obs_buf_and_releases_spaces(device):
    """``DirectRLEnv.close`` should release the cached observation buffer and the
    batched/single gym space attributes attached to the env."""
    env_cfg = parse_env_cfg("Isaac-Cartpole-Direct-v0", device=device, num_envs=2)
    env = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg)
    env.reset()
    # Step once so ``_get_observations`` populates ``obs_buf``.
    action = torch.zeros((2, env.action_space.shape[-1]), device=env.unwrapped.device)
    env.step(action)
    assert env.unwrapped.obs_buf, "precondition: obs_buf should be populated after step"
    assert env.unwrapped.observation_space is not None, "precondition: observation_space should be set"
    assert env.unwrapped.action_space is not None, "precondition: action_space should be set"

    env.close()

    assert not env.unwrapped.obs_buf, "obs_buf was not cleared on close"
    assert env.unwrapped.observation_space is None, "observation_space was not released on close"
    assert env.unwrapped.action_space is None, "action_space was not released on close"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_direct_marl_env_close_clears_obs_and_releases_spaces(device):
    """``DirectMARLEnv.close`` should release the cached per-agent observation/state
    buffers and the per-agent gym space dicts attached to the env."""
    env_cfg = parse_env_cfg("Isaac-Cart-Double-Pendulum-Direct-v0", device=device, num_envs=2)
    env = gym.make("Isaac-Cart-Double-Pendulum-Direct-v0", cfg=env_cfg)
    env.reset()
    # Step once so ``_get_observations`` / state computation populate the buffers.
    actions = {
        agent: torch.zeros((2, *space.shape), device=env.unwrapped.device)
        for agent, space in env.unwrapped.action_spaces.items()
    }
    env.step(actions)
    assert env.unwrapped.obs_dict, "precondition: obs_dict should be populated after step"
    assert env.unwrapped.observation_spaces is not None, "precondition: observation_spaces should be set"
    assert env.unwrapped.action_spaces is not None, "precondition: action_spaces should be set"

    env.close()

    assert not env.unwrapped.obs_dict, "obs_dict was not cleared on close"
    assert env.unwrapped.observation_spaces is None, "observation_spaces was not released on close"
    assert env.unwrapped.action_spaces is None, "action_spaces was not released on close"
