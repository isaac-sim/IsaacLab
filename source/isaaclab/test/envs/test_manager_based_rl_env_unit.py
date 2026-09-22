# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based RL environments."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from isaaclab.envs import ManagerBasedRLEnv

pytestmark = pytest.mark.unit


def test_non_concatenated_obs_space_contains_all_terms_with_clip_bounds():
    """Non-concatenated groups expose every term as a Box bounded by its clip range (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict.
    """
    num_envs = 2
    # (name, shape, clip)
    terms = [
        ("image", (4, 5, 3), (0.0, 1.0)),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("vector", (3,), None),
    ]

    # uninitialized env whose observation manager stubs drive the space setup
    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=num_envs)
    env.observation_manager = SimpleNamespace(
        active_terms={"policy": [name for name, _, _ in terms]},
        group_obs_concatenate={"policy": False},
        group_obs_dim={"policy": [shape for _, shape, _ in terms]},
        _group_obs_term_cfgs={"policy": [SimpleNamespace(clip=clip) for _, _, clip in terms]},
    )
    env.action_manager = SimpleNamespace(action_term_dim=[0])
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    assert isinstance(env.observation_space, gym.spaces.Dict)
    policy_space = env.observation_space.spaces["policy"]
    assert isinstance(policy_space, gym.spaces.Dict)
    assert list(policy_space.spaces) == [name for name, _, _ in terms]
    for name, shape, clip in terms:
        term_space = policy_space.spaces[name]
        low, high = (-np.inf, np.inf) if clip is None else clip
        assert isinstance(term_space, gym.spaces.Box)
        assert term_space.shape == (num_envs, *shape)
        assert np.all(term_space.low == low) and np.all(term_space.high == high)
    assert env.action_space.shape == (num_envs, 0)
