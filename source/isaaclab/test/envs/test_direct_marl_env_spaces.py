# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Unit tests for direct multi-agent environment space construction."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest

from isaaclab.envs import DirectMARLEnv
from isaaclab.test.env_cfgs import make_empty_direct_marl_env_cfg

pytestmark = pytest.mark.unit


def test_agent_and_space_configuration():
    """Agent counts and spaces are configured without initializing the simulator."""
    env = object.__new__(DirectMARLEnv)
    env._is_closed = True
    env.cfg = make_empty_direct_marl_env_cfg(device="cpu")
    env.scene = SimpleNamespace(num_envs=env.cfg.scene.num_envs)
    env.sim = SimpleNamespace(device=env.cfg.sim.device)

    env._configure_env_spaces()

    assert env.agents == ["agent_0", "agent_1"]
    assert env.possible_agents == ["agent_0", "agent_1"]
    assert env.num_agents == 2
    assert env.max_num_agents == 2
    assert len(env.observation_spaces) == 2
    assert len(env.action_spaces) == 2
    assert all(isinstance(space, gym.spaces.Box) for space in env.observation_spaces.values())
    assert all(isinstance(space, gym.spaces.Box) for space in env.action_spaces.values())
    assert env.observation_spaces["agent_0"].shape == (3,)
    assert env.observation_spaces["agent_1"].shape == (4,)
    assert env.action_spaces["agent_0"].shape == (1,)
    assert env.action_spaces["agent_1"].shape == (2,)
    assert isinstance(env.state_space, gym.spaces.Box)
    assert env.state_space.shape == (7,)
