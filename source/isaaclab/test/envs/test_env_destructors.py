# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv
from isaaclab.envs.utils.marl import multi_agent_to_single_agent, multi_agent_with_one_agent

pytestmark = pytest.mark.unit


class _FakeMultiAgentEnv:
    possible_agents = ["agent_0", "agent_1"]
    observation_spaces = {
        "agent_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
        "agent_1": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)),
    }
    action_spaces = {
        "agent_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "agent_1": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,)),
    }
    render_mode = None

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(state_space=1)
        self.sim = object()
        self.scene = SimpleNamespace(num_envs=2)
        self.closed_count = 0

    def close(self):
        self.closed_count += 1


@pytest.mark.parametrize("env_cls", [DirectRLEnv, DirectMARLEnv, ManagerBasedEnv])
@pytest.mark.parametrize(
    ("is_closed", "shutting_down", "expect_close"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (None, False, False),
    ],
    ids=["open", "already_closed", "import_shutdown", "init_failed_early"],
)
def test_env_destructor(env_cls, is_closed, shutting_down, expect_close, monkeypatch):
    """The destructor closes open environments exactly when it is safe to do so.

    It must skip already closed environments, environments whose ``__init__`` raised before setting
    ``_is_closed``, and interpreter shutdown (``sys.meta_path`` is None).
    """
    closed = False

    def close(_self):
        nonlocal closed
        closed = True

    env = object.__new__(env_cls)
    if is_closed is not None:
        env._is_closed = is_closed
    monkeypatch.setattr(env_cls, "close", close)
    if shutting_down:
        monkeypatch.setattr("sys.meta_path", None)

    env.__del__()

    assert closed is expect_close


@pytest.mark.parametrize("converter", [multi_agent_to_single_agent, multi_agent_with_one_agent])
def test_marl_adapter_destructor_closes_wrapped_env_once(converter):
    """MARL adapters inherit env destructors without running base env __init__."""
    env = _FakeMultiAgentEnv()
    converted_env = converter(env)

    converted_env.__del__()
    converted_env.__del__()

    assert converted_env._is_closed
    assert env.closed_count == 1
