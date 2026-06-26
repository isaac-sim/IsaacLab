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
def test_env_destructor_closes_before_shutdown(env_cls, monkeypatch):
    """Environment destructors should still close open envs during normal runtime."""

    closed = False

    def close(_self):
        nonlocal closed
        closed = True

    env = object.__new__(env_cls)
    env._is_closed = False
    monkeypatch.setattr(env_cls, "close", close)

    env.__del__()

    assert closed


@pytest.mark.parametrize("env_cls", [DirectRLEnv, DirectMARLEnv, ManagerBasedEnv])
def test_env_destructor_skips_close_when_already_closed(env_cls, monkeypatch):
    """Environment destructors should not re-enter close after normal cleanup."""

    def close(_self):
        raise AssertionError("close should not be called for an already closed env")

    env = object.__new__(env_cls)
    env._is_closed = True
    monkeypatch.setattr(env_cls, "close", close)

    env.__del__()


@pytest.mark.parametrize("env_cls", [DirectRLEnv, DirectMARLEnv, ManagerBasedEnv])
def test_env_destructor_skips_close_after_import_shutdown(env_cls, monkeypatch):
    """Environment destructors should not run cleanup after import machinery is torn down."""

    def close(_self):
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    env = object.__new__(env_cls)
    env._is_closed = False
    monkeypatch.setattr(env_cls, "close", close)
    monkeypatch.setattr("sys.meta_path", None)

    env.__del__()


@pytest.mark.parametrize("converter", [multi_agent_to_single_agent, multi_agent_with_one_agent])
def test_marl_adapter_destructor_closes_wrapped_env_once(converter):
    """MARL adapters inherit env destructors without running base env __init__."""

    env = _FakeMultiAgentEnv()
    converted_env = converter(env)

    converted_env.__del__()
    converted_env.__del__()

    assert converted_env._is_closed
    assert env.closed_count == 1
