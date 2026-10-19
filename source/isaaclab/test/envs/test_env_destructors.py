# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from builtins import ExceptionGroup
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import pytest

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.utils.marl import multi_agent_to_single_agent, multi_agent_with_one_agent

pytestmark = pytest.mark.unit


def test_manager_env_close_finishes_after_processor_cleanup_error():
    """Processor errors propagate after releasing the environment, and close remains idempotent."""
    error = ExceptionGroup("Failed to close observation pipelines.", [RuntimeError("Processor cleanup failed")])
    observation_manager = SimpleNamespace(close=Mock(side_effect=error))
    sim = SimpleNamespace(stop=Mock(), clear_instance=Mock())
    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = False
    env._window = object()
    env.sim = sim
    env.scene = object()
    env.action_manager = object()
    env.observation_manager = observation_manager
    env.event_manager = object()
    env.recorder_manager = object()
    env.command_manager = object()
    env.reward_manager = object()
    env.termination_manager = object()
    env.curriculum_manager = object()
    env.obs_buf = {"policy": object()}

    try:
        with pytest.raises(ExceptionGroup) as raised:
            env.close()
        assert raised.value is error
        sim.clear_instance.assert_called_once_with()
        assert env._is_closed
        assert env._window is None
        assert not hasattr(env, "scene")
        assert not hasattr(env, "observation_manager")
        assert not env.obs_buf
        env.close()
        sim.stop.assert_called_once_with()
        observation_manager.close.assert_called_once_with()
    finally:
        env._is_closed = True


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


@pytest.mark.parametrize("env_cls", [DirectRLEnv, DirectMARLEnv, ManagerBasedEnv])
def test_env_destructor_skips_close_when_init_failed_early(env_cls, monkeypatch):
    """Environment destructors should tolerate an env whose ``__init__`` raised before it set ``_is_closed``.

    Subclasses may run fallible setup (e.g. buffer allocation) before delegating to the base
    ``__init__`` that assigns the flag, so the destructor must not mask that original failure.
    """

    def close(_self):
        raise AssertionError("close should not be called for a partially constructed env")

    env = object.__new__(env_cls)
    monkeypatch.setattr(env_cls, "close", close)

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
