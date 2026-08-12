# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless unit tests for the manager-based MARL environment."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.envs import ManagerBasedMARLEnv

pytestmark = pytest.mark.unit


class _Manager:
    """Small manager double that records reset calls."""

    def __init__(self, name: str, observation: torch.Tensor | None = None):
        self.name = name
        self.observation = observation
        self.reset_calls = 0
        self.total_action_dim = 1
        self.active_terms = {"policy": ["term"]}
        self.group_obs_dim = {"policy": (1,)}
        self.group_obs_concatenate = {"policy": True}

    def compute(self, **kwargs):
        if self.observation is not None:
            return {"policy": self.observation}
        return torch.zeros(2, dtype=torch.bool)

    def reset(self, env_ids):
        self.reset_calls += 1
        return {self.name: 1.0}


def _make_env() -> ManagerBasedMARLEnv:
    """Build an uninitialized MARL environment with manager doubles."""
    env = object.__new__(ManagerBasedMARLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(decimation=1, state=None)
    env.scene = SimpleNamespace(num_envs=2)
    env.sim = SimpleNamespace(device="cpu")
    env.possible_agents = ["left", "right"]
    env.agents = list(env.possible_agents)
    env.extras = {agent: {} for agent in env.possible_agents}
    agent_cfg = SimpleNamespace(actions=object(), observations=object(), rewards=object(), terminations=object())
    env._agents = {agent: ManagerBasedMARLEnv.Agent(agent, agent_cfg, env) for agent in env.possible_agents}
    env.action_managers = {agent: _Manager(f"{agent}_action") for agent in env.possible_agents}
    env.observation_managers = {
        agent: _Manager(f"{agent}_observation", torch.ones(2, 1)) for agent in env.possible_agents
    }
    env.reward_managers = {agent: _Manager(f"{agent}_reward") for agent in env.possible_agents}
    env.termination_managers = {agent: _Manager(f"{agent}_termination") for agent in env.possible_agents}
    env.state_manager = None
    return env


def test_initialization_exposes_fixed_agents_and_spaces():
    """The fixed agent order and unwrapped per-agent spaces are exposed."""
    env = _make_env()

    env._configure_gym_env_spaces()

    assert env.agents == ["left", "right"]
    assert env.possible_agents == ["left", "right"]
    assert env.num_agents == 2
    assert isinstance(env.observation_space("left"), gym.spaces.Box)
    assert env.action_space("right").shape == (1,)
    assert env.get_agent("left").extras is env.extras["left"]


def test_step_rejects_missing_and_extra_agent_actions():
    """Action dictionaries must exactly match the fixed agent identifiers."""
    env = _make_env()

    with pytest.raises(ValueError, match=r"missing=\['right'\].*unexpected=\['extra'\]"):
        env.step({"left": torch.zeros(2, 1), "extra": torch.zeros(2, 1)})


def test_step_returns_per_agent_rewards_and_dones():
    """The manager dictionaries retain independent reward and done buffers."""
    env = _make_env()

    dones = {agent: manager.compute() for agent, manager in env.termination_managers.items()}
    rewards = {agent: torch.zeros(2) for agent in env.reward_managers}

    assert list(dones) == env.possible_agents
    assert list(rewards) == env.possible_agents


def test_reset_resets_shared_scene_once_and_all_agent_managers():
    """Reset statistics are namespaced by agent while shared state resets once."""
    env = _make_env()
    env._sim_step_counter = 0
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    scene_resets = []
    env.scene.reset = lambda env_ids: scene_resets.append(env_ids)
    env.event_manager = SimpleNamespace(available_modes=[], reset=lambda env_ids: {"event": 1.0})
    env.curriculum_manager = SimpleNamespace(compute=lambda env_ids: None, reset=lambda env_ids: {"curriculum": 1.0})
    env.command_manager = SimpleNamespace(reset=lambda env_ids: {"command": 1.0})
    env.recorder_manager = SimpleNamespace(reset=lambda env_ids: {"recorder": 1.0})
    env.sim.render_context = SimpleNamespace(reset_scene_state_cadence=lambda: None)

    env._reset_idx(torch.tensor([0, 1]))

    assert len(scene_resets) == 1
    assert all(manager.reset_calls == 1 for manager in env.action_managers.values())
    assert all("log" in env.extras[agent] for agent in env.possible_agents)
    assert env.extras["log"]["curriculum"] == 1.0


def test_state_returns_configured_group_and_none_when_disabled():
    """Centralized state is the one configured state group and otherwise None."""
    env = _make_env()

    assert env.state() is None
    env.state_manager = _Manager("state", torch.full((2, 1), 3.0))
    assert torch.equal(env.state(), torch.full((2, 1), 3.0))


def test_compute_final_obs_is_namespaced_per_agent():
    """Terminal-observation computation retains one payload per agent."""
    env = _make_env()

    final_obs = env._get_observations()
    for agent, observation in final_obs.items():
        env.extras[agent]["final_obs"] = observation

    assert torch.equal(env.extras["left"]["final_obs"], torch.ones(2, 1))
    assert set(env.extras["left"]) == {"final_obs"}
    assert set(env.extras["right"]) == {"final_obs"}


def test_ambiguous_recorder_or_curriculum_does_not_select_first_agent():
    """Shared terms cannot accidentally use the first agent's local managers."""
    env = _make_env()

    with pytest.raises(ValueError, match="get_agent"):
        _ = env.action_manager
    with pytest.raises(ValueError, match="get_agent"):
        _ = env.reward_manager


def test_close_releases_agent_managers_and_simulation_context():
    """Close drops per-agent manager references before clearing the simulation singleton."""
    env = _make_env()
    env._is_closed = False
    env.obs_dict = {}
    env.video_recorders = []
    calls = []
    env.sim = SimpleNamespace(stop=lambda: calls.append("stop"), clear_instance=lambda: calls.append("clear"))
    env.curriculum_manager = object()
    env.command_manager = object()
    env.recorder_manager = object()
    env.event_manager = object()
    env.scene = object()
    env._window = None

    env.close()

    assert calls == ["stop", "clear"]
    assert not env.action_managers
    assert not env.observation_managers
