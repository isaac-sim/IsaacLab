# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.envs.utils.marl import multi_agent_to_single_agent

pytestmark = pytest.mark.unit


class _FakeMultiAgentEnv:
    possible_agents = ["agent_0", "agent_1"]
    observation_spaces = {
        "agent_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,)),
        "agent_1": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
    }
    action_spaces = {
        "agent_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        "agent_1": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
    }
    render_mode = None

    def __init__(self, compute_final_obs=False):
        self.unwrapped = self
        self.cfg = SimpleNamespace(state_space=2, compute_final_obs=compute_final_obs)
        self.extras = {agent: {} for agent in self.possible_agents}
        self.state_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.sim = object()
        self.scene = SimpleNamespace(num_envs=2)
        self.episode_length_buf = torch.tensor([1, 2])
        self.obs_dict = {
            "agent_0": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "agent_1": torch.tensor([[5.0], [6.0]]),
        }
        self.last_actions = None

    def reset(self, seed=None, options=None):
        return self.obs_dict, self.extras

    def step(self, actions):
        self.last_actions = actions
        # shift the observations so a step is distinguishable from a reset
        self.obs_dict = {agent: obs + 10.0 for agent, obs in self.obs_dict.items()}
        if self.cfg.compute_final_obs:
            # per-agent terminal observations, distinguishable from the returned (post-reset) observations
            for agent, obs in self.obs_dict.items():
                self.extras[agent]["final_obs"] = obs + 100.0
        rewards = {agent: torch.ones(2) for agent in self.possible_agents}
        dones = {agent: torch.tensor([True, False]) for agent in self.possible_agents}
        return self.obs_dict, rewards, dones, dones, self.extras

    def state(self):
        return torch.tensor([[7.0, 8.0], [9.0, 10.0]])

    def close(self):
        pass


def test_concatenates_agents_and_tracks_latest_observations():
    """Reset and step concatenate the agents' observations and expose them through ``obs_buf``."""
    source_env = _FakeMultiAgentEnv()
    env = multi_agent_to_single_agent(source_env)
    assert env.single_action_space.shape == (2,)
    assert env.single_observation_space["policy"].shape == (3,)

    reset_obs, _ = env.reset()
    torch.testing.assert_close(reset_obs["policy"], torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]))
    torch.testing.assert_close(env.obs_buf["policy"], reset_obs["policy"])

    step_obs, rewards, terminated, time_outs, _ = env.step(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    torch.testing.assert_close(step_obs["policy"], torch.tensor([[11.0, 12.0, 15.0], [13.0, 14.0, 16.0]]))
    torch.testing.assert_close(env.obs_buf["policy"], step_obs["policy"])
    # the single-agent action is split per agent, rewards summed and dones combined with AND
    torch.testing.assert_close(source_env.last_actions["agent_0"], torch.tensor([[0.1], [0.3]]))
    torch.testing.assert_close(source_env.last_actions["agent_1"], torch.tensor([[0.2], [0.4]]))
    torch.testing.assert_close(rewards, torch.full((2,), 2.0))
    assert terminated.tolist() == time_outs.tolist() == [True, False]


def test_state_as_observation_tracks_steps():
    """The state-as-observation mode returns the environment state on reset and on every step."""
    env = multi_agent_to_single_agent(_FakeMultiAgentEnv(), state_as_observation=True)
    expected_state = torch.tensor([[7.0, 8.0], [9.0, 10.0]])

    reset_obs, _ = env.reset()
    torch.testing.assert_close(reset_obs["policy"], expected_state)
    step_obs = env.step(torch.zeros(2, 2))[0]
    torch.testing.assert_close(step_obs["policy"], expected_state)
    torch.testing.assert_close(env.obs_buf["policy"], expected_state)


def test_forwards_episode_lengths():
    """RSL-RL episode randomization should update the wrapped environment buffer in place."""
    source_env = _FakeMultiAgentEnv()
    wrapped_buffer = source_env.episode_length_buf
    env = multi_agent_to_single_agent(source_env)
    episode_lengths = torch.tensor([3, 4])

    env.episode_length_buf = episode_lengths

    torch.testing.assert_close(env.episode_length_buf, episode_lengths)
    torch.testing.assert_close(source_env.episode_length_buf, episode_lengths)
    # written in place, so references taken before the assignment observe the new values
    assert source_env.episode_length_buf is wrapped_buffer


def test_final_obs_conversion():
    """Per-agent terminal observations surface as ``extras["final_obs"]`` only when captured and concatenable."""
    source_env = _FakeMultiAgentEnv(compute_final_obs=True)
    env = multi_agent_to_single_agent(source_env)

    extras = env.step(torch.zeros(2, 2))[4]
    expected = torch.tensor([[111.0, 112.0, 115.0], [113.0, 114.0, 116.0]])
    torch.testing.assert_close(extras["final_obs"]["policy"], expected)
    torch.testing.assert_close(env.extras["final_obs"]["policy"], expected)
    # the per-agent extras of the wrapped environment are left untouched
    assert "final_obs" not in source_env.extras

    # not captured, or the state is used as observation: no entry is added
    env = multi_agent_to_single_agent(_FakeMultiAgentEnv())
    assert "final_obs" not in env.step(torch.zeros(2, 2))[4]
    env = multi_agent_to_single_agent(_FakeMultiAgentEnv(compute_final_obs=True), state_as_observation=True)
    assert "final_obs" not in env.step(torch.zeros(2, 2))[4]
