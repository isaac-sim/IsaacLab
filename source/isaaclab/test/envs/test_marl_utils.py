# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import gymnasium as gym
import torch

from isaaclab.envs.utils.marl import multi_agent_to_single_agent


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

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(state_space=2)
        self.state_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.sim = object()
        self.scene = SimpleNamespace(num_envs=2)
        self.episode_length_buf = torch.tensor([1, 2])
        self.obs_dict = {
            "agent_0": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "agent_1": torch.tensor([[5.0], [6.0]]),
        }

    def reset(self, seed=None, options=None):
        return self.obs_dict, {}

    def state(self):
        return torch.tensor([[7.0, 8.0], [9.0, 10.0]])

    def close(self):
        pass


def test_multi_agent_to_single_agent_reset_concatenates_agents():
    """The adapter reset should concatenate the agents' observations."""
    env = multi_agent_to_single_agent(_FakeMultiAgentEnv())

    observations, _ = env.reset()

    torch.testing.assert_close(observations["policy"], torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]))


def test_multi_agent_to_single_agent_reset_can_use_state():
    """The adapter reset should support the state-as-observation mode."""
    env = multi_agent_to_single_agent(_FakeMultiAgentEnv(), state_as_observation=True)

    observations, _ = env.reset()

    torch.testing.assert_close(observations["policy"], torch.tensor([[7.0, 8.0], [9.0, 10.0]]))


def test_multi_agent_to_single_agent_forwards_episode_lengths():
    """RSL-RL episode randomization should update the wrapped environment buffer."""
    source_env = _FakeMultiAgentEnv()
    env = multi_agent_to_single_agent(source_env)
    episode_lengths = torch.tensor([3, 4])

    env.episode_length_buf = episode_lengths

    assert env.episode_length_buf is episode_lengths
    assert source_env.episode_length_buf is episode_lengths


def test_multi_agent_to_single_agent_exposes_latest_observations():
    """The public observation buffer should reflect the wrapped environment's buffer."""
    env = multi_agent_to_single_agent(_FakeMultiAgentEnv())

    torch.testing.assert_close(env.obs_buf["policy"], torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]))
