# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedMARLEnv
from isaaclab.envs.utils.marl import multi_agent_to_single_agent, multi_agent_with_one_agent


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

    def step(self, actions):
        # shift the observations so a step is distinguishable from a reset
        self.obs_dict = {agent: obs + 10.0 for agent, obs in self.obs_dict.items()}
        rewards = {agent: torch.zeros(2) for agent in self.possible_agents}
        dones = {agent: torch.zeros(2, dtype=torch.bool) for agent in self.possible_agents}
        return self.obs_dict, rewards, dones, dones, {}

    def state(self):
        return torch.tensor([[7.0, 8.0], [9.0, 10.0]])

    def close(self):
        pass


class _FakeManagerBasedMARLEnv(ManagerBasedMARLEnv):
    """Manager-based MARL stub that skips simulator initialization."""

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
        self._is_closed = True
        self.cfg = SimpleNamespace()
        self.state_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.sim = SimpleNamespace(device="cpu")
        self.scene = SimpleNamespace(num_envs=2)
        self.episode_length_buf = torch.tensor([1, 2])
        self.obs_dict = {
            "agent_0": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "agent_1": torch.tensor([[5.0], [6.0]]),
        }
        self.actions = None
        self.closed_count = 0

    def reset(self, seed=None, options=None):
        return self.obs_dict, {}

    def step(self, actions):
        self.actions = actions
        rewards = {
            "agent_0": torch.tensor([1.0, 2.0]),
            "agent_1": torch.tensor([3.0, 4.0]),
        }
        terminated = {
            "agent_0": torch.tensor([True, True]),
            "agent_1": torch.tensor([True, False]),
        }
        time_outs = {
            "agent_0": torch.tensor([False, True]),
            "agent_1": torch.tensor([True, True]),
        }
        return self.obs_dict, rewards, terminated, time_outs, {}

    def state(self):
        return torch.tensor([[7.0, 8.0], [9.0, 10.0]])

    def close(self):
        self.closed_count += 1


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


def test_multi_agent_to_single_agent_exposes_latest_observations():
    """The public observation buffer should track the observations last returned by reset and step."""
    env = multi_agent_to_single_agent(_FakeMultiAgentEnv())

    reset_obs, _ = env.reset()
    torch.testing.assert_close(env.obs_buf["policy"], reset_obs["policy"])
    torch.testing.assert_close(env.obs_buf["policy"], torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]))

    step_obs = env.step(torch.zeros(2, 2))[0]
    torch.testing.assert_close(env.obs_buf["policy"], step_obs["policy"])
    torch.testing.assert_close(env.obs_buf["policy"], torch.tensor([[11.0, 12.0, 15.0], [13.0, 14.0, 16.0]]))


def test_multi_agent_to_single_agent_state_observation_tracks_steps():
    """The state-as-observation mode should also refresh the buffer on every transition."""
    env = multi_agent_to_single_agent(_FakeMultiAgentEnv(), state_as_observation=True)

    reset_obs, _ = env.reset()
    torch.testing.assert_close(env.obs_buf["policy"], reset_obs["policy"])

    step_obs = env.step(torch.zeros(2, 2))[0]
    torch.testing.assert_close(env.obs_buf["policy"], step_obs["policy"])


def test_manager_based_marl_conversion_splits_actions_and_combines_outputs():
    """The single-agent adapter preserves manager-based MARL action and transition semantics."""
    source_env = _FakeManagerBasedMARLEnv()
    env = multi_agent_to_single_agent(source_env)

    _, rewards, terminated, time_outs, _ = env.step(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    torch.testing.assert_close(source_env.actions["agent_0"], torch.tensor([[10.0], [30.0]]))
    torch.testing.assert_close(source_env.actions["agent_1"], torch.tensor([[20.0], [40.0]]))
    torch.testing.assert_close(rewards, torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(terminated, torch.tensor([True, False]))
    torch.testing.assert_close(time_outs, torch.tensor([False, True]))


def test_manager_based_marl_single_agent_adapter_preserves_transition_semantics():
    """The one-agent adapter preserves manager-based MARL action and transition semantics."""
    source_env = _FakeManagerBasedMARLEnv()
    env = multi_agent_with_one_agent(source_env)

    observations, rewards, terminated, time_outs, _ = env.step(
        {"single-agent": torch.tensor([[10.0, 20.0], [30.0, 40.0]])}
    )

    torch.testing.assert_close(observations["single-agent"], torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]))
    torch.testing.assert_close(source_env.actions["agent_0"], torch.tensor([[10.0], [30.0]]))
    torch.testing.assert_close(source_env.actions["agent_1"], torch.tensor([[20.0], [40.0]]))
    torch.testing.assert_close(rewards["single-agent"], torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(terminated["single-agent"], torch.tensor([True, False]))
    torch.testing.assert_close(time_outs["single-agent"], torch.tensor([False, True]))


def test_manager_based_marl_conversion_closes_source_once():
    """Repeated adapter cleanup closes a manager-based MARL source only once."""
    source_env = _FakeManagerBasedMARLEnv()
    env = multi_agent_to_single_agent(source_env)

    env.close()
    env.close()

    assert source_env.closed_count == 1
