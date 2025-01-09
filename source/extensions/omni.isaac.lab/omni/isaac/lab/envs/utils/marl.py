# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import math
import numpy as np
import torch
from typing import Any

from ..common import ActionType, AgentID, EnvStepReturn, ObsType, StateType, VecEnvObs, VecEnvStepReturn
from ..direct_marl_env import DirectMARLEnv
from ..direct_rl_env import DirectRLEnv


def multi_agent_to_single_agent(env: DirectMARLEnv, state_as_observation: bool = False) -> DirectRLEnv:
    """Convert the multi-agent environment instance to a single-agent environment instance.

    The converted environment will be an instance of the single-agent environment interface class (:class:`DirectRLEnv`).
    As part of the conversion process, the following operations are carried out:

    * The observations of all the agents in the original multi-agent environment are concatenated to compose
        the single-agent observation. If the use of the environment state is defined as the observation,
        it is returned as is.
    * The terminations and time-outs of all the agents in the original multi-agent environment are multiplied
        (``AND`` operation) to compose the corresponding single-agent values.
    * The rewards of all the agents in the original multi-agent environment are summed to compose the
        single-agent reward.
    * The action taken by the single-agent is split to compose the actions of each agent in the original
        multi-agent environment before stepping it.

    Args:
        env: The environment to convert to.
        state_as_observation: Weather to use the multi-agent environment state as single-agent observation.

    Returns:
        Single-agent environment instance.

    Raises:
        AssertionError: If the environment state cannot be used as observation since it was explicitly defined
            as unconstructed (:attr:`DirectMARLEnvCfg.state_space`).
    """

    class Env(DirectRLEnv):
        def __init__(self, env: DirectMARLEnv) -> None:
            self.env: DirectMARLEnv = env.unwrapped

            # check if it is possible to use the multi-agent environment state as single-agent observation
            self._state_as_observation = state_as_observation
            if self._state_as_observation:
                assert self.env.cfg.state_space != 0, (
                    "The environment state cannot be used as observation since it was explicitly defined as"
                    " unconstructed"
                )

            # create single-agent properties to expose in the converted environment
            self.cfg = self.env.cfg
            self.sim = self.env.sim
            self.scene = self.env.scene
            self.render_mode = self.env.render_mode

            self.single_observation_space = gym.spaces.Dict()
            if self._state_as_observation:
                self.single_observation_space["policy"] = self.env.state_space
            else:
                self.single_observation_space["policy"] = gym.spaces.flatten_space(
                    gym.spaces.Tuple([self.env.observation_spaces[agent] for agent in self.env.possible_agents])
                )
            self.single_action_space = gym.spaces.flatten_space(
                gym.spaces.Tuple([self.env.action_spaces[agent] for agent in self.env.possible_agents])
            )

            # batch the spaces for vectorized environments
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space["policy"], self.num_envs
            )
            self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
            obs, extras = self.env.reset(seed, options)

            # use environment state as observation
            if self._state_as_observation:
                obs = {"policy": self.env.state()}
            # concatenate agents' observations
            # FIXME: This implementation assumes the spaces are fundamental ones. Fix it to support composite spaces
            else:
                obs = {
                    "policy": torch.cat(
                        [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents], dim=-1
                    )
                }

            return obs, extras

        def step(self, action: torch.Tensor) -> VecEnvStepReturn:
            # split single-agent actions to build the multi-agent ones
            # FIXME: This implementation assumes the spaces are fundamental ones. Fix it to support composite spaces
            index = 0
            _actions = {}
            for agent in self.env.possible_agents:
                delta = gym.spaces.flatdim(self.env.action_spaces[agent])
                _actions[agent] = action[:, index : index + delta]
                index += delta

            # step the environment
            obs, rewards, terminated, time_outs, extras = self.env.step(_actions)

            # use environment state as observation
            if self._state_as_observation:
                obs = {"policy": self.env.state()}
            # concatenate agents' observations
            # FIXME: This implementation assumes the spaces are fundamental ones. Fix it to support composite spaces
            else:
                obs = {
                    "policy": torch.cat(
                        [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents], dim=-1
                    )
                }

            # process environment outputs to return single-agent data
            rewards = sum(rewards.values())
            terminated = math.prod(terminated.values()).to(dtype=torch.bool)
            time_outs = math.prod(time_outs.values()).to(dtype=torch.bool)

            return obs, rewards, terminated, time_outs, extras

        def render(self, recompute: bool = False) -> np.ndarray | None:
            return self.env.render(recompute)

        def close(self) -> None:
            self.env.close()

    return Env(env)


def multi_agent_with_one_agent(env: DirectMARLEnv, state_as_observation: bool = False) -> DirectMARLEnv:
    """Convert the multi-agent environment instance to a multi-agent environment instance with only one agent.

    The converted environment will be an instance of the multi-agent environment interface class
    (:class:`DirectMARLEnv`) but with only one agent available (with ID: ``"single-agent"``).
    As part of the conversion process, the following operations are carried out:

    * The observations of all the agents in the original multi-agent environment are concatenated to compose
        the agent observation. If the use of the environment state is defined as the observation, it is returned as is.
    * The terminations and time-outs of all the agents in the original multi-agent environment are multiplied
        (``AND`` operation) to compose the corresponding agent values.
    * The rewards of all the agents in the original multi-agent environment are summed to compose the agent reward.
    * The action taken by the agent is split to compose the actions of each agent in the original
        multi-agent environment before stepping it.

    Args:
        env: The environment to convert to.
        state_as_observation: Weather to use the multi-agent environment state as agent observation.

    Returns:
        Multi-agent environment instance with only one agent.

    Raises:
        AssertionError: If the environment state cannot be used as observation since it was explicitly defined
            as unconstructed (:attr:`DirectMARLEnvCfg.state_space`).
    """

    class Env(DirectMARLEnv):
        def __init__(self, env: DirectMARLEnv) -> None:
            self.env: DirectMARLEnv = env.unwrapped

            # check if it is possible to use the multi-agent environment state as agent observation
            self._state_as_observation = state_as_observation
            if self._state_as_observation:
                assert self.env.cfg.state_space != 0, (
                    "The environment state cannot be used as observation since it was explicitly defined as"
                    " unconstructed"
                )

            # create agent properties to expose in the converted environment
            self._agent_id = "single-agent"
            self._exported_agents = [self._agent_id]
            self._exported_possible_agents = [self._agent_id]
            if self._state_as_observation:
                self._exported_observation_spaces = {self._agent_id: self.env.state_space}
            else:
                self._exported_observation_spaces = {
                    self._agent_id: gym.spaces.flatten_space(
                        gym.spaces.Tuple([self.env.observation_spaces[agent] for agent in self.env.possible_agents])
                    )
                }
            self._exported_action_spaces = {
                self._agent_id: gym.spaces.flatten_space(
                    gym.spaces.Tuple([self.env.action_spaces[agent] for agent in self.env.possible_agents])
                )
            }

        def __getattr__(self, key: str) -> Any:
            return getattr(self.env, key)

        @property
        def agents(self) -> list[AgentID]:
            return self._exported_agents

        @property
        def possible_agents(self) -> list[AgentID]:
            return self._exported_possible_agents

        @property
        def observation_spaces(self) -> dict[AgentID, gym.Space]:
            return self._exported_observation_spaces

        @property
        def action_spaces(self) -> dict[AgentID, gym.Space]:
            return self._exported_action_spaces

        def reset(
            self, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
            obs, extras = self.env.reset(seed, options)

            # use environment state as observation
            if self._state_as_observation:
                obs = {self._agent_id: self.env.state()}
            # concatenate agents' observations
            # FIXME: This implementation assumes the spaces are fundamental ones. Fix it to support composite spaces
            else:
                obs = {
                    self._agent_id: torch.cat(
                        [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents], dim=-1
                    )
                }

            return obs, extras

        def step(self, actions: dict[AgentID, ActionType]) -> EnvStepReturn:
            # split agent actions to build the multi-agent ones
            # FIXME: This implementation assumes the spaces are fundamental ones. Fix it to support composite spaces
            index = 0
            _actions = {}
            for agent in self.env.possible_agents:
                delta = gym.spaces.flatdim(self.env.action_spaces[agent])
                _actions[agent] = actions[self._agent_id][:, index : index + delta]
                index += delta

            # step the environment
            obs, rewards, terminated, time_outs, extras = self.env.step(_actions)

            # use environment state as observation
            if self._state_as_observation:
                obs = {self._agent_id: self.env.state()}
            # concatenate agents' observations
            # FIXME: This implementation assumes the spaces are fundamental ones. Fix it to support composite spaces
            else:
                obs = {
                    self._agent_id: torch.cat(
                        [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents], dim=-1
                    )
                }

            # process environment outputs to return agent data
            rewards = {self._agent_id: sum(rewards.values())}
            terminated = {self._agent_id: math.prod(terminated.values()).to(dtype=torch.bool)}
            time_outs = {self._agent_id: math.prod(time_outs.values()).to(dtype=torch.bool)}

            return obs, rewards, terminated, time_outs, extras

        def state(self) -> StateType | None:
            return self.env.state()

        def render(self, recompute: bool = False) -> np.ndarray | None:
            self.env.render(recompute)

        def close(self) -> None:
            self.env.close()

    return Env(env)
