# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`RLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.orbit_envs.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""

from __future__ import annotations

import gym
import gym.spaces
import torch

from omni.isaac.orbit.envs import RLEnv


class RslRlVecEnvWrapper(gym.Wrapper):
    """Wraps around Isaac Orbit environment for RSL-RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_states` (int)
    and :attr:`state_space` (:obj:`gym.spaces.Box`). These are used by the learning agent to allocate buffers in
    the trajectory memory. Additionally, the method :meth:`_get_observations()` should have the key "critic"
    which corresponds to the privileged observations. Since this is optional for some environments, the wrapper
    checks if these attributes exist. If they don't then the wrapper defaults to zero as number of privileged
    observations.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: RLEnv):
        """Initializes the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`RLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, RLEnv):
            raise ValueError(f"The environment must be inherited from RLEnv. Environment type: {type(env)}")
        # initialize the wrapper
        gym.Wrapper.__init__(self, env)
        # store information required by wrapper
        orbit_env: RLEnv = self.env.unwrapped
        self.num_envs = orbit_env.num_envs
        self.num_actions = orbit_env.action_manager.total_action_dim
        self.num_obs = orbit_env.observation_manager.group_obs_dim["policy"][0]
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    """
    Properties
    """

    def get_observations(self) -> torch.Tensor:
        """Returns the current observations of the environment."""
        obs_dict = self.env.unwrapped.observation_manager.compute()
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note: This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.env.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def reset(self) -> tuple[torch.Tensor, dict]:
        # reset the environment
        obs_dict = self.env.reset()
        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_dict, rew, dones, extras = self.env.step(actions)
        # return step information
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        return obs, rew, dones, extras
