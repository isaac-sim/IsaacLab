# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`IsaacEnv` instance to RL-Games vectorized environment.

The following example shows how to wrap an environment for RL-Games and register the environment construction
for RL-Games :class:`Runner` class:

.. code-block:: python

    from rl_games.common import env_configurations, vecenv

    from omni.isaac.orbit_envs.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # configuration parameters
    rl_device = "cuda:0"
    clip_obs = 10.0
    clip_actions = 1.0

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

"""


import gym
import torch
from typing import Dict, Union

from rl_games.common import env_configurations
from rl_games.common.vecenv import IVecEnv

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvObs

__all__ = ["RlGamesVecEnvWrapper", "RlGamesGpuEnv"]


"""
Vectorized environment wrapper.
"""


class RlGamesVecEnvWrapper(gym.Wrapper):
    """Wraps around Isaac Orbit environment for RL-Games.

    This class wraps around the Isaac Orbit environment. Since RL-Games works directly on
    GPU buffers, the wrapper handles moving of buffers from the simulation environment
    to the same device as the learning agent. Additionally, it performs clipping of
    observations and actions.

    For algorithms like asymmetric actor-critic, RL-Games expects a dictionary for
    observations. This dictionary contains "obs" and "states" which typically correspond
    to the actor and critic observations respectively.

    To use asymmetric actor-critic, the environment instance must have the attributes
    :attr:`num_states` (int) and :attr:`state_space` (:obj:`gym.spaces.Box`). These are
    used by the learning agent to allocate buffers in the trajectory memory. Additionally,
    the method :meth:`_get_observations()` should have the key "critic" which corresponds
    to the privileged observations. Since this is optional for some environments, the wrapper
    checks if these attributes exist. If they don't then the wrapper defaults to zero as number
    of privileged observations.

    Reference:
        https://github.com/Denys88/rl_games/blob/master/rl_games/common/ivecenv.py
        https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
    """

    def __init__(self, env: IsaacEnv, rl_device: str, clip_obs: float, clip_actions: float):
        """Initializes the wrapper instance.

        Args:
            env (IsaacEnv): The environment to wrap around.
            rl_device (str): The device on which agent computations are performed.
            clip_obs (float): The clipping value for observations.
            clip_actions (float): The clipping value for actions.

        Raises:
            ValueError: The environment is not inherited from :class:`IsaacEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, IsaacEnv):
            raise ValueError(f"The environment must be inherited from IsaacEnv. Environment type: {type(env)}")
        # initialize gym wrapper
        gym.Wrapper.__init__(self, env)
        # initialize rl-games vec-env
        IVecEnv.__init__(self)
        # store provided arguments
        self._rl_device = rl_device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        # information about spaces for the wrapper
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # information for privileged observations
        self.state_space = getattr(self.env, "state_space", None)
        self.num_states = getattr(self.env, "num_states", 0)
        # print information about wrapper
        print("[INFO]: RL-Games Environment Wrapper:")
        print(f"\t\t Observations clipping: {clip_obs}")
        print(f"\t\t Actions clipping     : {clip_actions}")
        print(f"\t\t Agent device         : {rl_device}")
        print(f"\t\t Asymmetric-learning  : {self.num_states != 0}")

    """
    Properties
    """

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        return getattr(self, "num_agents", 1)

    def get_env_info(self) -> dict:
        """Returns the Gym spaces for the environment."""
        # fill the env info dict
        env_info = {"observation_space": self.observation_space, "action_space": self.action_space}
        # add information about privileged observations space
        if self.num_states > 0:
            env_info["state_space"] = self.state_space

        return env_info

    """
    Operations - MDP
    """

    def reset(self):  # noqa: D102
        obs_dict = self.env.reset()
        # process observations and states
        return self._process_obs(obs_dict)

    def step(self, actions):  # noqa: D102
        # clip the actions
        actions = torch.clamp(actions.clone(), -self._clip_actions, self._clip_actions)
        # perform environment step
        obs_dict, rew, dones, extras = self.env.step(actions)
        # process observations and states
        obs_and_states = self._process_obs(obs_dict)
        # move buffers to rl-device
        # note: we perform clone to prevent issues when rl-device and sim-device are the same.
        rew = rew.to(self._rl_device)
        dones = dones.to(self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        }

        return obs_and_states, rew, dones, extras

    """
    Helper functions
    """

    def _process_obs(self, obs_dict: VecEnvObs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Processing of the observations and states from the environment.

        Note:
            States typically refers to privileged observations for the critic function. It is typically used in
            asymmetric actor-critic algorithms [1].

        Args:
            obs (VecEnvObs): The current observations from environment.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: If environment provides states, then a dictionary
                containing the observations and states is returned. Otherwise just the observations tensor
                is returned.

        Reference:
            1. Pinto, Lerrel, et al. "Asymmetric actor critic for image-based robot learning."
               arXiv preprint arXiv:1710.06542 (2017).
        """
        # process policy obs
        obs = obs_dict["policy"]
        # clip the observations
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)
        # move the buffer to rl-device
        obs = obs.to(self._rl_device).clone()

        # check if asymmetric actor-critic or not
        if self.num_states > 0:
            # acquire states from the environment if it exists
            try:
                states = obs_dict["critic"]
            except AttributeError:
                raise NotImplementedError("Environment does not define key `critic` for privileged observations.")
            # clip the states
            states = torch.clamp(states, -self._clip_obs, self._clip_obs)
            # move buffers to rl-device
            states = states.to(self._rl_device).clone()
            # convert to dictionary
            return {"obs": obs, "states": states}
        else:
            return obs


"""
Environment Handler.
"""


class RlGamesGpuEnv(IVecEnv):
    """Thin wrapper to create instance of the environment to fit RL-Games runner."""

    # TODO: Adding this for now but do we really need this?

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        """Initialize the environment.

        Args:
            config_name (str): The name of the environment configuration.
            num_actors (int): The number of actors in the environment. This is not used in this wrapper.
        """
        self.env: RlGamesVecEnvWrapper = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, action):  # noqa: D102
        return self.env.step(action)

    def reset(self):  # noqa: D102
        return self.env.reset()

    def get_number_of_agents(self) -> int:
        """Get number of agents in the environment.

        Returns:
            int: The number of agents in the environment.
        """
        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict:
        """Get the Gym spaces for the environment.

        Returns:
            dict: The Gym spaces for the environment.
        """
        return self.env.get_env_info()
