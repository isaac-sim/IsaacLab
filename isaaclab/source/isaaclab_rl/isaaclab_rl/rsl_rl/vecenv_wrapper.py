# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the RSL-RL library

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """

        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        # reset the environment
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # return the step information
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other
        #   action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
