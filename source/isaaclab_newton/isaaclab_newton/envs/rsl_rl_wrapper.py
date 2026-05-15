# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight rsl_rl VecEnv adapter for :class:`CatheterStateEnv`.

Unlike Isaac Lab's :class:`RslRlVecEnvWrapper` this does **not** require
a :class:`DirectRLEnv` or :class:`SimulationContext` — it wraps the
standalone gymnasium env directly.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from .catheter_state_env import CatheterStateEnv


class CatheterRslRlVecEnvWrapper(VecEnv):
    """Wraps :class:`CatheterStateEnv` for rsl_rl."""

    def __init__(self, env: CatheterStateEnv, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = gym.spaces.flatdim(env.single_action_space)

        obs_dim = gym.spaces.flatdim(env.single_observation_space)
        self.num_obs = obs_dim
        self.num_privileged_obs = None

        self.rew_buf: torch.Tensor | None = None
        self.reset_buf: torch.Tensor | None = None
        self.extras: dict = {}

        obs, _ = self.env.reset()
        self.obs_td = self._make_obs_td(obs)
        self.episode_length_buf = self.env.episode_length_buf

    def _make_obs_td(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    # ------------------------------------------------------------------
    def get_observations(self):
        return self.obs_td

    def reset(self):
        obs, _ = self.env.reset()
        self.obs_td = self._make_obs_td(obs)
        return self.obs_td, {}

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)

        obs, rew, terminated, truncated, infos = self.env.step(actions)

        self.obs_td = self._make_obs_td(obs)
        self.rew_buf = rew
        self.reset_buf = (terminated | truncated).long()
        self.episode_length_buf = self.env.episode_length_buf
        self.extras = {"time_outs": truncated}

        return self.obs_td, self.rew_buf, self.reset_buf, self.extras

    def close(self):
        pass
