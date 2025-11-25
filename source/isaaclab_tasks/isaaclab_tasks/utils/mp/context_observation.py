# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class ContextObsWrapper(gym.ObservationWrapper):
    """Extract a context slice from policy observations using a boolean mask."""

    def __init__(self, env: gym.Env, mask: torch.Tensor | np.ndarray):
        super().__init__(env)
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask, dtype=bool)
        self.mask = mask_np

        base_space = None
        if isinstance(env.observation_space, gym.spaces.Dict) and "policy" in env.observation_space.spaces:
            base_space = env.observation_space.spaces["policy"]
        elif isinstance(env.observation_space, gym.spaces.Box):
            base_space = env.observation_space

        if isinstance(base_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=base_space.low[..., self.mask],
                high=base_space.high[..., self.mask],
                dtype=base_space.dtype,
            )
        else:
            self.observation_space = env.observation_space

    def observation(self, observation):
        obs = observation
        if isinstance(obs, dict) and "policy" in obs:
            obs = obs["policy"]
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if isinstance(obs, np.ndarray) and obs.ndim >= 1:
            obs = obs[..., self.mask]
        return obs
