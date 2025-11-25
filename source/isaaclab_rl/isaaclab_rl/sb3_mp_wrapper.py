# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from typing import Any

from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from isaaclab_rl.sb3 import Sb3VecEnvWrapper


def _find_attr_across_wrappers(env, names, default=None):
    head = env
    while head is not None:
        for name in names:
            if hasattr(head, name):
                return getattr(head, name)
        if not hasattr(head, "env"):
            break
        head = head.env
    return default


class Sb3MPVecEnvWrapper(Sb3VecEnvWrapper):
    """MP-aware variant of Sb3VecEnvWrapper that prefers masked observation spaces from the wrapper stack."""

    def _process_spaces(self):
        # Prefer masked single observation space from the wrapper stack; fallback to unwrapped
        single_obs_space = _find_attr_across_wrappers(self.env, ["single_observation_space"])
        if single_obs_space is None:
            single_obs_space = self.unwrapped.single_observation_space

        # Select policy entry if dict
        if isinstance(single_obs_space, gym.spaces.Dict):
            observation_space = single_obs_space.spaces.get("policy", single_obs_space)
        else:
            observation_space = single_obs_space

        # Apply context mask if present
        mask = _find_attr_across_wrappers(self.env, ["context_mask"])
        if mask is not None and isinstance(observation_space, gym.spaces.Box):
            mask_np = (
                mask.detach().cpu().numpy().astype(bool) if hasattr(mask, "detach") else np.asarray(mask, dtype=bool)
            )
            # Only mask when mask length matches the last dimension (i.e., space is still unmasked)
            if observation_space.low.shape[-1] == mask_np.size:
                observation_space = gym.spaces.Box(
                    low=observation_space.low[..., mask_np],
                    high=observation_space.high[..., mask_np],
                    dtype=observation_space.dtype,
                )

        # Image handling (same as base wrapper)
        if isinstance(observation_space, gym.spaces.Dict):
            for obs_key, obs_space in observation_space.spaces.items():
                processors: list[callable[[torch.Tensor], Any]] = []
                if is_image_space(obs_space, check_channels=True, normalized_image=True):
                    actually_normalized = np.all(obs_space.low == -1.0) and np.all(obs_space.high == 1.0)
                    if not actually_normalized:
                        if np.any(obs_space.low != 0) or np.any(obs_space.high != 255):
                            raise ValueError(
                                "Your image observation is not normalized in environment, and will not be"
                                "normalized by sb3 if its min is not 0 and max is not 255."
                            )
                        if obs_space.dtype != np.uint8:
                            processors.append(lambda obs: obs.to(torch.uint8))
                        observation_space.spaces[obs_key] = gym.spaces.Box(0, 255, obs_space.shape, np.uint8)
                    else:
                        if not is_image_space_channels_first(obs_space):

                            def tranp(img: torch.Tensor) -> torch.Tensor:
                                return img.permute(2, 0, 1) if len(img.shape) == 3 else img.permute(0, 3, 1, 2)

                            processors.append(tranp)
                            h, w, c = obs_space.shape
                            observation_space.spaces[obs_key] = gym.spaces.Box(-1.0, 1.0, (c, h, w), obs_space.dtype)

                    def chained_processor(obs: torch.Tensor, procs=processors):
                        for proc in procs:
                            obs = proc(obs)
                        return obs

                    if len(processors) > 0:
                        self.observation_processors[obs_key] = chained_processor

        # Prefer MP parameter space if available; otherwise fallback to unwrapped single action space
        action_space = _find_attr_across_wrappers(self.env, ["traj_gen_action_space", "single_action_space"])
        if action_space is None:
            action_space = self.unwrapped.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)

        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
