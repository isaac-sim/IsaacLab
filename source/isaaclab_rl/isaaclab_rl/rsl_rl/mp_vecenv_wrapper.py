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
import torch
from gym.vector.utils import batch_space
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


def _find_attr_across_wrappers(env, names, default=None):
    """Traverse a wrapper stack to retrieve the first matching attribute."""
    head = env
    while head is not None:
        for name in names:
            if hasattr(head, name):
                return getattr(head, name)
        if not hasattr(head, "env"):
            break
        head = head.env
    return default


class RslRlMPVecEnvWrapper(VecEnv):
    """MP-aware variant of :class:`RslRlVecEnvWrapper`.

    It exposes the masked policy observation and MP-parameter action space to RSL-RL without
    modifying the underlying step environment spaces.
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        # validate input
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        self.env = env
        self.clip_actions = clip_actions

        # core metadata
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # resolve policy observation space (masked if needed)
        self._context_mask = self._resolve_context_mask()
        self._policy_observation_space = self._resolve_policy_observation_space()
        self._single_observation_space = gym.spaces.Dict({"policy": self._policy_observation_space})
        self._observation_space = self._batch_space(self._single_observation_space, self.num_envs)

        # resolve MP action space
        self._single_action_space = self._resolve_single_action_space()
        self._action_space = self._maybe_clip_action_space(self._single_action_space)
        self._action_space = self._batch_space(self._action_space, self.num_envs)
        self.num_actions = gym.spaces.flatdim(self._single_action_space)

        # expose single-space handles for downstream consumers
        self.single_action_space = self._single_action_space
        self.single_observation_space = self._single_observation_space

        # reset at the start since the RSL-RL runner does not call reset.
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
        return getattr(self.env, "render_mode", None)

    @property
    def observation_space(self) -> gym.Space:
        """Returns the masked policy observation space."""
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the MP parameter action space (batched)."""
        return self._action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper."""
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
        """Set the episode length buffer."""
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        obs, extras = self.env.reset()
        obs_dict = self._format_policy_obs(obs)
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        if hasattr(self.env, "get_observations"):
            obs = self.env.get_observations()
        elif hasattr(self.unwrapped, "observation_manager"):
            obs = self.unwrapped.observation_manager.compute()
        else:
            obs = self.unwrapped._get_observations()
        obs_dict = self._format_policy_obs(obs)
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions if requested
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs, rew, terminated, truncated, extras = self.env.step(actions)
        obs_dict = self._format_policy_obs(obs)

        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _resolve_context_mask(self) -> torch.Tensor | None:
        """Resolve a boolean context mask from the wrapper stack."""
        mask = _find_attr_across_wrappers(self.env, ["context_mask"])
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        return mask.to(dtype=torch.bool)

    def _resolve_policy_observation_space(self) -> gym.Space:
        """Locate and optionally mask the policy observation space."""
        single_obs_space = _find_attr_across_wrappers(self.env, ["single_observation_space"])
        if single_obs_space is None:
            single_obs_space = getattr(self.unwrapped, "single_observation_space", None)
        if single_obs_space is None:
            single_obs_space = getattr(self.unwrapped, "observation_space", None)

        if isinstance(single_obs_space, gym.spaces.Dict):
            policy_space = single_obs_space.spaces.get("policy", single_obs_space)
        else:
            policy_space = single_obs_space

        policy_space = self._apply_mask_to_space(policy_space)
        if policy_space is None:
            raise ValueError("Could not determine policy observation space for MP wrapper.")
        return policy_space

    def _apply_mask_to_space(self, space: gym.Space | None) -> gym.Space | None:
        """Apply the context mask to a Box space when it is still unmasked."""
        if space is None or not isinstance(space, gym.spaces.Box):
            return space

        if self._context_mask is None:
            return space

        mask_np = self._context_mask.detach().cpu().numpy().astype(bool)
        # Only mask when the last dimension matches the mask length (i.e., still unmasked)
        if space.shape[-1] != mask_np.size:
            return space

        return gym.spaces.Box(low=space.low[..., mask_np], high=space.high[..., mask_np], dtype=space.dtype)

    def _resolve_single_action_space(self) -> gym.Space:
        """Prefer MP parameter action space when available."""
        action_space = _find_attr_across_wrappers(self.env, ["traj_gen_action_space", "single_action_space"])
        if action_space is None:
            action_space = getattr(self.unwrapped, "single_action_space", None)
        if action_space is None:
            action_space = getattr(self.unwrapped, "action_space", None)
        if action_space is None:
            raise ValueError("Could not determine action space for MP wrapper.")

        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100.0, high=100.0, shape=action_space.shape, dtype=action_space.dtype)
        return action_space

    def _maybe_clip_action_space(self, space: gym.Space) -> gym.Space:
        """Return a clipped copy of the action space when clip_actions is set."""
        if self.clip_actions is None:
            return space
        if isinstance(space, gym.spaces.Box):
            return gym.spaces.Box(low=-self.clip_actions, high=self.clip_actions, shape=space.shape, dtype=space.dtype)
        return space

    def _batch_space(self, space: gym.Space, batch_size: int) -> gym.Space:
        """Batch a space when possible without mutating the original environment."""
        if space is None:
            return space
        try:
            return batch_space(space, batch_size)
        except Exception:
            return space

    def _format_policy_obs(self, obs) -> dict[str, torch.Tensor]:
        """Extract and mask the policy observation, returning a dict for RSL-RL."""
        policy_obs = obs
        if isinstance(obs, dict):
            if "policy" in obs:
                policy_obs = obs["policy"]
            elif len(obs) == 1:
                policy_obs = next(iter(obs.values()))
        elif isinstance(obs, TensorDict):
            if "policy" in obs.keys():
                policy_obs = obs.get("policy")
        # Mask only when the observation is still unmasked.
        if torch.is_tensor(policy_obs) and self._context_mask is not None:
            if policy_obs.shape[-1] == self._context_mask.numel():
                policy_obs = policy_obs[..., self._context_mask.to(policy_obs.device)]
        return {"policy": policy_obs}
