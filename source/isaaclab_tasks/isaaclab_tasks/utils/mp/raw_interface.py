# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch


class RawMPInterface(gym.Wrapper):
    """Base wrapper exposing the minimum interface required by BlackBoxWrapper.

    Subclasses should override context_mask, current_pos, current_vel, and optionally
    preprocessing hooks. All values are expected to be torch tensors living on the
    same device as the underlying environment buffers.
    """

    @property
    def context_mask(self) -> torch.Tensor:
        """Boolean mask over the policy observation; defaults to using all entries."""
        obs_space = getattr(self.env, "observation_space", None)
        device = getattr(self.env, "device", None)
        if obs_space is None:
            return torch.tensor([True], dtype=torch.bool, device=device)
        if isinstance(obs_space, gym.spaces.Box):
            return torch.ones(obs_space.shape[-1], dtype=torch.bool, device=device)
        return torch.tensor([True], dtype=torch.bool, device=device)

    @property
    def current_pos(self) -> float | int | torch.Tensor | tuple:
        """Current position of the controlled dofs."""
        raise NotImplementedError

    @property
    def current_vel(self) -> float | int | torch.Tensor | tuple:
        """Current velocity of the controlled dofs."""
        raise NotImplementedError

    @property
    def dt(self) -> float:
        """Control interval for the MP rollout."""
        # Prefer explicit step_dt from IsaacLab envs, otherwise fallback.
        if hasattr(self.env, "step_dt"):
            return float(self.env.step_dt)
        if hasattr(self.env, "dt"):
            return float(self.env.dt)
        return 0.02

    def preprocessing_and_validity_callback(
        self,
        action: torch.Tensor,
        pos_traj: torch.Tensor,
        vel_traj: torch.Tensor,
        tau_bound: list | None = None,
        delay_bound: list | None = None,
    ) -> tuple[bool, torch.Tensor, torch.Tensor]:
        """Optional preprocessing and validity check for a trajectory."""
        return True, pos_traj, vel_traj

    def set_episode_arguments(
        self, action: torch.Tensor, pos_traj: torch.Tensor, vel_traj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Set episode-level arguments before rollout (can modify trajectories)."""
        return pos_traj, vel_traj

    def invalid_traj_callback(
        self,
        action: torch.Tensor,
        pos_traj: torch.Tensor,
        vel_traj: torch.Tensor,
        tau_bound: list | None,
        delay_bound: list | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Return fallback (obs, reward, terminated, truncated, info) for invalid trajectories."""
        num_envs = int(pos_traj.shape[0]) if hasattr(pos_traj, "shape") else getattr(self.env, "num_envs", 1)
        obs_space = getattr(self, "observation_space", None) or getattr(self.env, "observation_space", None)
        obs_shape = (1,)
        if (
            isinstance(obs_space, gym.spaces.Dict)
            and "policy" in obs_space.spaces
            and isinstance(obs_space.spaces["policy"], gym.spaces.Box)
        ):
            obs_shape = obs_space.spaces["policy"].shape
        elif isinstance(obs_space, gym.spaces.Box):
            obs_shape = obs_space.shape

        zeros = torch.zeros((num_envs,) + tuple(obs_shape), device=pos_traj.device)
        reward = torch.zeros((num_envs,), device=pos_traj.device)
        terminated = torch.ones((num_envs,), dtype=torch.bool, device=pos_traj.device)
        truncated = torch.zeros((num_envs,), dtype=torch.bool, device=pos_traj.device)
        return zeros, reward, terminated, truncated, {}

    def episode_callback(self, action: torch.Tensor, pos_traj: torch.Tensor, vel_traj: torch.Tensor) -> tuple[bool]:
        """Hook to split MP parameters from auxiliary parameters if needed."""
        return True

    def get_wrapper_attr(self, name: str, default=None):
        """Traverse wrapper stack to retrieve attribute."""
        head = self
        while head is not None:
            if hasattr(head, name):
                return getattr(head, name)
            if not hasattr(head, "env"):
                break
            head = head.env
        return default

    @property
    def action_bounds(self):
        """Optional tuple of (low, high) for step-action clamping."""
        return getattr(self, "_action_bounds", None)
