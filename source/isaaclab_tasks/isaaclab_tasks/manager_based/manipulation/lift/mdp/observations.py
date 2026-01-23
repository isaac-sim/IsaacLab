# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b




class ObservationNormalizer:
    """NEW: Normalizes observations for stable training."""

    def __init__(self, obs_dim: int, num_envs: int, clip_range: float = 10.0, device: str = "cuda"):
        self.obs_mean = torch.zeros(obs_dim, device=device)
        self.obs_var = torch.ones(obs_dim, device=device)
        self.count = 0
        self.clip_range = clip_range
        self.device = device

    def normalize(self, obs: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """Normalize observations using running mean and variance."""
        if update_stats and self.count < 10000:  # Update stats for first 10k steps
            batch_mean = obs.mean(dim=0)
            batch_var = obs.var(dim=0)
            
            # Update running statistics
            self.count += obs.shape[0]
            delta = batch_mean - self.obs_mean
            self.obs_mean += delta * obs.shape[0] / self.count
            self.obs_var = (self.obs_var * (self.count - obs.shape[0]) + 
                           batch_var * obs.shape[0]) / self.count
        
        # Normalize and clip
        normalized = (obs - self.obs_mean) / (torch.sqrt(self.obs_var) + 1e-8)
        return torch.clamp(normalized, -self.clip_range, self.clip_range)


class ObservationHistory:
    """NEW: Maintains history of observations for temporal context."""

    def __init__(self, obs_dim: int, num_envs: int, history_length: int = 3, device: str = "cuda"):
        self.history_length = history_length
        self.history = torch.zeros((num_envs, history_length, obs_dim), device=device)
        self.device = device

    def add(self, obs: torch.Tensor):
        """Add new observation and shift history."""
        self.history = torch.roll(self.history, shifts=1, dims=1)
        self.history[:, 0] = obs

    def get_flat(self) -> torch.Tensor:
        """Get flattened history [num_envs, history_length * obs_dim]."""
        return self.history.reshape(self.history.shape[0], -1)

    def reset(self, env_ids: torch.Tensor = None):
        """Reset history for specific environments."""
        if env_ids is None:
            self.history.zero_()
        else:
            self.history[env_ids] = 0.0


def add_noise_to_observations(
    env: ManagerBasedRLEnv,
    obs: torch.Tensor,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """NEW: Add domain randomization noise to observations.
    
    Helps with sim-to-real transfer.
    """
    if env.training:
        noise = torch.randn_like(obs) * noise_std
        return obs + noise
    return obs
