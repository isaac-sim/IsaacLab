# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action utility functions for safe and smooth robot control."""

from __future__ import annotations
import torch
from typing import Optional


class ActionSmoother:
    """Smooths actions using exponential moving average."""

    def __init__(self, action_dim: int, num_envs: int, smoothing_factor: float = 0.7, device: str = "cuda"):
        self.smoothing_factor = smoothing_factor
        self.prev_actions = torch.zeros((num_envs, action_dim), device=device)

    def smooth(self, actions: torch.Tensor) -> torch.Tensor:
        smoothed = self.smoothing_factor * actions + (1 - self.smoothing_factor) * self.prev_actions
        self.prev_actions = smoothed.clone()
        return smoothed

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            self.prev_actions.zero_()
        else:
            self.prev_actions[env_ids] = 0.0


class ActionClipper:
    """Clips actions to safe bounds and limits rate of change."""

    def __init__(self, action_dim: int, num_envs: int, action_low: float = -1.0, 
                 action_high: float = 1.0, max_delta: Optional[float] = None, device: str = "cuda"):
        self.action_low = action_low
        self.action_high = action_high
        self.max_delta = max_delta
        self.prev_actions = torch.zeros((num_envs, action_dim), device=device)

    def clip(self, actions: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(actions, self.action_low, self.action_high)
        
        if self.max_delta is not None:
            delta = clipped - self.prev_actions
            delta = torch.clamp(delta, -self.max_delta, self.max_delta)
            clipped = self.prev_actions + delta
            clipped = torch.clamp(clipped, self.action_low, self.action_high)
        
        self.prev_actions = clipped.clone()
        return clipped

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            self.prev_actions.zero_()
        else:
            self.prev_actions[env_ids] = 0.0


def validate_actions(actions: torch.Tensor, action_low: float = -1.0, action_high: float = 1.0) -> bool:
    """Check if actions are valid (no NaN/Inf, within bounds)."""
    if not torch.isfinite(actions).all():
        return False
    if (actions < action_low).any() or (actions > action_high).any():
        return False
    return True
