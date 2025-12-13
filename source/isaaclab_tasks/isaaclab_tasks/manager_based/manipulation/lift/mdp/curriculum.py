# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum learning utilities for progressive task difficulty."""

from __future__ import annotations
import torch
from typing import Dict, Any


class CurriculumScheduler:
    """Manages progressive difficulty increases during training."""

    def __init__(self, initial_params: Dict[str, Any], target_params: Dict[str, Any], 
                 success_threshold: float = 0.8, window_size: int = 100):
        self.initial_params = initial_params
        self.target_params = target_params
        self.current_params = initial_params.copy()
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.success_history = []
        
    def update(self, success_rate: float) -> Dict[str, Any]:
        """Update curriculum based on success rate."""
        self.success_history.append(success_rate)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
        
        avg_success = sum(self.success_history) / len(self.success_history)
        
        # Increase difficulty if performing well
        if avg_success > self.success_threshold and len(self.success_history) >= self.window_size:
            for key in self.current_params:
                current = self.current_params[key]
                target = self.target_params[key]
                # Move 10% closer to target
                self.current_params[key] = current + 0.1 * (target - current)
        
        return self.current_params


class TaskDifficultyManager:
    """Manages task-specific difficulty parameters."""

    def __init__(self, num_envs: int, device: str = "cuda"):
        self.num_envs = num_envs
        self.device = device
        self.object_mass = torch.ones(num_envs, device=device) * 0.5
        self.object_size_scale = torch.ones(num_envs, device=device) * 1.0
        self.friction = torch.ones(num_envs, device=device) * 0.8
        
    def set_easy_mode(self, env_ids: torch.Tensor = None):
        """Set easy parameters for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.object_mass[env_ids] = 0.3
        self.friction[env_ids] = 0.9
        
    def set_medium_mode(self, env_ids: torch.Tensor = None):
        """Set medium difficulty."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.object_mass[env_ids] = 0.5
        self.friction[env_ids] = 0.7
        
    def set_hard_mode(self, env_ids: torch.Tensor = None):
        """Set hard parameters."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.object_mass[env_ids] = torch.rand(len(env_ids), device=self.device) * 1.5 + 0.5
        self.friction[env_ids] = torch.rand(len(env_ids), device=self.device) * 0.4 + 0.4
