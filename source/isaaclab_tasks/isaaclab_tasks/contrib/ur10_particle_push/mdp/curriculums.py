# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for UR10 particle pushing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from ..ur10_particle_push_env import UR10ParticlePushEnv


class SinglePushCurriculum(ManagerTermBase):
    """Sample a persistent mixture of single-pile reset distributions.

    Every level contains one pile outside the bin and requires the same one-sweep objective. Sampling
    all levels throughout training preserves easy and intermediate examples while exposing the policy
    to the full reset distribution from the first rollout.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: UR10ParticlePushEnv) -> None:
        super().__init__(cfg, env)
        level_count = len(env.cfg.reset_randomization_scales)
        self._probabilities = torch.tensor(
            env.cfg.reset_level_probabilities,
            dtype=torch.float32,
            device=env.device,
        )
        if self._probabilities.shape != (level_count,):
            raise ValueError("reset_level_probabilities must contain one value per reset level.")
        if not bool(torch.all(torch.isfinite(self._probabilities))) or bool(torch.any(self._probabilities < 0.0)):
            raise ValueError("reset_level_probabilities must contain finite, non-negative values.")
        if not bool(torch.isclose(self._probabilities.sum(), self._probabilities.new_tensor(1.0))):
            raise ValueError("reset_level_probabilities must sum to one.")
        self._levels = self._sample_levels(env.num_envs)
        self._scales = torch.tensor(env.cfg.reset_randomization_scales, device=env.device)
        self._episode_count = torch.zeros(level_count, dtype=torch.long, device=env.device)
        self._success_count = torch.zeros_like(self._episode_count)

    def __call__(
        self,
        env: UR10ParticlePushEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, torch.Tensor]:
        """Credit completed outcomes, draw new reset levels, and return logging state."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        if env.common_step_counter > 0 and env_ids.numel() > 0:
            completed_levels = self._levels[env_ids]
            self._episode_count += torch.bincount(completed_levels, minlength=self._probabilities.numel())
            self._success_count += torch.bincount(
                completed_levels[env.success_this_step[env_ids]],
                minlength=self._probabilities.numel(),
            )
        if env_ids.numel() > 0:
            self._levels[env_ids] = self._sample_levels(env_ids.numel())

        state = {
            "mean_level": self._levels.float().mean(),
            "randomization_scale": self._scales[self._levels].mean(),
            "full_randomization_fraction": (self._levels == self._probabilities.numel() - 1).float().mean(),
        }
        success_rate = self._success_count.float() / self._episode_count.clamp_min(1)
        for level in range(self._probabilities.numel()):
            state[f"level_{level}_success_rate"] = success_rate[level]
        return state

    @property
    def levels(self) -> torch.Tensor:
        """Per-environment reset curriculum levels."""
        return self._levels

    def _sample_levels(self, count: int) -> torch.Tensor:
        """Draw reset levels from the configured persistent mixture."""
        return torch.multinomial(self._probabilities, count, replacement=True)
