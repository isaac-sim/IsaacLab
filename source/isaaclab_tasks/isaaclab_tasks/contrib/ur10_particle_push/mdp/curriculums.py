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
    """Promote successful environments through single-pile reset distributions.

    The curriculum only changes reset difficulty. Every level contains one pile outside the bin
    and requires the same one-sweep objective, so the learned behavior and success definition stay
    constant throughout training.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: UR10ParticlePushEnv) -> None:
        super().__init__(cfg, env)
        self._initial_level = int(cfg.params.get("initial_level", 0))
        level_count = len(env.cfg.reset_randomization_scales)
        if not 0 <= self._initial_level < level_count:
            raise ValueError(f"initial_level must lie in [0, {level_count - 1}].")
        self._levels = torch.full(
            (env.num_envs,),
            self._initial_level,
            dtype=torch.long,
            device=env.device,
        )
        self._maximum_level = level_count - 1
        self._scales = torch.tensor(env.cfg.reset_randomization_scales, device=env.device)

    def __call__(
        self,
        env: UR10ParticlePushEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
        initial_level: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Promote each successful environment by one level and return logging state."""
        if initial_level != self._initial_level:
            raise ValueError("The curriculum initial level changed after initialization.")
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        if env.common_step_counter > 0 and env_ids.numel() > 0:
            self._levels[env_ids] += env.success_this_step[env_ids].long()
            self._levels[env_ids].clamp_(max=self._maximum_level)
        return {
            "mean_level": self._levels.float().mean(),
            "randomization_scale": self._scales[self._levels].mean(),
            "full_randomization_fraction": (self._levels == self._maximum_level).float().mean(),
        }

    @property
    def levels(self) -> torch.Tensor:
        """Per-environment reset curriculum levels."""
        return self._levels
