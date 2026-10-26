# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional terminal-success curriculum for MPM resolution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


class PourMPMResolutionCurriculum(ManagerTermBase):
    """Apply optional global resolution stages using completed-episode success.

    Each rank uses its own terminal-success cohort, independently of the reset
    sampler's learning-progress signal. Transitions affect every environment
    between physics steps, including episodes that are still running.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        self._stages = env.cfg.mpm_resolution_stages
        self._minimum_episodes = env.cfg.mpm_resolution_min_episodes
        self._stage = 0
        self._assignments = 0
        self._counts = torch.zeros(2, device=env.device, dtype=torch.int64)
        self._success_rate = 0.0
        self._transition_step = -1

    def __call__(self, env: FrankaPourEnv, env_ids: Sequence[int] | torch.Tensor | slice) -> dict[str, float]:
        """Record terminal outcomes and apply at most one resolution transition."""
        if env.cfg.curriculum_freeze or self._stage >= len(self._stages):
            return self._metrics()
        num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        completed = env.episode_length_buf[env_ids] > 0
        self._counts[0] += completed.sum()
        self._counts[1] += (env.episode_succeeded[env_ids] & completed).sum()
        self._assignments += num_envs
        if self._assignments < self._minimum_episodes:
            return self._metrics()
        self._assignments = 0
        episodes, successes = self._counts.tolist()
        if episodes < self._minimum_episodes:
            return self._metrics()
        self._counts.zero_()
        self._success_rate = successes / episodes
        stage = self._stages[self._stage]
        if self._success_rate < stage.success_rate:
            return self._metrics()
        if stage.voxel_size is not None:
            env.sim.backend_manager.set_voxel_size(stage.voxel_size, entry_name="media")
        if stage.particle_radius is not None:
            env.sim.backend_manager.set_particle_radius(stage.particle_radius, entry_name="media")
        self._stage += 1
        self._transition_step = env.common_step_counter
        return self._metrics()

    def _metrics(self) -> dict[str, float]:
        """Return cached logging values without synchronizing physics."""
        return {
            "stage": float(self._stage),
            "success_rate": self._success_rate,
            "last_transition_policy_step": float(self._transition_step),
        }
