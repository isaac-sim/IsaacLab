# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive sampling over the Franka Pour reset-state dataset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

from ..reset_sampler import ResetDatasetSamplerCfg, _ResetDatasetSampler

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


class PourResetDatasetCurriculum(ManagerTermBase):
    """Sample reset rows and learn from rank-local asynchronous episode outcomes."""

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        states = env._reset_dataset_states
        category = states["category"]
        objective = states["objective"]
        if category.ndim != 1 or category.numel() == 0:
            raise ValueError("Reset dataset category must be a non-empty one-dimensional tensor.")
        if objective.shape != category.shape:
            raise ValueError("Reset dataset category and objective tensors must be row-aligned.")
        self._row_count = int(category.numel())
        self._device = category.device

        self._sampling_mode = env.cfg.reset_dataset_sampling_mode
        if self._sampling_mode not in ("adaptive", "uniform"):
            raise ValueError("reset_dataset_sampling_mode must be 'adaptive' or 'uniform'.")
        sampler_cfg = env.cfg.reset_dataset_sampler.copy()
        if not isinstance(sampler_cfg, ResetDatasetSamplerCfg):
            raise TypeError("reset_dataset_sampler must be ResetDatasetSamplerCfg.")
        self._sampler = _ResetDatasetSampler(self._row_count, self._device, sampler_cfg)

        self._frozen_rows = torch.arange(self._row_count, device=self._device, dtype=torch.long)
        top_grasp_count = env.cfg.reset_dataset_top_grasp_count
        if top_grasp_count is not None:
            grasp_rows = torch.nonzero(category == 1, as_tuple=False).flatten()
            if top_grasp_count > grasp_rows.numel():
                raise ValueError(
                    "reset_dataset_top_grasp_count exceeds the dataset grasp count: "
                    f"{top_grasp_count} > {grasp_rows.numel()}."
                )
            grasp_order = torch.argsort(objective[grasp_rows], descending=True, stable=True)
            self._frozen_rows = grasp_rows[grasp_order[:top_grasp_count]]
        if self._frozen_rows.numel() == 0:
            raise RuntimeError("The reset dataset must expose at least one playback row.")

        self._metrics_cache: dict[str, float] = {}
        self._local_reset_assignments_since_metrics = 0
        self._local_metrics_refresh_interval = max(1, env.num_envs)
        self._refresh_metrics_cache()

    @staticmethod
    def _env_ids(
        env: FrankaPourEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> torch.Tensor:
        """Normalize manager-provided environment IDs on the simulation device."""
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()

    def __call__(
        self,
        env: FrankaPourEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, float]:
        """Record completed episodes and assign the next reset rows."""
        ids = self._env_ids(env, env_ids)
        if ids.numel() == 0:
            return self._metrics(env)

        completed = (env.episode_length_buf[ids] > 0) & (env.reset_dataset_row_id[ids] >= 0)
        completed_ids = ids[completed]
        if completed_ids.numel() > 0 and not env.cfg.curriculum_freeze:
            progress = env.termination_manager.get_term_cfg("learning_progress_context").func
            rows = env.reset_dataset_row_id[completed_ids]
            local_progress = progress.ever_success[completed_ids]
            self._apply_outcomes(rows, local_progress)

        if env.cfg.curriculum_freeze:
            slots = torch.randint(self._frozen_rows.numel(), (ids.numel(),), device=self._device)
            rows = self._frozen_rows[slots]
        elif self._sampling_mode == "uniform":
            rows = torch.randint(self._row_count, (ids.numel(),), device=self._device)
        else:
            rows = self._sampler._sample_with_uniform_replay(ids.numel())
            if rows.shape != ids.shape or rows.dtype != torch.long:
                raise RuntimeError("Adaptive sampler returned invalid reset-row IDs.")

        env.reset_dataset_row_id[ids] = rows
        env.pour_target_frac[ids] = float(env.cfg.pour_target_frac)

        if not env.cfg.curriculum_freeze:
            self._local_reset_assignments_since_metrics += int(ids.numel())
            if self._local_reset_assignments_since_metrics >= self._local_metrics_refresh_interval:
                self._refresh_metrics_cache()
        return self._metrics(env)

    def _metrics(self, env: FrankaPourEnv) -> dict[str, float]:
        """Return playback or adaptive-sampler metrics."""
        if env.cfg.curriculum_freeze:
            return {
                "frozen_pool_fraction": float(self._frozen_rows.numel() / self._row_count),
                "frozen_pool_size": float(self._frozen_rows.numel()),
            }
        return self._metrics_cache

    def _apply_outcomes(self, rows: torch.Tensor, learning_progress: torch.Tensor) -> None:
        """Apply completed outcomes to the adaptive sampler."""
        self._sampler._record_validated(rows, learning_progress)

    def _refresh_metrics_cache(self) -> None:
        """Refresh host logging metrics after a bounded number of reset assignments."""
        self._metrics_cache = self._sampler.metrics()
        self._local_reset_assignments_since_metrics = 0
