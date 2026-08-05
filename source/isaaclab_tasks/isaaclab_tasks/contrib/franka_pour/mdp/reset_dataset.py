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

_RESET_REGION_NAMES = ("reaching", "near_object", "grasped_transport", "near_goal")
_RESET_REGION_COUNT = len(_RESET_REGION_NAMES)
_OUTCOME_NAMES = (
    "terminal_success",
    "local_progress",
    "reached_grasp_point",
    "bilateral_grasp",
    "lifted_grasp",
)


class PourResetDatasetCurriculum(ManagerTermBase):
    """Sample reset rows and learn from rank-local asynchronous episode outcomes."""

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        states = env._reset_dataset_states
        category = states["category"]
        objective = states["objective"]
        reset_region = states["reset_region"]
        if category.ndim != 1 or category.numel() == 0:
            raise ValueError("Reset dataset category must be a non-empty one-dimensional tensor.")
        if objective.shape != category.shape or reset_region.shape != category.shape:
            raise ValueError("Reset dataset category, objective, and region tensors must be row-aligned.")
        self._row_count = int(category.numel())
        self._device = category.device
        self._reset_region = reset_region

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

        self._recent_assignment_counts = torch.zeros(_RESET_REGION_COUNT, device=self._device, dtype=torch.long)
        self._recent_episode_counts = torch.zeros_like(self._recent_assignment_counts)
        self._recent_outcome_counts = torch.zeros(
            (_RESET_REGION_COUNT, len(_OUTCOME_NAMES)),
            device=self._device,
            dtype=torch.long,
        )
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
            self._record_episode_outcomes(
                env,
                completed_ids,
                rows,
                local_progress,
            )
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
            self._record_assignments(rows)
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

    def _record_episode_outcomes(
        self,
        env: FrankaPourEnv,
        env_ids: torch.Tensor,
        rows: torch.Tensor,
        local_progress: torch.Tensor,
    ) -> None:
        """Accumulate completed task outcomes by the restored reset region."""
        regions = self._reset_region[rows].long()
        self._recent_episode_counts.add_(torch.bincount(regions, minlength=_RESET_REGION_COUNT))
        outcomes = torch.stack(
            (
                env.episode_succeeded[env_ids],
                local_progress,
                env._episode_reached_grasp_point[env_ids],
                env._episode_bilateral_grasp[env_ids],
                env._episode_lifted_grasp[env_ids],
            ),
            dim=1,
        ).to(dtype=torch.long)
        self._recent_outcome_counts.index_add_(0, regions, outcomes)

    def _record_assignments(self, rows: torch.Tensor) -> None:
        """Accumulate selected reset regions for sampling-distribution logging."""
        regions = self._reset_region[rows].long()
        self._recent_assignment_counts.add_(torch.bincount(regions, minlength=_RESET_REGION_COUNT))

    def _recent_metrics(self) -> dict[str, float]:
        """Return reset-region rates accumulated since the previous refresh."""
        completed_count = self._recent_episode_counts.sum()
        overall_rates = self._recent_outcome_counts.sum(dim=0).float() / completed_count.clamp_min(1)
        region_rates = self._recent_outcome_counts.float() / self._recent_episode_counts[:, None].clamp_min(1)
        assignment_fractions = self._recent_assignment_counts.float() / self._recent_assignment_counts.sum().clamp_min(
            1
        )

        names = [
            "episodes/recent_completed_count",
            "episodes/recent_terminal_success_rate",
            "episodes/recent_local_progress_rate",
        ]
        values = [
            completed_count.float(),
            overall_rates[0],
            overall_rates[1],
        ]
        for region_id, region_name in enumerate(_RESET_REGION_NAMES):
            names.append(f"sampler/recent_{region_name}_assignment_fraction")
            values.append(assignment_fractions[region_id])
            names.append(f"episodes/{region_name}_completed_count")
            values.append(self._recent_episode_counts[region_id].float())
            for outcome_id, outcome_name in enumerate(_OUTCOME_NAMES):
                names.append(f"episodes/{region_name}_{outcome_name}_rate")
                values.append(region_rates[region_id, outcome_id])
        return dict(zip(names, torch.stack(values).tolist(), strict=True))

    def _refresh_metrics_cache(self) -> None:
        """Refresh host logging metrics after a bounded number of reset assignments."""
        self._metrics_cache = self._sampler.metrics() | self._recent_metrics()
        self._recent_assignment_counts.zero_()
        self._recent_episode_counts.zero_()
        self._recent_outcome_counts.zero_()
        self._local_reset_assignments_since_metrics = 0
