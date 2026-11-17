# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-monitored reset sampling for cube stacking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from isaaclab.utils.configclass import instantiate

from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg

from .reset_events import StackResetRecipe, StackResetStateTable
from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StackResetTableCurriculum(ManagerTermBase):
    """Mix guaranteed table starts with target-rate reset sampling."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset_term = env.event_manager.get_term_cfg("reset_from_state_buffer").func
        if not isinstance(reset_term, StackResetStateTable):
            raise RuntimeError("StackResetTableCurriculum requires StackResetStateTable.")
        self._reset_term = reset_term
        monitor_cfg = cfg.params["success_monitor"]
        if not isinstance(monitor_cfg, SuccessMonitorCfg):
            raise TypeError("StackResetTableCurriculum requires a SuccessMonitorCfg.")
        self._progress_monitor = instantiate(
            monitor_cfg,
            num_partitions=1,
            partition_size=reset_term.row_count,
            device=env.device,
        )
        self._attempts = torch.zeros(reset_term.row_count, dtype=torch.long, device=env.device)
        self._progress_successes = torch.zeros_like(self._attempts)
        self._table_sampling_probability = float(cfg.params["table_sampling_probability"])
        if not 0.0 < self._table_sampling_probability < 1.0:
            raise ValueError("table_sampling_probability must lie strictly between zero and one.")
        self._table_rows = reset_term.recipe_ids == StackResetRecipe.TABLE
        if not bool(torch.any(self._table_rows)) or bool(torch.all(self._table_rows)):
            raise RuntimeError("The stack reset table must contain both table and intermediate rows.")
        self._layout_count = reset_term.layout_count
        self._metric_partitions: dict[str, tuple[tuple[str, torch.Tensor], ...]] = {
            "recipe": tuple(
                (name, reset_term.recipe_ids == recipe) for recipe, name in enumerate(reset_term.recipe_names)
            )
        }
        self._full_task_successes_by_row = torch.zeros_like(self._attempts)

    def _sampling_distribution(self) -> torch.Tensor:
        """Mix uniform table starts with layout-balanced intermediate starts."""
        adaptive = self._progress_monitor.target_weights()
        adaptive[self._table_rows] = 0.0
        layout_ids = self._reset_term.layout_ids
        layout_mass = torch.zeros(self._layout_count, dtype=adaptive.dtype, device=adaptive.device)
        layout_mass.scatter_add_(0, layout_ids, adaptive)
        adaptive /= layout_mass[layout_ids].clamp_min(torch.finfo(adaptive.dtype).tiny)
        adaptive /= adaptive.sum()
        table = self._table_rows.to(dtype=adaptive.dtype)
        table /= table.sum()
        probabilities = (1.0 - self._table_sampling_probability) * adaptive + self._table_sampling_probability * table
        return probabilities

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        success_monitor: SuccessMonitorCfg,
        success_context_name: str = "learning_progress_context",
        final_success_context_name: str = "progress_context",
        table_sampling_probability: float = 0.35,
    ) -> dict[str, torch.Tensor]:
        """Credit completed episodes and update the reset event's distribution."""
        del (
            success_monitor,
            table_sampling_probability,
        )
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if env_ids is not None else slice(None)]
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        metrics: dict[str, torch.Tensor] = {
            "batch_table_full_task_attempts": torch.zeros((), device=env.device),
        }
        if ids.numel():
            state = get_stack_reset_runtime_state(env)
            completed = (state.row_ids[ids] >= 0) & (env.episode_length_buf[ids] > 0) & env.reset_buf[ids]
            completed_ids = ids[completed]
            if completed_ids.numel():
                success_context = env.termination_manager.get_term_cfg(success_context_name).func
                succeeded = success_context.ever_success[completed_ids]
                final_success_context = env.termination_manager.get_term_cfg(final_success_context_name).func
                final_succeeded = final_success_context.ever_success[completed_ids]
                metrics["batch_success_rate"] = succeeded.float().mean()
                metrics["batch_full_task_success_rate"] = final_succeeded.float().mean()
                completed_rows = state.row_ids[completed_ids]
                completed_table = self._table_rows[completed_rows]
                table_attempts = completed_table.sum()
                metrics["batch_table_full_task_attempts"] = table_attempts
                if bool(table_attempts):
                    metrics["batch_table_full_task_success_rate"] = final_succeeded[completed_table].float().mean()
                self._progress_monitor.success_update(completed_rows, succeeded)
                self._attempts.add_(torch.bincount(completed_rows, minlength=self._reset_term.row_count))
                self._progress_successes.add_(
                    torch.bincount(completed_rows[succeeded], minlength=self._reset_term.row_count)
                )
                self._full_task_successes_by_row.add_(
                    torch.bincount(completed_rows[final_succeeded], minlength=self._reset_term.row_count)
                )

        probabilities = self._sampling_distribution()
        self._reset_term.update_sampling_probabilities(probabilities)

        attempts = self._attempts
        observed = attempts > 0
        success_rate = self._progress_successes.sum().float() / attempts.sum().clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        entropy /= math.log(probabilities.numel())
        active_rows = ~self._table_rows
        unseen_rows = active_rows & ~observed
        rolling_rates = self._progress_monitor.get_success_rate()
        observed_count = observed.sum().clamp_min(1)
        metrics.update(
            {
                "row_coverage": observed.float().mean(),
                "row_success_rate": success_rate,
                "rolling_row_success_rate": (rolling_rates * observed).sum() / observed_count,
                "sampling_entropy": entropy,
                "unseen_row_probability_mass": probabilities[unseen_rows].sum(),
                "table_probability": probabilities[self._table_rows].sum(),
                "target_band_fraction": (
                    ((rolling_rates - self._progress_monitor.cfg.target_success_rate).abs() <= 0.1) & observed
                ).sum()
                / observed_count,
                "full_task_attempts": attempts.sum().float(),
                "full_task_success_rate": self._full_task_successes_by_row.sum().float() / attempts.sum().clamp_min(1),
                "table_curriculum_success_rate": self._progress_successes[self._table_rows].sum().float()
                / self._attempts[self._table_rows].sum().clamp_min(1),
                "table_full_task_success_rate": self._full_task_successes_by_row[self._table_rows].sum().float()
                / attempts[self._table_rows].sum().clamp_min(1),
                "table_full_task_attempts": attempts[self._table_rows].sum().float(),
            }
        )
        for prefix, partitions in self._metric_partitions.items():
            for name, rows in partitions:
                partition_attempts = attempts[rows].sum()
                metric_prefix = f"{prefix}_{name}"
                metrics[f"{metric_prefix}_attempts"] = partition_attempts
                metrics[f"{metric_prefix}_full_stack_attempts"] = partition_attempts
                metrics[f"{metric_prefix}_probability"] = probabilities[rows].sum()
                metrics[f"{metric_prefix}_curriculum_success"] = self._progress_successes[
                    rows
                ].sum().float() / partition_attempts.clamp_min(1)
                metrics[f"{metric_prefix}_full_stack_success"] = self._full_task_successes_by_row[
                    rows
                ].sum().float() / partition_attempts.clamp_min(1)
        return metrics
