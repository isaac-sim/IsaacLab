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

from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg

from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StackResetTableCurriculum(ManagerTermBase):
    """Mix guaranteed table starts with target-rate reset sampling."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset_term = env.event_manager.get_term_cfg("reset_from_state_buffer").func
        if not hasattr(reset_term, "row_count"):
            raise RuntimeError("StackResetTableCurriculum requires StackResetStateTable.")
        self._reset_term = reset_term
        monitor_cfg = cfg.params.get("success_monitor")
        if not isinstance(monitor_cfg, SuccessMonitorCfg):
            raise TypeError("StackResetTableCurriculum requires a SuccessMonitorCfg.")
        self._progress_monitor = monitor_cfg.class_type(
            monitor_cfg,
            num_partitions=1,
            partition_size=reset_term.row_count,
            device=env.device,
        )
        self._attempts = torch.zeros(reset_term.row_count, dtype=torch.long, device=env.device)
        self._progress_successes = torch.zeros_like(self._attempts)
        self._table_sampling_probability = float(cfg.params.get("table_sampling_probability", 0.35))
        if not 0.0 < self._table_sampling_probability < 1.0:
            raise ValueError("table_sampling_probability must lie strictly between zero and one.")
        self._global_sampling = bool(cfg.params.get("global_sampling", False))
        table_recipe_id = reset_term.recipe_names.index("table")
        self._table_rows = reset_term.recipe_ids == table_recipe_id
        if not bool(torch.any(self._table_rows)) or bool(torch.all(self._table_rows)):
            raise RuntimeError("The stack reset table must contain both table and intermediate rows.")
        self._layout_count = reset_term.layout_count
        self._metric_partitions: dict[str, tuple[tuple[str, torch.Tensor], ...]] = {
            "recipe": tuple(
                (name, reset_term.recipe_ids == recipe) for recipe, name in enumerate(reset_term.recipe_names)
            )
        }
        pair_ids = getattr(reset_term, "grasp_pair_ids", None)
        if pair_ids is not None:
            self._metric_partitions["pair"] = (("index_thumb", pair_ids == 0),)
        orientation_ids = getattr(reset_term, "orientation_bin_ids", None)
        if orientation_ids is not None:
            self._metric_partitions["orientation"] = tuple(
                (str(orientation_id), orientation_ids == orientation_id) for orientation_id in range(8)
            )
        resolved_tilt_azimuth_ids = getattr(reset_term, "tilt_azimuth_bin_ids", None)
        tilt_azimuth_ids = getattr(
            reset_term,
            "authored_tilt_azimuth_bin_ids",
            resolved_tilt_azimuth_ids,
        )
        if tilt_azimuth_ids is not None:
            self._metric_partitions["tilt_azimuth"] = tuple(
                (str(azimuth_id), tilt_azimuth_ids == azimuth_id) for azimuth_id in range(8)
            )
        if hasattr(reset_term, "authored_tilt_azimuth_bin_ids"):
            self._metric_partitions["resolved_tilt_azimuth"] = tuple(
                (str(azimuth_id), resolved_tilt_azimuth_ids == azimuth_id) for azimuth_id in range(8)
            )
        tilt_magnitude_ids = getattr(reset_term, "tilt_magnitude_bin_ids", None)
        if tilt_magnitude_ids is not None:
            self._metric_partitions["tilt_magnitude"] = tuple(
                (str(magnitude_id), tilt_magnitude_ids == magnitude_id) for magnitude_id in range(4)
            )
        self._continuation_attempts = torch.zeros((), dtype=torch.long, device=env.device)
        self._continuation_successes = torch.zeros((), dtype=torch.long, device=env.device)
        self._full_task_attempts_by_row = torch.zeros(reset_term.row_count, dtype=torch.long, device=env.device)
        self._full_task_successes_by_row = torch.zeros_like(self._full_task_attempts_by_row)

    def _sampling_distribution(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mixture probabilities and target-rate weights.

        Layout-balanced tasks normalize within each workspace layout. Reset
        KUKA uses one flat distribution over active rows because its reset bank
        already contains a balanced wrist/layout grid.
        """
        target_weights = self._progress_monitor.target_weights()
        adaptive = target_weights.clone()
        adaptive[self._table_rows] = 0.0
        layout_ids = self._reset_term.layout_ids
        if self._global_sampling:
            # Normalize one target-rate weight vector over the complete active
            # table without layout quotas.
            pass
        else:
            layout_mass = torch.zeros(self._layout_count, dtype=adaptive.dtype, device=adaptive.device)
            layout_mass.scatter_add_(0, layout_ids, adaptive)
            adaptive /= layout_mass[layout_ids].clamp_min(torch.finfo(adaptive.dtype).tiny)
        adaptive[self._table_rows] = 0.0
        adaptive /= adaptive.sum()
        table = self._table_rows.to(dtype=adaptive.dtype)
        table /= table.sum()
        probabilities = (1.0 - self._table_sampling_probability) * adaptive + self._table_sampling_probability * table
        return probabilities, target_weights

    def _sampling_probabilities(self) -> torch.Tensor:
        """Return the fixed-table/adaptive-intermediate sampling mixture."""
        return self._sampling_distribution()[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_monitor: SuccessMonitorCfg,
        success_context_name: str = "learning_progress_context",
        final_success_context_name: str = "progress_context",
        table_sampling_probability: float = 0.35,
        global_sampling: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Record training outcomes, select new rows, and report coverage."""
        del (
            success_monitor,
            table_sampling_probability,
            global_sampling,
        )
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        batch_success_rate = torch.zeros((), device=env.device)
        batch_full_task_success_rate = torch.zeros((), device=env.device)
        batch_table_full_task_success_rate = torch.zeros((), device=env.device)
        batch_table_full_task_attempts = torch.zeros((), device=env.device)
        if ids.numel():
            state = get_stack_reset_runtime_state(env)
            initialized = state.initialized
            row_ids = state.row_ids
            completed = initialized[ids] & (env.episode_length_buf[ids] > 0)
            completed_ids = ids[completed]
            if completed_ids.numel():
                success_context = env.termination_manager.get_term_cfg(success_context_name).func
                succeeded = success_context.ever_success[completed_ids]
                final_success_context = env.termination_manager.get_term_cfg(final_success_context_name).func
                final_succeeded = final_success_context.ever_success[completed_ids]
                batch_success_rate = succeeded.float().mean()
                batch_full_task_success_rate = final_succeeded.float().mean()
                self._continuation_attempts.add_(completed_ids.numel())
                self._continuation_successes.add_(final_succeeded.sum())
                completed_rows = row_ids[completed_ids]
                completed_table = self._table_rows[completed_rows]
                batch_table_full_task_attempts = completed_table.sum()
                if bool(torch.any(completed_table)):
                    batch_table_full_task_success_rate = final_succeeded[completed_table].float().mean()
                self._progress_monitor.success_update(completed_rows, succeeded)
                self._attempts.add_(torch.bincount(completed_rows, minlength=self._reset_term.row_count))
                self._progress_successes.add_(
                    torch.bincount(completed_rows[succeeded], minlength=self._reset_term.row_count)
                )
                self._full_task_attempts_by_row.add_(
                    torch.bincount(completed_rows, minlength=self._reset_term.row_count)
                )
                self._full_task_successes_by_row.add_(
                    torch.bincount(completed_rows[final_succeeded], minlength=self._reset_term.row_count)
                )

            probabilities, _ = self._sampling_distribution()
            rows = torch.multinomial(probabilities, ids.numel(), replacement=True)
            row_ids[ids] = rows
        else:
            probabilities, _ = self._sampling_distribution()

        attempts = self._attempts
        observed = attempts > 0
        success_rate = self._progress_successes.sum().float() / attempts.sum().clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        entropy /= math.log(probabilities.numel())
        active_rows = ~self._table_rows
        unseen_rows = active_rows & ~observed
        rolling_rates = self._progress_monitor.success_rate
        observed_count = observed.sum().clamp_min(1)
        metrics: dict[str, torch.Tensor] = {
            "row_coverage": observed.float().mean(),
            "row_success_rate": success_rate,
            "rolling_row_success_rate": (rolling_rates * observed).sum() / observed_count,
            "batch_success_rate": batch_success_rate,
            "batch_full_task_success_rate": batch_full_task_success_rate,
            "batch_table_full_task_success_rate": batch_table_full_task_success_rate,
            "batch_table_full_task_attempts": batch_table_full_task_attempts,
            "sampling_entropy": entropy,
            "unseen_row_probability_mass": probabilities[unseen_rows].sum(),
            "table_probability": probabilities[self._table_rows].sum(),
            "target_band_fraction": (
                ((rolling_rates - self._progress_monitor.cfg.target_success_rate).abs() <= 0.1) & observed
            ).sum()
            / observed_count,
            "full_task_attempts": self._continuation_attempts.float(),
            "full_task_success_rate": self._continuation_successes.float() / self._continuation_attempts.clamp_min(1),
            "table_curriculum_success_rate": self._progress_successes[self._table_rows].sum().float()
            / self._attempts[self._table_rows].sum().clamp_min(1),
            "table_full_task_success_rate": self._full_task_successes_by_row[self._table_rows].sum().float()
            / self._full_task_attempts_by_row[self._table_rows].sum().clamp_min(1),
            "table_full_task_attempts": self._full_task_attempts_by_row[self._table_rows].sum().float(),
        }
        for prefix, partitions in self._metric_partitions.items():
            for name, rows in partitions:
                partition_attempts = attempts[rows].sum()
                full_task_attempts = self._full_task_attempts_by_row[rows].sum()
                metric_prefix = f"{prefix}_{name}"
                metrics[f"{metric_prefix}_attempts"] = partition_attempts
                metrics[f"{metric_prefix}_full_stack_attempts"] = full_task_attempts
                metrics[f"{metric_prefix}_probability"] = probabilities[rows].sum()
                metrics[f"{metric_prefix}_curriculum_success"] = self._progress_successes[
                    rows
                ].sum().float() / partition_attempts.clamp_min(1)
                metrics[f"{metric_prefix}_full_stack_success"] = self._full_task_successes_by_row[
                    rows
                ].sum().float() / full_task_attempts.clamp_min(1)
        return metrics

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return monitor evidence and replay coverage for an RL checkpoint."""
        return {
            **self._progress_monitor.get_state(),
            "total_successes": self._progress_successes.clone(),
            "total_attempts": self._attempts.clone(),
            "continuation_attempts": self._continuation_attempts.clone(),
            "continuation_successes": self._continuation_successes.clone(),
            "full_task_attempts_by_row": self._full_task_attempts_by_row.clone(),
            "full_task_successes_by_row": self._full_task_successes_by_row.clone(),
        }

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore adaptive evidence and replay coverage from an RL checkpoint."""
        monitor_state_names = ("success_history", "history_pointer", "history_size")
        targets = {
            "total_successes": self._progress_successes,
            "total_attempts": self._attempts,
            "continuation_attempts": self._continuation_attempts,
            "continuation_successes": self._continuation_successes,
            "full_task_attempts_by_row": self._full_task_attempts_by_row,
            "full_task_successes_by_row": self._full_task_successes_by_row,
        }
        for name, target in targets.items():
            if name not in state:
                raise KeyError(f"Reset-table curriculum checkpoint is missing '{name}'.")
            if state[name].shape != target.shape:
                raise ValueError(
                    f"Reset-table curriculum checkpoint '{name}' has shape {state[name].shape}; "
                    f"expected {target.shape}."
                )
        self._progress_monitor.set_state({name: state[name] for name in monitor_state_names if name in state})
        for name, target in targets.items():
            target.copy_(state[name].to(device=target.device, dtype=target.dtype))
