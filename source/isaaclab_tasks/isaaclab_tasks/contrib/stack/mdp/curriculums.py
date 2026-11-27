# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-monitored reset sampling for cube stacking."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.reset import ResetStateSampler, ResetStateSamplerCfg, SamplingScoreCfg
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from isaaclab.utils import configclass

from isaaclab_tasks.utils.success_monitor import SuccessMonitorCfg

from .reset_events import StackResetRecipe, StackResetStateTable
from .runtime_state import get_stack_reset_runtime_state, torch_warp_stream

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _StackSamplingScore:
    """Preserve the curriculum's table share and balance intermediate layouts."""

    def __init__(self, cfg: _StackSamplingScoreCfg):
        self._cfg = cfg

    def score(self, metrics: Mapping[str, wp.array], out: wp.array) -> None:
        rate = wp.to_torch(metrics["success_rate"])
        table_rows = wp.to_torch(metrics["table_rows"])
        layout_ids = wp.to_torch(metrics["layout_ids"])
        table_probability = wp.to_torch(metrics["table_probability"])[0]

        cfg = self._cfg.monitor
        target = min(max(cfg.target_success_rate, 0.0), 1.0)
        kappa = max(cfg.kappa, 0.0)
        eps = 1.0e-4
        adaptive = (rate + eps).pow(kappa * target) * (1.0 - rate + eps).pow(kappa * (1.0 - target))
        adaptive = adaptive.clamp_min(eps).pow(1.0 / max(cfg.temperature, 1.0))
        adaptive.masked_fill_(table_rows, 0.0)
        layout_mass = adaptive.new_zeros(self._cfg.layout_count)
        layout_mass.scatter_add_(0, layout_ids, adaptive)
        adaptive /= layout_mass[layout_ids].clamp_min(torch.finfo(adaptive.dtype).tiny)
        adaptive /= adaptive.sum()
        table = table_rows.to(dtype=adaptive.dtype)
        table /= table.sum()
        wp.to_torch(out).copy_(adaptive * (1.0 - table_probability) + table * table_probability)


@configclass
class _StackSamplingScoreCfg(SamplingScoreCfg):
    class_type: type = _StackSamplingScore
    monitor: SuccessMonitorCfg = SuccessMonitorCfg()
    layout_count: int = 1


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
        self._target_success_rate = monitor_cfg.target_success_rate
        self._progress_successes = torch.zeros(reset_term.row_count, dtype=torch.long, device=env.device)
        self._full_task_successes_by_row = torch.zeros_like(self._progress_successes)
        self._table_rows = reset_term.recipe_ids == StackResetRecipe.TABLE
        if not bool(torch.any(self._table_rows)) or bool(torch.all(self._table_rows)):
            raise RuntimeError("The stack reset table must contain both table and intermediate rows.")
        self._recipe_rows = tuple(
            (name, reset_term.recipe_ids == recipe) for recipe, name in enumerate(reset_term.recipe_names)
        )
        table_probability = cfg.params.get("table_sampling_probability", 0.35)
        if not 0.0 < table_probability < 1.0:
            raise ValueError("table_sampling_probability must lie strictly between zero and one.")
        self._table_probability = torch.full((reset_term.row_count,), table_probability, device=env.device)
        sampler_cfg = ResetStateSamplerCfg(
            history_length=monitor_cfg.monitored_history_len,
            scores=[_StackSamplingScoreCfg(monitor=monitor_cfg, layout_count=reset_term.layout_count)],
            uniform_weight=0.0,
            seed=getattr(env.cfg, "seed", 0) or 0,
        )
        with torch_warp_stream(env.device):
            self._sampler = ResetStateSampler(
                sampler_cfg,
                num_envs=env.num_envs,
                num_states=reset_term.row_count,
                device=env.device,
                metrics={
                    "table_rows": self._table_rows,
                    "layout_ids": reset_term.layout_ids,
                    "table_probability": self._table_probability,
                },
            )
        reset_term.set_sampler(self._sampler)

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
        del success_monitor
        if not 0.0 < table_sampling_probability < 1.0:
            raise ValueError("table_sampling_probability must lie strictly between zero and one.")
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if env_ids is not None else slice(None)]
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        metrics: dict[str, torch.Tensor] = {
            "batch_table_full_task_attempts": torch.zeros((), device=env.device),
        }
        state = get_stack_reset_runtime_state(env)
        completed = (state.row_ids[ids] >= 0) & (env.episode_length_buf[ids] > 0) & env.reset_buf[ids]
        completed_ids = ids[completed]
        success = None
        if completed_ids.numel():
            success = env.termination_manager.get_term_cfg(success_context_name).func.ever_success
            succeeded = success[completed_ids]
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
            self._progress_successes.add_(
                torch.bincount(completed_rows[succeeded], minlength=self._reset_term.row_count)
            )
            self._full_task_successes_by_row.add_(
                torch.bincount(completed_rows[final_succeeded], minlength=self._reset_term.row_count)
            )

        # Keep assignments when recording outcomes. The reset event draws for the actual restoration mask.
        self._table_probability.fill_(table_sampling_probability)
        with torch_warp_stream(env.device):
            self._sampler.reset(env_ids=completed_ids, success=success, state_ids=state.row_ids)
            probabilities = self._sampler.probabilities.torch
            attempts = self._sampler.episode_counts.torch
            rolling_rates = self._sampler.success_rate.torch
        observed = attempts > 0
        total_attempts = attempts.sum()
        table_attempts = attempts[self._table_rows].sum()
        total_denominator = total_attempts.clamp_min(1)
        table_denominator = table_attempts.clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        entropy /= math.log(probabilities.numel())
        active_rows = ~self._table_rows
        unseen_rows = active_rows & ~observed
        observed_count = observed.sum().clamp_min(1)
        metrics.update(
            {
                "row_coverage": observed.float().mean(),
                "row_success_rate": self._progress_successes.sum().float() / total_denominator,
                "rolling_row_success_rate": (rolling_rates * observed).sum() / observed_count,
                "sampling_entropy": entropy,
                "unseen_row_probability_mass": probabilities[unseen_rows].sum(),
                "table_probability": probabilities[self._table_rows].sum(),
                "target_band_fraction": (((rolling_rates - self._target_success_rate).abs() <= 0.1) & observed).sum()
                / observed_count,
                "full_task_attempts": total_attempts.float(),
                "full_task_success_rate": self._full_task_successes_by_row.sum().float() / total_denominator,
                "table_curriculum_success_rate": self._progress_successes[self._table_rows].sum().float()
                / table_denominator,
                "table_full_task_success_rate": self._full_task_successes_by_row[self._table_rows].sum().float()
                / table_denominator,
                "table_full_task_attempts": table_attempts.float(),
            }
        )
        for name, rows in self._recipe_rows:
            recipe_attempts = attempts[rows].sum()
            denominator = recipe_attempts.clamp_min(1)
            metrics[f"recipe_{name}_attempts"] = recipe_attempts
            metrics[f"recipe_{name}_probability"] = probabilities[rows].sum()
            metrics[f"recipe_{name}_curriculum_success"] = self._progress_successes[rows].sum().float() / denominator
            metrics[f"recipe_{name}_full_stack_success"] = (
                self._full_task_successes_by_row[rows].sum().float() / denominator
            )
        return metrics
