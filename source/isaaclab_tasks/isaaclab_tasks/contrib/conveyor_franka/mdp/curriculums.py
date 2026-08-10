# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive reset-state curriculum for conveyor transfer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from .reset_events import BELT_DEPLOYMENT_VARIANT, CUBE_COUNT, ConveyorResetRecipe, reset_variant_counts

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ring_append_bool_count_rate(
    data: torch.Tensor,
    stream_ids: torch.Tensor,
    values: torch.Tensor,
    pointer: torch.Tensor,
    size: torch.Tensor,
    true_count: torch.Tensor,
    rate: torch.Tensor,
) -> None:
    """Append a batch to exact per-row Boolean rolling windows."""
    if stream_ids.numel() == 0:
        return

    capacity = data.shape[1]
    unique_ids, inverse, counts = torch.unique(stream_ids, return_inverse=True, return_counts=True)
    if unique_ids.numel() == stream_ids.numel():
        columns = pointer[stream_ids].long()
        overwritten = torch.where(
            size[stream_ids] == capacity,
            data[stream_ids, columns].to(dtype=true_count.dtype),
            torch.zeros_like(true_count[stream_ids]),
        )
        new_true_counts = true_count[stream_ids] - overwritten + values.to(dtype=true_count.dtype)
        data[stream_ids, columns] = values
        pointer[stream_ids] = ((columns + 1) % capacity).to(dtype=pointer.dtype)
        size[stream_ids] = (size[stream_ids] + 1).clamp(max=capacity)
        true_count[stream_ids] = new_true_counts
        rate[stream_ids] = new_true_counts.to(rate.dtype) / size[stream_ids].clamp(min=1)
        return

    order = torch.argsort(inverse, stable=True)
    sorted_ids = stream_ids[order]
    sorted_values = values[order]
    group_starts = counts.cumsum(0) - counts
    local_rank = torch.arange(stream_ids.numel(), device=data.device) - torch.repeat_interleave(group_starts, counts)
    inverse_sorted = inverse[order]
    counts_sorted = counts[inverse_sorted]
    true_added = torch.zeros(unique_ids.shape, device=data.device, dtype=true_count.dtype)
    true_added.scatter_add_(0, inverse, values.to(dtype=true_count.dtype))

    keep_start = (counts - capacity).clamp(min=0)
    keep = local_rank >= torch.repeat_interleave(keep_start, counts)
    true_kept = torch.zeros_like(true_added)
    true_kept.scatter_add_(0, inverse_sorted[keep], sorted_values[keep].to(dtype=true_count.dtype))

    overwrite_start = capacity - size[sorted_ids].long()
    overwrite_mask = (counts_sorted < capacity) & (local_rank >= overwrite_start)
    overwritten = torch.zeros_like(true_added)
    overwrite_ids = sorted_ids[overwrite_mask]
    overwrite_columns = (pointer[overwrite_ids].long() + local_rank[overwrite_mask]) % capacity
    overwritten.scatter_add_(
        0,
        inverse_sorted[overwrite_mask],
        data[overwrite_ids, overwrite_columns].to(dtype=true_count.dtype),
    )

    kept_ids = sorted_ids[keep]
    kept_columns = (pointer[kept_ids].long() + local_rank[keep]) % capacity
    data[kept_ids, kept_columns] = sorted_values[keep]
    replace = counts >= capacity
    new_true_counts = torch.where(replace, true_kept, true_count[unique_ids] - overwritten + true_added)
    new_size = (size[unique_ids].long() + counts).clamp(max=capacity)
    pointer[unique_ids] = ((pointer[unique_ids].long() + counts) % capacity).to(dtype=pointer.dtype)
    size[unique_ids] = new_size.to(dtype=size.dtype)
    true_count[unique_ids] = new_true_counts
    rate[unique_ids] = new_true_counts.to(rate.dtype) / new_size.clamp(min=1).to(rate.dtype)


def reset_sampling_probabilities(
    recipe_ids: torch.Tensor,
    variant_ids: torch.Tensor,
    target_cube_ids: torch.Tensor,
    source_side_ids: torch.Tensor,
    attempts: torch.Tensor,
    successes: torch.Tensor,
    deployment_probability: float | torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Mix guaranteed deployment starts with adaptive intermediate rows."""
    if not (
        recipe_ids.shape
        == variant_ids.shape
        == target_cube_ids.shape
        == source_side_ids.shape
        == attempts.shape
        == successes.shape
    ):
        raise ValueError("Reset row metadata and outcomes must have matching shapes.")
    deployment_probability = torch.as_tensor(
        deployment_probability,
        dtype=torch.float32,
        device=attempts.device,
    )
    if deployment_probability.numel() != 1:
        raise ValueError("deployment_probability must be a scalar.")
    deployment_probability = deployment_probability.reshape(())
    if bool((deployment_probability <= 0.0) | (deployment_probability >= 1.0)):
        raise ValueError("deployment_probability must lie strictly between zero and one.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    deployment_rows = (recipe_ids == int(ConveyorResetRecipe.BELT)) & (variant_ids == BELT_DEPLOYMENT_VARIANT)
    if not bool(torch.any(deployment_rows)) or bool(torch.all(deployment_rows)):
        raise ValueError("Reset table must contain deployment and intermediate rows.")

    rates = successes.float() / attempts.clamp_min(1).float()
    frontier = 4.0 * rates * (1.0 - rates)
    adaptive = frontier + epsilon
    adaptive[deployment_rows] = 0.0

    # Mirror Franka Stack's recipe/layout balancing: success in one physical
    # phase must not starve another cube identity or transfer direction.
    command_ids = 2 * target_cube_ids + source_side_ids
    command_count = 2 * CUBE_COUNT
    if bool(torch.any((command_ids < 0) | (command_ids >= command_count))):
        raise ValueError("Reset table contains an invalid cube or source-side id.")
    stratum_ids = recipe_ids * command_count + command_ids
    stratum_count = len(ConveyorResetRecipe) * command_count
    stratum_mass = torch.zeros(stratum_count, dtype=adaptive.dtype, device=adaptive.device)
    stratum_mass.scatter_add_(0, stratum_ids, adaptive)
    if bool(torch.any(stratum_mass <= 0.0)):
        raise ValueError("Every recipe, cube, and source-side stratum must have intermediate reset rows.")
    adaptive /= stratum_mass[stratum_ids]
    adaptive[deployment_rows] = 0.0
    adaptive *= (1.0 - deployment_probability) / adaptive.sum()
    deployment = deployment_rows.to(dtype=adaptive.dtype)
    deployment *= deployment_probability / deployment.sum()
    return adaptive + deployment


def deployment_probability_from_progress(
    progress_rate: torch.Tensor,
    row_coverage: torch.Tensor,
    initial_probability: float = 0.35,
    final_probability: float = 0.90,
    progress_start: float = 0.45,
    progress_end: float = 0.80,
    coverage_target: float = 0.50,
) -> torch.Tensor:
    """Interpolate deployment sampling from rolling competence and row coverage."""
    if progress_rate.numel() != 1 or row_coverage.numel() != 1:
        raise ValueError("progress_rate and row_coverage must be scalar tensors.")
    if not 0.0 < initial_probability <= final_probability < 1.0:
        raise ValueError("Deployment probabilities must be ordered strictly inside (0, 1).")
    if not 0.0 <= progress_start < progress_end <= 1.0:
        raise ValueError("Deployment progress thresholds must be ordered inside [0, 1].")
    if not 0.0 < coverage_target <= 1.0:
        raise ValueError("coverage_target must lie inside (0, 1].")
    if bool((progress_rate < 0.0) | (progress_rate > 1.0) | (row_coverage < 0.0) | (row_coverage > 1.0)):
        raise ValueError("Rolling progress and row coverage must lie inside [0, 1].")

    progress_fraction = ((progress_rate - progress_start) / (progress_end - progress_start)).clamp(0.0, 1.0)
    coverage_fraction = (row_coverage / coverage_target).clamp(0.0, 1.0)
    readiness = progress_fraction * coverage_fraction
    readiness = readiness.square() * (3.0 - 2.0 * readiness)
    return initial_probability + (final_probability - initial_probability) * readiness


class ConveyorResetCurriculum(ManagerTermBase):
    """Record row outcomes and sample the next physical reset states."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reset_term = env.event_manager.get_term_cfg("reset_from_state_table").func
        if not hasattr(reset_term, "row_count"):
            raise RuntimeError("ConveyorResetCurriculum requires ConveyorResetStateTable.")
        self._reset_term = reset_term
        self._attempts = torch.zeros(reset_term.row_count, dtype=torch.long, device=env.device)
        self._progress_successes = torch.zeros_like(self._attempts)
        self._final_successes = torch.zeros_like(self._attempts)
        history_len = int(cfg.params.get("monitored_history_len", 50))
        if history_len < 1:
            raise ValueError("monitored_history_len must be positive.")
        self._progress_history = torch.zeros((reset_term.row_count, history_len), dtype=torch.bool, device=env.device)
        self._history_pointer = torch.zeros(reset_term.row_count, dtype=torch.int32, device=env.device)
        self._history_size = torch.zeros_like(self._history_pointer)
        self._history_success_count = torch.zeros_like(self._history_pointer)
        self._rolling_progress_rates = torch.zeros(reset_term.row_count, dtype=torch.float32, device=env.device)
        variant_counts = reset_variant_counts()
        self._diagnostic_variant_rows = tuple(
            (
                recipe.name.lower(),
                variant_id,
                (reset_term.recipe_ids == int(recipe)) & (reset_term.variant_ids == variant_id),
            )
            for recipe in (ConveyorResetRecipe.PREGRASP, ConveyorResetRecipe.BELT)
            for variant_id in range(variant_counts[int(recipe)])
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        progress_context_name: str = "learning_progress_context",
        final_success_context_name: str = "transfer_success_context",
        deployment_probability_initial: float = 0.35,
        deployment_probability_final: float = 0.90,
        deployment_progress_start: float = 0.45,
        deployment_progress_end: float = 0.80,
        deployment_coverage_target: float = 0.50,
        epsilon: float = 0.05,
        monitored_history_len: int = 50,
        fixed_source_side_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Update adaptive evidence, sample rows, and expose diagnostics."""
        del monitored_history_len
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        state = env.conveyor_transfer_state
        batch_progress = torch.zeros((), dtype=torch.float32, device=env.device)
        batch_success = torch.zeros((), dtype=torch.float32, device=env.device)
        completed = state.initialized[ids] & (env.episode_length_buf[ids] > 0)
        completed_ids = ids[completed]
        if completed_ids.numel():
            progress_context = env.termination_manager.get_term_cfg(progress_context_name).func
            final_success = env.termination_manager.get_term_cfg(final_success_context_name).func
            progressed = progress_context.ever_success[completed_ids]
            succeeded = final_success.ever_success[completed_ids]
            rows = state.row_ids[completed_ids]
            _ring_append_bool_count_rate(
                self._progress_history,
                rows,
                progressed,
                self._history_pointer,
                self._history_size,
                self._history_success_count,
                self._rolling_progress_rates,
            )
            self._attempts.scatter_add_(0, rows, torch.ones_like(rows))
            self._progress_successes.scatter_add_(0, rows, progressed.long())
            self._final_successes.scatter_add_(0, rows, succeeded.long())
            batch_progress = progressed.float().mean()
            batch_success = succeeded.float().mean()

        attempted_rows = self._history_size > 0
        row_coverage = attempted_rows.float().mean()
        total_progress = self._history_success_count.sum().float() / self._history_size.sum().clamp_min(1)
        deployment_probability = deployment_probability_from_progress(
            total_progress,
            row_coverage,
            initial_probability=deployment_probability_initial,
            final_probability=deployment_probability_final,
            progress_start=deployment_progress_start,
            progress_end=deployment_progress_end,
            coverage_target=deployment_coverage_target,
        )
        probabilities = reset_sampling_probabilities(
            self._reset_term.recipe_ids,
            self._reset_term.variant_ids,
            self._reset_term.target_cube_ids,
            self._reset_term.source_side_ids,
            self._history_size,
            self._history_success_count,
            deployment_probability,
            epsilon,
        )
        if fixed_source_side_id is not None:
            if fixed_source_side_id not in (0, 1):
                raise ValueError("fixed_source_side_id must be 0 (left) or 1 (right).")
            probabilities *= self._reset_term.source_side_ids == fixed_source_side_id
            probabilities /= probabilities.sum()
        if ids.numel():
            state.row_ids[ids] = torch.multinomial(probabilities, ids.numel(), replacement=True)

        cumulative_progress = self._progress_successes.sum().float() / self._attempts.sum().clamp_min(1)
        total_success = self._final_successes.sum().float() / self._attempts.sum().clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        entropy /= math.log(probabilities.numel())
        metrics: dict[str, torch.Tensor] = {
            "batch_progress_rate": batch_progress,
            "batch_success_rate": batch_success,
            "batch_transfer_count": (
                state.transfer_counts[completed_ids].float().mean()
                if completed_ids.numel()
                else torch.zeros((), dtype=torch.float32, device=env.device)
            ),
            "batch_left_to_right_transfers": (
                state.direction_transfer_counts[completed_ids, 0].float().mean()
                if completed_ids.numel()
                else torch.zeros((), dtype=torch.float32, device=env.device)
            ),
            "batch_right_to_left_transfers": (
                state.direction_transfer_counts[completed_ids, 1].float().mean()
                if completed_ids.numel()
                else torch.zeros((), dtype=torch.float32, device=env.device)
            ),
            "deployment_probability": deployment_probability,
            "row_coverage": row_coverage,
            "overall_progress_rate": total_progress,
            "cumulative_progress_rate": cumulative_progress,
            "overall_success_rate": total_success,
            "sampling_entropy": entropy,
        }
        for recipe in ConveyorResetRecipe:
            mask = self._reset_term.recipe_ids == int(recipe)
            recipe_attempts = self._attempts[mask].sum()
            metrics[f"recipe_{recipe.name.lower()}_probability"] = probabilities[mask].sum()
            recipe_history_size = self._history_size[mask].sum()
            metrics[f"recipe_{recipe.name.lower()}_progress_rate"] = self._history_success_count[
                mask
            ].sum().float() / recipe_history_size.clamp_min(1)
            metrics[f"recipe_{recipe.name.lower()}_success_rate"] = self._final_successes[
                mask
            ].sum().float() / recipe_attempts.clamp_min(1)
        for side_id, side_name in ((0, "left_to_right"), (1, "right_to_left")):
            mask = self._reset_term.source_side_ids == side_id
            attempts = self._attempts[mask].sum()
            history_size = self._history_size[mask].sum()
            metrics[f"direction_{side_name}_probability"] = probabilities[mask].sum()
            metrics[f"direction_{side_name}_progress_rate"] = self._history_success_count[
                mask
            ].sum().float() / history_size.clamp_min(1)
            metrics[f"direction_{side_name}_success_rate"] = self._final_successes[
                mask
            ].sum().float() / attempts.clamp_min(1)
        for recipe_name, variant_id, mask in self._diagnostic_variant_rows:
            attempts = self._attempts[mask].sum()
            history_size = self._history_size[mask].sum()
            prefix = f"recipe_{recipe_name}_variant_{variant_id}"
            metrics[f"{prefix}_probability"] = probabilities[mask].sum()
            metrics[f"{prefix}_progress_rate"] = self._history_success_count[
                mask
            ].sum().float() / history_size.clamp_min(1)
            metrics[f"{prefix}_success_rate"] = self._final_successes[mask].sum().float() / attempts.clamp_min(1)
        return metrics

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return curriculum evidence for checkpointing."""
        return {
            "attempts": self._attempts.clone(),
            "progress_successes": self._progress_successes.clone(),
            "final_successes": self._final_successes.clone(),
            "progress_history": self._progress_history.clone(),
            "history_pointer": self._history_pointer.clone(),
            "history_size": self._history_size.clone(),
            "history_success_count": self._history_success_count.clone(),
            "rolling_progress_rates": self._rolling_progress_rates.clone(),
        }

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore curriculum evidence from a checkpoint."""
        targets = {
            "attempts": self._attempts,
            "progress_successes": self._progress_successes,
            "final_successes": self._final_successes,
            "progress_history": self._progress_history,
            "history_pointer": self._history_pointer,
            "history_size": self._history_size,
            "history_success_count": self._history_success_count,
            "rolling_progress_rates": self._rolling_progress_rates,
        }
        for name, target in targets.items():
            if name not in state or state[name].shape != target.shape:
                raise ValueError(f"Conveyor curriculum checkpoint has invalid '{name}'.")
        history_len = self._progress_history.shape[1]
        if bool(torch.any((state["history_pointer"] < 0) | (state["history_pointer"] >= history_len))):
            raise ValueError("Conveyor curriculum checkpoint has invalid history pointers.")
        if bool(torch.any((state["history_size"] < 0) | (state["history_size"] > history_len))):
            raise ValueError("Conveyor curriculum checkpoint has invalid history sizes.")
        if bool(
            torch.any((state["history_success_count"] < 0) | (state["history_success_count"] > state["history_size"]))
        ):
            raise ValueError("Conveyor curriculum checkpoint has invalid rolling success counts.")
        for name, target in targets.items():
            target.copy_(state[name].to(device=target.device, dtype=target.dtype))
