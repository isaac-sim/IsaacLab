# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-aware sampling over the Franka Pour reset-state cache."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.configclass import configclass

__all__ = [
    "ResetDatasetSamplerCfg",
]


@configclass
class ResetDatasetSamplerCfg:
    """Configuration for the Franka Pour reset-dataset sampler."""

    monitored_history_len: int = 50
    """Number of recent Boolean outcomes retained independently for every reset row."""

    target_success_rate: float = 0.5
    """Per-row rolling success rate at which the Beta sampling kernel peaks."""

    kappa: float = 1.0
    """Concentration of the Beta sampling kernel."""

    epsilon: float = 1.0e-4
    """Positive per-row weight floor that prevents reset-row starvation."""

    uniform_fraction: float = 0.25
    """Fraction reserved for exact shuffled cyclic replay."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self.validate_values()

    def validate_values(self) -> None:
        """Validate the sampling parameters after runtime overrides."""
        if (
            isinstance(self.monitored_history_len, bool)
            or not isinstance(self.monitored_history_len, int)
            or self.monitored_history_len < 1
        ):
            raise ValueError("monitored_history_len must be a positive integer.")
        target_success_rate = float(self.target_success_rate)
        if (
            isinstance(self.target_success_rate, bool)
            or not math.isfinite(target_success_rate)
            or not 0.0 < target_success_rate < 1.0
        ):
            raise ValueError("target_success_rate must lie strictly between zero and one.")
        for name in ("kappa", "epsilon"):
            value = getattr(self, name)
            numeric_value = float(value)
            if isinstance(value, bool) or not math.isfinite(numeric_value) or numeric_value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        uniform_fraction = float(self.uniform_fraction)
        if (
            isinstance(self.uniform_fraction, bool)
            or not math.isfinite(uniform_fraction)
            or not 0.0 < uniform_fraction < 1.0
        ):
            raise ValueError("uniform_fraction must lie strictly between zero and one.")


def _ring_append_bool_count_rate(
    data: torch.Tensor,
    stream_ids: torch.Tensor,
    values: torch.Tensor,
    pointer: torch.Tensor,
    size: torch.Tensor,
    true_count: torch.Tensor,
    rate: torch.Tensor,
) -> None:
    """Append outcomes to per-row Boolean rings and update rolling success rates."""
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


class _ResetDatasetSampler:
    """Sample reset rows using a rolling-success kernel and exact cyclic replay."""

    def __init__(
        self,
        row_count: int,
        device: str | torch.device,
        cfg: ResetDatasetSamplerCfg | None = None,
    ) -> None:
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
            raise ValueError("row_count must be a positive integer.")
        self.cfg = cfg if cfg is not None else ResetDatasetSamplerCfg()
        self.cfg.validate_values()
        self.row_count = row_count
        self._alpha_minus_one = self.cfg.kappa * self.cfg.target_success_rate
        self._beta_minus_one = self.cfg.kappa * (1.0 - self.cfg.target_success_rate)

        self._success_rates = torch.zeros(row_count, dtype=torch.float32, device=device)
        self._success_history = torch.zeros(
            (row_count, self.cfg.monitored_history_len),
            dtype=torch.bool,
            device=device,
        )
        self._history_pointers = torch.zeros(row_count, dtype=torch.int32, device=device)
        self._history_sizes = torch.zeros_like(self._history_pointers)
        self._history_success_counts = torch.zeros_like(self._history_pointers)
        self._uniform_order = torch.randperm(row_count, device=device)
        self._uniform_cursor = 0
        self._uniform_credit = 0.0

    def _probabilities(self) -> torch.Tensor:
        """Return normalized Beta scores with a per-row epsilon floor."""
        weights = self._success_rates.pow(self._alpha_minus_one)
        weights *= (1.0 - self._success_rates).pow(self._beta_minus_one)
        weights.add_(self.cfg.epsilon)
        return weights / weights.sum()

    def _sample_with_uniform_replay(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample rows with a mixture of adaptive and exact cyclic replay."""
        device = self._success_rates.device
        rows = torch.empty(count, device=device, dtype=torch.long)
        if count == 0:
            return rows

        uniform_count = self._uniform_assignment_count(count)
        assignment_order = torch.randperm(count, device=device, generator=generator)
        uniform_positions = assignment_order[:uniform_count]
        adaptive_positions = assignment_order[uniform_count:]
        if uniform_count > 0:
            rows[uniform_positions] = self._take_uniform_rows(uniform_count, generator)
        if adaptive_positions.numel() > 0:
            rows[adaptive_positions] = torch.multinomial(
                self._probabilities(),
                adaptive_positions.numel(),
                replacement=True,
                generator=generator,
            )
        return rows

    def _record_validated(self, rows: torch.Tensor, success: torch.Tensor) -> None:
        """Append outcomes whose shape, dtype, device, and row values are already trusted."""
        _ring_append_bool_count_rate(
            self._success_history,
            rows,
            success,
            self._history_pointers,
            self._history_sizes,
            self._history_success_counts,
            self._success_rates,
        )

    def metrics(self) -> dict[str, float]:
        """Return sampling-concentration and cyclic-replay metrics."""
        replay_fraction = float(self.cfg.uniform_fraction)
        probabilities = self._probabilities()
        probabilities.mul_(1.0 - replay_fraction).add_(replay_fraction / self.row_count)
        top_count = max(1, math.ceil(0.01 * self.row_count))
        values = torch.stack(
            (
                probabilities.square().sum().reciprocal() / self.row_count,
                torch.topk(probabilities, top_count, sorted=False).values.sum(),
                probabilities.new_tensor(replay_fraction),
            )
        )
        names = (
            "sampler/effective_pool_fraction",
            "sampler/top_1_percent_mass",
            "sampler/uniform_replay_fraction",
        )
        return dict(zip(names, values.tolist(), strict=True))

    def _uniform_assignment_count(self, unforced_count: int) -> int:
        """Reserve the configured exact fraction for cyclic assignments over time."""
        replay_credit = self._uniform_credit + self.cfg.uniform_fraction * unforced_count
        uniform_count = math.floor(replay_credit + 1.0e-12)
        self._uniform_credit = replay_credit - uniform_count
        return min(unforced_count, uniform_count)

    def _take_uniform_rows(
        self,
        count: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Take rows from shuffled complete-cache cycles without replacement within a cycle."""
        if count == 0:
            return torch.empty(0, device=self._success_rates.device, dtype=torch.long)
        remaining = count
        chunks = []
        while remaining > 0:
            available = self.row_count - self._uniform_cursor
            take = min(remaining, available)
            chunks.append(self._uniform_order[self._uniform_cursor : self._uniform_cursor + take])
            self._uniform_cursor += take
            remaining -= take
            if self._uniform_cursor == self.row_count:
                self._uniform_order = torch.randperm(
                    self.row_count,
                    device=self._success_rates.device,
                    generator=generator,
                )
                self._uniform_cursor = 0
        return torch.cat(chunks)
