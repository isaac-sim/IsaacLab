# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-aware sampling over the Franka Pour reset-state cache."""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.utils import SuccessMonitor, SuccessMonitorCfg
from isaaclab_tasks.utils.sampling import BetaSamplingStrategyCfg, Sampler, SamplerCfg

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


class ResetDatasetSampler:
    """Combine shared outcome weighting with Pour's exact cyclic replay."""

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
        self._success_monitor = SuccessMonitor(
            SuccessMonitorCfg(monitored_history_len=self.cfg.monitored_history_len), 1, row_count, device
        )
        self._sampler = Sampler(
            SamplerCfg(
                strategies=[BetaSamplingStrategyCfg(target=self.cfg.target_success_rate, kappa=self.cfg.kappa)],
                eps=self.cfg.epsilon,
            ),
            self._success_monitor.success_rate,
        )
        self._uniform_order = torch.randperm(row_count, device=device)
        self._uniform_cursor = 0
        self._uniform_credit = 0.0

    def _probabilities(self) -> torch.Tensor:
        """Return normalized Beta scores with a per-row epsilon floor."""
        return self._sampler.probabilities()

    def _sample_with_uniform_replay(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample rows with a mixture of adaptive and exact cyclic replay."""
        device = self._uniform_order.device
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
        self._success_monitor.success_update(rows, success)

    def metrics(self) -> dict[str, float]:
        """Return sampling-concentration and cyclic-replay metrics."""
        replay_fraction = float(self.cfg.uniform_fraction)
        probabilities = self._probabilities() * (1.0 - replay_fraction) + replay_fraction / self.row_count
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
            return torch.empty(0, device=self._uniform_order.device, dtype=torch.long)
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
                    device=self._uniform_order.device,
                    generator=generator,
                )
                self._uniform_cursor = 0
        return torch.cat(chunks)
