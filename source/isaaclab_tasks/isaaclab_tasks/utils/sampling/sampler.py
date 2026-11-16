# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Weighted sampling over a fixed item catalog and caller-owned metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import instantiate

from .sampling_strategies import SamplingScore, SamplingStrategy
from .sampling_strategies_cfg import BetaSamplingStrategyCfg, SamplingScoreCfg, UniformSamplingStrategyCfg

if TYPE_CHECKING:
    from .sampler_cfg import SamplerCfg


class Sampler:
    """Compose weighted score terms over per-item metrics.

    Constructor-bound custom strategies remain supported with a success-rate tensor.
    To use item-count construction, derive their configs from :class:`SamplingScoreCfg`,
    accept only the config at construction, and implement ``score(metrics, out)``.

    Args:
        cfg: Score terms and the additive score floor.
        num_items: Number of catalog items. A success-rate tensor retains the legacy
            binding and makes metrics optional in subsequent calls.
        device: Device for score and probability buffers.
        success_rates: Keyword form of the legacy success-rate tensor binding.
    """

    def __init__(
        self,
        cfg: SamplerCfg,
        num_items: int | torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        *,
        success_rates: torch.Tensor | None = None,
    ) -> None:
        if success_rates is not None:
            if num_items is not None:
                raise TypeError("Pass either num_items or legacy success_rates, not both.")
            num_items = success_rates
        self._metrics: Mapping[str, torch.Tensor] = {}
        if isinstance(num_items, torch.Tensor):
            success_rates = num_items
            self._metrics = {"success_rate": success_rates}
            device = success_rates.device
            num_items = len(success_rates)
        if not isinstance(num_items, int) or num_items < 1:
            raise ValueError("num_items must be a positive integer.")

        self.strategies: list[tuple[SamplingScore | SamplingStrategy, float]] = []
        self._legacy_strategies: list[bool] = []
        for strategy_cfg in cfg.strategies:
            legacy = success_rates is not None and (
                isinstance(strategy_cfg, (BetaSamplingStrategyCfg, UniformSamplingStrategyCfg))
                or not isinstance(strategy_cfg, SamplingScoreCfg)
            )
            strategy = instantiate(strategy_cfg, success_rates) if legacy else instantiate(strategy_cfg)
            self.strategies.append((strategy, float(strategy_cfg.weight)))
            self._legacy_strategies.append(legacy)
        self.eps = float(cfg.eps)
        self.names = [strategy.name for strategy, _ in self.strategies]
        self._score_rows = torch.empty((len(self.strategies), num_items), device=device, dtype=torch.float32)
        self._weighted = torch.empty(num_items, device=device, dtype=torch.float32)
        self._probs = torch.empty_like(self._weighted)

    def scores(self, metrics: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Return reusable score rows shaped ``[num_strategies, num_items]``.

        Each metric must have shape [num_items] on the sampler's device. Scorers
        choose their metric keys; uniform scoring accepts an empty mapping.
        The returned buffer is overwritten by the next score or probability call.
        """
        if metrics is None:
            metrics = self._metrics
        for name, metric in metrics.items():
            if metric.shape != self._weighted.shape or metric.device != self._weighted.device:
                raise ValueError(
                    f"Metric {name!r} must have shape {tuple(self._weighted.shape)} on {self._weighted.device}."
                )
        for i, (strategy, _) in enumerate(self.strategies):
            if self._legacy_strategies[i]:
                strategy.score(self._score_rows[i])
            else:
                strategy.score(metrics, self._score_rows[i])
        return self._score_rows

    def probabilities(self, metrics: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Return the reusable normalized probability buffer, shape [num_items]."""
        scores = self.scores(metrics)
        self._weighted.zero_()
        for i, (_, weight) in enumerate(self.strategies):
            self._weighted.add_(scores[i], alpha=max(0.0, weight))
        self._weighted.add_(self.eps)
        torch.div(self._weighted, self._weighted.sum(), out=self._probs)
        return self._probs

    def sample(self, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample item indices from ``probs``."""
        return torch.multinomial(probs, num_samples, replacement=True)

    def probabilities_and_sample(
        self, num_samples: int, metrics: Mapping[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probabilities from the supplied metrics and sampled item indices."""
        probs = self.probabilities(metrics)
        return probs, self.sample(probs, num_samples)
