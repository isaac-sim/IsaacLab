# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Weighted sampling over a fixed success-rate table."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.contrib.nist.utils.sampling.sampling_strategies import SamplingStrategy

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.nist.utils.sampling.sampler_cfg import SamplerCfg


class Sampler:
    """Compose weighted sampling strategies over per-item success rates."""

    def __init__(self, cfg: SamplerCfg, success_rates: torch.Tensor) -> None:
        self.strategies: list[tuple[SamplingStrategy, float]] = [
            (strategy_cfg.class_type(strategy_cfg, success_rates), float(strategy_cfg.weight))
            for strategy_cfg in cfg.strategies
        ]
        self.eps = float(cfg.eps)
        self.names = [strategy.name for strategy, _ in self.strategies]
        self._score_rows = torch.empty(
            (len(self.strategies), len(success_rates)), device=success_rates.device, dtype=torch.float32
        )
        self._weighted = torch.empty_like(success_rates, dtype=torch.float32)
        self._probs = torch.empty_like(success_rates, dtype=torch.float32)

    def scores(self) -> torch.Tensor:
        """Return contiguous per-strategy score rows shaped ``[num_strategies, num_items]``."""
        for i, (strategy, _) in enumerate(self.strategies):
            strategy.score(self._score_rows[i])
        return self._score_rows

    def probabilities(self) -> torch.Tensor:
        """Return ``[num_items]`` probability vector summing to 1."""
        scores = self.scores()
        self._weighted.zero_()
        for i, (_, weight) in enumerate(self.strategies):
            self._weighted.add_(scores[i], alpha=max(0.0, weight))
        self._weighted.add_(self.eps)
        torch.div(self._weighted, self._weighted.sum(), out=self._probs)
        return self._probs

    def sample(self, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample item indices from ``probs``."""
        return torch.multinomial(probs, num_samples, replacement=True)

    def probabilities_and_sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probabilities and sampled item indices."""
        probs = self.probabilities()
        return probs, self.sample(probs, num_samples)
