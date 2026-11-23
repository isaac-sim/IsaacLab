# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-state weights for the NIST variant task's joint-cell balancing."""

from __future__ import annotations

from dataclasses import field

import torch

from isaaclab.utils import configclass


@configclass
class BetaSamplingStrategyCfg:
    """Weight states near the target success rate."""

    weight: float = 1.0
    target: float = 0.66
    kappa: float = 1.0


@configclass
class UniformSamplingStrategyCfg:
    """Give every state the same weight."""

    weight: float = 1.0


@configclass
class SamplerCfg:
    """Weighted sampling strategies applied before joint-cell balancing."""

    strategies: list[BetaSamplingStrategyCfg | UniformSamplingStrategyCfg] = field(default_factory=list)
    eps: float = 1e-3


class Sampler:
    """Build probabilities that the variant reset term can balance by cell."""

    def __init__(self, cfg: SamplerCfg, success_rates: torch.Tensor) -> None:
        self.cfg = cfg
        self.success_rates = success_rates
        self._score = torch.empty_like(success_rates, dtype=torch.float32)
        self._probs = torch.empty_like(success_rates, dtype=torch.float32)

    def probabilities(self) -> torch.Tensor:
        """Return a reused probability vector over reset states."""
        self._probs.fill_(float(self.cfg.eps))
        for strategy in self.cfg.strategies:
            weight = max(0.0, float(strategy.weight))
            if isinstance(strategy, UniformSamplingStrategyCfg):
                self._probs.add_(weight)
            else:
                target = max(0.0, min(1.0, float(strategy.target)))
                kappa = max(0.0, float(strategy.kappa))
                self._score.copy_(self.success_rates)
                self._score.pow_(kappa * target)
                self._score.mul_((1.0 - self.success_rates).pow(kappa * (1.0 - target)))
                self._probs.add_(self._score, alpha=weight)
        self._probs.div_(self._probs.sum())
        return self._probs

    def sample(self, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Draw reset-state indices from the supplied probabilities."""
        return torch.multinomial(probs, num_samples, replacement=True)
