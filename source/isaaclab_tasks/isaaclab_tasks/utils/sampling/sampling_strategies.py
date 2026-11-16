# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-item scoring from caller-owned metrics, plus success-bound compatibility strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from .sampling_strategies_cfg import (
        BetaSamplingStrategyCfg,
        TargetSuccessScoreCfg,
        UniformSamplingStrategyCfg,
        UniformScoreCfg,
    )


class SamplingScore(Protocol):
    """Score items from current caller-owned metrics without retaining their tensors.

    Scorers accept only their config at construction. Each call writes non-negative
    scores into the caller's output buffer, shape [num_items]. All supplied metrics
    have the same item shape and device; each scorer chooses its metric keys.
    """

    name: str
    """Short identifier used in diagnostic and log keys."""

    def score(self, metrics: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
        """Write scores into ``out`` without modifying the supplied metrics."""
        ...


class TargetSuccessScore:
    """Beta-shaped target-success score with an analytic peak of one."""

    name = "target_success"

    def __init__(self, cfg: TargetSuccessScoreCfg) -> None:
        self._metric = cfg.metric
        self._target = max(0.0, min(1.0, float(cfg.target)))
        self._kappa = max(0.0, float(cfg.kappa))

    def score(self, metrics: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
        """Write a peak-normalized kernel, evaluated in log space for large concentration."""
        if self._kappa == 0.0:
            out.fill_(1.0)
            return

        rates = metrics[self._metric]
        if self._target == 0.0:
            out.copy_(rates).neg_().log1p_().mul_(self._kappa).exp_()
        elif self._target == 1.0:
            out.copy_(rates).log_().mul_(self._kappa).exp_()
        else:
            # Centering before log1p keeps the analytic peak at one, even for large kappa.
            delta = rates - self._target
            out.copy_(delta).div_(self._target).log1p_().mul_(self._target)
            delta.neg_().div_(1.0 - self._target).log1p_().mul_(1.0 - self._target)
            out.add_(delta).mul_(self._kappa).clamp_(max=0.0).exp_()
            out.masked_fill_((rates == 0.0) | (rates == 1.0), 0.0)


class UniformScore:
    """Constant score of one, requiring no metrics."""

    name = "uniform"

    def __init__(self, cfg: UniformScoreCfg) -> None:
        del cfg

    def score(self, metrics: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
        """Fill the caller's output buffer with ones."""
        out.fill_(1.0)


class SamplingStrategy(Protocol):
    """Per-item scorer for sampler probability construction.

    :meth:`score` writes non-negative scores into ``out`` using the success-rate
    tensor supplied at construction.
    """

    name: str
    """Short identifier used in diagnostic / log keys."""

    def score(self, out: torch.Tensor) -> None:
        """Write ``[num_items]`` non-negative unnormalized scores into ``out``."""
        ...


class BetaSamplingStrategy:
    """Per-item Beta-kernel score peaked at a target success rate."""

    name = "beta"

    def __init__(self, cfg: BetaSamplingStrategyCfg, success_rates: torch.Tensor | None = None) -> None:
        target = max(0.0, min(1.0, float(cfg.target)))
        kappa = max(0.0, float(cfg.kappa))
        self._a = 1.0 + kappa * target
        self._b = 1.0 + kappa * (1.0 - target)
        self._rates = success_rates

    def score(
        self, metrics: Mapping[str, torch.Tensor] | torch.Tensor | None = None, out: torch.Tensor | None = None
    ) -> None:
        """Write raw Beta scores from supplied metrics or the live constructor-bound rates."""
        if metrics is None:
            rates = self._rates
        elif out is None:
            out = metrics
            rates = self._rates
        else:
            rates = metrics["success_rate"]
        out.copy_(rates)
        out.pow_(self._a - 1.0)
        out.mul_((1.0 - rates).pow(self._b - 1.0))


class UniformSamplingStrategy:
    """Constant 1.0 per item -- the trivial baseline / floor."""

    name = "uniform"

    def __init__(self, cfg: UniformSamplingStrategyCfg, success_rates: torch.Tensor | None = None) -> None:
        del cfg, success_rates

    def score(
        self, metrics: Mapping[str, torch.Tensor] | torch.Tensor | None = None, out: torch.Tensor | None = None
    ) -> None:
        """Write ones using either the metric-based or constructor-bound calling convention."""
        if out is None:
            out = metrics
        out.fill_(1.0)
