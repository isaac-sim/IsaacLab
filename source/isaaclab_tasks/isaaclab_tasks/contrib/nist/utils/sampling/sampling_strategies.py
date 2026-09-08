# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime sampling strategies.

:meth:`SamplingStrategy.score` writes a per-item non-negative score tensor into
the pre-allocated ``out`` buffer; the sampler sums weighted scores and
normalizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.nist.utils.sampling.sampling_strategies_cfg import (
        BetaSamplingStrategyCfg,
        UniformSamplingStrategyCfg,
    )


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

    def __init__(self, cfg: BetaSamplingStrategyCfg, success_rates: torch.Tensor) -> None:
        target = max(0.0, min(1.0, float(cfg.target)))
        kappa = max(0.0, float(cfg.kappa))
        self._a = 1.0 + kappa * target
        self._b = 1.0 + kappa * (1.0 - target)
        self._rates = success_rates

    def score(self, out: torch.Tensor) -> None:
        out.copy_(self._rates)
        out.pow_(self._a - 1.0)
        out.mul_((1.0 - self._rates).pow(self._b - 1.0))


class UniformSamplingStrategy:
    """Constant 1.0 per item -- the trivial baseline / floor."""

    name = "uniform"

    def __init__(self, cfg: UniformSamplingStrategyCfg, success_rates: torch.Tensor) -> None:
        del cfg, success_rates

    def score(self, out: torch.Tensor) -> None:
        out.fill_(1.0)
