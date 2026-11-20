# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native scores for selecting reset states from caller-owned metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol

import warp as wp

from .sampling_scores_cfg import BetaSamplingScoreCfg


@wp.func
def beta_sampling_score(rate: float, target: float, kappa: float) -> float:
    """Evaluate a Beta kernel scaled to one at its mode, ``target``."""
    if kappa == 0.0:
        return 1.0
    rate = wp.clamp(rate, 0.0, 1.0)
    if target == 0.0:
        return wp.pow(1.0 - rate, kappa)
    if target == 1.0:
        return wp.pow(rate, kappa)
    if rate == 0.0 or rate == 1.0:
        return 0.0
    log_score = target * wp.log(rate / target)
    log_score += (1.0 - target) * wp.log((1.0 - rate) / (1.0 - target))
    return wp.exp(wp.min(0.0, kappa * log_score))


@wp.kernel
def _beta_sampling_score(metric: wp.array(dtype=Any), target: float, kappa: float, out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = beta_sampling_score(float(metric[i]), target, kappa)


class SamplingScore(Protocol):
    """Write finite non-negative scores into caller-owned Warp storage without mutating metrics."""

    def score(self, metrics: Mapping[str, wp.array], out: wp.array) -> None:
        """Write one score per catalog entry; all storage is allocated before capture."""
        ...


class BetaSamplingScore:
    """Score the configured metric through :func:`beta_sampling_score`."""

    def __init__(self, cfg: BetaSamplingScoreCfg) -> None:
        if not math.isfinite(cfg.target) or not 0.0 <= cfg.target <= 1.0:
            raise ValueError("target must be finite and in [0, 1].")
        if not math.isfinite(cfg.kappa) or cfg.kappa < 0.0:
            raise ValueError("kappa must be finite and non-negative.")
        self.cfg = cfg

    def score(self, metrics: Mapping[str, wp.array], out: wp.array) -> None:
        """Write scores using the current metric values, without device allocation or synchronization."""
        wp.launch(
            _beta_sampling_score,
            out.shape,
            [metrics[self.cfg.metric], self.cfg.target, self.cfg.kappa, out],
            device=out.device,
        )
