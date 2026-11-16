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

from .sampling_cfg import TargetSuccessScoreCfg, UniformScoreCfg


@wp.func
def target_success_score(rate: float, target: float, kappa: float) -> float:
    """Return a target-success kernel with analytic peak one, suitable for inlining in MDP terms."""
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


@wp.func
def sample_cdf(cumulative: wp.array(dtype=Any), start: int, count: int, uniform: float) -> int:
    """Draw from a cumulative-weight segment using a supplied sample in [0, 1).

    ``count`` must be positive. A zero-total segment falls back to uniform selection.
    Both float32 and float64 storage can be used by inlined MDP terms.
    """
    end = start + count
    base = wp.float64(0.0)
    if start > 0:
        base = wp.float64(cumulative[start - 1])
    total = wp.float64(cumulative[end - 1]) - base
    if total <= wp.float64(0.0):
        return start + wp.min(int(uniform * float(count)), count - 1)
    draw = base + wp.float64(uniform) * total
    lo = start
    hi = end - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if wp.float64(cumulative[mid]) <= draw:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel
def _target_success(metric: wp.array(dtype=Any), target: float, kappa: float, out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = target_success_score(float(metric[i]), target, kappa)


class SamplingScore(Protocol):
    """Write non-negative scores into caller-provided Warp storage without changing metric inputs."""

    name: str

    def score(self, metrics: Mapping[str, wp.array], out: wp.array) -> None:
        """Write one score per catalog entry; all storage is allocated before capture."""
        ...


class TargetSuccessScore:
    """Score current success rates through :func:`target_success_score`."""

    name = "target_success"

    def __init__(self, cfg: TargetSuccessScoreCfg) -> None:
        if not math.isfinite(cfg.target) or not 0.0 <= cfg.target <= 1.0:
            raise ValueError("target must be finite and in [0, 1].")
        if not math.isfinite(cfg.kappa) or cfg.kappa < 0.0:
            raise ValueError("kappa must be finite and non-negative.")
        self.cfg = cfg

    def score(self, metrics: Mapping[str, wp.array], out: wp.array) -> None:
        """Write scores using the current metric values, without device allocation or synchronization."""
        wp.launch(
            _target_success,
            out.shape,
            [metrics[self.cfg.metric], self.cfg.target, self.cfg.kappa, out],
            device=out.device,
        )


class UniformScore:
    """Fill caller-provided storage with one."""

    name = "uniform"

    def __init__(self, cfg: UniformScoreCfg) -> None:
        del cfg

    def score(self, metrics: Mapping[str, wp.array], out: wp.array) -> None:
        """Write constant scores without reading metrics."""
        out.fill_(1.0)
