# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific score for balancing NIST reset labels and assembly variants."""

from __future__ import annotations

import math
from collections.abc import Mapping

import warp as wp

from isaaclab.envs.reset import BetaSamplingScore, BetaSamplingScoreCfg, SamplingScoreCfg
from isaaclab.utils import configclass


@configclass
class BalancedVariantScoreCfg(SamplingScoreCfg):
    """Balance both marginals of the reset-label by asset-variant table."""

    class_type: type | str = "{DIR}.variant_sampling:BalancedVariantScore"
    num_labels: int = 1
    num_variants: int = 1
    target: float = 0.66
    kappa: float = 1.0
    beta_weight: float = 1.0
    floor: float = 1e-4


class BalancedVariantScore:
    """Supply balanced slot weights to the shared reset-state sampler."""

    def __init__(self, cfg: BalancedVariantScoreCfg) -> None:
        if cfg.num_labels < 1 or cfg.num_variants < 1:
            raise ValueError("The NIST reset score requires positive label and variant counts.")
        if not math.isfinite(cfg.beta_weight) or cfg.beta_weight < 0 or not math.isfinite(cfg.floor) or cfg.floor <= 0:
            raise ValueError("The NIST reset score requires nonnegative Beta weight and a positive uniform floor.")
        self.cfg = cfg
        self._beta = BetaSamplingScore(
            BetaSamplingScoreCfg(metric="monitor_success_rate", target=cfg.target, kappa=cfg.kappa)
        )
        self._beta_scale = (
            cfg.beta_weight
            * cfg.target ** (cfg.kappa * cfg.target)
            * (1 - cfg.target) ** (cfg.kappa * (1 - cfg.target))
        )
        self._raw_cells: wp.array | None = None
        self._balanced_cells: wp.array | None = None

    def score(self, metrics: Mapping[str, wp.array], out: wp.array) -> None:
        """Apply the original four row/column normalization passes to the Beta weights."""
        if self._raw_cells is None:
            self._raw_cells = wp.zeros(self.cfg.num_labels * self.cfg.num_variants, dtype=float, device=out.device)
            self._balanced_cells = wp.empty_like(self._raw_cells)
        self._beta.score(metrics, out)
        self._raw_cells.zero_()
        wp.launch(
            _accumulate_cells,
            out.shape,
            [out, metrics["cell_indices"], self._beta_scale, self.cfg.floor, self._raw_cells],
            device=out.device,
        )
        wp.copy(self._balanced_cells, self._raw_cells)
        wp.launch(
            _balance_cells, 1, [self._balanced_cells, self.cfg.num_labels, self.cfg.num_variants], device=out.device
        )
        wp.launch(
            _spread_cell_weights,
            out.shape,
            [out, metrics["cell_indices"], self._raw_cells, self._balanced_cells],
            device=out.device,
        )


@wp.kernel
def _accumulate_cells(
    weights: wp.array(dtype=float),
    cells: wp.array(dtype=wp.int64),
    beta_scale: float,
    floor: float,
    totals: wp.array(dtype=float),
):
    i = wp.tid()
    weight = beta_scale * weights[i] + floor
    weights[i] = weight
    wp.atomic_add(totals, wp.int32(cells[i]), weight)


@wp.kernel
def _balance_cells(cells: wp.array(dtype=float), num_labels: int, num_variants: int):
    for _ in range(4):
        for row in range(num_labels):
            start = row * num_variants
            total = float(0.0)
            for column in range(num_variants):
                total += cells[start + column]
            for column in range(num_variants):
                cells[start + column] /= total * float(num_labels)
        for column in range(num_variants):
            total = float(0.0)
            for row in range(num_labels):
                total += cells[row * num_variants + column]
            for row in range(num_labels):
                cells[row * num_variants + column] /= total * float(num_variants)


@wp.kernel
def _spread_cell_weights(
    weights: wp.array(dtype=float),
    cells: wp.array(dtype=wp.int64),
    raw_cells: wp.array(dtype=float),
    balanced_cells: wp.array(dtype=float),
):
    i = wp.tid()
    cell = wp.int32(cells[i])
    weights[i] *= balanced_cells[cell] / raw_cells[cell]
