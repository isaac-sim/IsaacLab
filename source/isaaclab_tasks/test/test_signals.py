# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Factory sampling signals."""

import torch

from isaaclab_tasks.contrib.nist.utils import BetaSamplingStrategyCfg, UniformSamplingStrategyCfg


def _score(cfg, rates: torch.Tensor) -> torch.Tensor:
    strategy = cfg.class_type(cfg, rates)
    scores = torch.empty_like(rates)
    strategy.score(scores)
    return scores


def test_beta_peaks_at_target():
    """Beta score maximum is at the configured success-rate target, and scores are non-negative over [0, 1]."""
    rates = torch.linspace(0.0, 1.0, 21)
    scores = _score(BetaSamplingStrategyCfg(target=0.5, kappa=4.0), rates)
    assert int(scores.argmax()) == 10
    assert (scores >= 0).all()


def test_uniform_returns_ones():
    """Uniform scoring is independent of success rates and writes into the caller-provided output dtype."""
    for dtype in (torch.float32, torch.float64):
        rates = torch.rand(100, dtype=dtype)
        scores = _score(UniformSamplingStrategyCfg(), rates)
        assert scores.dtype == dtype
        assert torch.equal(scores, torch.ones_like(rates))
