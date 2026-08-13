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
    """Beta score maximum is at the configured success-rate target."""
    rates = torch.linspace(0.0, 1.0, 21)
    scores = _score(BetaSamplingStrategyCfg(target=0.5, kappa=4.0), rates)
    assert int(scores.argmax()) == 10


def test_beta_uniform_input_returns_uniform_output():
    """Equal success rates produce equal Beta scores."""
    rates = torch.full((100,), 0.42)
    scores = _score(BetaSamplingStrategyCfg(target=0.66, kappa=1.0), rates)
    assert float(scores.std()) < 1e-7


def test_beta_score_is_non_negative():
    """Beta scores are non-negative over the valid success-rate range."""
    scores = _score(BetaSamplingStrategyCfg(target=0.66, kappa=1.0), torch.rand(100))
    assert (scores >= 0).all()


def test_uniform_returns_ones():
    """Uniform scoring is independent of success rates."""
    rates = torch.rand(100)
    assert torch.equal(_score(UniformSamplingStrategyCfg(), rates), torch.ones_like(rates))


def test_uniform_preserves_dtype():
    """Uniform scoring writes into the caller-provided output dtype."""
    for dtype in (torch.float32, torch.float64):
        rates = torch.rand(64, dtype=dtype)
        assert _score(UniformSamplingStrategyCfg(), rates).dtype == dtype
