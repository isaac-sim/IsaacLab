# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for metric-based scores and legacy Factory sampling signals."""

import pytest
import torch

from isaaclab.utils import instantiate

from isaaclab_tasks.contrib.nist.utils import BetaSamplingStrategyCfg, UniformSamplingStrategyCfg
from isaaclab_tasks.utils.sampling import TargetSuccessScoreCfg, UniformScoreCfg

pytestmark = pytest.mark.kitless


def _score(cfg, rates: torch.Tensor) -> torch.Tensor:
    strategy = instantiate(cfg, rates)
    scores = torch.empty_like(rates)
    strategy.score(scores)
    return scores


def test_beta_peaks_at_target():
    """Legacy Beta scoring keeps its raw amplitude and observes updates to its bound rates."""
    rates = torch.linspace(0.0, 1.0, 21)
    strategy = instantiate(BetaSamplingStrategyCfg(target=0.5, kappa=4.0), rates)
    scores = torch.empty_like(rates)
    strategy.score(scores)
    assert int(scores.argmax()) == 10
    torch.testing.assert_close(scores[10], torch.tensor(0.0625))
    assert (scores >= 0).all()
    rates.zero_()
    strategy.score(out=scores)
    assert torch.equal(scores, torch.zeros_like(rates))


@pytest.mark.parametrize(
    "target,kappa,rates,expected",
    [
        (0.5, 2.0, [0.25, 0.75], [0.75, 0.75]),
        (0.0, 2.0, [0.0, 0.5, 1.0], [1.0, 0.25, 0.0]),
        (1.0, 2.0, [0.0, 0.5, 1.0], [0.0, 0.25, 1.0]),
        (0.66, 0.0, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0]),
        (0.66, 1.0e6, [0.0, 0.65, 0.66, 0.67, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_target_success_has_analytic_unit_peak(target, kappa, rates, expected):
    """Peak normalization handles absent catalog targets, endpoints, and large concentration."""
    rates = torch.tensor(rates)
    original_rates = rates.clone()
    score = instantiate(TargetSuccessScoreCfg(metric="completion", target=target, kappa=kappa))
    out = torch.empty_like(rates)
    score.score({"completion": rates}, out)
    torch.testing.assert_close(out, torch.tensor(expected))
    assert torch.equal(rates, original_rates)


def test_uniform_returns_ones():
    """Uniform scoring needs no metric, while legacy calls retain the output dtype contract."""
    scores = torch.empty(4)
    instantiate(UniformScoreCfg()).score({}, scores)
    assert torch.equal(scores, torch.ones(4))

    rates = torch.rand(4, dtype=torch.float64)
    scores = _score(UniformSamplingStrategyCfg(), rates)
    assert scores.dtype == torch.float64
    assert torch.equal(scores, torch.ones_like(rates))
