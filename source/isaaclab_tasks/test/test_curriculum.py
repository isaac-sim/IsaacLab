# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Factory sampling and curriculum scheduling."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils import configclass, instantiate

from isaaclab_tasks.contrib.nist.mdp.curriculums import DifficultyScheduler
from isaaclab_tasks.contrib.nist.utils import (
    BetaSamplingStrategyCfg,
    Sampler,
    SamplerCfg,
    UniformSamplingStrategyCfg,
)
from isaaclab_tasks.utils.sampling import SamplingScoreCfg, TargetSuccessScoreCfg, UniformScoreCfg

pytestmark = pytest.mark.kitless


class _MetricScore:
    """Example application score based on a metric unrelated to success."""

    name = "novelty"

    def __init__(self, cfg: SamplingScoreCfg) -> None:
        del cfg

    def score(self, metrics, out: torch.Tensor) -> None:
        out.copy_(metrics["novelty"])


class _LegacySuccessScore:
    """Example constructor-bound strategy from the original sampling API."""

    name = "legacy_success"

    def __init__(self, cfg, success_rates: torch.Tensor) -> None:
        del cfg
        self._rates = success_rates

    def score(self, out: torch.Tensor) -> None:
        out.copy_(self._rates).square_()


@configclass
class _LegacySuccessScoreCfg:
    class_type: type = _LegacySuccessScore
    weight: float = 4.0


def test_uniform_only_sampler_is_uniform():
    """Uniform-only sampling constructs and scores a catalog without any success-rate tensor."""
    cfg = SamplerCfg(strategies=[UniformScoreCfg(weight=1.0)], eps=0.0)
    probs = instantiate(cfg, 100, "cpu").probabilities({})
    torch.testing.assert_close(probs, torch.full_like(probs, 1.0 / 100))


def test_custom_metric_score_uses_current_values():
    """Custom score configs consume arbitrary caller-owned metrics on every probability call."""
    cfg = SamplerCfg(strategies=[SamplingScoreCfg(class_type=_MetricScore, weight=2.0)], eps=0.0)
    sampler = instantiate(cfg, 3, "cpu")
    novelty = torch.tensor([1.0, 2.0, 1.0])
    torch.testing.assert_close(sampler.probabilities({"novelty": novelty}), torch.tensor([0.25, 0.5, 0.25]))
    assert torch.equal(novelty, torch.tensor([1.0, 2.0, 1.0]))
    novelty = torch.tensor([0.0, 0.0, 1.0])
    scores = sampler.scores({"novelty": novelty})
    torch.testing.assert_close(scores, torch.tensor([[0.0, 0.0, 1.0]]))
    probs, samples = sampler.probabilities_and_sample(12, metrics={"novelty": novelty})
    torch.testing.assert_close(probs, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(samples, torch.full((12,), 2))
    assert torch.equal(novelty, torch.tensor([0.0, 0.0, 1.0]))


def test_target_success_concentration_preserves_mixture_peak():
    """Concentration narrows the kernel without changing its strength relative to a uniform term."""
    metrics = {"success_rate": torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])}
    cfg = SamplerCfg(
        strategies=[TargetSuccessScoreCfg(weight=0.25, target=0.5, kappa=2.0), UniformScoreCfg(weight=0.05)],
        eps=1.0e-4,
    )
    sampler = instantiate(cfg, 5, "cpu")
    broad = sampler.probabilities(metrics).clone()
    torch.testing.assert_close(broad, torch.tensor([0.057224, 0.271388, 0.342776, 0.271388, 0.057224]))
    cfg.strategies[0].kappa = 1000.0
    concentrated = instantiate(cfg, 5, "cpu").probabilities(metrics)
    torch.testing.assert_close(concentrated[2] / concentrated[0], broad[2] / broad[0])
    assert concentrated[2] > broad[2]
    assert concentrated[1] < broad[1]


def test_sampler_probabilities_are_finite_nonnegative_and_normalized():
    rates = torch.rand(50)
    cfg = SamplerCfg(
        strategies=[
            BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
            UniformSamplingStrategyCfg(weight=0.1),
        ],
        eps=1e-3,
    )
    sampler = instantiate(cfg, rates)
    assert sampler.names == ["beta", "uniform"]
    probs = sampler.probabilities()
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


def test_legacy_sampler_uses_live_bound_success_rates():
    """NIST sampler calls retain raw Beta weights, their epsilon floor, and live rate binding."""
    rates = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    cfg = SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.5, kappa=2.0), UniformSamplingStrategyCfg(weight=0.05)],
        eps=1.0e-4,
    )
    sampler = Sampler(cfg, success_rates=rates)
    torch.testing.assert_close(
        sampler.probabilities(), torch.tensor([0.057224, 0.271388, 0.342776, 0.271388, 0.057224])
    )
    rates.copy_(torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(sampler.scores()[0], torch.tensor([0.0, 0.25, 0.0, 0.0, 0.0]))
    probabilities, choices = sampler.probabilities_and_sample(12)
    assert probabilities.argmax() == 1
    assert choices.shape == (12,)
    assert torch.all((choices >= 0) & (choices < 5))


def test_legacy_custom_strategy_keeps_constructor_bound_rates():
    """Legacy custom strategies retain live constructor binding beside canonical score configs."""
    rates = torch.tensor([0.0, 0.5, 1.0])
    cfg = SamplerCfg(strategies=[_LegacySuccessScoreCfg(), UniformScoreCfg()], eps=0.0)
    sampler = Sampler(cfg, rates)
    torch.testing.assert_close(sampler.probabilities(), torch.tensor([0.125, 0.25, 0.625]))
    rates.copy_(torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(sampler.scores(), torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    torch.testing.assert_close(sampler.probabilities(), torch.tensor([5.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0]))


def test_sampler_clamps_negative_strategy_weight():
    """A negative strategy weight contributes no probability mass."""
    rates = torch.rand(50)
    beta = BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0)
    beta_only = Sampler(SamplerCfg(strategies=[beta], eps=1e-6), rates).probabilities()
    with_negative = Sampler(
        SamplerCfg(strategies=[beta, UniformSamplingStrategyCfg(weight=-1.0)], eps=1e-6), rates
    ).probabilities()
    torch.testing.assert_close(with_negative, beta_only)


def test_difficulty_scheduler_waits_for_accumulator_rates():
    """Initial curriculum updates preserve difficulty until reset rates exist."""
    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    reset_accumulator = SimpleNamespace(monitor_success_rate=None)
    env = SimpleNamespace(
        device=torch.device("cpu"),
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator)),
    )

    result = DifficultyScheduler.__call__(
        scheduler,
        env,
        torch.arange(3),
        "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
        max_difficulty=10,
    )

    assert result is scheduler.difficulty_frac
    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 2)


def test_difficulty_scheduler_averages_ready_success_rates():
    """Curriculum updates average the reset bank's per-state rates."""
    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    reset_accumulator = SimpleNamespace(monitor_success_rate=torch.ones(4))
    env = SimpleNamespace(
        device=torch.device("cpu"),
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator)),
    )

    result = DifficultyScheduler.__call__(
        scheduler,
        env,
        torch.arange(3),
        "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
        max_difficulty=10,
    )

    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 3)
    torch.testing.assert_close(result, torch.tensor(0.3))
