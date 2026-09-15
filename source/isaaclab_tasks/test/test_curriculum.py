# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Factory sampling and curriculum scheduling."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.nist.mdp.curriculums import DifficultyScheduler
from isaaclab_tasks.contrib.nist.utils import (
    BetaSamplingStrategyCfg,
    Sampler,
    SamplerCfg,
    UniformSamplingStrategyCfg,
)


def test_uniform_only_sampler_is_uniform():
    """A uniform-only sampler with no floor returns exact uniform probabilities."""
    rates = torch.rand(100)
    cfg = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0)
    probs = cfg.class_type(cfg, rates).probabilities()
    torch.testing.assert_close(probs, torch.full_like(probs, 1.0 / 100))


def test_sampler_probabilities_are_finite_nonnegative_and_normalized():
    rates = torch.rand(50)
    sampler = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                UniformSamplingStrategyCfg(weight=0.1),
            ],
            eps=1e-3,
        ),
        rates,
    )
    probs = sampler.probabilities()
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


def test_sampler_clamps_negative_strategy_weight():
    """A negative strategy weight contributes no probability mass."""
    rates = torch.rand(50)
    positive = Sampler(SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=1e-6), rates).probabilities()
    negative = Sampler(
        SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=-1.0)], eps=1e-6), rates
    ).probabilities()
    torch.testing.assert_close(positive, negative)


def test_sampler_cfg_builds_runtime_sampler():
    rates = torch.rand(50)
    cfg = SamplerCfg(strategies=[BetaSamplingStrategyCfg(target=0.5, kappa=2.0)])
    sampler = cfg.class_type(cfg, rates)
    assert isinstance(sampler, Sampler)
    assert sampler.names == ["beta"]


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
