# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Factory sampling and curriculum scheduling."""

from types import SimpleNamespace

import torch

from isaaclab.utils import instantiate

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
    probs = instantiate(cfg, rates).probabilities()
    torch.testing.assert_close(probs, torch.full_like(probs, 1.0 / 100))


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


def test_choice_preset_curriculum_reads_the_selected_reset_term():
    """The ``choice`` preset resets through ``TermChoice``, whose rates its curriculum callback reads."""
    from isaaclab_tasks.contrib.nist.factory_env_cfg import FactoryEnvCfg
    from isaaclab_tasks.contrib.nist.utils import ChainedResetTerms, TermChoice
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = resolve_presets(FactoryEnvCfg(), ("choice",))
    reset_cfg = cfg.events.reset_strategies
    assert reset_cfg.func is ChainedResetTerms

    env = SimpleNamespace(num_envs=3, device=torch.device("cpu"))
    choice_cfg = reset_cfg.params["terms"]["reset_strategies"]
    choice_cfg.func = TermChoice(choice_cfg, env)
    choice_cfg.func.term_success_rate.fill_(1.0)
    reset_cfg.func = ChainedResetTerms(reset_cfg, env)
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: getattr(cfg.events, name))

    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    params = cfg.curriculum.difficulty_scheduler.params
    result = DifficultyScheduler.__call__(
        scheduler, env, torch.arange(3), params["success_rate_callback"], max_difficulty=params["max_difficulty"]
    )

    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 3)
    torch.testing.assert_close(result, torch.tensor(0.3))
