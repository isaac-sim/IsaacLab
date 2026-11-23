# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Unit tests for Factory curriculum scheduling."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.nist.mdp.curriculums import DifficultyScheduler

pytestmark = pytest.mark.kitless


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
