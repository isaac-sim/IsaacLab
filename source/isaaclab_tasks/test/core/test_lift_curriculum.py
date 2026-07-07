# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the adaptive rigid-object lift curriculum."""

from types import SimpleNamespace

import torch

from isaaclab.managers import CurriculumTermCfg
from isaaclab_tasks.core.lift.mdp.curriculums import LiftDifficultyScheduler


class _TerminationManager:
    def __init__(self, success: torch.Tensor):
        self.success = success

    def get_term(self, _: str) -> torch.Tensor:
        return self.success


def test_lift_difficulty_requires_success_termination() -> None:
    """Only the task's success termination should advance the curriculum."""
    termination_manager = _TerminationManager(torch.tensor([False, True]))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        termination_manager=termination_manager,
    )
    cfg = CurriculumTermCfg(
        func=LiftDifficultyScheduler,
        params={
            "success_termination_name": "success",
            "max_difficulty": 20,
            "successes_to_promote": 3,
        },
    )
    scheduler = LiftDifficultyScheduler(cfg, env)
    env_ids = torch.arange(2)

    for _ in range(3):
        scheduler(env, env_ids, **cfg.params)

    assert scheduler.difficulties.tolist() == [0, 1]
    assert scheduler.success_streak.tolist() == [0, 0]


def test_lift_difficulty_resets_streak_after_failure() -> None:
    """Successes separated by a failed episode should not promote difficulty."""
    termination_manager = _TerminationManager(torch.tensor([True]))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        termination_manager=termination_manager,
    )
    cfg = CurriculumTermCfg(
        func=LiftDifficultyScheduler,
        params={
            "success_termination_name": "success",
            "max_difficulty": 20,
            "successes_to_promote": 3,
        },
    )
    scheduler = LiftDifficultyScheduler(cfg, env)

    scheduler(env, torch.tensor([0]), **cfg.params)
    termination_manager.success[0] = False
    scheduler(env, torch.tensor([0]), **cfg.params)
    termination_manager.success[0] = True
    scheduler(env, torch.tensor([0]), **cfg.params)

    assert scheduler.difficulties.item() == 0
    assert scheduler.success_streak.item() == 1
