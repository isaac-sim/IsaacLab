# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive curricula for rigid-object lift tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LiftDifficultyScheduler(ManagerTermBase):
    """Advance each environment after task success at its current level.

    Difficulty is promotion-only so environments cannot retreat to easy resets.
    The task's success termination is authoritative: reaching the goal ends the
    episode, advances the reset distribution, and prevents any post-success drop
    from affecting curriculum progression.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        initial = cfg.params.get("initial_difficulty", 0)
        self.difficulties = torch.full((env.num_envs,), initial, dtype=torch.long, device=env.device)
        self.success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_termination_name: str,
        initial_difficulty: int = 0,
        max_difficulty: int = 20,
        successes_to_promote: int = 3,
    ) -> dict[str, torch.Tensor]:
        """Update levels and return aggregate curriculum metrics."""
        del initial_difficulty
        succeeded = env.termination_manager.get_term(success_termination_name)[env_ids]
        self.success_streak[env_ids] = torch.where(
            succeeded,
            self.success_streak[env_ids] + 1,
            torch.zeros_like(self.success_streak[env_ids]),
        )
        promote = self.success_streak[env_ids] >= successes_to_promote
        self.difficulties[env_ids] = (self.difficulties[env_ids] + promote.long()).clamp(max=max_difficulty)
        self.success_streak[env_ids] = torch.where(
            promote,
            torch.zeros_like(self.success_streak[env_ids]),
            self.success_streak[env_ids],
        )
        fractions = self.difficulties.float() / max(max_difficulty, 1)
        return {
            "mean_level": self.difficulties.float().mean(),
            "mean_fraction": fractions.mean(),
            "at_final_level": (self.difficulties == max_difficulty).float().mean(),
            "mastery_rate": succeeded.float().mean(),
        }

    def difficulty_fraction(self, env_ids: Sequence[int], max_difficulty: int | None = None) -> torch.Tensor:
        """Return normalized per-environment difficulty in ``[0, 1]``."""
        if max_difficulty is None:
            max_difficulty = self.cfg.params.get("max_difficulty", 20)
        return self.difficulties[env_ids].float() / max(max_difficulty, 1)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return curriculum state for checkpointing."""
        return {"difficulties": self.difficulties, "success_streak": self.success_streak}

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore curriculum state from a checkpoint."""
        self.difficulties.copy_(state["difficulties"].to(self.device))
        self.success_streak.copy_(state["success_streak"].to(self.device))


def lift_difficulty_fraction(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return the active lift curriculum's normalized per-environment difficulty."""
    term: LiftDifficultyScheduler = env.curriculum_manager.cfg.lift_difficulty.func
    return term.difficulty_fraction(env_ids)
