# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Outcome-driven curriculum for UR10 particle pushing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from ..ur10_particle_push_env import UR10ParticlePushEnv


def update_curriculum_levels(
    level: torch.Tensor,
    success_streak: torch.Tensor,
    failure_streak: torch.Tensor,
    episode_success: torch.Tensor,
    *,
    max_level: int,
    successes_to_promote: int,
    failures_to_demote: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance or back off independent competence levels from completed episodes."""
    next_success_streak = torch.where(episode_success, success_streak + 1, torch.zeros_like(success_streak))
    next_failure_streak = torch.where(episode_success, torch.zeros_like(failure_streak), failure_streak + 1)
    promotion_ready = next_success_streak >= successes_to_promote
    demotion_ready = next_failure_streak >= failures_to_demote
    promoted = promotion_ready & (level < max_level)
    demoted = demotion_ready & (level > 0)
    next_level = (level + promoted.long() - demoted.long()).clamp(0, max_level)
    next_success_streak = torch.where(
        promotion_ready | demoted,
        torch.zeros_like(next_success_streak),
        next_success_streak,
    )
    next_failure_streak = torch.where(
        demotion_ready | promoted,
        torch.zeros_like(next_failure_streak),
        next_failure_streak,
    )
    return next_level, next_success_streak, next_failure_streak, promoted, demoted


class PushCurriculum(ManagerTermBase):
    """Update per-world reverse-curriculum levels from completed outcomes."""

    @staticmethod
    def _env_ids(
        env: UR10ParticlePushEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()

    def __call__(
        self,
        env: UR10ParticlePushEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, float]:
        """Record completed outcomes and select levels for the next resets."""
        ids = self._env_ids(env, env_ids)
        if ids.numel() == 0:
            return self._metrics(env)

        completed = env.reset_initialized[ids] & (env.episode_length_buf[ids] > 0)
        completed_ids = ids[completed]
        cycled_level = env.sample_reset_curriculum_levels(ids.numel())
        promoted = torch.zeros(completed_ids.numel(), dtype=torch.bool, device=env.device)
        demoted = torch.zeros_like(promoted)
        if completed_ids.numel() > 0:
            completed_level = env.curriculum_level[completed_ids].clone()
            if cycled_level is not None:
                next_level = cycled_level[completed]
            elif env.cfg.curriculum_level_override is None:
                next_level, success_streak, failure_streak, promoted, demoted = update_curriculum_levels(
                    completed_level,
                    env.curriculum_success_streak[completed_ids],
                    env.curriculum_failure_streak[completed_ids],
                    env.episode_success[completed_ids],
                    max_level=len(env.cfg.curriculum_pile_center_x) - 1,
                    successes_to_promote=env.cfg.curriculum_successes_to_promote,
                    failures_to_demote=env.cfg.curriculum_failures_to_demote,
                )
                env.curriculum_success_streak[completed_ids] = success_streak
                env.curriculum_failure_streak[completed_ids] = failure_streak
            else:
                next_level = torch.full_like(completed_level, env.cfg.curriculum_level_override)
            env.curriculum_level[completed_ids] = next_level

        if cycled_level is not None:
            env.curriculum_level[ids] = cycled_level
            env.curriculum_success_streak[ids] = 0
            env.curriculum_failure_streak[ids] = 0
        elif env.cfg.curriculum_level_override is not None:
            env.curriculum_level[ids] = env.cfg.curriculum_level_override
            env.curriculum_success_streak[ids] = 0
            env.curriculum_failure_streak[ids] = 0

        metrics = self._metrics(env)
        if completed_ids.numel() > 0:
            metrics.update(
                success_rate=float(env.episode_success[completed_ids].float().mean()),
                promotion_rate=float(promoted.float().mean()),
                demotion_rate=float(demoted.float().mean()),
            )
            split_episode = env.split_source_episode[completed_ids]
            transition_episode = split_episode | env.post_first_sweep_reset[completed_ids]
            if bool(split_episode.any()):
                metrics["first_sweep_rate"] = float(
                    env.first_sweep_complete[completed_ids][split_episode].float().mean()
                )
            if bool(transition_episode.any()):
                metrics["second_pile_reach_rate"] = float(
                    env.second_pile_reached[completed_ids][transition_episode].float().mean()
                )
                metrics["second_push_rate"] = float(
                    env.second_push_started[completed_ids][transition_episode].float().mean()
                )
            post_first_sweep = env.post_first_sweep_reset[completed_ids]
            if bool(post_first_sweep.any()):
                metrics["post_first_sweep_success_rate"] = float(
                    env.episode_success[completed_ids][post_first_sweep].float().mean()
                )
            for level in completed_level.unique(sorted=True).tolist():
                level_mask = completed_level == level
                metrics[f"level_{level}_success_rate"] = float(
                    env.episode_success[completed_ids][level_mask].float().mean()
                )
        return metrics

    @staticmethod
    def _metrics(env: UR10ParticlePushEnv) -> dict[str, float]:
        return {"mean_level": float(env.curriculum_level.float().mean())}
