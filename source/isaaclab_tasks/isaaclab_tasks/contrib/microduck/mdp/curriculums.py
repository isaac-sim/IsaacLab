# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking curriculums."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.envs.mdp import modify_term_cfg

from .commands import MicroDuckRoughVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def staged_value(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], old_value: Any, stages: Sequence[tuple[int, Any]]
) -> Any:
    """``modify_fn`` for :class:`~isaaclab.envs.mdp.modify_term_cfg` that steps through a schedule.

    Args:
        stages: ``(start_step, value)`` pairs in increasing step order; the last reached value applies.
    """
    value = old_value
    for start_step, stage_value in stages:
        if env.common_step_counter >= start_step:
            value = stage_value
    return modify_term_cfg.NO_CHANGE if value == old_value else value


def microduck_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    min_distance: float = 1.0,
    min_walking_time: float = 5.0,
    promote_error_ratio: float = 0.5,
    demote_error_ratio: float = 0.8,
    promote_yaw_error: float = 0.5,
    demote_yaw_error: float = 0.8,
) -> dict[str, torch.Tensor]:
    """Advance terrain after surviving and tracking sufficient commanded walking.

    Accumulated path length allows changing direction without losing progress. Mostly stationary episodes
    hold their level; failures demote.

    Args:
        env: The environment.
        env_ids: Environments about to reset.
        min_distance: Minimum commanded and actual walking distance for promotion [m].
        min_walking_time: Minimum translation-command duration [s], also at least 25% of the episode.
        promote_error_ratio: Maximum integrated XY error divided by commanded distance for promotion.
        demote_error_ratio: Integrated XY error divided by commanded distance above which to demote.
        promote_yaw_error: Maximum episode-mean yaw-rate error for promotion [rad/s].
        demote_yaw_error: Episode-mean yaw-rate error above which to demote [rad/s].
    """
    command: MicroDuckRoughVelocityCommand = env.command_manager.get_term("base_velocity")
    command.record_terrain_progress(env_ids)
    progress = {name: value[env_ids] for name, value in command.terrain_progress.items()}
    error_ratio = progress["error_distance"] / progress["commanded_distance"].clamp_min(1e-6)
    yaw_error = progress["yaw_error"] / progress["time"].clamp_min(env.step_dt)
    eligible = (
        (progress["commanded_distance"] >= min_distance)
        & (progress["walking_time"] >= min_walking_time)
        & (progress["walking_time"] >= 0.25 * progress["time"])
    )
    failed = env.termination_manager.terminated[env_ids]
    move_up = (
        eligible
        & ~failed
        & env.termination_manager.time_outs[env_ids]
        & (progress["walked_distance"] >= min_distance)
        & (error_ratio <= promote_error_ratio)
        & (yaw_error <= promote_yaw_error)
    )
    move_down = (
        (progress["time"] > 0.0)
        & (failed | (eligible & ((error_ratio > demote_error_ratio) | (yaw_error > demote_yaw_error))))
        & ~move_up
    )
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)
    return {
        "mean": terrain.terrain_levels.float().mean(),
        "promoted": move_up.float().mean(),
        "demoted": move_down.float().mean(),
        "eligible": eligible.float().mean(),
        "walked_distance": progress["walked_distance"].nan_to_num().mean(),
    }
