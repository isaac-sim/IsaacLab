# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the manager-based handover task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def handover_reward(goal_distance: torch.Tensor, distance_scale: float) -> torch.Tensor:
    """Return one hand's Direct reward for the current object-goal distance."""
    return 2.0 * torch.exp(-distance_scale * goal_distance)


@torch.jit.script
def evaluate_handover_success(
    object_position: torch.Tensor, target_position: torch.Tensor, success_distance_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate handover success while exposing its physical error.

    Args:
        object_position: Object positions [m].
        target_position: Goal positions [m].
        success_distance_threshold: Exclusive successful goal-distance threshold [m].

    Returns:
        Per-environment success flags and object-to-goal distances [m].
    """
    goal_distance = torch.linalg.norm(object_position - target_position, ord=2, dim=-1)
    return goal_distance < success_distance_threshold, goal_distance


def handover_goal_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward both hands for holding the object near its goal.

    The Direct environment sums one identical reward per hand, so this returns twice the
    single-hand value. The command term owns the episode success bookkeeping behind
    ``Metrics/success_rate``.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        distance_scale: Exponential decay rate of the distance reward [1/m].
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
    goal_pos = env.command_manager.get_command(command_name)[:, :3]
    goal_distance = torch.linalg.norm(object_pos - goal_pos, ord=2, dim=-1)
    return 2.0 * handover_reward(goal_distance, distance_scale)
