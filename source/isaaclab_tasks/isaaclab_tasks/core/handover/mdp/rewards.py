# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the manager-based handover task."""

from __future__ import annotations

import torch


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
