# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-5000, 5000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-5000, 5000)


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance_reached: torch.Tensor = getattr(context_term, "z_distance_reached")
    return torch.where(orientation_aligned & position_centered & z_distance_reached, 1.0, 0.0)
