# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase transitions and rewards shared by the H2 tabletop tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def advance_phase(
    phase: torch.Tensor, hold_counter: torch.Tensor, current: int, condition: torch.Tensor, hold_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance after consecutive matching steps; other phases leave the counter unchanged."""
    hold_counter = torch.where(phase == current, torch.where(condition, hold_counter + 1, 0), hold_counter)
    advance = (phase == current) & (hold_counter >= hold_steps)
    return torch.where(advance, current + 1, phase), torch.where(advance, 0, hold_counter)


def phase_reward(env: ManagerBasedRLEnv, from_phase: int) -> torch.Tensor:
    """Reward one transition, measured by the task-success term before rewards are evaluated."""
    progress = env.termination_manager.get_term_cfg("task_success").func
    return ((progress.previous_phase == from_phase) & (progress.phase > from_phase)).float() / env.step_dt
