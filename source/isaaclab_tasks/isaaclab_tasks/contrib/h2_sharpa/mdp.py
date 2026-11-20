# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations and phase rewards shared by the H2 tabletop tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_camera_response(
    image: torch.Tensor,
    response_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
    response_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Apply the fitted color response to a raw uint8 RGB observation."""
    output_dtype = image.dtype
    image = image.to(torch.float32)
    image = (image / 255.0).clamp(0.0, 1.0)
    image = image - 0.18 * image.pow(3)
    warm_balance = image.new_tensor((1.01, 0.99, 0.96))
    image = (image * warm_balance).clamp_(0.0, 1.0)

    # Match the real camera's sRGB response, white balance, and exposure.
    gamma = image.new_tensor(response_gamma)
    gain = image.new_tensor(response_gain)
    image = image.pow(gamma) * gain

    return (image * 255.0).clamp_(0.0, 255.0).to(output_dtype)


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
