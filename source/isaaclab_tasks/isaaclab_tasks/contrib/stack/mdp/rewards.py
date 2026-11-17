# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for ordered cube stacking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .terminations import _cube_pair_is_aligned

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stack_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward reaching, lifting, and placing the next cube in the stack.

    Compare the world-frame positions of ``cube_1``, ``cube_2``, ``cube_3``, and ``ee_frame``.
    The ``success`` termination term supplies the target height and alignment tolerances.

    Args:
        env: The Franka cube stacking environment.

    Returns:
        One progress reward per environment, with shape ``(num_envs,)``.
    """
    bottom = env.scene["cube_1"].data.root_pos_w.torch
    middle = env.scene["cube_2"].data.root_pos_w.torch
    top = env.scene["cube_3"].data.root_pos_w.torch
    ee = env.scene["ee_frame"].data.target_pos_w.torch[:, 0]

    # The termination manager resolves omitted defaults when it prepares the success term.
    success_params = env.termination_manager.get_term_cfg("success").params
    height = success_params["height_diff"]
    # Use alignment alone so closing the gripper for the next pick keeps the new goal.
    middle_placed = _cube_pair_is_aligned(
        middle - bottom,
        success_params["xy_threshold"],
        height,
        success_params["height_threshold"],
        success_params["symmetric_height_tolerance"],
    )
    cube = torch.where(middle_placed[:, None], top, middle)
    support = torch.where(middle_placed[:, None], middle, bottom)
    target_error = cube - support
    target_error[:, 2] -= height

    reach = 1.0 - torch.tanh(torch.linalg.vector_norm(cube - ee, dim=1) / 0.1)
    # Measure lift relative to the bottom cube's center.
    lift = ((cube[:, 2] - bottom[:, 2]) / height).clamp(0.0, 1.0)
    place = 1.0 - torch.tanh(torch.linalg.vector_norm(target_error, dim=1) / 0.1)
    return middle_placed.float() + 0.25 * ((1.0 - lift) * reach + lift) + 0.75 * lift * place
