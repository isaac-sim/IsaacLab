# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for ordered cube stacking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .terminations import cube_pair_is_aligned

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stack_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward reaching, lifting, and placing the next cube in the stack."""
    bottom = env.scene["cube_1"].data.root_pos_w.torch
    middle = env.scene["cube_2"].data.root_pos_w.torch
    top = env.scene["cube_3"].data.root_pos_w.torch
    ee = env.scene["ee_frame"].data.target_pos_w.torch[:, 0]

    # Switch goals on alignment alone so closing the gripper for the next pick keeps the new goal.
    success = env.termination_manager.get_term_cfg("success").params
    height = success["height_diff"]
    middle_placed = cube_pair_is_aligned(middle - bottom, success["xy_threshold"], height, success["height_threshold"])
    cube = torch.where(middle_placed[:, None], top, middle)
    support = torch.where(middle_placed[:, None], middle, bottom)
    target_error = cube - support
    target_error[:, 2] -= height

    reach = 1.0 - torch.tanh(torch.linalg.vector_norm(cube - ee, dim=1) / 0.1)
    # Measure lift above the bottom cube's center while it remains on the table.
    lift = ((cube[:, 2] - bottom[:, 2]) / height).clamp(0.0, 1.0)
    place = 1.0 - torch.tanh(torch.linalg.vector_norm(target_error, dim=1) / 0.1)
    return middle_placed.float() + 0.25 * ((1.0 - lift) * reach + lift) + 0.75 * lift * place
