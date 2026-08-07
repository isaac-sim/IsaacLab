# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed runtime state shared by conveyor-transfer MDP terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class ConveyorTransferState:
    """Episode-local transfer command and reset metadata."""

    row_ids: torch.Tensor
    recipe_ids: torch.Tensor
    target_cube_ids: torch.Tensor
    source_side_ids: torch.Tensor
    held_cube_ids: torch.Tensor
    initialized: torch.Tensor


def create_transfer_state(env: ManagerBasedRLEnv, row_count: int) -> ConveyorTransferState:
    """Create and attach the environment's transfer-state owner."""
    state = ConveyorTransferState(
        row_ids=torch.randint(row_count, (env.num_envs,), dtype=torch.long, device=env.device),
        recipe_ids=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        target_cube_ids=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        source_side_ids=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        held_cube_ids=torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        initialized=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    env.conveyor_transfer_state = state
    return state
