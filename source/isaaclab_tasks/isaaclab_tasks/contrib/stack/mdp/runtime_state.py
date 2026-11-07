# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed runtime state shared by the Franka stack reset MDP terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class StackResetRuntimeState:
    """Episode-local reset metadata shared across stack MDP managers."""

    row_ids: torch.Tensor
    recipes: torch.Tensor
    target_potentials: torch.Tensor
    held_cube_ids: torch.Tensor
    grasp_pair_ids: torch.Tensor
    role_to_cube: torch.Tensor
    initialized: torch.Tensor


def create_stack_reset_runtime_state(
    env: ManagerBasedRLEnv,
) -> StackResetRuntimeState:
    """Create and attach one typed reset-state owner to an environment."""
    state = StackResetRuntimeState(
        row_ids=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        recipes=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        target_potentials=torch.ones(env.num_envs, dtype=torch.float32, device=env.device),
        held_cube_ids=torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        grasp_pair_ids=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        role_to_cube=torch.arange(3, dtype=torch.long, device=env.device).repeat(env.num_envs, 1),
        initialized=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    env.stack_reset_state = state
    return state


def get_stack_reset_runtime_state(env: ManagerBasedRLEnv) -> StackResetRuntimeState:
    """Return the reset state created by :class:`StackResetStateTable`."""
    state = getattr(env, "stack_reset_state", None)
    if state is None:
        raise AttributeError("Stack reset runtime state is unavailable before the reset-table event is initialized.")
    return state
