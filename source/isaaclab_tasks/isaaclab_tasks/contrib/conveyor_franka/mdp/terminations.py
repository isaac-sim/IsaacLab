# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actual episode termination and truncation terms for conveyor transfer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from ..conveyor_geometry import (
    BELT_CENTER_X,
    BELT_HALF_STRAIGHT,
    BELT_TURN_RADIUS,
    BELT_WIDTH,
    GUARD_THICKNESS,
)
from .reset_events import CUBE_COUNT, CUBE_SIZE

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


_TRACK_X_CLEARANCE = BELT_TURN_RADIUS + 0.5 * BELT_WIDTH + GUARD_THICKNESS + CUBE_SIZE


def subgoal_time_out(
    env: ManagerBasedRLEnv,
    timeout_s: float = 20.0,
    command_name: str = "transfer",
) -> torch.Tensor:
    """Truncate environments that make no transfer within one subgoal timeout [s]."""
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive.")
    command = env.command_manager.get_term(command_name)
    command.evaluate()
    timeout_steps = math.ceil(timeout_s / env.step_dt)
    elapsed = env.episode_length_buf - command.subgoal_start_steps
    return (elapsed >= timeout_steps) & ~command.pending_success


def transfer_sequence_time_out(
    env: ManagerBasedRLEnv,
    maximum_transfers: int = 8,
    command_name: str = "transfer",
) -> torch.Tensor:
    """Truncate completed sequences so reset-state coverage remains fresh."""
    if maximum_transfers < 1:
        raise ValueError("maximum_transfers must be positive.")
    command = env.command_manager.get_term(command_name)
    command.evaluate()
    return command.transfer_counts >= maximum_transfers


def cube_out_of_workspace(
    env: ManagerBasedRLEnv,
    minimum: tuple[float, float, float] = (
        BELT_CENTER_X - BELT_HALF_STRAIGHT - _TRACK_X_CLEARANCE,
        -1.05,
        -0.05,
    ),
    maximum: tuple[float, float, float] = (
        BELT_CENTER_X + BELT_HALF_STRAIGHT + _TRACK_X_CLEARANCE,
        1.05,
        0.80,
    ),
) -> torch.Tensor:
    """Terminate when any cube leaves the complete guarded racetrack workspace."""
    cubes: tuple[RigidObject, ...] = tuple(env.scene[f"cube_{cube_id}"] for cube_id in range(CUBE_COUNT))
    positions = torch.stack(tuple(cube.data.root_pos_w.torch for cube in cubes), dim=1)
    positions -= env.scene.env_origins.unsqueeze(1)
    lower = positions.new_tensor(minimum)
    upper = positions.new_tensor(maximum)
    return torch.any((positions < lower) | (positions > upper), dim=(1, 2))


def nonfinite_scene_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate environments containing nonfinite robot or cube state."""
    robot: Articulation = env.scene[robot_cfg.name]
    invalid = ~torch.all(torch.isfinite(robot.data.joint_pos.torch), dim=1)
    invalid |= ~torch.all(torch.isfinite(robot.data.joint_vel.torch), dim=1)
    for cube_id in range(CUBE_COUNT):
        cube: RigidObject = env.scene[f"cube_{cube_id}"]
        invalid |= ~torch.all(torch.isfinite(cube.data.root_state_w.torch), dim=1)
    return invalid
