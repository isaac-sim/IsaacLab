# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-conditioned observations for conveyor transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .kinematics import end_effector_pose, tool_velocity
from .reset_events import CUBE_COUNT, TRANSFER_X, side_inner_y

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _transfer_command(env: ManagerBasedRLEnv, command_name: str = "transfer"):
    """Return the configured transfer command."""
    return env.command_manager.get_term(command_name)


def _cube_assets(env: ManagerBasedRLEnv) -> tuple[RigidObject, ...]:
    """Return the four cubes in stable identity order."""
    return tuple(env.scene[f"cube_{cube_id}"] for cube_id in range(CUBE_COUNT))


def _cube_state(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack cube world positions, orientations, and spatial velocities."""
    cubes = _cube_assets(env)
    return (
        torch.stack(tuple(cube.data.root_pos_w.torch for cube in cubes), dim=1),
        torch.stack(tuple(cube.data.root_quat_w.torch for cube in cubes), dim=1),
        torch.stack(tuple(cube.data.root_vel_w.torch for cube in cubes), dim=1),
    )


def _active_cube_values(values: torch.Tensor, target_cube_ids: torch.Tensor) -> torch.Tensor:
    """Gather one cube row for every vectorized environment."""
    shape = (values.shape[0], 1, *values.shape[2:])
    index = target_cube_ids.view(values.shape[0], 1, *([1] * (values.ndim - 2))).expand(shape)
    return torch.gather(values, 1, index).squeeze(1)


def target_cube_one_hot(env: ManagerBasedRLEnv, command_name: str = "transfer") -> torch.Tensor:
    """Encode which numbered cube the policy must transfer."""
    command = _transfer_command(env, command_name)
    return torch.nn.functional.one_hot(command.target_cube_ids.long(), num_classes=CUBE_COUNT).float()


def target_side_one_hot(env: ManagerBasedRLEnv, command_name: str = "transfer") -> torch.Tensor:
    """Encode the destination conveyor, opposite the reset source side."""
    command = _transfer_command(env, command_name)
    return torch.nn.functional.one_hot(1 - command.source_side_ids.long(), num_classes=2).float()


def classify_cube_conveyors(local_positions: torch.Tensor, transit_half_width: float = 0.14) -> torch.Tensor:
    """Classify positions as left conveyor, in transit, or right conveyor."""
    if local_positions.shape[-1] != 3:
        raise ValueError("Cube positions must end in xyz coordinates.")
    side_ids = torch.full(local_positions.shape[:-1], 1, dtype=torch.long, device=local_positions.device)
    side_ids[local_positions[..., 1] > transit_half_width] = 0
    side_ids[local_positions[..., 1] < -transit_half_width] = 2
    return torch.nn.functional.one_hot(side_ids, num_classes=3).float()


def cube_conveyor_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return left/transit/right one-hot state for every numbered cube."""
    positions, _, _ = _cube_state(env)
    local_positions = positions - env.scene.env_origins.unsqueeze(1)
    return classify_cube_conveyors(local_positions).flatten(start_dim=1)


def transfer_object_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Describe all four cubes in stable identity slots.

    The observation contains local positions, tool-relative positions, local
    up axes, and linear/angular velocities. Cube identity does not change
    during an episode; :func:`target_cube_one_hot` selects the active slot.
    """
    positions, quaternions, velocities = _cube_state(env)
    local_positions = positions - env.scene.env_origins.unsqueeze(1)
    tool_position, _ = end_effector_pose(env)
    tool_relative = positions - tool_position.unsqueeze(1)
    rotations = math_utils.matrix_from_quat(quaternions.flatten(end_dim=1)).view(env.num_envs, CUBE_COUNT, 3, 3)
    up_axes = rotations[..., 2]
    return torch.cat(
        (
            local_positions.flatten(start_dim=1),
            tool_relative.flatten(start_dim=1),
            up_axes.flatten(start_dim=1),
            velocities.flatten(start_dim=1),
        ),
        dim=1,
    )


def active_transfer_features(env: ManagerBasedRLEnv, command_name: str = "transfer") -> torch.Tensor:
    """Return active-cube and destination-relative position features [m]."""
    command = _transfer_command(env, command_name)
    positions, _, _ = _cube_state(env)
    active_position = _active_cube_values(positions, command.target_cube_ids.long())
    local_active_position = active_position - env.scene.env_origins
    tool_position, _ = end_effector_pose(env)
    target_side_ids = 1 - command.source_side_ids.long()
    target_position = torch.stack(
        (
            torch.full_like(target_side_ids, TRANSFER_X, dtype=active_position.dtype),
            side_inner_y(target_side_ids),
            torch.full_like(target_side_ids, 0.06, dtype=active_position.dtype),
        ),
        dim=1,
    )
    return torch.cat(
        (
            local_active_position,
            active_position - tool_position,
            target_position - local_active_position,
        ),
        dim=1,
    )


def gripper_joint_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint[1-2]"]),
) -> torch.Tensor:
    """Return the two Franka finger positions [m]."""
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.joint_pos.torch[:, robot_cfg.joint_ids]


def end_effector_axes(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return continuous tool-frame x and z axes."""
    _, orientation = end_effector_pose(env)
    rotation = math_utils.matrix_from_quat(orientation)
    return torch.cat((rotation[:, :, 0], rotation[:, :, 2]), dim=1)


def end_effector_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return tool-center linear and angular velocity [m/s, rad/s]."""
    return tool_velocity(env)
