# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def gripper_to_box_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    box_name: str,
    grasp_offset: list[float],
) -> torch.Tensor:
    """Compute distance between gripper and desired grasp point on box.

    Args:
        env: The environment instance
        robot_cfg: Configuration for the robot (specifies which gripper to use)
        box_name: Name of the box object
        grasp_offset: Offset from box center for desired grasp point [x,y,z]

    Returns:
        Distance between gripper and grasp point
    """
    # Get box pose
    box: RigidObject = env.scene[box_name]
    box_pos = box.data.root_state_w[:, :3]
    box_quat = box.data.root_state_w[:, 3:7]

    # Transform grasp offset to world frame
    grasp_offset_local = torch.tensor(grasp_offset, device=env.device).repeat(
        env.num_envs, 1
    )
    grasp_pos_w, _ = combine_frame_transforms(box_pos, box_quat, grasp_offset_local)

    # Get gripper position
    robot: RigidObject = env.scene[robot_cfg.name]
    gripper_pos = robot.data.body_state_w[:, robot_cfg.body_ids[0], :3]  # type: ignore

    # Compute distance
    return torch.norm(gripper_pos - grasp_pos_w, dim=1)


def box_height(
    env: ManagerBasedRLEnv, box_name: str, target_height: float
) -> torch.Tensor:
    """Reward for lifting the box to a target height.

    Args:
        env: The environment instance
        box_name: Name of the box object
        target_height: Desired height for the box

    Returns:
        Height-based reward scaled by how close box is to target height
    """
    box: RigidObject = env.scene[box_name]
    box_height = box.data.root_state_w[:, 2]  # z-coordinate
    height_error = torch.abs(box_height - target_height)
    return torch.exp(
        -5.0 * height_error
    )  # Exponential reward for getting close to target


def box_height_threshold(
    env: ManagerBasedRLEnv, box_name: str, min_height: float
) -> torch.Tensor:
    """Check if box has fallen below minimum height (for termination).

    Args:
        env: The environment instance
        box_name: Name of the box object
        min_height: Minimum allowed height

    Returns:
        Boolean tensor indicating if box is below minimum height
    """
    box: RigidObject = env.scene[box_name]
    box_height = box.data.root_state_w[:, 2]  # z-coordinate
    return box_height < min_height
