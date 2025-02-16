# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
