# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return distance < threshold


def object_reached_goal_with_stability(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    position_threshold: float = 0.02,
    velocity_threshold: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """NEW: Termination condition with velocity stability check.

    This ensures the object has not only reached the goal but is also stable (low velocity).

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        position_threshold: The position threshold for goal reach. Defaults to 0.02.
        velocity_threshold: The velocity threshold for stability. Defaults to 0.1 m/s.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
    """
    # extract the used quantities
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    
    # IMPROVEMENT 1: Check position distance
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    position_reached = distance < position_threshold
    
    # IMPROVEMENT 2: Check velocity for stability
    velocity = torch.norm(object.data.root_lin_vel_w, dim=1)
    is_stable = velocity < velocity_threshold
    
    # Both conditions must be met
    return position_reached & is_stable


def object_dropped(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """NEW: Termination if object is dropped (falls below minimal height).

    This helps identify failed grasps early.

    Args:
        env: The environment.
        minimal_height: The minimal height threshold. Defaults to 0.05.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
    """
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2] < minimal_height


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_distance: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """NEW: Termination if object moves too far from robot.

    Prevents unproductive exploration.

    Args:
        env: The environment.
        max_distance: Maximum allowed distance from robot. Defaults to 2.0 meters.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Calculate horizontal distance (ignore z-axis)
    robot_pos_xy = robot.data.root_pos_w[:, :2]
    object_pos_xy = object.data.root_pos_w[:, :2]
    distance = torch.norm(object_pos_xy - robot_pos_xy, dim=1)
    
    return distance > max_distance
