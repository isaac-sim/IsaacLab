# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def align_grasp_around_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Bonus for correct hand orientation around the object.

    The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Target object position: (num_envs, 3)
    object_pos = object.data.root_pos_w#[..., 0, :]

    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = ee_frame.data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < object_pos[:, 2]) & (lfinger_pos[:, 2] > object_pos[:, 2])

    # bonus if left finger is above the drawer handle and right below
    return is_graspable


def approach_gripper_object_to_grasp(
    env: ManagerBasedRLEnv, 
    offset: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the robot's gripper reaching the object with the right pose.

    This function returns the distance of fingertips to the object when the fingers are in a grasping orientation
    (i.e., fingers are around the cube/object). Otherwise, it returns zero.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Target object position: (num_envs, 3)
    object_pos = object.data.root_pos_w#[..., 0, :]

    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = ee_frame.data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Compute the distance of each finger from the handle (top and bottom, TODO: change this)
    lfinger_dist = torch.abs(lfinger_pos[:, 2] - object_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - object_pos[:, 2])

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < object_pos[:, 2]) & (lfinger_pos[:, 2] > object_pos[:, 2])
    is_approached_in_right_pose = is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))

    return is_approached_in_right_pose


def grasp_object(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    open_joint_pos: float, 
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the object.

    The :attr:`threshold` is the distance from the object at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_tcp_pos = ee_frame.data.target_pos_w[..., 0, :]
    object_pos = object.data.root_pos_w#[..., 0, :]
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold
    # NOTE: Maximize difference between the gripper pose and its open position.
    # The only way is to close it as much as possible.
    is_grasped = is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)

    return is_grasped

def orientation_command_error(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg
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

def position_command_error(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg
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
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w

    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

# NOTE: EXPERIMENTAL
def is_ee_stationary(
    env: ManagerBasedRLEnv,
    threshold: float, 
    prev_joint_vel: torch.Tensor, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize stationary ee joint velocities"""
    # NOTE: track the joint velocities of the end-effector and see if it stays stationary for 2
    # successive iterations
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]

    curr_joint_vel = robot.data.body_vel_w[:, -1, :]
    difference = torch.norm(curr_joint_vel - prev_joint_vel.to(curr_joint_vel.device), dim=-1, p=2) #L2
    is_stationary = difference <= threshold
    prev_joint_vel = curr_joint_vel.clone()

    return is_stationary
