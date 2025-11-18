# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab_tasks.manager_based.box_pushing.box_pushing_env import BoxPushingEnv


# TODO resolve import
def action_scaled_l2(env: BoxPushingEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action * env.cfg.actions.body_joint_effort.scale),
        dim=1,
    )


def object_ee_distance(
    env: BoxPushingEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    return torch.clamp(object_ee_distance, min=0.05, max=100)


def object_goal_position_distance(
    env: BoxPushingEnv,
    command_name: str,
    end_ep: bool = False,
    end_ep_weight: float = 100.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3], command[:, 3:]
    )
    # distance of the object to the target: (num_envs,)
    distance = torch.linalg.norm(object.data.root_pos_w - des_pos_w, dim=1)

    #  If there is a different weighting only to be computed at the end of an episode
    if end_ep:
        #  compute only for terminated envs
        terminated = env.termination_manager.dones
        distance = torch.where(terminated, distance, 0.0) * end_ep_weight
    return distance


def object_goal_orientation_distance(
    env: BoxPushingEnv,
    command_name: str,
    end_ep: bool = False,
    end_ep_weight: float = 100.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired orientation in the world frame
    _, des_or_w = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3], command[:, 3:]
    )
    # compute orientation distance
    rot_dist = quat_error_magnitude(des_or_w, object.data.root_quat_w)

    #  If there is a different weighting only to be computed at the end of an episode
    if end_ep:
        #  compute only for terminated envs
        terminated = env.termination_manager.dones
        rot_dist = torch.where(terminated, rot_dist, 0.0) * end_ep_weight
    return rot_dist


# TODO somehow asset_cfg.joint_ids is None so has to be replaced with :
def joint_pos_limits_bp(env: BoxPushingEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(asset.data.joint_pos[:, :] - asset.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (asset.data.joint_pos[:, :] - asset.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def end_ep_vel(
    env: BoxPushingEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    #  retrieving velocity
    asset: Articulation = env.scene[asset_cfg.name]
    vel = torch.abs(asset.data.joint_vel[:, :7])

    reward = torch.linalg.norm(vel, dim=1)

    #  compute only for terminated envs
    terminated = env.termination_manager.dones
    reward = torch.where(terminated, reward, 0.0)

    return reward


def joint_vel_limits_bp(
    env: BoxPushingEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # max joint velocities
    arm_dof_vel_max = convert_to_torch([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], device=env.device)
    # compute out of limits constraints
    out_of_limits = torch.abs(asset.data.joint_vel[:, :7]) - arm_dof_vel_max

    mask = out_of_limits > 0
    out_of_limits = torch.where(mask, out_of_limits, 0)

    return soft_ratio * torch.sum(out_of_limits, dim=1)


def rod_inclined_angle(
    env: BoxPushingEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    desired_rod_quat = convert_to_torch([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    rot_dist = quat_error_magnitude(desired_rod_quat, ee_quat)
    return torch.where(rot_dist > torch.pi / 4.0, rot_dist / torch.pi, rot_dist * 0)
