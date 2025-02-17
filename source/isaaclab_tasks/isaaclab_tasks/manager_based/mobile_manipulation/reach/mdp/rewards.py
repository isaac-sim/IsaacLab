# Copyright (c) 2022-2025, Elevate Robotics
# All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize end-effector position error using L2-norm.

    Computes position error between desired position (from command) and current
    end-effector position in world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w[:, :3], asset.data.root_quat_w, des_pos_b
    )
    body_id = asset_cfg.body_ids[0]
    curr_pos_w = asset.data.body_state_w[:, body_id, :3]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward end-effector position tracking using tanh kernel.

    Maps position error through tanh for smoother gradients and bounded rewards.
    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w[:, :3], asset.data.root_quat_w, des_pos_b
    )
    body_id = asset_cfg.body_ids[0]
    curr_pos_w = asset.data.body_state_w[:, body_id, :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def base_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize base velocity using L2-norm.

    Encourages smooth base movement.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute velocity magnitude (linear and angular)
    lin_vel = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    ang_vel = torch.abs(asset.data.root_ang_vel_w[:, 2])
    return lin_vel + 0.5 * ang_vel


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocity using L2-norm.

    Encourages smooth joint movement.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.norm(asset.data.joint_vel, dim=1)
    return joint_vel


def arm_manipulability(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining good arm manipulability.

    Encourages keeping the arm in a good configuration away from singularities
    by penalizing joint limits and extreme configurations.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Get joint positions
    joint_pos = asset.data.joint_pos
    joint_limits = asset.data.soft_joint_pos_limits
    # Compute distance from center of joint range
    joint_range = joint_limits[..., 1] - joint_limits[..., 0]
    joint_center = (joint_limits[..., 1] + joint_limits[..., 0]) / 2
    joint_dist = torch.abs(joint_pos - joint_center) / (joint_range / 2)
    # Penalize being close to limits
    return -torch.mean(torch.square(joint_dist), dim=1)


def base_heading_to_target(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Reward for base heading aligned with target direction.

    Encourages the base to face the target location for better manipulation.
    """
    # extract the used quantities (to enable type-hinting)
    command = env.command_manager.get_command(command_name)

    # Get target position relative to base
    target_pos_b = command[:, :2]  # XY only
    target_dist = torch.norm(target_pos_b, dim=1)
    target_dir = target_pos_b / (target_dist.unsqueeze(-1) + 1e-6)

    # Get base forward vector
    base_forward = torch.tensor([1.0, 0.0], device=env.device).expand(env.num_envs, 2)

    # Compute alignment
    alignment = torch.sum(target_dir * base_forward, dim=1)

    # Only reward alignment when target is far enough
    return torch.where(target_dist > 0.5, alignment, torch.zeros_like(alignment))


def base_position_error(
    env: ManagerBasedRLEnv,
    body_name: str = "base_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize tracking of the base position error using L2-norm.

    The function computes the position error between a fixed desired position (1,0,0) and the
    current position of the asset's body (in world frame), accounting for environment tiling.
    """
    # Get the asset and current position
    asset: RigidObject = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    curr_pos_w = asset.data.body_state_w[:, body_idx, :3]  # type: ignore

    # Get environment origins from the scene
    env_origins = env.scene.env_origins

    # Subtract environment origins to get local positions
    local_pos_w = curr_pos_w - env_origins

    # Define desired position in local coordinates
    des_pos_w = torch.tensor([1.0, 0.0, 0.0], device=curr_pos_w.device).expand(
        curr_pos_w.shape[0], -1
    )
    xy_distance = torch.norm(local_pos_w[:, :2] - des_pos_w[:, :2], dim=1)
    return xy_distance


def reached_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
) -> torch.Tensor:
    """Check if end-effector has reached the target position."""
    # Get position error
    error = base_position_error(env)
    return error < threshold
