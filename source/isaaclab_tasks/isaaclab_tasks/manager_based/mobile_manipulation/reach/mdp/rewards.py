# Copyright (c) 2022-2025, Elevate Robotics
# All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
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
        asset.data.root_pos_w[:, :3],
        asset.data.root_quat_w,
        des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
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
        asset.data.root_pos_w[:, :3],
        asset.data.root_quat_w,
        des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for base heading aligned with target direction.

    Encourages the base to face the target location for better manipulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
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


def reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Check if end-effector has reached the target position."""
    # Get position error
    error = position_command_error(env, command_name, asset_cfg)
    return error < threshold