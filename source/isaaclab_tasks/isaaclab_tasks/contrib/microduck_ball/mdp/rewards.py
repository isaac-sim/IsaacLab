# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the MicroDuck on a ball."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from .observations import ball_pos_b, ball_vel_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the base for staying upright: ``exp(-|horizontal gravity in the base frame|^2 / std^2)``."""
    gravity_xy = env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, :2]
    return torch.exp(-torch.sum(torch.square(gravity_xy), dim=1) / std**2)


def ball_under_feet(
    env: ManagerBasedRLEnv, std: float, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward the ball for staying under the middle of the feet: ``exp(-distance^2 / std^2)``."""
    distance_sq = torch.sum(torch.square(ball_pos_b(env, robot_cfg, ball_cfg)), dim=1)
    return torch.exp(-distance_sq / std**2)


def ball_slip_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared horizontal speed of the ball relative to the robot base [m^2/s^2]."""
    return torch.sum(torch.square(ball_vel_b(env, robot_cfg, ball_cfg)), dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical power of the joints, ``sum(|torque * velocity|)`` [W]."""
    asset = env.scene[asset_cfg.name]
    torque = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def feet_slip(
    env: ManagerBasedRLEnv, contact_height: float, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Sum of the speeds at which the feet slide over the ball [m/s].

    A foot slides when it is closer to the ball surface than ``contact_height`` [m]. Its slip speed is its velocity
    relative to the ball surface point below it, without the part pointing into the ball.
    """
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    offset = robot.data.body_pos_w.torch[:, robot_cfg.body_ids] - ball.data.root_pos_w.torch.unsqueeze(1)
    normal = offset / torch.linalg.norm(offset, dim=-1, keepdim=True)
    surface_velocity = ball.data.root_lin_vel_w.torch.unsqueeze(1) + torch.cross(
        ball.data.root_ang_vel_w.torch.unsqueeze(1).expand_as(offset), offset, dim=-1
    )
    relative = robot.data.body_lin_vel_w.torch[:, robot_cfg.body_ids] - surface_velocity
    tangential = relative - torch.sum(relative * normal, dim=-1, keepdim=True) * normal
    on_ball = _foot_height_on_ball(env, robot_cfg, ball_cfg) < contact_height
    return torch.sum(torch.linalg.norm(tangential, dim=-1) * on_ball, dim=1)


def gait_clock_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    clearance: float,
    rest_height: float,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the feet for following the stepping clock of the velocity command.

    While the command asks for a motion, each foot follows a half sine of height ``clearance`` [m] over its resting
    height ``rest_height`` [m] during the first half of its cycle and rests on the ball during the second half. With no
    motion commanded both feet rest. The reward is ``exp(-sum of squared height errors / std^2)``.
    """
    command = env.command_manager.get_term(command_name)
    swing = clearance * torch.clamp(torch.sin(2 * torch.pi * command.foot_phase), min=0.0)
    swing = swing * command.is_moving.unsqueeze(-1)
    height = _foot_height_on_ball(env, robot_cfg, ball_cfg) - rest_height
    return torch.exp(-torch.sum(torch.square(height - swing), dim=1) / std**2)


def idle_feet_lift(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Height difference between the feet while the command asks for no motion [m]."""
    height = _foot_height_on_ball(env, robot_cfg, ball_cfg)
    idle = ~env.command_manager.get_term(command_name).is_moving
    return torch.abs(height[:, 0] - height[:, 1]) * idle


def lean_into_motion(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    lean_per_speed: float,
    lean_per_error: float,
    max_lean: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward leaning the base towards where the ball has to travel or accelerate.

    The ball under the robot behaves like a wheeled inverted pendulum, which accelerates where its mass leans. The
    wanted lean is ``lean_per_speed * target + lean_per_error * (target - velocity)`` [rad], limited to ``max_lean``,
    with the target and the velocity in the robot frame. It is compared with the horizontal gravity in the base frame
    and the reward is ``exp(-|difference|^2 / std^2)``.
    """
    asset = env.scene[asset_cfg.name]
    target = env.command_manager.get_term(command_name).target_velocity_b
    error = target - asset.data.root_lin_vel_b.torch[:, :2]
    wanted = torch.clamp(lean_per_speed * target + lean_per_error * error, -max_lean, max_lean)
    lean = asset.data.projected_gravity_b.torch[:, :2]
    return torch.exp(-torch.sum(torch.square(lean - wanted), dim=1) / std**2)


def heading_error_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the heading of the robot and the reference heading of the velocity command [rad]."""
    reference = env.command_manager.get_term(command_name).reference_heading
    return torch.abs(wrap_to_pi(env.scene[asset_cfg.name].data.heading_w.torch - reference))


def track_lin_vel_xy_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the world velocity of the base for matching the target velocity: ``exp(-error^2 / std^2)``."""
    target = env.command_manager.get_term(command_name).target_velocity_w
    error = torch.sum(torch.square(target - env.scene[asset_cfg.name].data.root_lin_vel_w.torch[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_lin_vel_xy_linear(
    env: ManagerBasedRLEnv, command_name: str, scale: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the world velocity of the base for matching the target velocity: ``max(0, 1 - error / scale)``.

    Unlike the exponential term it keeps a gradient far from the target. ``scale`` is the error [m/s] at which the
    reward reaches zero.
    """
    target = env.command_manager.get_term(command_name).target_velocity_w
    error = torch.linalg.norm(target - env.scene[asset_cfg.name].data.root_lin_vel_w.torch[:, :2], dim=1)
    return torch.clamp(1.0 - error / scale, min=0.0)


def track_ang_vel_z_linear(
    env: ManagerBasedRLEnv, command_name: str, scale: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the yaw rate of the base for matching the command: ``max(0, 1 - error / scale)``.

    ``scale`` is the error [rad/s] at which the reward reaches zero.
    """
    command = env.command_manager.get_command(command_name)
    error = torch.abs(command[:, 2] - env.scene[asset_cfg.name].data.root_ang_vel_b.torch[:, 2])
    return torch.clamp(1.0 - error / scale, min=0.0)


def _foot_height_on_ball(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance of each foot (its body frame origin) from the ball surface [m], shape (num_envs, number of feet)."""
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    offset = robot.data.body_pos_w.torch[:, robot_cfg.body_ids] - ball.data.root_pos_w.torch.unsqueeze(1)
    return torch.linalg.norm(offset, dim=-1) - ball.cfg.spawn.radius
