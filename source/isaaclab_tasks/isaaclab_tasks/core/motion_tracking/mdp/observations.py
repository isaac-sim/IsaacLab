# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from HybridRobotics/whole_body_tracking (MIT).
# See docs/licenses/dependencies/motion-tracking-license.txt.

"""Reference and privileged observations in the robot anchor frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Position of the robot bodies in the anchor frame. Shape is (num_envs, num_bodies * 3)."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Orientation of the robot bodies in the anchor frame. Shape is (num_envs, num_bodies * 6).

    The returned orientation is in the 6-DoF representation of SO(3).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Desired position of the robot anchor in the current anchor frame. Shape is (num_envs, 3)."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Desired orientation of the robot anchor in the current anchor frame. Shape is (num_envs, 6).

    The returned orientation is in the 6-DoF representation of SO(3).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Noise-free root link pose and velocity and joint state of the robot.

    Order: root position relative to the environment origin, root XYZW quaternion, root linear and angular velocity
    in the world frame, joint positions and joint velocities.
    """
    data = env.scene["robot"].data
    return torch.cat(
        (
            data.root_link_pos_w.torch - env.scene.env_origins,
            data.root_link_quat_w.torch,
            data.root_link_lin_vel_w.torch,
            data.root_link_ang_vel_w.torch,
            data.joint_pos.torch,
            data.joint_vel.torch,
        ),
        dim=-1,
    )


def motion_reference_action(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the current reference pose in the action manager's joint order and units."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    action = env.action_manager.get_term(env.action_manager.active_terms[0])
    return (command.joint_pos[..., action._joint_ids] - action._offset) / action._scale


def motion_frame(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Index of the current reference frame. Shape is (num_envs, 1)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.time_steps[:, None].float()


def self_collision_events(env: ManagerBasedEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Self-collision indicator per physics frame of the last control step, as counted by ``self_collision_cost``."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.force_matrix_w_history.torch[:, :, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1).amax(dim=(-1, -2)) > threshold).float()


def body_mass(env: ManagerBasedEnv) -> torch.Tensor:
    """Privileged body masses of the robot [kg]."""
    return env.scene["robot"].data.body_mass.torch
