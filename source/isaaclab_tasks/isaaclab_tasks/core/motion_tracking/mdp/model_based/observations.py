# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native state extraction and physical objectives for the reference algorithm."""

import torch

from isaaclab_rl.model_based.kinematics import from_native, matrix_from_quat, quat_apply_inverse

from isaaclab_tasks.core.motion_tracking.mdp.model_based.rewards import tracking_reward


def full_state(env) -> torch.Tensor:
    command = env.command_manager.get_term("motion")
    data = command.robot.data
    pose, twist = data.root_link_pose_w.torch, data.root_link_vel_w.torch
    return torch.cat(
        (
            pose[:, :3] - env.scene.env_origins,
            from_native(pose[:, 3:]),
            data.joint_pos.torch,
            twist,
            data.joint_vel.torch,
        ),
        -1,
    )


def body_state(env) -> tuple[torch.Tensor, ...]:
    command = env.command_manager.get_term("motion")
    data, ids = command.robot.data, command.body_indexes
    pose, twist = data.body_link_pose_w.torch[:, ids], data.body_link_vel_w.torch[:, ids]
    return (pose[..., :3] - env.scene.env_origins[:, None], from_native(pose[..., 3:]), twist[..., :3], twist[..., 3:])


def maximal_state(env) -> torch.Tensor:
    return torch.cat([value.flatten(1) for value in body_state(env)], -1)


def actor_sensor(env) -> torch.Tensor:
    command = env.command_manager.get_term("motion")
    data = command.robot.data
    quat = from_native(data.root_link_pose_w.torch[:, 3:])
    gravity = torch.zeros(env.num_envs, 3, device=env.device)
    gravity[:, 2] = -1
    return torch.cat(
        (
            quat_apply_inverse(quat, gravity),
            quat_apply_inverse(quat, data.root_link_vel_w.torch[:, 3:]),
            data.joint_pos.torch - data.default_joint_pos.torch + command.encoder_bias,
            data.joint_vel.torch,
            env.action_manager.action,
        ),
        -1,
    )


def reward(env) -> torch.Tensor:
    """Use the same reward function on simulated and imagined post-step states."""
    command = env.command_manager.get_term("motion")
    pos, quat, vel, ang_vel = body_state(env)
    data = command.robot.data
    return tracking_reward(
        torch.cat((pos[:, 0], quat[:, 0]), -1),
        pos,
        matrix_from_quat(quat),
        vel,
        ang_vel,
        torch.cat((data.joint_pos.torch, data.joint_vel.torch), -1),
        env.action_manager.action,
        env.action_manager.prev_action,
        *command.motion.body_state(command.time_steps),
        data.soft_joint_pos_limits.torch[0],
        1.0,
    )


def anchor_height_failure(env, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_term("motion")
    height = command.robot.data.root_link_pose_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (height - command.motion.root_pos_w[command.time_steps, 2]).abs() > threshold


def anchor_orientation_failure(env, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_term("motion")
    quat = from_native(command.robot.data.root_link_pose_w.torch[:, 3:])
    target = command.motion.root_quat_w[command.time_steps]
    return (2 * (quat[:, 1:3].square().sum(-1) - target[:, 1:3].square().sum(-1))).abs() > threshold


def motion_end(env) -> torch.Tensor:
    command = env.command_manager.get_term("motion")
    clip = torch.searchsorted(command.motion.motion_ends, command.time_steps, right=True)
    return command.time_steps >= command.motion.motion_ends[clip] - 1
