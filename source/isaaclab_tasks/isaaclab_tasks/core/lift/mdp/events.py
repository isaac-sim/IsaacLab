# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum reset events for rigid-object lift tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .curriculums import lift_difficulty_fraction

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def reset_franka_lift_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    closed_finger_position: float = 0.016,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset a Franka lift episode according to its adaptive difficulty.

    The reference joint configurations were solved with Newton's articulation
    Jacobians. The curriculum starts with a lifted cube already in the gripper,
    moves the grasp to the table, opens the gripper, and finally restores the
    default arm pose and full object-position randomization.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    difficulty = lift_difficulty_fraction(env, env_ids).unsqueeze(-1)

    high_arm = torch.tensor(
        [0.001342, -0.362166, -0.003198, -2.751293, -0.008638, 3.183486, 0.747609],
        device=env.device,
    )
    low_arm = torch.tensor(
        [0.008757, 0.465495, -0.007622, -2.340756, -0.011284, 2.768182, 0.738861],
        device=env.device,
    )
    high_object = torch.tensor([0.499620, 0.0, 0.349140], device=env.device)
    low_object = torch.tensor([0.499772, 0.0, 0.029296], device=env.device)
    table_object = torch.tensor([0.5, 0.0, 0.0174], device=env.device)

    move_to_table = _smoothstep((difficulty - 0.30) / 0.25)
    release_object = _smoothstep((difficulty - 0.55) / 0.20)
    restore_default = _smoothstep((difficulty - 0.75) / 0.25)

    joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    arm_ids = robot.find_joints([f"panda_joint{index}" for index in range(1, 8)])[0]
    finger_ids = robot.find_joints("panda_finger_joint.*")[0]
    grasp_arm = torch.lerp(high_arm, low_arm, move_to_table)
    joint_pos[:, arm_ids] = torch.lerp(grasp_arm, joint_pos[:, arm_ids], restore_default)
    closed_fingers = torch.full((len(env_ids), len(finger_ids)), closed_finger_position, device=env.device)
    joint_pos[:, finger_ids] = torch.lerp(closed_fingers, joint_pos[:, finger_ids], release_object)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)

    grasp_object = torch.lerp(high_object, low_object, move_to_table)
    object_pos_b = torch.lerp(grasp_object, table_object, release_object)
    random_xy = torch.empty((len(env_ids), 2), device=env.device)
    random_xy[:, 0].uniform_(-0.1, 0.1)
    random_xy[:, 1].uniform_(-0.25, 0.25)
    object_pos_b[:, :2] += restore_default * random_xy

    object_pose = object.data.default_root_pose.torch[env_ids].clone()
    object_pose[:, :3] = object_pos_b + env.scene.env_origins[env_ids]
    object_velocity = torch.zeros((len(env_ids), 6), device=env.device)
    object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
    object.write_root_velocity_to_sim_index(root_velocity=object_velocity, env_ids=env_ids)
