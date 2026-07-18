# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for state-based in-hand reorientation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.utils import random_xy_rotation, sample_joint_positions_within_limits

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def reset_reorient_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    position_noise: float,
    joint_position_noise: float,
    joint_velocity_noise: float,
    action_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the object and hand with the Direct task's distributions.

    Args:
        env: Environment containing the robot and object.
        env_ids: Environment indices to reset.
        position_noise: Object-position noise half-width [m].
        joint_position_noise: Scale applied to sampled joint-position deltas.
        joint_velocity_noise: Joint-velocity noise half-width [rad/s].
        action_name: Action term whose terminal raw action is retained in the reset observation.
        robot_cfg: Robot scene entity.
        object_cfg: Object scene entity.
    """
    raw_action = env.action_manager.get_term(action_name).raw_actions
    if not hasattr(env, "_reorient_reset_action"):
        env._reorient_reset_action = torch.zeros_like(raw_action)
        env._reorient_reset_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=raw_action.device)
    env._reorient_reset_action[env_ids] = raw_action[env_ids]
    env._reorient_reset_step[env_ids] = env.common_step_counter

    object_asset: Articulation | RigidObject = env.scene[object_cfg.name]
    object_pose = object_asset.data.default_root_pose.torch[env_ids].clone()
    object_velocity = torch.zeros_like(object_asset.data.default_root_vel.torch[env_ids])
    position_delta = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=env.device)
    object_pose[:, :3] += position_noise * position_delta + env.scene.env_origins[env_ids]
    object_pose[:, 3:7] = random_xy_rotation(len(env_ids), env.device)
    object_asset.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
    object_asset.write_root_velocity_to_sim_index(root_velocity=object_velocity, env_ids=env_ids)

    robot: Articulation = env.scene[robot_cfg.name]
    default_position = robot.data.default_joint_pos.torch[env_ids]
    limits = robot.data.joint_limits.torch[env_ids]
    joint_position = sample_joint_positions_within_limits(default_position, limits, joint_position_noise)
    velocity_sample = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), robot.num_joints), device=env.device)
    joint_velocity = robot.data.default_joint_vel.torch[env_ids] + joint_velocity_noise * velocity_sample
    robot.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
    robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
