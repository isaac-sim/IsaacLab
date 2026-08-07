# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the manager-based handover task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.reorient.mdp.events import random_xy_rotation, sample_joint_positions_within_limits

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def reset_handover_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    position_noise: float,
    joint_position_noise: float,
    joint_velocity_noise: float,
    action_names: tuple[str, ...],
    right_hand_cfg: SceneEntityCfg = SceneEntityCfg("right_hand"),
    left_hand_cfg: SceneEntityCfg = SceneEntityCfg("left_hand"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the object and both hands with the Direct task's distributions.

    Args:
        env: Environment containing both hands and the object.
        env_ids: Environment indices to reset.
        position_noise: Object-position noise half-width [m].
        joint_position_noise: Scale applied to sampled joint-position deltas.
        joint_velocity_noise: Joint-velocity noise half-width [rad/s].
        action_names: Action terms whose pre-reset raw actions are retained in reset observations.
        right_hand_cfg: Right-hand scene entity.
        left_hand_cfg: Left-hand scene entity.
        object_cfg: Object scene entity.
    """
    if not hasattr(env, "_handover_reset_actions"):
        env._handover_reset_actions = {}
    for action_name in action_names:
        raw_action = env.action_manager.get_term(action_name).raw_actions
        if action_name not in env._handover_reset_actions:
            env._handover_reset_actions[action_name] = torch.zeros_like(raw_action)
        env._handover_reset_actions[action_name][env_ids] = raw_action[env_ids]

    object_asset: RigidObject = env.scene[object_cfg.name]
    object_pose = object_asset.data.default_root_pose.torch[env_ids].clone()
    object_velocity = torch.zeros_like(object_asset.data.default_root_vel.torch[env_ids])
    position_delta = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=env.device)
    object_pose[:, :3] += position_noise * position_delta + env.scene.env_origins[env_ids]
    object_pose[:, 3:7] = random_xy_rotation(len(env_ids), env.device)
    object_asset.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
    object_asset.write_root_velocity_to_sim_index(root_velocity=object_velocity, env_ids=env_ids)

    for hand_cfg in (right_hand_cfg, left_hand_cfg):
        hand: Articulation = env.scene[hand_cfg.name]
        default_position = hand.data.default_joint_pos.torch[env_ids]
        limits = hand.data.joint_limits.torch[env_ids]
        joint_position = sample_joint_positions_within_limits(default_position, limits, joint_position_noise)
        velocity_sample = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), hand.num_joints), device=env.device)
        joint_velocity = hand.data.default_joint_vel.torch[env_ids] + joint_velocity_noise * velocity_sample

        hand.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
        hand.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        hand.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
