# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_multitask_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    Supports heterogeneous scenes where per-task assets only exist in a
    subset of environments.  Uses the centralized :class:`EnvLayout` for
    global-to-local env-id mapping rather than per-asset filter methods.

    If :attr:`reset_joint_targets` is True, the joint position and velocity
    targets of the articulations are also reset to their default values.
    """
    layout = env.scene.layout
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        key = rigid_object._layout_key
        local_ids, global_ids = layout.filter_and_split(key, env_ids) if key else (env_ids, env_ids)
        if local_ids.numel() == 0:
            continue
        default_pose = wp.to_torch(rigid_object.data.default_root_pose)[local_ids].clone()
        default_vel = wp.to_torch(rigid_object.data.default_root_vel)[local_ids].clone()
        default_pose[:, :3] += env.scene.env_origins[global_ids]
        rigid_object.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=local_ids)
        rigid_object.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=local_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        key = articulation_asset._layout_key
        local_ids, global_ids = layout.filter_and_split(key, env_ids) if key else (env_ids, env_ids)
        if local_ids.numel() == 0:
            continue
        default_pose = wp.to_torch(articulation_asset.data.default_root_pose)[local_ids].clone()
        default_vel = wp.to_torch(articulation_asset.data.default_root_vel)[local_ids].clone()
        default_pose[:, :3] += env.scene.env_origins[global_ids]
        articulation_asset.write_root_pose_to_sim_index(root_pose=default_pose, env_ids=local_ids)
        articulation_asset.write_root_velocity_to_sim_index(root_velocity=default_vel, env_ids=local_ids)
        default_joint_pos = wp.to_torch(articulation_asset.data.default_joint_pos)[local_ids].clone()
        default_joint_vel = wp.to_torch(articulation_asset.data.default_joint_vel)[local_ids].clone()
        articulation_asset.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=local_ids)
        articulation_asset.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=local_ids)
        if reset_joint_targets:
            articulation_asset.set_joint_position_target_index(target=default_joint_pos, env_ids=local_ids)
            articulation_asset.set_joint_velocity_target_index(target=default_joint_vel, env_ids=local_ids)
