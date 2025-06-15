# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, axis_angle_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position and quaternion of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w
    object_quat_w = object.data.root_quat_w
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w, object_quat_w
    )
    return torch.cat((object_pos_b, object_quat_b), dim=1)


def object_idx(env: ManagerBasedRLEnv) -> torch.Tensor:
    num_unique_objects = len(env.scene.cfg.object.spawn.assets_cfg)
    multi_object_idx = torch.remainder(torch.arange(env.num_envs), num_unique_objects).to(env.device)
    multi_object_idx_onehot = F.one_hot(multi_object_idx, num_classes=num_unique_objects).float()
    return multi_object_idx_onehot


def projected_joint_force(env: ManagerBasedRLEnv, asset_cfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_dof_projected_joint_forces()[:, asset_cfg.joint_ids]

def body_state_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pose = asset.data.body_state_w[:, asset_cfg.body_ids].clone()
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)
    return pose.reshape(env.num_envs, -1)

def all_ones(env: ManagerBasedRLEnv):
    return torch.ones((env.num_envs, 1), device=env.device)