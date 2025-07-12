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
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse

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

def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].clone().view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].clone().view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1) 
    return out.view(env.num_envs, -1)


class object_scale(ManagerTermBase):

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        import isaacsim.core.utils.prims as prim_utils
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg('object'))
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.scale = torch.zeros((env.num_envs, 3), device=self.device)

        for i in range(env.num_envs):
            object_cfg = self.object.cfg
            prim_path = object_cfg.prim_path
            prim = prim_utils.get_prim_at_path(prim_path.replace(".*", str(i)))
            # prim_spec = Sdf.CreatePrimInLayer(stage_utils.get_current_stage().GetRootLayer(), )
            scale = prim.GetAttribute("xformOp:scale").Get()
            self.scale[i] = torch.tensor(scale, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
        return self.scale




def all_ones(env: ManagerBasedRLEnv):
    return torch.ones((env.num_envs, 1), device=env.device)