# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from pxr import UsdGeom

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData,RigidObject,Articulation
from isaaclab.managers import SceneEntityCfg

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils

from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

    return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)

def gripper_state(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    threshold = 0.075
    is_open = torch.norm((finger_joint_1 - finger_joint_2), dim=1) >= threshold
    
    return is_open.float()  


def waypoints(env: ManagerBasedRLEnv,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
              referance: str = "root") -> torch.Tensor:
    """
    waypoint_idx_gripperaction
    
    收集场景中所有的路点，并选择是否将其转换为相对于机器人底座坐标系
    夹爪动作：
        - 1.0: open
        - 0.0: close
        - -1.0: 不动
    Returns:
        A tensor of shape (N, 8), where each row = [pose(7), gripper_action]
    """
    def find_all_waypoint_paths(max_depth=6):
        all_paths = []
        for depth in range(1, max_depth + 1):
            if depth == 1:
                pattern = "/waypoint_.*"
            else:
                prefix = "/".join([".*?"] * (depth - 1))
                pattern = f"/{prefix}/waypoint_.*"
            matches = prim_utils.find_matching_prim_paths(pattern)
            all_paths.extend(matches)
        return list(set(all_paths))  # 去重    
    

    # 这一行可以手动刷新一下场景树
    stage = stage_utils.get_current_stage()
    # 获取所有的路点节点路径
    all_waypoint_paths = find_all_waypoint_paths(max_depth=10)
    
    waypoint_states = []
    
    asset: Articulation = env.scene[asset_cfg.name]
    # (1,3), (1,4)
    root_link_pos , root_link_quat = asset.data.root_state_w[..., :3],asset.data.root_state_w[..., 3:7]  

    waypoint_states = torch.empty(len(all_waypoint_paths), 8, device=env.device)

    for path in all_waypoint_paths:
        waypoint_name = path.split("/")[-1]  # 使用 path，而非 rigid_obj_name
        segments = waypoint_name.split("_")
        waypoint_idx = int(segments[1])  # 路点索引

        # 获取到路点的prim
        waypoint_prim = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        world_transform = waypoint_prim.ComputeLocalToWorldTransform(0)
        world_pos = list(world_transform.ExtractTranslation())
        world_quat = world_transform.ExtractRotationQuat()
        
        world_quat = [world_quat.GetReal(),*world_quat.GetImaginary()]
        # (7,)
        waypoint_pose = torch.tensor(world_pos + world_quat, dtype=torch.float32, device=env.device)
        
        # 变换到以root坐标系为参考系
        if referance == "root":
            # 1. 四元数变换（旋转变换到 root 坐标系）
            rel_quat = math_utils.quat_mul(
                math_utils.quat_inv(root_link_quat),
                waypoint_pose[None,3:7]
            )
            # 2. 平移向量变换（位置减去 root）
            rel_pos = waypoint_pose[:3] - root_link_pos

            # 拼接为新的 pose
            waypoint_pose = torch.cat([rel_pos, rel_quat], dim=-1).squeeze(0)
        

        # gripper flag: open=1.0, close=0.0, default=-1.0 不动
        if len(segments) > 2:
            gripper_flag = 1.0 if segments[-1] == "open" else 0.0
        else:
            gripper_flag = -1.0

        gripper_tensor = torch.tensor([gripper_flag], device=waypoint_pose.device)
        full_state = torch.cat([waypoint_pose, gripper_tensor], dim=-1)
        waypoint_states[waypoint_idx,:] = full_state


    return waypoint_states




