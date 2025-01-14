# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedEnv
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import ArticulationData
from omni.isaac.lab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def rel_nut_bolt_distance(env: ManagerBasedRLEnv, bolt_part_name: str) -> torch.Tensor:
    nut_tf_data: FrameTransformerData = env.scene["nut_frame"].data
    bolt_tf_data: FrameTransformerData = env.scene["bolt_frame"].data
    bolt_id = bolt_tf_data.target_frame_names.index(bolt_part_name)
    dis = nut_tf_data.target_pos_w[..., 0, :] - bolt_tf_data.target_pos_w[..., bolt_id, :]
    return dis


def rel_nut_bolt_bottom_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return rel_nut_bolt_distance(env, "bolt_bottom")


def rel_nut_bolt_tip_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return rel_nut_bolt_distance(env, "bolt_tip")


def robot_tool_pose(env: ManagerBasedEnv):
    tool_w = env.unwrapped.scene["robot"].read_body_state_w("victor_left_tool0")[:, 0, :7]
    tool_w[:, :3] = tool_w[:, :3] - env.unwrapped.scene.env_origins
    return tool_w


# def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The distance between the end-effector and the object."""
#     ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
#     object_data: ArticulationData = env.scene["object"].data

#     return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


# def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The distance between the end-effector and the object."""
#     ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
#     cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

#     return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


# def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The position of the fingertips relative to the environment origins."""
#     ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
#     fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

#     return fingertips_pos.view(env.num_envs, -1)


# def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The position of the end-effector relative to the environment origins."""
#     ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
#     ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

#     return ee_pos


# def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
#     """The orientation of the end-effector in the environment frame.

#     If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
#     """
#     ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
#     ee_quat = ee_tf_data.target_quat_w[..., 0, :]
#     # make first element of quaternion positive
#     return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat
