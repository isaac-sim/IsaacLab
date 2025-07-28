# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs(
    env: ManagerBasedRLEnv,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef_to object,
    """

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def get_left_eef_pos(
    env: ManagerBasedRLEnv,
    link_name: str
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_left_eef_quat(
    env: ManagerBasedRLEnv,
    link_name: str
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_right_eef_pos(
    env: ManagerBasedRLEnv,
    link_name: str
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    return right_eef_pos


def get_right_eef_quat(
    env: ManagerBasedRLEnv,
    link_name: str
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    right_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    right_eef_quat = body_quat_w[:, right_eef_idx]

    return right_eef_quat


def get_hand_state(
    env: ManagerBasedRLEnv,
    hand_joint_names: List[str],
) -> torch.Tensor:
    # hand_joint_names is a list of regex, use find_joints
    indexes, _ = env.scene["robot"].find_joints(hand_joint_names)
    indexes = torch.tensor(indexes, dtype=torch.long)
    hand_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return hand_joint_states


def get_head_state(
    env: ManagerBasedRLEnv,
    head_joint_names: List[str]
) -> torch.Tensor:
    robot_joint_names = env.scene["robot"].data.joint_names
    indexes = torch.tensor([robot_joint_names.index(name) for name in head_joint_names], dtype=torch.long)
    head_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return head_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos
