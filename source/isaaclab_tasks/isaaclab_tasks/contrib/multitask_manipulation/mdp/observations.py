# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware observations for heterogeneous manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from ..selection_utils import SceneEntitySelectionCfg
from .utils import _offset_body_pose

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def task_encoding(env: ManagerBasedRLEnv, task_asset_cfgs: tuple[SceneEntitySelectionCfg, ...]) -> torch.Tensor:
    """Return a one-hot identity for the selected manipulation task."""
    encoding = torch.zeros(env.num_envs, len(task_asset_cfgs), device=env.device)
    for task_id, asset_cfg in enumerate(task_asset_cfgs):
        encoding[asset_cfg.instance_ids >= 0, task_id] = 1.0
    return encoding


def selected_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntitySelectionCfg) -> torch.Tensor:
    """Return relative joint positions in global environment order."""
    asset: Articulation = env.scene[asset_cfg.name]
    values = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return asset_cfg.scatter_to_envs(values)


def selected_joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntitySelectionCfg) -> torch.Tensor:
    """Return relative joint velocities in global environment order."""
    asset: Articulation = env.scene[asset_cfg.name]
    values = (
        asset.data.joint_vel.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_vel.torch[:, asset_cfg.joint_ids]
    )
    return asset_cfg.scatter_to_envs(values)


def lift_object_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    object_cfg: SceneEntitySelectionCfg,
) -> torch.Tensor:
    """Return the lift object's pose in the Franka root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    robot_rows = robot_cfg.instance_ids[object_cfg.env_ids]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch[robot_rows],
        robot.data.root_quat_w.torch[robot_rows],
        object_asset.data.root_pos_w.torch,
        object_asset.data.root_quat_w.torch,
    )
    return object_cfg.scatter_to_envs(torch.cat((pos_b, quat_b), dim=-1))


def lift_object_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    object_cfg: SceneEntitySelectionCfg,
) -> torch.Tensor:
    """Return the lift object's position [m] in the OpenArm root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    robot_rows = robot_cfg.instance_ids[object_cfg.env_ids]
    pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch[robot_rows],
        robot.data.root_quat_w.torch[robot_rows],
        object_asset.data.root_pos_w.torch,
    )
    return object_cfg.scatter_to_envs(pos_b)


def cabinet_drawer_state(env: ManagerBasedRLEnv, cabinet_cfg: SceneEntitySelectionCfg) -> torch.Tensor:
    """Return top-drawer position and velocity in global environment order."""
    cabinet: Articulation = env.scene[cabinet_cfg.name]
    values = torch.cat(
        (
            cabinet.data.joint_pos.torch[:, cabinet_cfg.joint_ids],
            cabinet.data.joint_vel.torch[:, cabinet_cfg.joint_ids],
        ),
        dim=-1,
    )
    return cabinet_cfg.scatter_to_envs(values)


def cabinet_ee_to_handle(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
) -> torch.Tensor:
    """Return the vector [m] from the cabinet Franka TCP to the drawer handle."""
    env_ids, ee_pos_w, _ = _offset_body_pose(env, robot_cfg, (0.0, 0.0, 0.1034))
    _, handle_pos_w, _ = _offset_body_pose(
        env,
        cabinet_cfg,
        (0.305, 0.0, 0.01),
        (0.5, -0.5, -0.5, 0.5),
    )
    values = handle_pos_w[cabinet_cfg.instance_ids[env_ids]] - ee_pos_w
    return robot_cfg.scatter_to_envs(values)
