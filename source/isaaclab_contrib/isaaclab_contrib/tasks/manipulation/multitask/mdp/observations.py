# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for multi-robot reach tasks.

Most observations (joint positions, joint velocities, command targets)
can be expressed with **standard** :mod:`isaaclab.envs.mdp` observation
functions combined with ``per_robot=True`` on the
:class:`~isaaclab.managers.ObservationTermCfg`.  The manager automatically
iterates :attr:`EnvLayout.robot_specs`, auto-injects ``asset_cfg`` /
``command_name``, and scatters results into the global tensor.

This module provides only the terms that have **no standard equivalent**:

* :func:`ee_pose_b` — end-effector pose in the robot root frame.
* :func:`ee_pos_error` — position error between EE and command target.
* :func:`multi_robot_type_onehot` — one-hot robot-type encoding (inherently
  global, not per-robot).
* :func:`ee_jacobian_b_padded` — body-frame Jacobian (advanced / reference).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


# ===========================================================
# Task-space observations (use with per_robot=True)
# ===========================================================


def ee_pose_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector pose in the robot root frame [m, -].

    The first body in ``asset_cfg.body_ids`` is treated as the
    end-effector.

    Returns:
        Shape ``(num_envs, 7)`` — ``(pos_x, y, z, quat_w, x, y, z)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos_w = wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids[0]]
    ee_quat_w = wp.to_torch(asset.data.body_quat_w)[:, asset_cfg.body_ids[0]]
    root_pos = wp.to_torch(asset.data.root_pos_w)
    root_quat = wp.to_torch(asset.data.root_quat_w)
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        root_pos,
        root_quat,
        ee_pos_w,
        ee_quat_w,
    )
    return torch.cat([pos_b, quat_b], dim=-1)


def ee_pos_error(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """EE position error vector ``(target − current)`` in the root frame [m].

    The first body in ``asset_cfg.body_ids`` is treated as the
    end-effector.  The command is expected to contain the target
    position in its first three columns (body-frame convention).

    Returns:
        Shape ``(num_envs, 3)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ee_pos_w = wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids[0]]
    root_pos = wp.to_torch(asset.data.root_pos_w)
    root_quat = wp.to_torch(asset.data.root_quat_w)
    cur_b, _ = math_utils.subtract_frame_transforms(
        root_pos,
        root_quat,
        ee_pos_w,
    )
    return cmd[:, :3] - cur_b


# ===========================================================
# Robot identity (inherently global — not suitable for per_robot)
# ===========================================================


def multi_robot_type_onehot(env: ManagerBasedEnv) -> torch.Tensor:
    """One-hot encoding of robot type, scattered across groups.

    Gives the policy an explicit signal to distinguish which robot it
    is controlling.  The number of classes equals ``len(layout.robot_specs)``.

    Returns:
        Shape ``(num_envs, num_robot_types)``.
    """
    layout = env.scene.layout
    specs = layout.robot_specs
    n_types = len(specs)
    out = torch.zeros(env.num_envs, n_types, device=env.device)
    for i, spec in enumerate(specs):
        gids = layout.asset_env_ids_t(spec[0])
        if gids is None:
            continue
        out[gids, i] = 1.0
    return out


# ===========================================================
# Jacobian observation (use with per_robot=True)
# ===========================================================


def ee_jacobian_b_padded(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body-frame EE Jacobian, flattened.

    This is a high-dimensional observation (``6 * num_joints``) that
    encodes the full kinematic structure.  For DiffIK-based reach tasks
    it is usually unnecessary because the IK controller already handles
    the Jacobian internally.

    Returns:
        Shape ``(num_envs, 6 * num_joints)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    bidx = asset_cfg.body_ids[0]
    jids = list(asset_cfg.joint_ids)
    nj = len(jids)

    if asset.is_fixed_base:
        jb_idx, jb_jids = bidx - 1, jids
    else:
        jb_idx = bidx
        jb_jids = [j + 6 for j in jids]

    jac = wp.to_torch(asset.root_view.get_jacobians())[:, jb_idx, :, jb_jids]
    rot = wp.to_torch(asset.data.root_quat_w)
    R = math_utils.matrix_from_quat(math_utils.quat_inv(rot))
    jac[:, :3, :] = torch.bmm(R, jac[:, :3, :])
    jac[:, 3:, :] = torch.bmm(R, jac[:, 3:, :])
    return jac.reshape(-1, 6 * nj)
