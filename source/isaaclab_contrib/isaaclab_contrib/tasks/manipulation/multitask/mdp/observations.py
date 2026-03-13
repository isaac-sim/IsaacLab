# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for multi-robot reach tasks.

Most observations (joint positions, joint velocities, command targets)
can be expressed with **standard** :mod:`isaaclab.envs.mdp` observation
functions combined with ``per_robot=True`` on the
:class:`~isaaclab.managers.ObservationTermCfg`.  The manager automatically
iterates :attr:`EnvLayout.robot_infos`, auto-injects ``asset_cfg`` /
``command_name``, and scatters results into the global tensor.

This module provides only the terms that have **no standard equivalent**:

* :func:`ee_pose_b` — end-effector pose in the robot root frame.
* :func:`ee_pos_error` — position error between EE and command target.
* :func:`multi_task_onehot` — one-hot task group encoding (inherently
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


def multi_task_onehot(env: ManagerBasedEnv) -> torch.Tensor:
    """One-hot encoding of task group, scattered across groups.

    Delegates to :meth:`EnvLayout.multi_task_onehot`.

    Returns:
        Shape ``(num_envs, num_task_groups)``.
    """
    return env.scene.layout.multi_task_onehot()
