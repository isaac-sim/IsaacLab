# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first reward terms for the reach task.

All functions follow the ``func(env, out, **params) -> None`` signature.
Command tensors are cached as zero-copy warp views on first call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# position_command_error
# ---------------------------------------------------------------------------


@wp.kernel
def _position_command_error_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    body_pos_w: wp.array(dtype=wp.vec3f, ndim=2),
    cmd: wp.array(dtype=wp.float32, ndim=2),
    body_idx: int,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    # desired position in body frame -> world frame
    des_b = wp.vec3f(cmd[i, 0], cmd[i, 1], cmd[i, 2])
    des_w = root_pos_w[i] + wp.quat_rotate(root_quat_w[i], des_b)
    # current end-effector position
    cur_w = body_pos_w[i, body_idx]
    dx = cur_w[0] - des_w[0]
    dy = cur_w[1] - des_w[1]
    dz = cur_w[2] - des_w[2]
    out[i] = wp.sqrt(dx * dx + dy * dy + dz * dz)


def position_command_error(env: ManagerBasedRLEnv, out, command_name: str, asset_cfg: SceneEntityCfg) -> None:
    """Penalize tracking of the position error using L2-norm."""
    fn = position_command_error
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.launch(
        kernel=_position_command_error_kernel,
        dim=env.num_envs,
        inputs=[
            asset.data.root_pos_w,
            asset.data.root_quat_w,
            asset.data.body_pos_w,
            fn._cmd_wp,
            asset_cfg.body_ids[0],
            out,
        ],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# position_command_error_tanh
# ---------------------------------------------------------------------------


@wp.kernel
def _position_command_error_tanh_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    body_pos_w: wp.array(dtype=wp.vec3f, ndim=2),
    cmd: wp.array(dtype=wp.float32, ndim=2),
    body_idx: int,
    inv_std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    des_b = wp.vec3f(cmd[i, 0], cmd[i, 1], cmd[i, 2])
    des_w = root_pos_w[i] + wp.quat_rotate(root_quat_w[i], des_b)
    cur_w = body_pos_w[i, body_idx]
    dx = cur_w[0] - des_w[0]
    dy = cur_w[1] - des_w[1]
    dz = cur_w[2] - des_w[2]
    dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
    out[i] = 1.0 - wp.tanh(dist * inv_std)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, out, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> None:
    """Reward tracking of the position using the tanh kernel."""
    fn = position_command_error_tanh
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.launch(
        kernel=_position_command_error_tanh_kernel,
        dim=env.num_envs,
        inputs=[
            asset.data.root_pos_w,
            asset.data.root_quat_w,
            asset.data.body_pos_w,
            fn._cmd_wp,
            asset_cfg.body_ids[0],
            1.0 / std,
            out,
        ],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# orientation_command_error
# ---------------------------------------------------------------------------


@wp.kernel
def _orientation_command_error_kernel(
    root_quat_w: wp.array(dtype=wp.quatf),
    body_quat_w: wp.array(dtype=wp.quatf, ndim=2),
    cmd: wp.array(dtype=wp.float32, ndim=2),
    body_idx: int,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    # desired quat in body frame -> world frame: q_des_w = q_root * q_des_b
    des_b = wp.quatf(cmd[i, 3], cmd[i, 4], cmd[i, 5], cmd[i, 6])
    des_w = root_quat_w[i] * des_b
    # current ee orientation
    cur_w = body_quat_w[i, body_idx]
    # shortest-path error: angle of q_err = cur^-1 * des
    q_err = wp.quat_inverse(cur_w) * des_w
    # error magnitude = 2 * acos(|w|)  (w component of the error quaternion)
    qw = wp.abs(q_err[3])
    qw = wp.clamp(qw, 0.0, 1.0)
    out[i] = 2.0 * wp.acos(qw)


def orientation_command_error(env: ManagerBasedRLEnv, out, command_name: str, asset_cfg: SceneEntityCfg) -> None:
    """Penalize tracking orientation error using shortest path."""
    fn = orientation_command_error
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.launch(
        kernel=_orientation_command_error_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_quat_w, asset.data.body_quat_w, fn._cmd_wp, asset_cfg.body_ids[0], out],
        device=env.device,
    )
