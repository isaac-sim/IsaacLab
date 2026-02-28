# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first reward functions for the velocity locomotion environment.

All functions follow the ``func(env, out, **params) -> None`` signature.
Cross-manager torch tensors (contact sensor, commands) are cached as zero-copy
warp views on first call via ``wp.from_torch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# feet_air_time
# ---------------------------------------------------------------------------


@wp.kernel
def _feet_air_time_kernel(
    last_air_time: wp.array(dtype=wp.float32, ndim=2),
    first_contact: wp.array(dtype=wp.float32, ndim=2),
    body_ids: wp.array(dtype=wp.int32),
    cmd_xy: wp.array(dtype=wp.float32, ndim=2),
    threshold: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    s = float(0.0)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        s += (last_air_time[i, b] - threshold) * first_contact[i, b]
    # gate by command magnitude
    cx = cmd_xy[i, 0]
    cy = cmd_xy[i, 1]
    cmd_norm = wp.sqrt(cx * cx + cy * cy)
    out[i] = wp.where(cmd_norm > 0.1, s, 0.0)


def feet_air_time(env: ManagerBasedRLEnv, out, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> None:
    """Reward long steps taken by the feet using L2-kernel."""
    fn = feet_air_time
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Cache command bridge (persistent pointer)
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    # Newton contact sensor returns persistent wp.arrays — use directly, no wp.from_torch needed
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    wp.launch(
        kernel=_feet_air_time_kernel,
        dim=env.num_envs,
        inputs=[contact_sensor.data.last_air_time, first_contact, sensor_cfg.body_ids_wp, fn._cmd_wp, threshold, out],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# feet_air_time_positive_biped
# ---------------------------------------------------------------------------


@wp.kernel
def _feet_air_time_positive_biped_kernel(
    air_time: wp.array(dtype=wp.float32, ndim=2),
    contact_time: wp.array(dtype=wp.float32, ndim=2),
    body_ids: wp.array(dtype=wp.int32),
    cmd_xy: wp.array(dtype=wp.float32, ndim=2),
    threshold: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n_feet = body_ids.shape[0]
    # count feet in contact and find single-stance min mode time
    n_contact = int(0)
    for k in range(n_feet):
        b = body_ids[k]
        if contact_time[i, b] > 0.0:
            n_contact += 1
    single_stance = n_contact == 1
    min_val = threshold  # clamp upper bound
    for k in range(n_feet):
        b = body_ids[k]
        in_contact = contact_time[i, b] > 0.0
        mode_time = wp.where(in_contact, contact_time[i, b], air_time[i, b])
        val = wp.where(single_stance, mode_time, 0.0)
        min_val = wp.min(min_val, val)
    # gate by command magnitude
    cx = cmd_xy[i, 0]
    cy = cmd_xy[i, 1]
    cmd_norm = wp.sqrt(cx * cx + cy * cy)
    out[i] = wp.where(cmd_norm > 0.1, min_val, 0.0)


def feet_air_time_positive_biped(env, out, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> None:
    """Reward long steps taken by the feet for bipeds."""
    fn = feet_air_time_positive_biped
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.launch(
        kernel=_feet_air_time_positive_biped_kernel,
        dim=env.num_envs,
        inputs=[
            contact_sensor.data.current_air_time,
            contact_sensor.data.current_contact_time,
            sensor_cfg.body_ids_wp,
            fn._cmd_wp,
            threshold,
            out,
        ],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# feet_slide
# ---------------------------------------------------------------------------


@wp.kernel
def _feet_slide_kernel(
    body_lin_vel_w: wp.array(dtype=wp.vec3f, ndim=2),
    net_forces_w: wp.array(dtype=wp.vec3f, ndim=3),
    body_ids: wp.array(dtype=wp.int32),
    n_history: int,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    s = float(0.0)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        # check if in contact: max force norm over history > 1.0
        max_force = float(0.0)
        for h in range(n_history):
            f = net_forces_w[i, h, b]
            f_norm = wp.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
            max_force = wp.max(max_force, f_norm)
        in_contact = wp.where(max_force > 1.0, 1.0, 0.0)
        # planar velocity norm
        vx = body_lin_vel_w[i, b][0]
        vy = body_lin_vel_w[i, b][1]
        vel_norm = wp.sqrt(vx * vx + vy * vy)
        s += vel_norm * in_contact
    out[i] = s


def feet_slide(env, out, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize feet sliding."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_feet_slide_kernel,
        dim=env.num_envs,
        inputs=[
            asset.data.body_lin_vel_w,
            contact_sensor.data.net_forces_w_history,
            sensor_cfg.body_ids_wp,
            contact_sensor.data.net_forces_w_history.shape[1],
            out,
        ],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# track_lin_vel_xy_yaw_frame_exp
# ---------------------------------------------------------------------------


@wp.kernel
def _track_lin_vel_xy_yaw_frame_exp_kernel(
    root_quat_w: wp.array(dtype=wp.quatf),
    root_lin_vel_w: wp.array(dtype=wp.vec3f),
    cmd: wp.array(dtype=wp.float32, ndim=2),
    inv_std_sq: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    q = root_quat_w[i]
    # extract yaw-only quaternion
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_half = wp.atan2(sin_yaw, cos_yaw) * 0.5
    yaw_q = wp.quatf(0.0, 0.0, wp.sin(yaw_half), wp.cos(yaw_half))
    # rotate world velocity into yaw frame (inverse = conjugate for unit quat)
    vel_w = root_lin_vel_w[i]
    vel_yaw = wp.quat_rotate(wp.quat_inverse(yaw_q), vel_w)
    # error
    ex = cmd[i, 0] - vel_yaw[0]
    ey = cmd[i, 1] - vel_yaw[1]
    err_sq = ex * ex + ey * ey
    out[i] = wp.exp(-err_sq * inv_std_sq)


def track_lin_vel_xy_yaw_frame_exp(
    env, out, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame."""
    fn = track_lin_vel_xy_yaw_frame_exp
    asset = env.scene[asset_cfg.name]
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.launch(
        kernel=_track_lin_vel_xy_yaw_frame_exp_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_quat_w, asset.data.root_lin_vel_w, fn._cmd_wp, 1.0 / (std * std), out],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# track_ang_vel_z_world_exp
# ---------------------------------------------------------------------------


@wp.kernel
def _track_ang_vel_z_world_exp_kernel(
    root_ang_vel_w: wp.array(dtype=wp.vec3f),
    cmd: wp.array(dtype=wp.float32, ndim=2),
    inv_std_sq: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    err = cmd[i, 2] - root_ang_vel_w[i][2]
    out[i] = wp.exp(-(err * err) * inv_std_sq)


def track_ang_vel_z_world_exp(
    env, out, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Reward tracking of angular velocity commands (yaw) in world frame."""
    fn = track_ang_vel_z_world_exp
    asset = env.scene[asset_cfg.name]
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.launch(
        kernel=_track_ang_vel_z_world_exp_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_ang_vel_w, fn._cmd_wp, 1.0 / (std * std), out],
        device=env.device,
    )


# ---------------------------------------------------------------------------
# stand_still_joint_deviation_l1
# ---------------------------------------------------------------------------


@wp.kernel
def _stand_still_joint_deviation_l1_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    default_joint_pos: wp.array(dtype=wp.float32, ndim=2),
    cmd: wp.array(dtype=wp.float32, ndim=2),
    command_threshold: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n_joints = joint_pos.shape[1]
    dev = float(0.0)
    for j in range(n_joints):
        dev += wp.abs(joint_pos[i, j] - default_joint_pos[i, j])
    # gate: only penalize when command is small
    cx = cmd[i, 0]
    cy = cmd[i, 1]
    cmd_norm = wp.sqrt(cx * cx + cy * cy)
    out[i] = wp.where(cmd_norm < command_threshold, dev, 0.0)


def stand_still_joint_deviation_l1(
    env, out, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Penalize offsets from the default joint positions when the command is very small."""
    fn = stand_still_joint_deviation_l1
    asset = env.scene[asset_cfg.name]
    if not hasattr(fn, "_cmd_wp") or fn._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        fn._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        fn._cmd_name = command_name
    wp.launch(
        kernel=_stand_still_joint_deviation_l1_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_pos, asset.data.default_joint_pos, fn._cmd_wp, command_threshold, out],
        device=env.device,
    )
