# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions (experimental).

All functions in this file follow the Warp-compatible reward signature expected by
`isaaclab_experimental.managers.RewardManager`:

- ``func(env, out, **params) -> None``

where ``out`` is a pre-allocated Warp array of shape ``(num_envs,)`` with ``float32`` dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.kernels.state_kernels import (
    body_ang_vel_from_root,
    body_lin_vel_from_root,
    rotate_vec_to_body_frame,
)

from isaaclab.assets import Articulation

from isaaclab_experimental.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
General.
"""


@wp.kernel
def _is_alive_kernel(terminated: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.where(terminated[i], 0.0, 1.0)


def is_alive(env: ManagerBasedRLEnv, out: wp.array(dtype=wp.float32)) -> None:
    """Reward for being alive. Writes into ``out`` (shape: (num_envs,))."""
    wp.launch(
        kernel=_is_alive_kernel,
        dim=env.num_envs,
        inputs=[env.termination_manager.terminated_wp, out],
        device=env.device,
    )


@wp.kernel
def _is_terminated_kernel(terminated: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.where(terminated[i], 1.0, 0.0)


def is_terminated(env: ManagerBasedRLEnv, out) -> None:
    """Penalize terminated episodes. Writes into ``out``."""
    wp.launch(
        kernel=_is_terminated_kernel,
        dim=env.num_envs,
        inputs=[env.termination_manager.terminated_wp, out],
        device=env.device,
    )


"""
Root penalties.
"""


# Inline Tier 1 access: these rewards derive body-frame quantities directly from
# root_link_pose_w (transformf) and root_com_vel_w (spatial_vectorf), avoiding the lazy
# TimestampedWarpBuffer properties which are not CUDA-graph-capturable.
# See GRAPH_CAPTURE_MIGRATION.md in isaaclab_newton for background.
# If ArticulationData Tier 2 lazy update is made graph-safe in the future, these can
# revert to reading the pre-computed .data buffers (simpler, avoids redundant rotations).


@wp.kernel
def _lin_vel_z_l2_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    vz = body_lin_vel_from_root(root_pose_w[i], root_vel_w[i])[2]
    out[i] = vz * vz


def lin_vel_z_l2(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_lin_vel_z_l2_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_link_pose_w, asset.data.root_com_vel_w, out],
        device=env.device,
    )


@wp.kernel
def _ang_vel_xy_l2_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    v = body_ang_vel_from_root(root_pose_w[i], root_vel_w[i])
    out[i] = v[0] * v[0] + v[1] * v[1]


def ang_vel_xy_l2(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_ang_vel_xy_l2_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_link_pose_w, asset.data.root_com_vel_w, out],
        device=env.device,
    )


@wp.kernel
def _flat_orientation_l2_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    gravity_w: wp.array(dtype=wp.vec3f),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    g = rotate_vec_to_body_frame(gravity_w[0], root_pose_w[i])
    out[i] = g[0] * g[0] + g[1] * g[1]


def flat_orientation_l2(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize non-flat base orientation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_flat_orientation_l2_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_link_pose_w, asset.data.GRAVITY_VEC_W, out],
        device=env.device,
    )


"""
Joint penalties.
"""


# TODO(warp-migration): Revisit whether 2D kernel + wp.atomic_add is faster than 1D with inner loop
#  for the following masked reduction kernels. Profile with typical joint counts (12-30).
@wp.kernel
def _sum_sq_masked_kernel(
    x: wp.array(dtype=wp.float32, ndim=2), joint_mask: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.float32)
):
    i = wp.tid()
    s = float(0.0)
    for j in range(x.shape[1]):
        if joint_mask[j]:
            s += x[i, j] * x[i, j]
    out[i] = s


def joint_torques_l2(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize joint torques applied on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_sum_sq_masked_kernel,
        dim=env.num_envs,
        inputs=[asset.data.applied_torque, asset_cfg.joint_mask, out],
        device=env.device,
    )


# TODO(warp-migration): Revisit 2D kernel + wp.atomic_add vs 1D inner loop.
@wp.kernel
def _sum_abs_masked_kernel(
    x: wp.array(dtype=wp.float32, ndim=2), joint_mask: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.float32)
):
    i = wp.tid()
    s = float(0.0)
    for j in range(x.shape[1]):
        if joint_mask[j]:
            s += wp.abs(x[i, j])
    out[i] = s


def joint_vel_l1(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg) -> None:
    """Penalize joint velocities on the articulation using an L1-kernel. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_sum_abs_masked_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_vel, asset_cfg.joint_mask, out],
        device=env.device,
    )


def joint_vel_l2(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_sum_sq_masked_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_vel, asset_cfg.joint_mask, out],
        device=env.device,
    )


def joint_acc_l2(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_sum_sq_masked_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_acc, asset_cfg.joint_mask, out],
        device=env.device,
    )


# TODO(warp-migration): Revisit 2D kernel + wp.atomic_add vs 1D inner loop.
@wp.kernel
def _sum_abs_diff_masked_kernel(
    a: wp.array(dtype=wp.float32, ndim=2),
    b: wp.array(dtype=wp.float32, ndim=2),
    joint_mask: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    s = float(0.0)
    for j in range(a.shape[1]):
        if joint_mask[j]:
            s += wp.abs(a[i, j] - b[i, j])
    out[i] = s


def joint_deviation_l1(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_sum_abs_diff_masked_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_pos, asset.data.default_joint_pos, asset_cfg.joint_mask, out],
        device=env.device,
    )


# TODO(warp-migration): Revisit 2D kernel + wp.atomic_add vs 1D inner loop.
@wp.kernel
def _joint_pos_limits_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    soft_joint_pos_limits: wp.array(dtype=wp.vec2f, ndim=2),
    joint_mask: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    s = float(0.0)
    for j in range(joint_pos.shape[1]):
        if joint_mask[j]:
            pos = joint_pos[i, j]
            lim = soft_joint_pos_limits[i, j]
            lower = lim.x
            upper = lim.y
            # penalty for exceeding lower limit
            below = lower - pos
            if below > 0.0:
                s += below
            # penalty for exceeding upper limit
            above = pos - upper
            if above > 0.0:
                s += above
    out[i] = s


def joint_pos_limits(env: ManagerBasedRLEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Penalize joint positions if they cross the soft limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_joint_pos_limits_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_pos, asset.data.soft_joint_pos_limits, asset_cfg.joint_mask, out],
        device=env.device,
    )


"""
Action penalties.
"""


# TODO(warp-migration): Revisit 2D kernel + wp.atomic_add vs 1D inner loop.
@wp.kernel
def _sum_sq_diff_2d_kernel(
    a: wp.array(dtype=wp.float32, ndim=2),
    b: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    s = float(0.0)
    for j in range(a.shape[1]):
        d = a[i, j] - b[i, j]
        s += d * d
    out[i] = s


def action_rate_l2(env: ManagerBasedRLEnv, out) -> None:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    wp.launch(
        kernel=_sum_sq_diff_2d_kernel,
        dim=env.num_envs,
        inputs=[env.action_manager.action, env.action_manager.prev_action, out],
        device=env.device,
    )


# TODO(warp-migration): Revisit 2D kernel + wp.atomic_add vs 1D inner loop.
@wp.kernel
def _sum_sq_2d_kernel(x: wp.array(dtype=wp.float32, ndim=2), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    s = float(0.0)
    for j in range(x.shape[1]):
        s += x[i, j] * x[i, j]
    out[i] = s


def action_l2(env: ManagerBasedRLEnv, out) -> None:
    """Penalize the actions using L2 squared kernel."""
    wp.launch(
        kernel=_sum_sq_2d_kernel,
        dim=env.num_envs,
        inputs=[env.action_manager.action, out],
        device=env.device,
    )


"""
Contact sensor.
"""


@wp.kernel
def _undesired_contacts_kernel(
    forces: wp.array(dtype=wp.vec3f, ndim=3),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
    out: wp.array(dtype=wp.float32),
):
    """Count bodies where max-over-history contact force norm exceeds threshold."""
    i = wp.tid()
    count = float(0.0)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        max_force = float(0.0)
        for h in range(forces.shape[1]):
            f = forces[i, h, b]
            norm = wp.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
            if norm > max_force:
                max_force = norm
        if max_force > threshold:
            count += 1.0
    out[i] = count


def undesired_contacts(env: ManagerBasedRLEnv, out, threshold: float, sensor_cfg: SceneEntityCfg) -> None:
    """Penalize undesired contacts as the number of violations above a threshold. Writes into ``out``.

    Warp-first override of :func:`isaaclab.envs.mdp.rewards.undesired_contacts`.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    wp.launch(
        kernel=_undesired_contacts_kernel,
        dim=env.num_envs,
        inputs=[contact_sensor.data.net_forces_w_history, sensor_cfg.body_ids_wp, threshold, out],
        device=env.device,
    )


"""
Velocity-tracking rewards.
"""


@wp.kernel
def _track_lin_vel_xy_exp_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    command: wp.array(dtype=wp.float32, ndim=2),
    std_sq_inv: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    v = body_lin_vel_from_root(root_pose_w[i], root_vel_w[i])
    dx = command[i, 0] - v[0]
    dy = command[i, 1] - v[1]
    error = dx * dx + dy * dy
    out[i] = wp.exp(-error * std_sq_inv)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    out,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel. Writes into ``out``.

    Warp-first override of :func:`isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # cache the warp view of the command tensor on first call (zero-copy)
    # TODO(warp-migration): Cross-manager access (reward → command). Replace with direct
    #  warp getter once all managers are guaranteed to be warp-native.
    if not hasattr(track_lin_vel_xy_exp, "_cmd_wp") or track_lin_vel_xy_exp._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        if isinstance(cmd, wp.array):
            track_lin_vel_xy_exp._cmd_wp = cmd
        else:
            track_lin_vel_xy_exp._cmd_wp = wp.from_torch(cmd)
        track_lin_vel_xy_exp._cmd_name = command_name
    wp.launch(
        kernel=_track_lin_vel_xy_exp_kernel,
        dim=env.num_envs,
        inputs=[
            asset.data.root_link_pose_w,
            asset.data.root_com_vel_w,
            track_lin_vel_xy_exp._cmd_wp,
            1.0 / (std * std),
            out,
        ],
        device=env.device,
    )


@wp.kernel
def _track_ang_vel_z_exp_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    command: wp.array(dtype=wp.float32, ndim=2),
    cmd_col: int,
    std_sq_inv: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dz = command[i, cmd_col] - body_ang_vel_from_root(root_pose_w[i], root_vel_w[i])[2]
    out[i] = wp.exp(-dz * dz * std_sq_inv)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    out,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel. Writes into ``out``.

    Warp-first override of :func:`isaaclab.envs.mdp.rewards.track_ang_vel_z_exp`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # TODO(warp-migration): Cross-manager access (reward → command). Replace with direct
    #  warp getter once all managers are guaranteed to be warp-native.
    if not hasattr(track_ang_vel_z_exp, "_cmd_wp") or track_ang_vel_z_exp._cmd_name != command_name:
        cmd = env.command_manager.get_command(command_name)
        if isinstance(cmd, wp.array):
            track_ang_vel_z_exp._cmd_wp = cmd
        else:
            track_ang_vel_z_exp._cmd_wp = wp.from_torch(cmd)
        track_ang_vel_z_exp._cmd_name = command_name
    wp.launch(
        kernel=_track_ang_vel_z_exp_kernel,
        dim=env.num_envs,
        inputs=[
            asset.data.root_link_pose_w,
            asset.data.root_com_vel_w,
            track_ang_vel_z_exp._cmd_wp,
            2,
            1.0 / (std * std),
            out,
        ],
        device=env.device,
    )
