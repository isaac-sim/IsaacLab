# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first observation terms for the humanoid task.

All observation functions follow the ``func(env, out, **params) -> None`` signature.
Dimensions are declared via ``out_dim`` on the ``@generic_io_descriptor_warp`` decorator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.envs.utils.io_descriptors import generic_io_descriptor_warp
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_newton.kernels.state_kernels import rotate_vec_to_body_frame

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@wp.kernel
def _base_yaw_roll_kernel(
    root_quat_w: wp.array(dtype=wp.quatf),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    """Extract yaw and roll angles from root quaternion (x, y, z, w layout)."""
    i = wp.tid()
    q = root_quat_w[i]
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    # roll = atan2(2*(qw*qx + qy*qz), 1 - 2*(qx^2 + qy^2))
    sin_roll = 2.0 * (qw * qx + qy * qz)
    cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = wp.atan2(sin_roll, cos_roll)
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = wp.atan2(sin_yaw, cos_yaw)
    out[i, 0] = yaw
    out[i, 1] = roll


@generic_io_descriptor_warp(out_dim=2, observation_type="RootState")
def base_yaw_roll(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Yaw and roll of the base in the simulation world frame. Shape: (num_envs, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_base_yaw_roll_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_quat_w, out],
        device=env.device,
    )


# Inline Tier 1 access: derives projected gravity directly from root_link_pose_w,
# avoiding the lazy TimestampedWarpBuffer which is not CUDA-graph-capturable.
# See GRAPH_CAPTURE_MIGRATION.md in isaaclab_newton for background.


@wp.kernel
def _base_up_proj_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    gravity_w: wp.array(dtype=wp.vec3f),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    """Project base up vector onto world up: -gravity_b[2]."""
    i = wp.tid()
    out[i, 0] = -rotate_vec_to_body_frame(gravity_w[0], root_pose_w[i])[2]


@generic_io_descriptor_warp(out_dim=1, observation_type="RootState")
def base_up_proj(env: ManagerBasedEnv, out, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Projection of the base up vector onto the world up vector. Shape: (num_envs, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_base_up_proj_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_link_pose_w, asset.data.GRAVITY_VEC_W, out],
        device=env.device,
    )


@wp.kernel
def _base_heading_proj_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    target_x: float,
    target_y: float,
    target_z: float,
    out: wp.array(dtype=wp.float32, ndim=2),
):
    """Dot product between robot forward and direction to target."""
    i = wp.tid()
    pos = root_pos_w[i]
    q = root_quat_w[i]
    # compute direction to target (zeroed z)
    dx = target_x - pos[0]
    dy = target_y - pos[1]
    dist = wp.sqrt(dx * dx + dy * dy)
    # avoid division by zero
    inv_dist = wp.where(dist > 1.0e-6, 1.0 / dist, 0.0)
    to_target_x = dx * inv_dist
    to_target_y = dy * inv_dist
    # compute forward vector via quaternion rotation of (1,0,0)
    fwd = wp.quat_rotate(q, wp.vec3f(1.0, 0.0, 0.0))
    # dot product (xy only)
    heading_proj = fwd[0] * to_target_x + fwd[1] * to_target_y
    out[i, 0] = heading_proj


@generic_io_descriptor_warp(out_dim=1, observation_type="RootState")
def base_heading_proj(
    env: ManagerBasedEnv,
    out,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Dot product between the base forward direction and direction to target. Shape: (num_envs, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_base_heading_proj_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_pos_w, asset.data.root_quat_w, target_pos[0], target_pos[1], target_pos[2], out],
        device=env.device,
    )


@wp.kernel
def _base_angle_to_target_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    target_x: float,
    target_y: float,
    out: wp.array(dtype=wp.float32, ndim=2),
):
    """Angle between base forward and vector to target, normalized to [-pi, pi]."""
    i = wp.tid()
    pos = root_pos_w[i]
    q = root_quat_w[i]
    # angle to target in world frame
    dx = target_x - pos[0]
    dy = target_y - pos[1]
    walk_target_angle = wp.atan2(dy, dx)
    # extract yaw from quaternion
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = wp.atan2(sin_yaw, cos_yaw)
    # normalize to [-pi, pi]
    angle = walk_target_angle - yaw
    out[i, 0] = wp.atan2(wp.sin(angle), wp.cos(angle))


@generic_io_descriptor_warp(out_dim=1, observation_type="RootState")
def base_angle_to_target(
    env: ManagerBasedEnv,
    out,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Angle between the base forward vector and the vector to the target. Shape: (num_envs, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_base_angle_to_target_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_pos_w, asset.data.root_quat_w, target_pos[0], target_pos[1], out],
        device=env.device,
    )
