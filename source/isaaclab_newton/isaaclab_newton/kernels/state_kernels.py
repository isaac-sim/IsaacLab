# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp helper functions for body-frame state computations.

These ``@wp.func`` helpers are used by warp-first MDP terms (observations,
rewards) that need to project root-frame quantities into body frames.
"""

import warp as wp


@wp.func
def rotate_vec_to_body_frame(vec_w: wp.vec3f, pose_w: wp.transformf) -> wp.vec3f:
    """Rotate a world-frame vector into the body frame defined by pose_w."""
    return wp.quat_rotate_inv(wp.transform_get_rotation(pose_w), vec_w)


@wp.func
def body_lin_vel_from_root(pose_w: wp.transformf, vel_w: wp.spatial_vectorf) -> wp.vec3f:
    """Extract body-frame linear velocity from root pose and spatial velocity."""
    return wp.quat_rotate_inv(wp.transform_get_rotation(pose_w), wp.spatial_top(vel_w))


@wp.func
def body_ang_vel_from_root(pose_w: wp.transformf, vel_w: wp.spatial_vectorf) -> wp.vec3f:
    """Extract body-frame angular velocity from root pose and spatial velocity."""
    return wp.quat_rotate_inv(wp.transform_get_rotation(pose_w), wp.spatial_bottom(vel_w))
