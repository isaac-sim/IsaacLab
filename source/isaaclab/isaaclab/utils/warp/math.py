# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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


@wp.func
def quat_error_magnitude(q1: wp.quatf, q2: wp.quatf) -> float:
    """Shortest rotation angle [rad] between nonzero ``xyzw`` quaternions.

    The result is in [0, pi] for finite inputs and is invariant to quaternion
    sign and scale. Non-finite inputs propagate to the result.
    """
    error = wp.quat_inverse(q1) * q2
    return 2.0 * wp.atan2(wp.length(wp.vec3f(error[0], error[1], error[2])), wp.abs(error[3]))


@wp.func
def velocity_at_offset(velocity: wp.spatial_vectorf, offset: wp.vec3f) -> wp.vec3f:
    """Linear velocity [m/s] at an offset [m]; all quantities share one frame.

    Spatial vectors store linear components first and angular components last.
    """
    return wp.spatial_top(velocity) + wp.cross(wp.spatial_bottom(velocity), offset)


@wp.func
def acceleration_at_offset(acceleration: wp.spatial_vectorf, angular_velocity: wp.vec3f, offset: wp.vec3f) -> wp.vec3f:
    """Rigid-body linear acceleration [m/s^2] at an offset [m], in the input frame.

    Includes tangential and centripetal acceleration. Gravity is the caller's policy.
    Spatial vectors store linear components first and angular components last.
    """
    return (
        wp.spatial_top(acceleration)
        + wp.cross(wp.spatial_bottom(acceleration), offset)
        + wp.cross(angular_velocity, wp.cross(angular_velocity, offset))
    )


@wp.func
def yaw_from_quat(q: wp.quatf) -> float:
    """Signed yaw angle [rad] of a unit ``xyzw`` quaternion."""
    return wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


@wp.func
def yaw_quat(q: wp.quatf) -> wp.quatf:
    """Unit ``xyzw`` quaternion containing only the input orientation's yaw."""
    half_yaw = 0.5 * yaw_from_quat(q)
    return wp.quatf(0.0, 0.0, wp.sin(half_yaw), wp.cos(half_yaw))


@wp.func
def heading_from_quat(forward_vec: wp.vec3f, quat: wp.quatf) -> float:
    """World-frame heading [rad] of a body-frame forward vector rotated by an ``xyzw`` quaternion."""
    forward_w = wp.quat_rotate(quat, forward_vec)
    return wp.atan2(forward_w[1], forward_w[0])


@wp.func
def scale_interval(bounds: wp.vec2f, factor: float) -> wp.vec2f:
    """Scale an interval's width about its midpoint without clamping the factor."""
    midpoint = (bounds[0] + bounds[1]) / 2.0
    width = bounds[1] - bounds[0]
    return wp.vec2f(midpoint - 0.5 * width * factor, midpoint + 0.5 * width * factor)


@wp.kernel
def shift_jacobian_reference(
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_offset: wp.int32,
    src: wp.array4d(dtype=wp.float32),
    dst: wp.array4d(dtype=wp.float32),
):
    """Shift world-frame Jacobian columns from COM to link origin.

    Inputs are already in public body/DoF order. src and dst have shape
    (environments, Jacobian bodies, 6, DoFs), with linear rows before angular rows.
    body_link_pose and body_com_pos_b include every body; link_offset maps a
    Jacobian body to that array (1 when a fixed root was omitted, otherwise 0).
    COM offsets [m] are in the link frame. Angular rows are unchanged.
    """
    env, body, dof = wp.tid()
    full_body = body + link_offset
    rotation = wp.transform_get_rotation(body_link_pose[env, full_body])
    offset_w = wp.quat_rotate(rotation, body_com_pos_b[env, full_body])
    linear = wp.vec3f(src[env, body, 0, dof], src[env, body, 1, dof], src[env, body, 2, dof])
    angular = wp.vec3f(src[env, body, 3, dof], src[env, body, 4, dof], src[env, body, 5, dof])
    shifted = velocity_at_offset(wp.spatial_vector(linear, angular), -offset_w)
    dst[env, body, 0, dof] = shifted[0]
    dst[env, body, 1, dof] = shifted[1]
    dst[env, body, 2, dof] = shifted[2]
    dst[env, body, 3, dof] = angular[0]
    dst[env, body, 4, dof] = angular[1]
    dst[env, body, 5, dof] = angular[2]


# Predeclare array launches without allocating storage or launching kernels. Zero-extent array
# descriptors carry rank and dtype without specializing singleton broadcasting; scalar values stay runtime inputs.
quat_rotate_inv = {
    rank: wp.map(
        wp.quat_rotate_inv,
        wp.array(dtype=wp.quatf, ndim=rank),
        wp.array(dtype=wp.vec3f, ndim=rank),
        out=wp.array(dtype=wp.vec3f, ndim=rank),
        return_kernel=True,
    )
    for rank in (1, 2)
}
heading = {
    rank: wp.map(
        heading_from_quat,
        wp.array(dtype=wp.vec3f, ndim=rank),
        wp.array(dtype=wp.quatf, ndim=rank),
        out=wp.array(dtype=wp.float32, ndim=rank),
        return_kernel=True,
    )
    for rank in (1, 2)
}
scale_intervals = wp.map(
    scale_interval, wp.array2d(dtype=wp.vec2f), 0.0, out=wp.array2d(dtype=wp.vec2f), return_kernel=True
)
