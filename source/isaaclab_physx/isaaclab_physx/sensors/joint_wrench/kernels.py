# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def joint_wrench_split_kernel(
    env_mask: wp.array(dtype=wp.bool),
    incoming_joint_wrench: wp.array(dtype=wp.spatial_vectorf, ndim=2),
    joint_pos_b: wp.array(dtype=wp.vec3f),
    joint_quat_b: wp.array(dtype=wp.quatf),
    out_force: wp.array(dtype=wp.vec3f, ndim=2),
    out_torque: wp.array(dtype=wp.vec3f, ndim=2),
):
    """Convert PhysX incoming joint spatial wrenches into the child-side joint frame."""
    env, body = wp.tid()
    if not env_mask[env]:
        return

    wrench = incoming_joint_wrench[env, body]
    force_b = wp.spatial_top(wrench)
    torque_b = wp.spatial_bottom(wrench)

    # PhysX reports the wrench in body1's frame, referenced at body1's origin.
    # Shift torque to the child-side joint anchor and rotate both components
    # into the child-side joint frame.
    torque_joint_anchor_b = torque_b - wp.cross(joint_pos_b[body], force_b)
    out_force[env, body] = wp.quat_rotate_inv(joint_quat_b[body], force_b)
    out_torque[env, body] = wp.quat_rotate_inv(joint_quat_b[body], torque_joint_anchor_b)


@wp.kernel
def joint_wrench_reset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    out_force: wp.array(dtype=wp.vec3f, ndim=2),
    out_torque: wp.array(dtype=wp.vec3f, ndim=2),
):
    """Zero force and torque entries for the environments selected by ``env_mask``."""
    env, body = wp.tid()
    if not env_mask[env]:
        return

    out_force[env, body] = wp.vec3f(0.0, 0.0, 0.0)
    out_torque[env, body] = wp.vec3f(0.0, 0.0, 0.0)
