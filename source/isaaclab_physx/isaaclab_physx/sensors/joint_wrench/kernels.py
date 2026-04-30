# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def joint_wrench_split_kernel(
    env_mask: wp.array(dtype=wp.bool),
    incoming_joint_wrench: wp.array(dtype=wp.spatial_vectorf, ndim=2),
    out_force: wp.array(dtype=wp.vec3f, ndim=2),
    out_torque: wp.array(dtype=wp.vec3f, ndim=2),
):
    """Split PhysX incoming joint spatial wrenches into force and torque components."""
    env, body = wp.tid()
    if not env_mask[env]:
        return

    wrench = incoming_joint_wrench[env, body]
    out_force[env, body] = wp.spatial_top(wrench)
    out_torque[env, body] = wp.spatial_bottom(wrench)


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
