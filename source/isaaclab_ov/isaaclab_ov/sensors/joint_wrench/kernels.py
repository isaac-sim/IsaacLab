# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

from isaaclab.assets.articulation.ordering_kernels import resolve_backend_index


@wp.kernel
def joint_wrench_split_kernel(
    # inputs
    env_mask: wp.array(dtype=wp.bool),
    incoming_joint_wrench: wp.array(dtype=wp.spatial_vectorf, ndim=2),
    public_to_native: wp.array(dtype=wp.int32),
    has_ordering: bool,
    timestamp: wp.array(dtype=wp.float32),
    # outputs
    out_force: wp.array(dtype=wp.vec3f, ndim=2),
    out_torque: wp.array(dtype=wp.vec3f, ndim=2),
):
    """Split OVPhysX wrenches, already expressed at the child-side joint anchor in its frame.

    Args:
        env_mask: Boolean mask selecting which environments to update.
        incoming_joint_wrench: Incoming joint spatial wrenches in the child-side joint frame, referenced at the
            joint anchor ``(num_envs, num_bodies)``.
        public_to_native: Map from public sensor entries to native sensor entries.
        has_ordering: Whether a nonidentity sensor ordering is active.
        timestamp: Current sensor timestamp per environment [s] ``(num_envs,)``.
        out_force: Output force in child-side joint frame [N] ``(num_envs, num_bodies)``.
        out_torque: Output torque in child-side joint frame [N·m] ``(num_envs, num_bodies)``.
    """
    env, body = wp.tid()
    if not env_mask[env]:
        return

    # Skip envs that have not been stepped since their last reset: OVPhysX's incoming joint
    # wrench still holds pre-reset values, so reading it now would inject stale data (#4970).
    if timestamp[env] == 0.0:
        return

    native_body = resolve_backend_index(body, public_to_native, has_ordering)
    wrench = incoming_joint_wrench[env, native_body]
    out_force[env, body] = wp.spatial_top(wrench)
    out_torque[env, body] = wp.spatial_bottom(wrench)


@wp.kernel
def joint_wrench_reset_kernel(
    # inputs
    env_mask: wp.array(dtype=wp.bool),
    # outputs
    out_force: wp.array(dtype=wp.vec3f, ndim=2),
    out_torque: wp.array(dtype=wp.vec3f, ndim=2),
):
    """Zero force and torque entries for the environments selected by ``env_mask``.

    Args:
        env_mask: Boolean mask selecting which environments to reset.
        out_force: Output force in child-side joint frame [N] ``(num_envs, num_bodies)``.
        out_torque: Output torque in child-side joint frame [N·m] ``(num_envs, num_bodies)``.
    """
    env, body = wp.tid()
    if not env_mask[env]:
        return

    out_force[env, body] = wp.vec3f(0.0, 0.0, 0.0)
    out_torque[env, body] = wp.vec3f(0.0, 0.0, 0.0)
