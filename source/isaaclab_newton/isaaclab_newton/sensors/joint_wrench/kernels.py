# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def joint_wrench_to_incoming_joint_frame_kernel(
    env_mask: wp.array(dtype=wp.bool),
    body_parent_f: wp.array(dtype=wp.spatial_vectorf, ndim=2),
    body_q: wp.array(dtype=wp.transformf, ndim=2),
    body_com: wp.array(dtype=wp.vec3f, ndim=2),
    joint_X_c: wp.array(dtype=wp.transformf, ndim=2),
    joint_child: wp.array(dtype=wp.int32),
    out_force: wp.array(dtype=wp.vec3f, ndim=2),
    out_torque: wp.array(dtype=wp.vec3f, ndim=2),
):
    """Convert Newton's ``body_parent_f`` to the INCOMING_JOINT_FRAME convention.

    Newton reports ``body_parent_f[env, body]`` as a spatial wrench in world
    frame, referenced at the child body's centre of mass. The output is that
    same wrench re-expressed in the child-side joint frame and with the
    child-side joint anchor as the reference point — matching what a 6-axis
    force/torque sensor mounted at the joint would measure.
    """
    env, j = wp.tid()
    if not env_mask[env]:
        return

    body_idx = joint_child[j]

    # Source wrench: (force, torque-about-COM) in world frame.
    src = body_parent_f[env, body_idx]
    f_world = wp.vec3f(src[0], src[1], src[2])
    tau_world_com = wp.vec3f(src[3], src[4], src[5])

    # Child link transform in world and COM offset in link frame.
    link_xform = body_q[env, body_idx]
    link_quat = wp.transform_get_rotation(link_xform)
    link_pos = wp.transform_get_translation(link_xform)
    com_world = link_pos + wp.quat_rotate(link_quat, body_com[env, body_idx])

    # Child-side joint frame in world = body link pose composed with joint_X_c.
    joint_xform_world = link_xform * joint_X_c[env, j]
    anchor_world = wp.transform_get_translation(joint_xform_world)
    joint_quat_world = wp.transform_get_rotation(joint_xform_world)

    # Shift torque reference from COM to joint anchor: tau_B = tau_A + (A - B) x f.
    r_com_to_anchor = com_world - anchor_world
    tau_world_anchor = tau_world_com + wp.cross(r_com_to_anchor, f_world)

    # Rotate both components into the child-side joint frame.
    out_force[env, j] = wp.quat_rotate_inv(joint_quat_world, f_world)
    out_torque[env, j] = wp.quat_rotate_inv(joint_quat_world, tau_world_anchor)


@wp.kernel
def joint_wrench_reset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    out_force: wp.array(dtype=wp.vec3f, ndim=2),
    out_torque: wp.array(dtype=wp.vec3f, ndim=2),
):
    """Zero force / torque entries for the environments selected by ``env_mask``."""
    env, joint = wp.tid()
    if not env_mask[env]:
        return
    out_force[env, joint] = wp.vec3f(0.0, 0.0, 0.0)
    out_torque[env, joint] = wp.vec3f(0.0, 0.0, 0.0)
