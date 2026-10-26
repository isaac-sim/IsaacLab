# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def pva_update_kernel(
    # indexing
    env_mask: wp.array(dtype=wp.bool),
    site_indices: wp.array(dtype=int),
    # model arrays
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    gravity: wp.array(dtype=wp.vec3),
    body_world: wp.array(dtype=wp.int32),
    # state arrays
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qdd: wp.array(dtype=wp.spatial_vector),
    # outputs
    out_pose_w: wp.array(dtype=wp.transformf),
    out_pos_w: wp.array(dtype=wp.vec3f),
    out_quat_w: wp.array(dtype=wp.quatf),
    out_projected_gravity_b: wp.array(dtype=wp.vec3f),
    out_lin_vel_b: wp.array(dtype=wp.vec3f),
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
    out_ang_acc_b: wp.array(dtype=wp.vec3f),
):
    idx = wp.tid()
    if not env_mask[idx]:
        return

    site_idx = site_indices[idx]
    body_idx = shape_body[site_idx]
    site_xform = shape_transform[site_idx]

    # 1. World-frame pose at sensor site
    body_xform = body_q[body_idx]
    body_quat = wp.transform_get_rotation(body_xform)
    sensor_pos = wp.transform_get_translation(body_xform) + wp.quat_rotate(body_quat, site_xform.p)
    sensor_quat = body_quat * site_xform.q

    # 2. Projected gravity (normalized to unit vector)
    world_idx = body_world[body_idx]
    g = gravity[wp.max(world_idx, 0)]
    projected_gravity_b = wp.quat_rotate_inv(sensor_quat, wp.normalize(g))

    # 3. Velocity at sensor site: v_site = v_com + omega x r
    #    r = lever arm from body COM to sensor site, in world frame.
    #    body_qd stores the spatial velocity at the COM, so the linear
    #    velocity at the site requires the cross-product correction.
    r = wp.quat_rotate(body_quat, site_xform.p - body_com[body_idx])
    ang_vel_w = wp.spatial_bottom(body_qd[body_idx])
    lin_vel_w = wp.spatial_top(body_qd[body_idx]) + wp.cross(ang_vel_w, r)

    lin_vel_b = wp.quat_rotate_inv(sensor_quat, lin_vel_w)
    ang_vel_b = wp.quat_rotate_inv(sensor_quat, ang_vel_w)

    # 4. Acceleration at sensor site via rigid-body transport (from body_qdd):
    #    a_site = a_com + alpha x r + omega x (omega x r)
    ang_acc_w = wp.spatial_bottom(body_qdd[body_idx])
    lin_acc_w = (
        wp.spatial_top(body_qdd[body_idx]) + wp.cross(ang_acc_w, r) + wp.cross(ang_vel_w, wp.cross(ang_vel_w, r))
    )

    lin_acc_b = wp.quat_rotate_inv(sensor_quat, lin_acc_w)
    ang_acc_b = wp.quat_rotate_inv(sensor_quat, ang_acc_w)

    # 5. Write outputs
    out_pose_w[idx] = wp.transform(sensor_pos, sensor_quat)
    out_pos_w[idx] = sensor_pos
    out_quat_w[idx] = sensor_quat
    out_projected_gravity_b[idx] = projected_gravity_b
    out_lin_vel_b[idx] = lin_vel_b
    out_ang_vel_b[idx] = ang_vel_b
    out_lin_acc_b[idx] = lin_acc_b
    out_ang_acc_b[idx] = ang_acc_b


@wp.kernel
def pva_reset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    out_pose_w: wp.array(dtype=wp.transformf),
    out_pos_w: wp.array(dtype=wp.vec3f),
    out_quat_w: wp.array(dtype=wp.quatf),
    out_projected_gravity_b: wp.array(dtype=wp.vec3f),
    out_lin_vel_b: wp.array(dtype=wp.vec3f),
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
    out_ang_acc_b: wp.array(dtype=wp.vec3f),
):
    idx = wp.tid()
    if not env_mask[idx]:
        return

    out_pose_w[idx] = wp.transform(wp.vec3f(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))
    out_pos_w[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_quat_w[idx] = wp.quatf(0.0, 0.0, 0.0, 1.0)
    out_projected_gravity_b[idx] = wp.vec3f(0.0, 0.0, -1.0)
    out_lin_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_ang_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_lin_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_ang_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
