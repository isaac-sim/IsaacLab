# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def imu_update_kernel(
    # indexing
    env_mask: wp.array(dtype=wp.bool),
    # PhysX view data
    transforms: wp.array(dtype=wp.transformf),
    velocities: wp.array(dtype=wp.spatial_vectorf),
    coms: wp.array(dtype=wp.transformf),
    # sensor config (per-env)
    offset_pos_b: wp.array(dtype=wp.vec3f),
    offset_quat_b: wp.array(dtype=wp.quatf),
    gravity_bias_w: wp.array(dtype=wp.vec3f),
    gravity_vec_w: wp.array(dtype=wp.vec3f),
    # previous velocities (read + write)
    prev_lin_vel_w: wp.array(dtype=wp.vec3f),
    prev_ang_vel_w: wp.array(dtype=wp.vec3f),
    # scalar
    inv_dt: wp.float32,
    # outputs (written in-place)
    out_pos_w: wp.array(dtype=wp.vec3f),
    out_quat_w: wp.array(dtype=wp.quatf),
    out_lin_vel_b: wp.array(dtype=wp.vec3f),
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
    out_ang_acc_b: wp.array(dtype=wp.vec3f),
    out_projected_gravity_b: wp.array(dtype=wp.vec3f),
):
    idx = wp.tid()
    if not env_mask[idx]:
        return

    # 1. Extract body pose
    body_pos = wp.transform_get_translation(transforms[idx])
    body_quat = wp.transform_get_rotation(transforms[idx])

    # 2. Apply sensor offset
    sensor_pos = body_pos + wp.quat_rotate(body_quat, offset_pos_b[idx])
    sensor_quat = body_quat * offset_quat_b[idx]

    # 3. Extract lin/ang velocity
    lin_vel_w = wp.spatial_top(velocities[idx])
    ang_vel_w = wp.spatial_bottom(velocities[idx])

    # 4. COM correction: lin_vel += cross(ang_vel, quat_rotate(body_quat, offset_pos - com_pos))
    com_pos_b = wp.transform_get_translation(coms[idx])
    lever_arm = wp.quat_rotate(body_quat, offset_pos_b[idx] - com_pos_b)
    lin_vel_w = lin_vel_w + wp.cross(ang_vel_w, lever_arm)

    # 5. Numerical differentiation (world frame)
    lin_acc_w = (lin_vel_w - prev_lin_vel_w[idx]) * inv_dt + gravity_bias_w[idx]
    ang_acc_w = (ang_vel_w - prev_ang_vel_w[idx]) * inv_dt

    # 6. Rotate world -> body using sensor orientation
    lin_vel_b = wp.quat_rotate_inv(sensor_quat, lin_vel_w)
    ang_vel_b = wp.quat_rotate_inv(sensor_quat, ang_vel_w)
    lin_acc_b = wp.quat_rotate_inv(sensor_quat, lin_acc_w)
    ang_acc_b = wp.quat_rotate_inv(sensor_quat, ang_acc_w)
    projected_gravity_b = wp.quat_rotate_inv(sensor_quat, gravity_vec_w[idx])

    # 7. Store results
    out_pos_w[idx] = sensor_pos
    out_quat_w[idx] = sensor_quat
    out_lin_vel_b[idx] = lin_vel_b
    out_ang_vel_b[idx] = ang_vel_b
    out_lin_acc_b[idx] = lin_acc_b
    out_ang_acc_b[idx] = ang_acc_b
    out_projected_gravity_b[idx] = projected_gravity_b

    # Update previous velocities
    prev_lin_vel_w[idx] = lin_vel_w
    prev_ang_vel_w[idx] = ang_vel_w


@wp.kernel
def imu_reset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    out_pos_w: wp.array(dtype=wp.vec3f),
    out_quat_w: wp.array(dtype=wp.quatf),
    out_lin_vel_b: wp.array(dtype=wp.vec3f),
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
    out_ang_acc_b: wp.array(dtype=wp.vec3f),
    out_projected_gravity_b: wp.array(dtype=wp.vec3f),
    prev_lin_vel_w: wp.array(dtype=wp.vec3f),
    prev_ang_vel_w: wp.array(dtype=wp.vec3f),
):
    idx = wp.tid()
    if not env_mask[idx]:
        return

    out_pos_w[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_quat_w[idx] = wp.quatf(0.0, 0.0, 0.0, 1.0)
    out_lin_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_ang_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_lin_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_ang_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_projected_gravity_b[idx] = wp.vec3f(0.0, 0.0, -1.0)
    prev_lin_vel_w[idx] = wp.vec3f(0.0, 0.0, 0.0)
    prev_ang_vel_w[idx] = wp.vec3f(0.0, 0.0, 0.0)
