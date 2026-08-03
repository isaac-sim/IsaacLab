# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def pva_update_kernel(
    # inputs
    env_mask: wp.array(dtype=wp.bool),
    transforms: wp.array(dtype=wp.transformf),
    velocities: wp.array(dtype=wp.spatial_vectorf),
    coms: wp.array(dtype=wp.transformf),
    offset_pos_b: wp.array(dtype=wp.vec3f),
    offset_quat_b: wp.array(dtype=wp.quatf),
    gravity_vec_w: wp.array(dtype=wp.vec3f),
    inv_dt: wp.float32,
    timestamp: wp.array(dtype=wp.float32),
    # inputs / outputs
    prev_lin_vel_w: wp.array(dtype=wp.vec3f),
    prev_ang_vel_w: wp.array(dtype=wp.vec3f),
    # outputs
    out_pos_w: wp.array(dtype=wp.vec3f),
    out_quat_w: wp.array(dtype=wp.quatf),
    out_lin_vel_b: wp.array(dtype=wp.vec3f),
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
    out_ang_acc_b: wp.array(dtype=wp.vec3f),
    out_projected_gravity_b: wp.array(dtype=wp.vec3f),
):
    """Update the PVA sensor data.

    Args:
        env_mask: Mask of environments to update.
        transforms: Transforms of the bodies.
        velocities: Velocities of the bodies.
        coms: COMs of the bodies.
        offset_pos_b: Offset positions of the sensors.
        offset_quat_b: Offset quaternions of the sensors.
        gravity_vec_w: Gravity direction unit vector in the world frame.
        inv_dt: Inverse of the time step.
        timestamp: Timestamp of the environment.
        prev_lin_vel_w: Previous linear velocity in the world frame.
        prev_ang_vel_w: Previous angular velocity in the world frame.
        out_pos_w: Output position in the world frame.
        out_quat_w: Output orientation in the world frame.
        out_lin_vel_b: Output linear velocity in the body frame.
        out_ang_vel_b: Output angular velocity in the body frame.
        out_lin_acc_b: Output linear acceleration in the body frame.
        out_ang_acc_b: Output angular acceleration in the body frame.
        out_projected_gravity_b: Output projected gravity in the body frame.
    """
    idx = wp.tid()
    if not env_mask[idx]:
        return

    # Skip envs that have not been stepped since their last reset: PhysX velocities still
    # hold pre-reset values, so finite-difference acceleration would be spurious.
    if timestamp[idx] == 0.0:
        return

    body_pos = wp.transform_get_translation(transforms[idx])
    body_quat = wp.transform_get_rotation(transforms[idx])

    sensor_pos = body_pos + wp.quat_rotate(body_quat, offset_pos_b[idx])
    sensor_quat = body_quat * offset_quat_b[idx]

    lin_vel_w = wp.spatial_top(velocities[idx])
    ang_vel_w = wp.spatial_bottom(velocities[idx])

    com_pos_b = wp.transform_get_translation(coms[idx])
    lever_arm = wp.quat_rotate(body_quat, offset_pos_b[idx] - com_pos_b)
    lin_vel_w = lin_vel_w + wp.cross(ang_vel_w, lever_arm)

    lin_acc_w = (lin_vel_w - prev_lin_vel_w[idx]) * inv_dt
    ang_acc_w = (ang_vel_w - prev_ang_vel_w[idx]) * inv_dt

    out_pos_w[idx] = sensor_pos
    out_quat_w[idx] = sensor_quat
    out_lin_vel_b[idx] = wp.quat_rotate_inv(sensor_quat, lin_vel_w)
    out_ang_vel_b[idx] = wp.quat_rotate_inv(sensor_quat, ang_vel_w)
    out_lin_acc_b[idx] = wp.quat_rotate_inv(sensor_quat, lin_acc_w)
    out_ang_acc_b[idx] = wp.quat_rotate_inv(sensor_quat, ang_acc_w)
    out_projected_gravity_b[idx] = wp.quat_rotate_inv(sensor_quat, gravity_vec_w[idx])

    # Update previous velocities.
    prev_lin_vel_w[idx] = lin_vel_w
    prev_ang_vel_w[idx] = ang_vel_w


@wp.kernel
def pva_reset_kernel(
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
    """Reset the PVA sensor data.

    Args:
        env_mask: Mask of environments to reset.
        out_pos_w: Output position in the world frame.
        out_quat_w: Output orientation in the world frame.
        out_lin_vel_b: Output linear velocity in the body frame.
        out_ang_vel_b: Output angular velocity in the body frame.
        out_lin_acc_b: Output linear acceleration in the body frame.
        out_ang_acc_b: Output angular acceleration in the body frame.
        out_projected_gravity_b: Output projected gravity in the body frame.
        prev_lin_vel_w: Previous linear velocity in the world frame.
        prev_ang_vel_w: Previous angular velocity in the world frame.
    """
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
