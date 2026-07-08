# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def imu_update_kernel(
    # inputs
    env_mask: wp.array(dtype=wp.bool),
    transforms: wp.array(dtype=wp.transformf),
    velocities: wp.array(dtype=wp.spatial_vectorf),
    coms: wp.array(dtype=wp.transformf),
    offset_pos_b: wp.array(dtype=wp.vec3f),
    offset_quat_b: wp.array(dtype=wp.quatf),
    gravity_bias_w: wp.array(dtype=wp.vec3f),
    inv_dt: wp.float32,
    timestamp: wp.array(dtype=wp.float32),
    # inputs / outputs
    prev_lin_vel_w: wp.array(dtype=wp.vec3f),
    # outputs
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
):
    """Update the IMU sensor data.

    Args:
        env_mask: Mask of environments to update.
        transforms: Transforms of the bodies.
        velocities: Velocities of the bodies.
        coms: COMs of the bodies.
        offset_pos_b: Offset positions of the sensors.
        offset_quat_b: Offset quaternions of the sensors.
        gravity_bias_w: Gravity bias in the world frame.
        inv_dt: Inverse of the time step.
        timestamp: Timestamp of the environment.
        prev_lin_vel_w: Previous linear velocity in the world frame.
        out_ang_vel_b: Output angular velocity in the body frame.
        out_lin_acc_b: Output linear acceleration in the body frame.
    """
    idx = wp.tid()
    if not env_mask[idx]:
        return

    # Skip envs that have not been stepped since their last reset: PhysX velocities still
    # hold pre-reset values, so the finite-difference acceleration would be spurious.
    if timestamp[idx] == 0.0:
        return

    body_quat = wp.transform_get_rotation(transforms[idx])

    lin_vel_w = wp.spatial_top(velocities[idx])
    ang_vel_w = wp.spatial_bottom(velocities[idx])

    com_pos_b = wp.transform_get_translation(coms[idx])
    lever_arm = wp.quat_rotate(body_quat, offset_pos_b[idx] - com_pos_b)
    lin_vel_w = lin_vel_w + wp.cross(ang_vel_w, lever_arm)
    lin_acc_w = (lin_vel_w - prev_lin_vel_w[idx]) * inv_dt + gravity_bias_w[idx]

    sensor_quat = body_quat * offset_quat_b[idx]
    out_ang_vel_b[idx] = wp.quat_rotate_inv(sensor_quat, ang_vel_w)
    out_lin_acc_b[idx] = wp.quat_rotate_inv(sensor_quat, lin_acc_w)

    # Update previous velocity
    prev_lin_vel_w[idx] = lin_vel_w


@wp.kernel
def imu_reset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
    prev_lin_vel_w: wp.array(dtype=wp.vec3f),
):
    """Reset the IMU sensor data.

    Args:
        env_mask: Mask of environments to reset.
        out_ang_vel_b: Output angular velocity in the body frame.
        out_lin_acc_b: Output linear acceleration in the body frame.
        prev_lin_vel_w: Previous linear velocity in the world frame.
    """
    idx = wp.tid()
    if not env_mask[idx]:
        return

    out_ang_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_lin_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    prev_lin_vel_w[idx] = wp.vec3f(0.0, 0.0, 0.0)
