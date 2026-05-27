# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def imu_copy_kernel(
    env_mask: wp.array(dtype=wp.bool),
    accelerometer: wp.array(dtype=wp.vec3f),
    gyroscope: wp.array(dtype=wp.vec3f),
    timestamp: wp.array(dtype=wp.float32),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
):
    idx = wp.tid()
    if not env_mask[idx]:
        return
    # Skip envs that have not been stepped since their last reset: Newton's sensor outputs
    # still reflect pre-reset state, so reading them now would inject stale data (#4970).
    if timestamp[idx] == 0.0:
        return
    out_lin_acc_b[idx] = accelerometer[idx]
    out_ang_vel_b[idx] = gyroscope[idx]


@wp.kernel
def imu_reset_kernel(
    env_mask: wp.array(dtype=wp.bool),
    out_lin_acc_b: wp.array(dtype=wp.vec3f),
    out_ang_vel_b: wp.array(dtype=wp.vec3f),
):
    idx = wp.tid()
    if not env_mask[idx]:
        return
    out_lin_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
    out_ang_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)
