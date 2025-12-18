# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel
def update_wrench_array(
    new_value: wp.array2d(dtype=wp.spatial_vectorf),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = new_value[env_index, body_index]


@wp.kernel
def update_wrench_array_with_value(
    value: wp.spatial_vectorf,
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = value


@wp.func
def update_wrench_with_force(
    wrench: wp.spatial_vectorf,
    force: wp.vec3f,
) -> wp.spatial_vectorf:
    return wp.spatial_vector(force, wp.spatial_bottom(wrench), wp.float32)


@wp.func
def update_wrench_with_torque(
    wrench: wp.spatial_vectorf,
    torque: wp.vec3f,
) -> wp.spatial_vectorf:
    return wp.spatial_vector(wp.spatial_top(wrench), torque, wp.float32)


@wp.kernel
def update_wrench_array_with_force(
    forces: wp.array2d(dtype=wp.vec3f),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = update_wrench_with_force(
            wrench[env_index, body_index], forces[env_index, body_index]
        )


@wp.kernel
def update_wrench_array_with_torque(
    torques: wp.array2d(dtype=wp.vec3f),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = update_wrench_with_torque(
            wrench[env_index, body_index], torques[env_index, body_index]
        )


@wp.kernel
def generate_mask_from_ids(
    mask: wp.array(dtype=wp.bool),
    ids: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    mask[ids[index]] = True


@wp.kernel
def populate_empty_array(
    input_array: wp.array(dtype=wp.float32),
    output_array: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    output_array[indices[index]] = input_array[index]
