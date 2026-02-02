# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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


@wp.kernel
def apply_gravity_compensation_force(
    body_mass: wp.array2d(dtype=wp.float32),
    gravcomp: wp.array2d(dtype=wp.float32),
    gravity: wp.vec3f,
    body_f: wp.array2d(dtype=wp.spatial_vectorf),
    root_link_index: wp.int32,
    root_comp_force: wp.array2d(dtype=wp.float32),
):
    """Apply gravity compensation force to body_f.

    The compensation force is: F = m * (-g) * gravcomp_factor
    where m is body mass, g is gravity vector, and gravcomp_factor is the compensation factor.
    The compensated weight is accumulated atomically into root_comp_force for later application to the root link.

    Args:
        root_comp_force: Accumulator array of shape (num_envs, 3) for x, y, z components.
            Must be zeroed before calling this kernel.
    """
    env_index, body_index = wp.tid()
    mass = body_mass[env_index, body_index]
    comp_factor = gravcomp[env_index, body_index]
    # Calculate the opposite of gravity force scaled by compensation factor
    comp_force = wp.vec3f(
        -gravity[0] * mass * comp_factor, -gravity[1] * mass * comp_factor, -gravity[2] * mass * comp_factor
    )
    # Add to existing wrench (keep torque, add to force)
    current_wrench = body_f[env_index, body_index]
    current_force = wp.spatial_top(current_wrench)
    new_force = wp.vec3f(
        current_force[0] + comp_force[0], current_force[1] + comp_force[1], current_force[2] + comp_force[2]
    )
    body_f[env_index, body_index] = wp.spatial_vector(new_force, wp.spatial_bottom(current_wrench), wp.float32)

    # Atomically accumulate the negative compensation force for the root link
    wp.atomic_add(root_comp_force, env_index, 0, -comp_force[0])
    wp.atomic_add(root_comp_force, env_index, 1, -comp_force[1])
    wp.atomic_add(root_comp_force, env_index, 2, -comp_force[2])


@wp.kernel
def apply_accumulated_root_force(
    root_comp_force: wp.array2d(dtype=wp.float32),
    body_f: wp.array2d(dtype=wp.spatial_vectorf),
    root_link_index: wp.int32,
):
    """Apply the accumulated compensation force to the root link.

    This kernel should be called after apply_gravity_compensation_force to apply
    the atomically accumulated forces to the root link.

    Args:
        root_comp_force: Accumulator array of shape (num_envs, 3) populated by
            apply_gravity_compensation_force.
    """
    env_index = wp.tid()
    current_wrench = body_f[env_index, root_link_index]
    current_force = wp.spatial_top(current_wrench)
    accumulated = wp.vec3f(root_comp_force[env_index, 0], root_comp_force[env_index, 1], root_comp_force[env_index, 2])
    new_force = wp.vec3f(
        current_force[0] + accumulated[0], current_force[1] + accumulated[1], current_force[2] + accumulated[2]
    )
    body_f[env_index, root_link_index] = wp.spatial_vector(new_force, wp.spatial_bottom(current_wrench), wp.float32)
