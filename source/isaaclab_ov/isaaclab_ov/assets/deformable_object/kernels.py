# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for OVPhysX deformable object buffers."""

import warp as wp

vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def write_nodal_vec3f_to_buffer(
    data: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    full_data: bool,
    out_data: wp.array2d(dtype=wp.vec3f),
):
    """Write selected nodal vectors into a complete body-state buffer.

    Args:
        data: Input nodal positions [m] or velocities [m/s].
        env_ids: Environment indices selected for the write.
        full_data: Whether :paramref:`data` contains rows for every environment.
        out_data: Complete output nodal buffer [m] or [m/s].
    """
    i, j = wp.tid()
    out_data[env_ids[i], j] = data[env_ids[i], j] if full_data else data[i, j]


@wp.kernel
def write_nodal_vec4f_to_buffer(
    data: wp.array2d(dtype=wp.vec4f),
    env_ids: wp.array(dtype=wp.int32),
    full_data: bool,
    out_data: wp.array2d(dtype=wp.vec4f),
):
    """Write selected kinematic targets into a complete target buffer.

    Args:
        data: Input nodal kinematic targets [m, dimensionless].
        env_ids: Environment indices selected for the write.
        full_data: Whether :paramref:`data` contains rows for every environment.
        out_data: Complete output kinematic-target buffer [m, dimensionless].
    """
    i, j = wp.tid()
    out_data[env_ids[i], j] = data[env_ids[i], j] if full_data else data[i, j]


@wp.kernel
def compute_nodal_state_w(
    nodal_pos: wp.array2d(dtype=wp.vec3f),
    nodal_vel: wp.array2d(dtype=wp.vec3f),
    nodal_state: wp.array2d(dtype=vec6f),
):
    """Combine nodal positions and velocities into six-component states.

    Args:
        nodal_pos: Nodal positions in simulation world frame [m].
        nodal_vel: Nodal velocities in simulation world frame [m/s].
        nodal_state: Output nodal state in simulation world frame [m, m/s].
    """
    i, j = wp.tid()
    position = nodal_pos[i, j]
    velocity = nodal_vel[i, j]
    nodal_state[i, j] = vec6f(position[0], position[1], position[2], velocity[0], velocity[1], velocity[2])


@wp.kernel
def compute_mean_vec3f_over_vertices(
    data: wp.array2d(dtype=wp.vec3f),
    num_vertices: int,
    result: wp.array(dtype=wp.vec3f),
):
    """Compute each body's mean vector over its simulation vertices.

    Args:
        data: Input nodal vectors [m] or [m/s].
        num_vertices: Number of simulation vertices per body.
        result: Output per-body mean vectors [m] or [m/s].
    """
    i = wp.tid()
    accumulator = wp.vec3f(0.0, 0.0, 0.0)
    for vertex_index in range(num_vertices):
        accumulator = accumulator + data[i, vertex_index]
    result[i] = accumulator / float(num_vertices)


@wp.kernel
def set_kinematic_flags_to_one(data: wp.array(dtype=wp.vec4f)):
    """Mark every kinematic target as a free simulation node.

    Args:
        data: Flattened nodal kinematic targets [m, dimensionless].
    """
    i = wp.tid()
    target = data[i]
    data[i] = wp.vec4f(target[0], target[1], target[2], 1.0)
