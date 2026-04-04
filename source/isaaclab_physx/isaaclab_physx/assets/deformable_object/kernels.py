# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def write_nodal_vec3f_to_buffer(
    data: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    out_data: wp.array2d(dtype=wp.vec3f),
):
    """Write nodal vec3f data (positions or velocities) to a buffer at specified indices.

    Args:
        data: Input array of nodal vec3f data. Shape is (num_envs, num_vertices) or
            (num_selected_envs, num_vertices) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        from_mask: Input flag indicating whether to use masked indexing.
        out_data: Output array where data is written. Shape is (num_envs, num_vertices).
    """
    i, j = wp.tid()
    if from_mask:
        out_data[env_ids[i], j] = data[env_ids[i], j]
    else:
        out_data[env_ids[i], j] = data[i, j]


@wp.kernel
def write_nodal_vec4f_to_buffer(
    data: wp.array2d(dtype=wp.vec4f),
    env_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    out_data: wp.array2d(dtype=wp.vec4f),
):
    """Write nodal vec4f data (kinematic targets) to a buffer at specified indices.

    Args:
        data: Input array of nodal vec4f data. Shape is (num_envs, num_vertices) or
            (num_selected_envs, num_vertices) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        from_mask: Input flag indicating whether to use masked indexing.
        out_data: Output array where data is written. Shape is (num_envs, num_vertices).
    """
    i, j = wp.tid()
    if from_mask:
        out_data[env_ids[i], j] = data[env_ids[i], j]
    else:
        out_data[env_ids[i], j] = data[i, j]


@wp.kernel
def compute_nodal_state_w(
    nodal_pos: wp.array2d(dtype=wp.vec3f),
    nodal_vel: wp.array2d(dtype=wp.vec3f),
    nodal_state: wp.array2d(dtype=vec6f),
):
    """Concatenate nodal positions and velocities into a 6-element state vector.

    Args:
        nodal_pos: Input array of nodal positions. Shape is (num_instances, num_vertices).
        nodal_vel: Input array of nodal velocities. Shape is (num_instances, num_vertices).
        nodal_state: Output array where concatenated state vectors are written.
            Shape is (num_instances, num_vertices).
    """
    i, j = wp.tid()
    p = nodal_pos[i, j]
    v = nodal_vel[i, j]
    nodal_state[i, j] = vec6f(p[0], p[1], p[2], v[0], v[1], v[2])


@wp.kernel
def compute_mean_vec3f_over_vertices(
    data: wp.array2d(dtype=wp.vec3f),
    num_vertices: int,
    result: wp.array(dtype=wp.vec3f),
):
    """Compute the mean of vec3f data over the vertex dimension.

    Args:
        data: Input array of vec3f data. Shape is (num_instances, num_vertices).
        num_vertices: Number of vertices per instance.
        result: Output array where mean values are written. Shape is (num_instances,).
    """
    i = wp.tid()
    acc = wp.vec3f(0.0, 0.0, 0.0)
    for j in range(num_vertices):
        acc = acc + data[i, j]
    result[i] = acc / float(num_vertices)


@wp.kernel
def set_kinematic_flags_to_one(
    data: wp.array(dtype=wp.vec4f),
):
    """Set the w-component (kinematic flag) of all vec4f entries to 1.0.

    This is used to initialize all vertices as non-kinematic (free) nodes.

    Args:
        data: Input/output array of vec4f kinematic targets. Shape is (N*V,).
    """
    i = wp.tid()
    v = data[i]
    data[i] = wp.vec4f(v[0], v[1], v[2], 1.0)
