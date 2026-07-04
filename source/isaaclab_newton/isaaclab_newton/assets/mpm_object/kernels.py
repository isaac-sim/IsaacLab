# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for Newton MPM object gather/scatter operations."""

import warp as wp

vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def gather_particles_vec3f(
    src: wp.array(dtype=wp.vec3f),
    offsets: wp.array(dtype=wp.int32),
    dst: wp.array2d(dtype=wp.vec3f),
):
    i, j = wp.tid()
    dst[i, j] = src[offsets[i] + j]


@wp.kernel
def scatter_particles_vec3f_index(
    src: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    full_data: bool,
    dst: wp.array(dtype=wp.vec3f),
):
    i, j = wp.tid()
    env_id = env_ids[i]
    if full_data:
        dst[offsets[env_id] + j] = src[env_id, j]
    else:
        dst[offsets[env_id] + j] = src[i, j]


@wp.kernel
def scatter_particles_vec3f_mask(
    src: wp.array2d(dtype=wp.vec3f),
    env_mask: wp.array(dtype=wp.bool),
    offsets: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.vec3f),
):
    i, j = wp.tid()
    if env_mask[i]:
        dst[offsets[i] + j] = src[i, j]


@wp.kernel
def scatter_particles_state_vec6f_index(
    src: wp.array2d(dtype=vec6f),
    env_ids: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    full_data: bool,
    particle_q: wp.array(dtype=wp.vec3f),
    particle_qd: wp.array(dtype=wp.vec3f),
):
    i, j = wp.tid()
    env_id = env_ids[i]
    src_id = env_id if full_data else i
    state = src[src_id, j]
    flat_idx = offsets[env_id] + j
    particle_q[flat_idx] = wp.vec3f(state[0], state[1], state[2])
    particle_qd[flat_idx] = wp.vec3f(state[3], state[4], state[5])


@wp.kernel
def scatter_particles_state_vec6f_mask(
    src: wp.array2d(dtype=vec6f),
    env_mask: wp.array(dtype=wp.bool),
    offsets: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3f),
    particle_qd: wp.array(dtype=wp.vec3f),
):
    i, j = wp.tid()
    if env_mask[i]:
        state = src[i, j]
        flat_idx = offsets[i] + j
        particle_q[flat_idx] = wp.vec3f(state[0], state[1], state[2])
        particle_qd[flat_idx] = wp.vec3f(state[3], state[4], state[5])


@wp.kernel
def compute_particle_state_w(
    particle_pos: wp.array2d(dtype=wp.vec3f),
    particle_vel: wp.array2d(dtype=wp.vec3f),
    particle_state: wp.array2d(dtype=vec6f),
):
    i, j = wp.tid()
    p = particle_pos[i, j]
    v = particle_vel[i, j]
    particle_state[i, j] = vec6f(p[0], p[1], p[2], v[0], v[1], v[2])


@wp.kernel
def compute_mean_vec3f_over_particles(
    data: wp.array2d(dtype=wp.vec3f),
    num_particles: int,
    result: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    acc = wp.vec3f(0.0, 0.0, 0.0)
    for j in range(num_particles):
        acc = acc + data[i, j]
    result[i] = acc / float(num_particles)
