# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for Newton deformable object gather/scatter operations."""

import warp as wp
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    evaluate_body_particle_contact as _evaluate_body_particle_contact,
)

vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def gather_particles_vec3f(
    src: wp.array(dtype=wp.vec3f),
    offsets: wp.array(dtype=wp.int32),
    num_particles: int,
    dst: wp.array2d(dtype=wp.vec3f),
):
    """Gather particle data from a flat array into a per-instance 2D array.

    Args:
        src: Flat source particle array (all instances concatenated). Shape is (total_particles,).
        offsets: Per-instance start offset into the flat array. Shape is (num_instances,).
        num_particles: Number of particles per instance.
        dst: Output 2D array. Shape is (num_instances, num_particles).
    """
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
    """Scatter per-instance particle data into the flat simulation array using indices.

    Args:
        src: Input 2D array. Shape is (len(env_ids), num_particles) or (num_instances, num_particles).
        env_ids: Environment indices to scatter to. Shape is (num_selected,).
        offsets: Per-instance start offset into the flat array. Shape is (num_instances,).
        full_data: If True, index src with env_ids[i]; otherwise index src with i.
        dst: Flat destination particle array. Shape is (total_particles,).
    """
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
    """Scatter per-instance particle data into the flat simulation array using a mask.

    Args:
        src: Input particle data [m] or [m/s]. Shape is (num_instances, num_particles).
        env_mask: Environment mask. Shape is (num_instances,).
        offsets: Per-instance start offset into the flat array. Shape is (num_instances,).
        dst: Flat destination particle array. Shape is (total_particles,).
    """
    i, j = wp.tid()
    if env_mask[i]:
        dst[offsets[i] + j] = src[i, j]


@wp.kernel
def scatter_particles_state_vec6f_mask(
    src: wp.array2d(dtype=vec6f),
    env_mask: wp.array(dtype=wp.bool),
    offsets: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3f),
    particle_qd: wp.array(dtype=wp.vec3f),
):
    """Scatter per-instance nodal state into the flat simulation arrays using a mask.

    Args:
        src: Input nodal state data [m, m/s]. Shape is (num_instances, num_particles).
        env_mask: Environment mask. Shape is (num_instances,).
        offsets: Per-instance start offset into the flat arrays. Shape is (num_instances,).
        particle_q: Flat destination particle positions [m]. Shape is (total_particles,).
        particle_qd: Flat destination particle velocities [m/s]. Shape is (total_particles,).
    """
    i, j = wp.tid()
    if env_mask[i]:
        state = src[i, j]
        flat_idx = offsets[i] + j
        particle_q[flat_idx] = wp.vec3f(state[0], state[1], state[2])
        particle_qd[flat_idx] = wp.vec3f(state[3], state[4], state[5])


@wp.kernel
def write_nodal_kinematic_target_mask(
    src: wp.array2d(dtype=wp.vec4f),
    env_mask: wp.array(dtype=wp.bool),
    dst: wp.array2d(dtype=wp.vec4f),
):
    """Write kinematic target data into the per-instance target buffer using a mask.

    Args:
        src: Input kinematic targets [m]. Shape is (num_instances, num_particles).
        env_mask: Environment mask. Shape is (num_instances,).
        dst: Destination kinematic target buffer [m]. Shape is (num_instances, num_particles).
    """
    i, j = wp.tid()
    if env_mask[i]:
        dst[i, j] = src[i, j]


@wp.kernel
def write_nodal_kinematic_target_index(
    src: wp.array2d(dtype=wp.vec4f),
    env_ids: wp.array(dtype=wp.int32),
    full_data: bool,
    dst: wp.array2d(dtype=wp.vec4f),
):
    """Write kinematic target data into the per-instance target buffer using indices.

    Args:
        src: Input kinematic targets [m]. Shape is (len(env_ids), num_particles)
            or (num_instances, num_particles).
        env_ids: Environment indices to write. Shape is (num_selected,).
        full_data: If True, index src with env_ids[i]; otherwise index src with i.
        dst: Destination kinematic target buffer [m]. Shape is (num_instances, num_particles).
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    if full_data:
        dst[env_id, j] = src[env_id, j]
    else:
        dst[env_id, j] = src[i, j]


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
def scatter_zero_vel_index(
    env_ids: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    num_particles: int,
    dst: wp.array(dtype=wp.vec3f),
):
    """Zero the velocity of particles for selected environments.

    Args:
        env_ids: Environment indices to zero velocities for. Shape is (num_selected,).
        offsets: Per-instance start offset into the flat array. Shape is (num_instances,).
        num_particles: Number of particles per instance.
        dst: Flat destination velocity array. Shape is (total_particles,).
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    dst[offsets[env_id] + j] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def scatter_default_pos_index(
    default_pos: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.vec3f),
):
    """Scatter default positions for selected environments into the flat simulation array.

    Args:
        default_pos: Default positions per instance. Shape is (num_instances, num_particles).
        env_ids: Environment indices to reset. Shape is (num_selected,).
        offsets: Per-instance start offset into the flat array. Shape is (num_instances,).
        dst: Flat destination particle array. Shape is (total_particles,).
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    dst[offsets[env_id] + j] = default_pos[env_id, j]


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


@wp.kernel
def enforce_kinematic_targets(
    targets: wp.array2d(dtype=wp.vec4f),
    offsets: wp.array(dtype=wp.int32),
    default_inv_mass: wp.array(dtype=wp.float32),
    default_flags: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3f),
    particle_qd: wp.array(dtype=wp.vec3f),
    particle_inv_mass: wp.array(dtype=wp.float32),
    particle_flags: wp.array(dtype=wp.int32),
):
    """Enforce kinematic targets on Newton particles.

    For each particle, reads the kinematic target flag (w-component):
    - flag == 0.0 (kinematic): set inv_mass to 0, particle_flags to 0, write target position, zero velocity.
    - flag != 0.0 (free): restore the default inv_mass and set particle_flags to 1 (ACTIVE).

    Args:
        targets: Per-instance kinematic targets. Shape is (num_instances, particles_per_body).
            Each vec4f contains (target_x, target_y, target_z, flag).
        offsets: Per-instance start offset into the flat particle array.
        default_inv_mass: Saved default inverse masses. Shape is (total_particles,).
        default_flags: Saved default particle flags. Shape is (total_particles,).
        particle_q: Flat particle positions to write. Shape is (total_particles,).
        particle_qd: Flat particle velocities to write. Shape is (total_particles,).
        particle_inv_mass: Flat particle inverse masses to write. Shape is (total_particles,).
        particle_flags: Flat particle flags to write. Shape is (total_particles,).
            0 = kinematic (solver skips integration), 1 = ACTIVE.
    """
    i, j = wp.tid()
    t = targets[i, j]
    flat_idx = offsets[i] + j
    flag = t[3]
    if flag == 0.0:
        particle_inv_mass[flat_idx] = 0.0
        particle_flags[flat_idx] = 0
        particle_q[flat_idx] = wp.vec3f(t[0], t[1], t[2])
        particle_qd[flat_idx] = wp.vec3f(0.0, 0.0, 0.0)
    else:
        particle_inv_mass[flat_idx] = default_inv_mass[flat_idx]
        particle_flags[flat_idx] = default_flags[flat_idx]


@wp.kernel
def _kernel_body_particle_reaction(
    contact_count: wp.array(dtype=wp.int32),
    contact_particle: wp.array(dtype=wp.int32),
    contact_shape: wp.array(dtype=wp.int32),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    shape_material_mu: wp.array(dtype=wp.float32),
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    friction_epsilon: float,
    dt: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Newton's-third-law reaction from soft particles onto rigid bodies.

    Delegates to Newton's ``evaluate_body_particle_contact()`` for the contact
    force computation (normal + damping + Coulomb friction) so the model stays
    in sync with the VBD solver. The force on the particle is negated and
    applied as a wrench on the rigid body via ``body_f``.

    One thread per contact slot; threads beyond the actual contact count
    early-exit.

    The "previous" particle position required by the contact model is
    reconstructed from the current velocity (``particle_q - particle_qd * dt``)
    rather than read from a stored previous-state array. VBD mutates
    ``particle_q`` in place during its iteration, so the swapped state's
    ``particle_q`` is no longer a reliable snapshot of the prior substep.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    p_idx = contact_particle[tid]
    s_idx = contact_shape[tid]
    body_idx = shape_body[s_idx]
    if body_idx < 0:
        return

    # Reconstruct previous particle position from velocity so that
    # dx = particle_qd * dt regardless of what VBD wrote into stored states.
    p_pos = particle_q[p_idx]
    p_pos_prev = p_pos - particle_qd[p_idx] * dt

    # Delegate to Newton's canonical contact model
    f_on_particle, _ = _evaluate_body_particle_contact(
        p_idx,
        p_pos,
        p_pos_prev,
        tid,
        soft_contact_ke,
        soft_contact_kd,
        soft_contact_mu,
        friction_epsilon,
        particle_radius,
        shape_material_mu,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        contact_shape,
        contact_body_pos,
        contact_body_vel,
        contact_normal,
        dt,
    )

    # Newton's third law: negate particle force → rigid body wrench
    X_wb = body_q[body_idx]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    com_w = wp.transform_point(X_wb, body_com[body_idx])

    reaction = -f_on_particle
    torque = wp.cross(bx - com_w, reaction)

    wp.atomic_add(
        body_f,
        body_idx,
        wp.spatial_vector(
            reaction[0],
            reaction[1],
            reaction[2],
            torque[0],
            torque[1],
            torque[2],
        ),
    )
