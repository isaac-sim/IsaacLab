# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernel used by the custom coupling manager."""

import warp as wp
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    evaluate_body_particle_contact as _evaluate_body_particle_contact,
)


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
    shape_margin: wp.array(dtype=wp.float32),
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    friction_epsilon: float,
    dt: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Apply body-particle contact reactions to rigid bodies.

    Newton's contact model evaluates normal, damping, and Coulomb friction forces on each particle. This kernel
    applies the equal and opposite force and torque to the contacted rigid body. One thread runs per allocated contact
    slot, and unused slots exit through ``contact_count``. Previous particle positions are reconstructed because VBD
    mutates them in place.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    particle_idx = contact_particle[tid]
    shape_idx = contact_shape[tid]
    body_idx = shape_body[shape_idx]
    if body_idx < 0:
        return

    # VBD mutates particle positions in place, so reconstruct the prior position.
    particle_pos = particle_q[particle_idx]
    particle_pos_prev = particle_pos - particle_qd[particle_idx] * dt
    # Use Newton's contact model to stay consistent with VBD.
    particle_force, _ = _evaluate_body_particle_contact(
        particle_idx,
        particle_pos,
        particle_pos_prev,
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
        shape_margin,
        dt,
    )

    # Apply the equal and opposite particle force as a rigid-body wrench.
    body_pose = body_q[body_idx]
    contact_pos = wp.transform_point(body_pose, contact_body_pos[tid])
    com_pos = wp.transform_point(body_pose, body_com[body_idx])
    reaction = -particle_force
    torque = wp.cross(contact_pos - com_pos, reaction)
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
