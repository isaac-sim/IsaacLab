# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous particle-surface pad contact, independent of grid-node inclusion."""

import warp as wp

from .mpm.hand_contact import local_velocity, surface_local


@wp.kernel
def pad_contact(
    x: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    following: wp.array[wp.transform],
    kind: wp.array[int],
    sizes: wp.array[wp.vec3],
    friction: wp.array[float],
    tangential_displacement: wp.array2d[wp.vec3],
    impulse: wp.array[wp.vec3],
    mass: float,
    spacing: float,
    young: float,
    dt: float,
):
    p = wp.tid()
    v = velocity[p]
    # A particle represents a finite tissue volume, not a zero-radius point.
    radius = 0.5 * spacing
    stiffness = young * spacing
    damping = wp.sqrt(mass * stiffness)
    for collider in range(poses.shape[0]):
        pose = poses[collider]
        bound = wp.length(sizes[collider]) + radius
        if kind[collider] == 1:
            bound = sizes[collider][0] + sizes[collider][1] + radius
        if wp.length_sq(x[p] - wp.transform_get_translation(pose)) > bound * bound:
            tangential_displacement[p, collider] = wp.vec3(0.0)
            continue
        local = wp.transform_point(wp.transform_inverse(pose), x[p])
        surface = surface_local(local, kind[collider], sizes[collider])
        depth = radius - surface[3]
        if depth > 0.0:
            normal = wp.transform_vector(pose, wp.vec3(surface[0], surface[1], surface[2]))
            relative = v - local_velocity(local, pose, following[collider], dt)
            vn = wp.dot(relative, normal)
            # Unilateral contact: no tensile force and no unloaded tangential grip.
            normal_impulse = dt * wp.max(0.0, stiffness * depth - damping * vn)
            tangent = relative - vn * normal
            # Elastic tangential displacement supports static friction; sliding is
            # returned to the Coulomb cone. History is local to the moving pad.
            displacement = wp.transform_vector(pose, tangential_displacement[p, collider]) + dt * tangent
            displacement -= wp.dot(displacement, normal) * normal
            tangent_force = -0.5 * stiffness * displacement - 0.5 * damping * tangent
            magnitude = wp.length(tangent_force)
            limit = friction[p] * normal_impulse / dt
            if magnitude > limit:
                tangent_force *= limit / wp.max(magnitude, 1.0e-12)
                displacement = -(tangent_force + 0.5 * damping * tangent) / (0.5 * stiffness)
            if limit <= 0.0:
                displacement = wp.vec3(0.0)
            tangential_displacement[p, collider] = wp.transform_vector(wp.transform_inverse(pose), displacement)
            change = normal_impulse * normal + dt * tangent_force
            v += change / mass
            wp.atomic_add(impulse, collider, change)
        else:
            tangential_displacement[p, collider] = wp.vec3(0.0)
    velocity[p] = v
