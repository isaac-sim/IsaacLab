# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Weak, breakable contact bonds activated by physical crushing damage.

Forces act on MPM particles before P2G. No rendering state enters this model.
This is a demo cohesive-contact approximation, not a calibrated juice model.
"""

import warp as wp


@wp.func
def activation(damage: float):
    t = wp.clamp((damage - 0.1) / 0.5, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def separation(delta: wp.vec3, normal: wp.vec3):
    # Hard contact handles compression; adhesion resists opening and slip.
    return delta - wp.min(wp.dot(delta, normal), 0.0) * normal


@wp.func
def cohesive_force(delta: wp.vec3, peak_distance: float, reach: float, maximum_force: float):
    # Bilinear traction: peak at 20% reach, then irreversible softening.
    s = peak_distance / reach
    envelope = wp.min(s / 0.2, (1.0 - s) / 0.8)
    return -maximum_force * wp.max(envelope, 0.0) * delta / wp.max(peak_distance, 1.0e-12)


@wp.kernel
def apply_contact(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    damage: wp.array[float],
    cube: wp.array[wp.vec3],
    half: float,
    use_cube: int,
    use_plane: int,
    tool_anchor: wp.array[wp.vec3],
    tool_face: wp.array[int],
    tool_peak: wp.array[float],
    plane_anchor: wp.array[wp.vec3],
    plane_active: wp.array[int],
    plane_peak: wp.array[float],
    tool_force: wp.array[wp.vec3],
    plane_force: wp.array[wp.vec3],
    tool_age: wp.array[float],
    strength: float,
    reach: float,
    spacing: float,
    mass: float,
    dt: float,
    lifetime: float,
    plane_scale: float,
    particle_weight: wp.array[float],
):
    p = wp.tid()
    spacing = spacing * wp.pow(particle_weight[p], 1.0 / 3.0)
    mass = mass * particle_weight[p]
    wet = activation(damage[p])
    ft, fp = wp.vec3(0.0), wp.vec3(0.0)
    cap = strength * spacing * spacing * wet
    tolerance = 0.1 * spacing
    if use_cube == 0:
        tool_face[p] = 0
    if use_plane == 0:
        plane_active[p] = 0
    if wet <= 0.0:
        tool_face[p] = 0
        plane_active[p] = 0
    else:
        if use_cube != 0:
            local = x[p] - cube[0]
            if tool_face[p] == 0:
                excess = wp.vec3(wp.abs(local[0]) - half, wp.abs(local[1]) - half, wp.abs(local[2]) - half)
                outside = wp.vec3(wp.max(excess[0], 0.0), wp.max(excess[1], 0.0), wp.max(excess[2], 0.0))
                axis = int(0)
                if excess[1] > excess[axis]:
                    axis = 1
                if excess[2] > excess[axis]:
                    axis = 2
                if tool_age[p] >= 1.0 and wp.length(outside) > 2.0 * tolerance:
                    tool_age[p] = 0.0
                if tool_age[p] < 1.0 and wp.length(outside) <= tolerance and excess[axis] >= -tolerance:
                    tool_face[p] = (axis + 1) * wp.where(local[axis] >= 0.0, 1, -1)
                    tool_anchor[p] = local
                    tool_peak[p] = 0.0
                    tool_age[p] = 0.0
            if tool_face[p] != 0:
                normal = wp.vec3(0.0)
                normal[wp.abs(tool_face[p]) - 1] = wp.where(tool_face[p] > 0, 1.0, -1.0)
                delta = separation(local - tool_anchor[p], normal)
                peak = wp.max(tool_peak[p], wp.length(delta))
                # Once peeling starts, unloading cannot pause bond expiry.
                if lifetime > 0.0 and (tool_age[p] > 0.0 or wp.length(delta) > 1.0e-6):
                    tool_age[p] += dt / lifetime
                if peak >= reach or tool_age[p] >= 1.0:
                    tool_face[p] = 0
                    tool_peak[p] = 0.0
                    tool_age[p] = 1.0
                else:
                    tool_peak[p] = peak
                    ft = cohesive_force(delta, peak, reach, cap * (1.0 - tool_age[p]) * (1.0 - tool_age[p]))
        if use_plane != 0:
            if plane_active[p] == 0 and x[p][2] <= 0.5 * spacing + tolerance:
                plane_anchor[p] = x[p]
                plane_active[p] = 1
                plane_peak[p] = 0.0
            if plane_active[p] != 0:
                delta = separation(x[p] - plane_anchor[p], wp.vec3(0.0, 0.0, 1.0))
                peak = wp.max(plane_peak[p], wp.length(delta))
                if peak >= reach:
                    plane_active[p] = 0
                    plane_peak[p] = 0.0
                else:
                    plane_peak[p] = peak
                    fp = cohesive_force(delta, peak, reach, cap * plane_scale)
    tool_force[p] = ft
    plane_force[p] = fp
    v[p] += dt / mass * (ft + fp)


class AdhesiveContact:
    def __init__(self, count, strength, reach, spacing, mass, lifetime=0.0, plane_scale=1.0):
        self.strength, self.reach, self.spacing, self.mass = strength, reach, spacing, mass
        self.lifetime, self.plane_scale = lifetime, plane_scale
        self.particle_weight = wp.ones(count, dtype=float)
        self.tool_anchor = wp.zeros(count, dtype=wp.vec3)
        self.tool_face = wp.zeros(count, dtype=int)
        self.tool_peak = wp.zeros(count, dtype=float)
        self.plane_anchor = wp.zeros(count, dtype=wp.vec3)
        self.plane_active = wp.zeros(count, dtype=int)
        self.plane_peak = wp.zeros(count, dtype=float)
        self.tool_force = wp.zeros(count, dtype=wp.vec3)
        self.plane_force = wp.zeros(count, dtype=wp.vec3)
        self.tool_age = wp.zeros(count, dtype=float)
        self.arrays = (
            self.tool_anchor,
            self.tool_face,
            self.tool_peak,
            self.plane_anchor,
            self.plane_active,
            self.plane_peak,
            self.tool_force,
            self.plane_force,
            self.tool_age,
        )

    def step(self, x, v, damage, cube, half, use_cube, use_plane, dt):
        wp.launch(
            apply_contact,
            dim=len(x),
            inputs=[
                x,
                v,
                damage,
                cube,
                half,
                use_cube,
                use_plane,
                *self.arrays,
                self.strength,
                self.reach,
                self.spacing,
                self.mass,
                dt,
                self.lifetime,
                self.plane_scale,
                self.particle_weight,
            ],
        )
