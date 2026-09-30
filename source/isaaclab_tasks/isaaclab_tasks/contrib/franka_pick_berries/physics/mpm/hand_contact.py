# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moving Allegro box/capsule contacts for physical MPM particles and grid."""

import numpy as np
import warp as wp

from .adhesion import activation, cohesive_force, separation


@wp.func
def surface_local(x: wp.vec3, kind: int, size: wp.vec3):
    normal = wp.vec3(0.0)
    distance = float(0.0)
    if kind == 1:
        delta = x - wp.vec3(0.0, 0.0, wp.clamp(x[2], -size[1], size[1]))
        length = wp.length(delta)
        normal = delta / wp.max(length, 1.0e-12)
        distance = length - size[0]
    else:
        excess = wp.vec3(wp.abs(x[0]) - size[0], wp.abs(x[1]) - size[1], wp.abs(x[2]) - size[2])
        outside = wp.vec3(wp.max(excess[0], 0.0), wp.max(excess[1], 0.0), wp.max(excess[2], 0.0))
        length = wp.length(outside)
        axis = int(0)
        if excess[1] > excess[axis]:
            axis = 1
        if excess[2] > excess[axis]:
            axis = 2
        distance = length + wp.min(excess[axis], 0.0)
        if length > 1.0e-12:
            normal = wp.cw_mul(outside / length, wp.vec3(wp.sign(x[0]), wp.sign(x[1]), wp.sign(x[2])))
        else:
            normal[axis] = wp.where(x[axis] >= 0.0, 1.0, -1.0)
    return wp.vec4(normal[0], normal[1], normal[2], distance)


@wp.func
def project_velocity(v: wp.vec3, normal: wp.vec3, friction: float):
    vn = wp.dot(v, normal)
    if vn < 0.0:
        tangent = v - vn * normal
        return tangent * wp.max(0.0, 1.0 + friction * vn / wp.max(wp.length(tangent), 1.0e-12))
    return v


@wp.func
def interpolate_pose(samples: wp.array2d[wp.transform], i: int, t: float):
    index = wp.min(int(t), samples.shape[0] - 2)
    u = wp.min(t - float(index), 1.0)
    a = samples[index, i]
    b = samples[index + 1, i]
    p = wp.lerp(wp.transform_get_translation(a), wp.transform_get_translation(b), u)
    q = wp.quat_slerp(wp.transform_get_rotation(a), wp.transform_get_rotation(b), u)
    return wp.transform(p, q)


@wp.kernel
def move_colliders(
    clock: wp.array[float],
    dt: float,
    samples: wp.array2d[wp.transform],
    rate: float,
    current: wp.array[wp.transform],
    following: wp.array[wp.transform],
):
    i = wp.tid()
    current[i] = interpolate_pose(samples, i, wp.max(clock[0] - dt, 0.0) * rate)
    following[i] = interpolate_pose(samples, i, wp.max(clock[0], 0.0) * rate)


@wp.func
def local_velocity(local: wp.vec3, a: wp.transform, b: wp.transform, dt: float):
    return (wp.transform_point(b, local) - wp.transform_point(a, local)) / dt


@wp.kernel
def grid_contact(
    active: wp.array[int],
    count: wp.array[int],
    nodes: int,
    fields: int,
    mass: wp.array[float],
    velocity: wp.array[wp.vec3],
    origin: wp.vec3,
    res: wp.vec3i,
    h: float,
    dt: float,
    poses: wp.array[wp.transform],
    following: wp.array[wp.transform],
    kind: wp.array[int],
    sizes: wp.array[wp.vec3],
    friction: wp.array[float],
    impulse: wp.array[wp.vec3],
):
    tid = wp.tid()
    if tid >= wp.min(count[0], active.shape[0]):
        return
    n = active[tid]
    coord = wp.vec3i(n // (res[1] * res[2]), (n // res[2]) % res[1], n % res[2])
    x = origin + h * wp.vec3(float(coord[0]), float(coord[1]), float(coord[2]))
    for collider in range(poses.shape[0]):
        a = poses[collider]
        if wp.length_sq(x - wp.transform_get_translation(a)) > wp.length_sq(sizes[collider]) * 2.5:
            continue
        local = wp.transform_point(wp.transform_inverse(a), x)
        surf = surface_local(local, kind[collider], sizes[collider])
        if surf[3] <= 1.0e-6:
            normal = wp.transform_vector(a, wp.vec3(surf[0], surf[1], surf[2]))
            speed = local_velocity(local, a, following[collider], dt)
            for region in range(fields):
                idx = region * nodes + n
                old = velocity[idx]
                velocity[idx] = speed + project_velocity(
                    old - speed, normal, friction[wp.min(region, friction.shape[0] - 1)]
                )
                wp.atomic_add(impulse, collider, mass[idx] * (velocity[idx] - old))


@wp.kernel
def particle_contact(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    dt: float,
    poses: wp.array[wp.transform],
    following: wp.array[wp.transform],
    kind: wp.array[int],
    sizes: wp.array[wp.vec3],
    friction: wp.array[float],
    mass: float,
    impulse: wp.array[wp.vec3],
    particle_weight: wp.array[float],
):
    p = wp.tid()
    xx = x[p] + wp.vec3(0.0)
    vv = v[p] + wp.vec3(0.0)
    for collider in range(poses.shape[0]):
        a = poses[collider]
        if wp.length_sq(xx - wp.transform_get_translation(a)) > wp.length_sq(sizes[collider]) * 2.5:
            continue
        local = wp.transform_point(wp.transform_inverse(a), xx)
        surf = surface_local(local, kind[collider], sizes[collider])
        if surf[3] < 0.0:
            normal = wp.transform_vector(a, wp.vec3(surf[0], surf[1], surf[2]))
            xx -= surf[3] * normal
            speed = local_velocity(local, a, following[collider], dt)
            old = vv
            vv = speed + project_velocity(vv - speed, normal, friction[p])
            wp.atomic_add(impulse, collider, mass * particle_weight[p] * (vv - old))
    x[p] = xx
    v[p] = vv


@wp.kernel
def sticky_contact(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    damage: wp.array[float],
    poses: wp.array[wp.transform],
    kind: wp.array[int],
    sizes: wp.array[wp.vec3],
    attached: wp.array[int],
    anchor: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    peak: wp.array[float],
    age: wp.array[float],
    force: wp.array[wp.vec3],
    strength: float,
    reach: float,
    lifetime: float,
    spacing: float,
    mass: float,
    dt: float,
    particle_weight: wp.array[float],
):
    p = wp.tid()
    spacing = spacing * wp.pow(particle_weight[p], 1.0 / 3.0)
    mass = mass * particle_weight[p]
    force[p] = wp.vec3(0.0)
    wet = activation(damage[p])
    if wet <= 0.0:
        attached[p] = -1
        return
    contact = int(attached[p])
    if contact < 0:
        nearest = float(1.0e10)
        best = int(-1)
        for i in range(poses.shape[0]):
            local = wp.transform_point(wp.transform_inverse(poses[i]), x[p])
            surf = surface_local(local, kind[i], sizes[i])
            if surf[3] > -0.1 * spacing and surf[3] < nearest:
                nearest = surf[3]
                best = i
        if age[p] >= 1.0 and nearest > 0.2 * spacing:
            age[p] = 0.0
        if age[p] < 1.0 and best >= 0 and nearest <= 0.1 * spacing:
            contact = best
            attached[p] = best
            local = wp.transform_point(wp.transform_inverse(poses[best]), x[p])
            anchor[p] = local
            surf = surface_local(local, kind[best], sizes[best])
            normals[p] = wp.vec3(surf[0], surf[1], surf[2])
            peak[p] = 0.0
            age[p] = 0.0
    if contact >= 0:
        a = poses[contact]
        normal = wp.transform_vector(a, normals[p])
        delta = separation(x[p] - wp.transform_point(a, anchor[p]), normal)
        distance = wp.max(peak[p], wp.length(delta))
        if lifetime > 0.0 and (age[p] > 0.0 or wp.length(delta) > 1.0e-6):
            age[p] += dt / lifetime
        if distance >= reach or age[p] >= 1.0:
            attached[p] = -1
            age[p] = 1.0
            peak[p] = 0.0
        else:
            peak[p] = distance
            force[p] = cohesive_force(
                delta, distance, reach, strength * spacing * spacing * wet * (1.0 - age[p]) * (1.0 - age[p])
            )
            v[p] += dt / mass * force[p]


class HandContact:
    def __init__(
        self, samples, kind, sizes, count, rate=120.0, friction=0.7, strength=400.0, reach=0.005, lifetime=1.5
    ):
        self.samples = wp.array(np.asarray(samples, np.float32), dtype=wp.transform)
        self.poses = wp.array(samples[0].astype(np.float32), dtype=wp.transform)
        self.following = wp.clone(self.poses)
        self.kind = wp.array(kind, dtype=int)
        self.sizes = wp.array(sizes, dtype=wp.vec3)
        self.rate, self.friction, self.strength, self.reach, self.lifetime = rate, friction, strength, reach, lifetime
        self.field_friction = wp.array([friction], dtype=float)
        self.particle_friction = wp.full(count, friction, dtype=float)
        self.particle_weight = wp.ones(count, dtype=float)
        self.attached = wp.full(count, -1, dtype=int)
        self.anchor = wp.zeros(count, dtype=wp.vec3)
        self.normals = wp.zeros(count, dtype=wp.vec3)
        self.peak = wp.zeros(count, dtype=float)
        self.age = wp.zeros(count, dtype=float)
        self.force = wp.zeros(count, dtype=wp.vec3)
        self.impulse = wp.zeros(len(kind), dtype=wp.vec3)
        self.state_arrays = (
            self.poses,
            self.following,
            self.attached,
            self.anchor,
            self.normals,
            self.peak,
            self.age,
            self.force,
            self.impulse,
        )

    def begin_step(self, sim):
        wp.launch(
            move_colliders,
            dim=len(self.kind),
            inputs=[sim.clock, sim.dt, self.samples, self.rate, self.poses, self.following],
        )
        if self.strength > 0:
            wp.launch(
                sticky_contact,
                dim=len(sim.rest),
                inputs=[
                    sim.x,
                    sim.v,
                    sim.damage,
                    self.poses,
                    self.kind,
                    self.sizes,
                    self.attached,
                    self.anchor,
                    self.normals,
                    self.peak,
                    self.age,
                    self.force,
                    self.strength,
                    self.reach,
                    self.lifetime,
                    sim.spacing,
                    sim.mass,
                    sim.dt,
                    self.particle_weight,
                ],
            )

    def grid_step(self, sim):
        wp.launch(
            grid_contact,
            dim=sim.capacity,
            inputs=[
                sim.active,
                sim.count,
                sim.nodes,
                sim.fields,
                sim.node_mass,
                sim.node_velocity,
                sim.origin,
                sim.res,
                sim.h,
                sim.dt,
                self.poses,
                self.following,
                self.kind,
                self.sizes,
                self.field_friction,
                self.impulse,
            ],
        )

    def particle_step(self, sim):
        wp.launch(
            particle_contact,
            dim=len(sim.rest),
            inputs=[
                sim.x,
                sim.v,
                sim.dt,
                self.poses,
                self.following,
                self.kind,
                self.sizes,
                self.particle_friction,
                sim.mass,
                self.impulse,
                self.particle_weight,
            ],
        )
