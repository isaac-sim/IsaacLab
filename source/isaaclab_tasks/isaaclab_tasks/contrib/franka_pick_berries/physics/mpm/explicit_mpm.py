# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental explicit APIC/MLS MPM with independent material velocity fields.

Quadratic transfer and fixed-corotated elasticity follow the MLS-MPM formulation:
https://github.com/yuanming-hu/taichi_mpm
Pairwise nodal contact follows the multi-field MPM principle; this is not a full
reproduction of Bardenhagen et al. (2001). Plasticity is a demo-tuned isochoric
log-strain projection. Appearance arrays never enter this module.
"""

import numpy as np
import warp as wp


@wp.func
def weights(t: float):
    return wp.vec3(0.5 * (1.5 - t) * (1.5 - t), 0.75 - (t - 1.0) * (t - 1.0), 0.5 * (t - 0.5) * (t - 0.5))


@wp.func
def derivatives(t: float):
    return wp.vec3(t - 1.5, -2.0 * (t - 1.0), t - 0.5)


@wp.func
def flatten(n: wp.vec3i, res: wp.vec3i):
    return (n[0] * res[1] + n[1]) * res[2] + n[2]


@wp.func
def friction_velocity(v: wp.vec3, normal: wp.vec3, friction: float):
    vn = wp.dot(v, normal)
    if vn < 0.0:
        tangent = v - vn * normal
        return tangent * wp.max(0.0, 1.0 + friction * vn / wp.max(wp.length(tangent), 1.0e-12))
    return v


@wp.func
def cube_normal(x: wp.vec3, center: wp.vec3, half: float):
    delta = x - center
    gap = wp.vec3(half - wp.abs(delta[0]), half - wp.abs(delta[1]), half - wp.abs(delta[2]))
    normal = wp.vec3(0.0)
    if wp.min(gap[0], wp.min(gap[1], gap[2])) > 0.0:
        axis = int(0)
        if gap[1] < gap[axis]:
            axis = 1
        if gap[2] < gap[axis]:
            axis = 2
        normal[axis] = wp.where(delta[axis] >= 0.0, 1.0, -1.0)
    return normal


@wp.kernel
def clear_grid(
    active: wp.array[int],
    count: wp.array[int],
    visited: wp.array[int],
    nodes: int,
    fields: int,
    mass: wp.array[float],
    momentum: wp.array[wp.vec3],
    gradient: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    lower: wp.array2d[float],
    upper: wp.array2d[float],
):
    t = wp.tid()
    if t < wp.min(count[0], active.shape[0]):
        n = active[t]
        visited[n] = 0
        for region in range(fields):
            i = region * nodes + n
            mass[i] = 0.0
            momentum[i] = wp.vec3(0.0)
            if fields > 1:
                gradient[i] = wp.vec3(0.0)
                moment[i] = wp.vec3(0.0)
                for axis in range(3):
                    lower[i, axis] = 1.0e30
                    upper[i, axis] = -1.0e30


@wp.kernel
def begin_step(
    count: wp.array[int],
    clock: wp.array[float],
    dt: float,
    top: float,
    depth: float,
    cube: wp.array[wp.vec3],
    cube_speed: wp.array[wp.vec3],
    tool_half: float,
    ticks: wp.array[int],
):
    count[0] = 0
    t = float(ticks[0]) * dt
    s, ds = float(0.0), float(0.0)
    if t >= 1.0 and t < 3.5:
        u = (t - 1.0) / 2.5
        s, ds = u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u) / 2.5
    elif t >= 3.5 and t < 4.0:
        s = 1.0
    elif t >= 4.0 and t < 5.5:
        u = (t - 4.0) / 1.5
        s, ds = 1.0 - u * u * (3.0 - 2.0 * u), -6.0 * u * (1.0 - u) / 1.5
    travel = -top * depth - 0.003
    cube[0] = wp.vec3(0.0, 0.0, top + tool_half + 0.003 + travel * s)
    cube_speed[0] = wp.vec3(0.0, 0.0, travel * ds)
    ticks[0] += 1
    clock[0] = float(ticks[0]) * dt


@wp.kernel
def particle_affines(
    c: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    affine: wp.array[wp.mat33],
    particle_mass: float,
    volume: float,
    mu: float,
    lam: float,
    h: float,
    dt: float,
    errors: wp.array[int],
    tear: wp.array[float],
    damage: wp.array[float],
    dose: wp.array[float],
    bruise_stress: float,
    bruise_rate: float,
):
    p = wp.tid()
    f = elastic[p]
    u, sigma, vv = wp.svd3(f)
    rotation = u @ wp.transpose(vv)
    j = wp.determinant(f)
    if j <= 0.0:
        wp.atomic_add(errors, 0, 1)
    stress = 2.0 * mu * (f - rotation) @ wp.transpose(f) + lam * j * (j - 1.0) * wp.identity(3, dtype=float)
    if bruise_stress > 0.0 and bruise_rate > 0.0:
        compression = float(0.0)
        for axis in range(3):
            value = 2.0 * mu / j * (sigma[axis] - 1.0) * sigma[axis] + lam * (j - 1.0)
            compression = wp.max(compression, -value)
        excess = wp.max(compression / bruise_stress - 1.0, 0.0)
        dose[p] += dt * bruise_rate * excess * excess
        damage[p] = wp.max(damage[p], 1.0 - wp.exp(-dose[p]))
    if tear[p] > 0.0:
        principal = wp.vec3(0.0)
        for axis in range(3):
            value = 2.0 * mu * (sigma[axis] - 1.0) * sigma[axis] + lam * j * (j - 1.0)
            # Local ductile failure weakens tensile traction, while preserving
            # compression support. It never releases an entire material region.
            principal[axis] = wp.where(value > 0.0, value * (1.0 - 0.98 * tear[p]), value)
        stress = u @ wp.diag(principal) @ wp.transpose(u)
    affine[p] = particle_mass * c[p] - dt * volume * 4.0 / (h * h) * stress


@wp.kernel
def particle_to_grid(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    affine: wp.array[wp.mat33],
    region: wp.array[int],
    particle_mass: float,
    h: float,
    origin: wp.vec3,
    res: wp.vec3i,
    nodes: int,
    mass: wp.array[float],
    momentum: wp.array[wp.vec3],
    gradient: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    active: wp.array[int],
    count: wp.array[int],
    visited: wp.array[int],
    errors: wp.array[int],
    radius: float,
    lower: wp.array2d[float],
    upper: wp.array2d[float],
    fields: int,
):
    tid = wp.tid()
    p = tid // 27
    offset = tid % 27
    a, b, cc = offset // 9, (offset // 3) % 3, offset % 3
    q = (x[p] - origin) / h
    base = wp.vec3i(int(wp.floor(q[0] - 0.5)), int(wp.floor(q[1] - 0.5)), int(wp.floor(q[2] - 0.5)))
    fx = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wx, wy, wz = weights(fx[0]), weights(fx[1]), weights(fx[2])
    dx, dy, dz = derivatives(fx[0]) / h, derivatives(fx[1]) / h, derivatives(fx[2]) / h
    cell = base + wp.vec3i(a, b, cc)
    if cell[0] < 0 or cell[1] < 0 or cell[2] < 0 or cell[0] >= res[0] or cell[1] >= res[1] or cell[2] >= res[2]:
        wp.atomic_add(errors, 1, 1)
    else:
        n = flatten(cell, res)
        i = region[p] * nodes + n
        if wp.atomic_cas(visited, n, 0, 1) == 0:
            slot = wp.atomic_add(count, 0, 1)
            if slot < active.shape[0]:
                active[slot] = n
            else:
                wp.atomic_add(errors, 2, 1)
        w = wx[a] * wy[b] * wz[cc]
        delta = (wp.vec3(float(a), float(b), float(cc)) - fx) * h
        wp.atomic_add(mass, i, w * particle_mass)
        wp.atomic_add(momentum, i, w * (particle_mass * v[p] + affine[p] @ delta))
        # These diagnostics feed only inter-field contact. The accepted berry
        # is one cohesive field; omit unused atomics without changing its forces.
        if fields > 1:
            wp.atomic_add(moment, i, w * particle_mass * x[p])
            wp.atomic_add(
                gradient,
                i,
                particle_mass * wp.vec3(dx[a] * wy[b] * wz[cc], wx[a] * dy[b] * wz[cc], wx[a] * wy[b] * dz[cc]),
            )
            if w > 1.0e-5:
                for axis in range(3):
                    wp.atomic_min(lower, i, axis, x[p][axis] - radius)
                    wp.atomic_max(upper, i, axis, x[p][axis] + radius)


@wp.func
def collider_velocity(
    v: wp.vec3,
    x: wp.vec3,
    cube: wp.vec3,
    cube_speed: wp.vec3,
    use_plane: int,
    use_cube: int,
    friction: float,
    tool_half: float,
):
    result = v
    if use_plane != 0 and x[2] <= 0.0:
        result = friction_velocity(result, wp.vec3(0.0, 0.0, 1.0), friction)
    if use_cube != 0:
        normal = cube_normal(x, cube, tool_half)
        if wp.length_sq(normal) > 0.5:
            result = cube_speed + friction_velocity(result - cube_speed, normal, friction)
    return result


@wp.kernel
def grid_solve(
    active: wp.array[int],
    count: wp.array[int],
    nodes: int,
    fields: int,
    mass: wp.array[float],
    momentum: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    gradient: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    released: wp.array[int],
    origin: wp.vec3,
    res: wp.vec3i,
    h: float,
    dt: float,
    gravity: wp.vec3,
    damping: float,
    friction: float,
    cube: wp.array[wp.vec3],
    cube_speed: wp.array[wp.vec3],
    use_plane: int,
    use_cube: int,
    lower: wp.array2d[float],
    upper: wp.array2d[float],
    tool_half: float,
    contact_friction: float,
    interfield_contact: int,
):
    tid = wp.tid()
    if tid >= wp.min(count[0], active.shape[0]):
        return
    n = active[tid]
    coord = wp.vec3i(n // (res[1] * res[2]), (n // res[2]) % res[1], n % res[2])
    x = origin + h * wp.vec3(float(coord[0]), float(coord[1]), float(coord[2]))
    total_m = float(0.0)
    total_p = wp.vec3(0.0)
    for a in range(fields):
        i = a * nodes + n
        velocity[i] = wp.vec3(0.0)
        if mass[i] > 1.0e-14:
            vel = (momentum[i] / mass[i] + dt * gravity) * wp.exp(-damping * dt)
            total_m += mass[i]
            total_p += mass[i] * vel
            velocity[i] = vel
    if released[0] == 0:
        common = collider_velocity(
            total_p / wp.max(total_m, 1.0e-14), x, cube[0], cube_speed[0], use_plane, use_cube, friction, tool_half
        )
        for a in range(fields):
            velocity[a * nodes + n] = common
    else:
        # One thread owns all fields at a node. Every pair projection dissipates
        # relative motion and conserves its momentum; separated motion is free.
        for sweep in range(3):
            for a in range(fields):
                i = a * nodes + n
                if mass[i] > 1.0e-14:
                    velocity[i] = collider_velocity(
                        velocity[i], x, cube[0], cube_speed[0], use_plane, use_cube, friction, tool_half
                    )
                    for b in range(a + 1, fields):
                        jj = b * nodes + n
                        if interfield_contact != 0 and mass[jj] > 1.0e-14:
                            normal = gradient[i] / mass[i] - gradient[jj] / mass[jj]
                            delta = moment[jj] / mass[jj] - moment[i] / mass[i]
                            if wp.length_sq(normal) < 1.0e-10:
                                normal = delta
                            if wp.dot(normal, delta) < 0.0:
                                normal = -normal
                            normal = wp.normalize(normal)
                            relative = velocity[jj] - velocity[i]
                            closing = wp.dot(relative, normal)
                            gap = float(0.0)
                            for axis in range(3):
                                pa = wp.where(normal[axis] >= 0.0, upper[i, axis], lower[i, axis])
                                pb = wp.where(normal[axis] >= 0.0, lower[jj, axis], upper[jj, axis])
                                gap += normal[axis] * (pb - pa)
                            # Kernel support overlap alone is not physical contact.
                            # Use the nodal particle-volume bounds as a coarse gap
                            # estimate, with a declared 50 micrometre margin.
                            if closing < 0.0 and gap <= wp.max(-closing * dt, 0.0) + 0.00005:
                                reduced = mass[i] * mass[jj] / (mass[i] + mass[jj])
                                tangent = relative - closing * normal
                                tangent *= wp.min(
                                    1.0, contact_friction * (-closing) / wp.max(wp.length(tangent), 1.0e-12)
                                )
                                impulse = reduced * (closing * normal + tangent)
                                velocity[i] += impulse / mass[i]
                                velocity[jj] -= impulse / mass[jj]


@wp.kernel
def grid_to_particle(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    c: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    frames: wp.array[wp.mat33],
    history: wp.array[float],
    damage: wp.array[float],
    interface: wp.array[int],
    region: wp.array[int],
    velocity: wp.array[wp.vec3],
    nodes: int,
    h: float,
    dt: float,
    origin: wp.vec3,
    res: wp.vec3i,
    bulk_yield: float,
    interface_yield: float,
    hardening: float,
    cube: wp.array[wp.vec3],
    cube_speed: wp.array[wp.vec3],
    use_plane: int,
    use_cube: int,
    friction: float,
    errors: wp.array[int],
    tear: wp.array[float],
    softening: float,
    tear_onset: float,
    tear_end: float,
    tool_half: float,
):
    p = wp.tid()
    q = (x[p] - origin) / h
    base = wp.vec3i(int(wp.floor(q[0] - 0.5)), int(wp.floor(q[1] - 0.5)), int(wp.floor(q[2] - 0.5)))
    fx = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wx, wy, wz = weights(fx[0]), weights(fx[1]), weights(fx[2])
    vv = wp.vec3(0.0)
    cc = wp.mat33(0.0)
    for a in range(3):
        for b in range(3):
            for k in range(3):
                node = base + wp.vec3i(a, b, k)
                if (
                    node[0] >= 0
                    and node[1] >= 0
                    and node[2] >= 0
                    and node[0] < res[0]
                    and node[1] < res[1]
                    and node[2] < res[2]
                ):
                    w = wx[a] * wy[b] * wz[k]
                    delta = (wp.vec3(float(a), float(b), float(k)) - fx) * h
                    vg = velocity[region[p] * nodes + flatten(node, res)]
                    vv += w * vg
                    cc += 4.0 * w / (h * h) * wp.outer(vg, delta)
    step_f = wp.identity(3, dtype=float) + dt * cc
    trial = step_f @ elastic[p]
    u, s, vrot = wp.svd3(trial)
    if wp.min(s[0], wp.min(s[1], s[2])) <= 0.0:
        wp.atomic_add(errors, 0, 1)
    log_s = wp.vec3(wp.log(wp.max(s[0], 1.0e-8)), wp.log(wp.max(s[1], 1.0e-8)), wp.log(wp.max(s[2], 1.0e-8)))
    mean = (log_s[0] + log_s[1] + log_s[2]) / 3.0
    dev = log_s - wp.vec3(mean)
    limit = wp.where(interface[p] != 0, interface_yield, bulk_yield) * (1.0 + hardening * history[p])
    limit *= 1.0 - softening * damage[p]
    norm = wp.length(dev)
    remaining = wp.min(1.0, limit / wp.max(norm, 1.0e-12))
    projected = wp.vec3(mean) + remaining * dev
    elastic[p] = (
        u @ wp.diag(wp.vec3(wp.exp(projected[0]), wp.exp(projected[1]), wp.exp(projected[2]))) @ wp.transpose(vrot)
    )
    history[p] += (1.0 - remaining) * norm
    onset = wp.where(interface[p] != 0, 0.08, 0.50)
    interval = wp.where(interface[p] != 0, 0.22, 1.0)
    damage[p] = wp.max(damage[p], wp.clamp((history[p] - onset) / interval, 0.0, 1.0))
    if tear_end > tear_onset:
        failure = wp.clamp((history[p] - tear_onset) / (tear_end - tear_onset), 0.0, 1.0)
        tear[p] = wp.max(tear[p], failure * failure * (3.0 - 2.0 * failure))
    new_x = x[p] + dt * vv
    if use_plane != 0 and new_x[2] < 0.00005:
        new_x[2] = 0.00005
        vv = friction_velocity(vv, wp.vec3(0.0, 0.0, 1.0), friction)
    if use_cube != 0:
        normal = cube_normal(new_x, cube[0], tool_half)
        if wp.length_sq(normal) > 0.5:
            for axis in range(3):
                if wp.abs(normal[axis]) > 0.5:
                    new_x[axis] = cube[0][axis] + tool_half * normal[axis]
            vv = cube_speed[0] + friction_velocity(vv - cube_speed[0], normal, friction)
    x[p] = new_x
    v[p] = vv
    c[p] = cc
    frames[p] = step_f @ frames[p]


class ExplicitMPM:
    def __init__(
        self,
        rest,
        regions=None,
        interface=None,
        spacing=0.001,
        h=0.002,
        hz=6000,
        young=12000.0,
        poisson=0.35,
        density=900.0,
        gravity=(0, 0, -9.81),
        damping=1.0,
        friction=0.25,
        plane=True,
        cube=True,
        bulk_yield=0.18,
        interface_yield=0.04,
        released=False,
        particle_volume=None,
        hardening=0.0,
        softening=0.0,
        tear_onset=0.0,
        tear_end=0.0,
        tool_size=0.018,
        adhesion=0.0,
        adhesion_range=0.0015,
        adhesion_lifetime=0.0,
        adhesion_plane_scale=1.0,
        contact=None,
        grid_res=None,
        bruise_stress=0.0,
        bruise_rate=0.0,
        frame_hz=30,
        contact_friction=None,
        interfield_contact=True,
    ):
        wp.init()
        wp.set_device("cuda:0")
        self.rest = np.asarray(rest, np.float32)
        self.regions = np.zeros(len(rest), np.int32) if regions is None else np.asarray(regions, np.int32)
        self.interface = np.zeros(len(rest), np.int32) if interface is None else np.asarray(interface, np.int32)
        self.fields = int(self.regions.max()) + 1
        self.spacing, self.h, self.hz, self.dt = spacing, h, hz, 1.0 / hz
        self.frame_hz = frame_hz
        if not np.isfinite(tool_size) or tool_size <= 0:
            raise ValueError("Positive finite tool size required")
        self.tool_half = tool_size / 2
        self.volume = spacing**3 if particle_volume is None else particle_volume
        self.mass = density * self.volume
        if not np.isfinite(adhesion) or adhesion < 0 or not np.isfinite(adhesion_range) or adhesion_range <= 0:
            raise ValueError("Require nonnegative finite adhesion pressure and positive finite adhesion range")
        self.adhesion = None
        if not np.isfinite(adhesion_lifetime) or adhesion_lifetime < 0 or not 0 <= adhesion_plane_scale <= 1:
            raise ValueError("Require nonnegative bond lifetime and floor adhesion scale in [0,1]")
        if adhesion > 0:
            from .adhesion import AdhesiveContact

            self.adhesion = AdhesiveContact(
                len(rest), adhesion, adhesion_range, spacing, self.mass, adhesion_lifetime, adhesion_plane_scale
            )
        self.contact = contact
        self.juice = None
        if not np.isfinite([bruise_stress, bruise_rate]).all() or bruise_stress < 0 or bruise_rate < 0:
            raise ValueError("Bruise threshold and rate must be finite and nonnegative")
        self.bruise_stress, self.bruise_rate = bruise_stress, bruise_rate
        self.mu, self.lam = young / (2 * (1 + poisson)), young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.cfl = self.dt * np.sqrt((self.lam + 2 * self.mu) / density) / h
        if hz % frame_hz or self.cfl > 0.45:
            raise ValueError(f"Require whole steps/frame and CFL <= .45, got {self.cfl}")
        self.origin = wp.vec3(-0.06, -0.06, -0.008)
        self.res = wp.vec3i(*(grid_res or (61, 61, 33)))
        self.nodes = int(np.prod(self.res))
        self.capacity = min(self.nodes, 32768)
        self.gravity = wp.vec3(*gravity)
        self.damping, self.friction = damping, friction
        self.contact_friction = friction if contact_friction is None else contact_friction
        if not np.isfinite(self.contact_friction) or self.contact_friction < 0:
            raise ValueError("Inter-field friction must be finite and nonnegative")
        self.interfield_contact = int(interfield_contact)
        self.use_plane, self.use_cube = int(plane), int(cube)
        self.bulk_yield, self.interface_yield = bulk_yield, interface_yield
        if hardening < 0:
            raise ValueError("Plastic hardening must be nonnegative")
        self.hardening = hardening
        if not 0 <= softening < 1:
            raise ValueError("Require 0 <= softening < 1")
        if tear_end != 0 and not 0 <= tear_onset < tear_end:
            raise ValueError("Tearing requires 0 <= onset < end, or end=0 to disable")
        self.softening, self.tear_onset, self.tear_end = softening, tear_onset, tear_end
        self.top = float(rest[:, 2].max() + spacing / 2)
        self.x = wp.array(rest, dtype=wp.vec3)
        self.v = wp.zeros(len(rest), dtype=wp.vec3)
        self.c = wp.zeros(len(rest), dtype=wp.mat33)
        self.affine = wp.empty(len(rest), dtype=wp.mat33)
        identity = np.broadcast_to(np.eye(3, dtype=np.float32), (len(rest), 3, 3)).copy()
        self.elastic = wp.array(identity, dtype=wp.mat33)
        self.frames = wp.array(identity, dtype=wp.mat33)
        self.history = wp.zeros(len(rest), dtype=float)
        self.damage = wp.zeros(len(rest), dtype=float)
        self.tear = wp.zeros(len(rest), dtype=float)
        self.bruise_dose = wp.zeros(len(rest), dtype=float)
        self.region_gpu = wp.array(self.regions, dtype=int)
        self.interface_gpu = wp.array(self.interface, dtype=int)
        self.released = wp.array([int(released)], dtype=int)
        self.clock = wp.zeros(1, dtype=float)
        self.ticks = wp.zeros(1, dtype=int)
        self.cube = wp.array([[0, 0, self.top + self.tool_half + 0.003]], dtype=wp.vec3)
        self.cube_speed = wp.zeros(1, dtype=wp.vec3)
        self.node_mass = wp.zeros(self.nodes * self.fields, dtype=float)
        self.node_momentum = wp.zeros(self.nodes * self.fields, dtype=wp.vec3)
        self.node_gradient = wp.zeros(self.nodes * self.fields, dtype=wp.vec3)
        self.node_moment = wp.zeros(self.nodes * self.fields, dtype=wp.vec3)
        self.node_velocity = wp.zeros(self.nodes * self.fields, dtype=wp.vec3)
        self.node_lower = wp.full((self.nodes * self.fields, 3), 1.0e30, dtype=float)
        self.node_upper = wp.full((self.nodes * self.fields, 3), -1.0e30, dtype=float)
        self.active = wp.empty(self.capacity, dtype=int)
        self.count = wp.zeros(1, dtype=int)
        self.visited = wp.zeros(self.nodes, dtype=int)
        self.errors = wp.zeros(3, dtype=int)
        self.graph = None

    def step(self, depth=0.85):
        wp.launch(
            clear_grid,
            dim=self.capacity,
            inputs=[
                self.active,
                self.count,
                self.visited,
                self.nodes,
                self.fields,
                self.node_mass,
                self.node_momentum,
                self.node_gradient,
                self.node_moment,
                self.node_lower,
                self.node_upper,
            ],
        )
        wp.launch(
            begin_step,
            dim=1,
            inputs=[
                self.count,
                self.clock,
                self.dt,
                self.top,
                depth,
                self.cube,
                self.cube_speed,
                self.tool_half,
                self.ticks,
            ],
        )
        if self.adhesion:
            self.adhesion.step(
                self.x, self.v, self.damage, self.cube, self.tool_half, self.use_cube, self.use_plane, self.dt
            )
        if self.contact:
            self.contact.begin_step(self)
        if self.juice:
            self.juice.begin_step(self)
        wp.launch(
            particle_affines,
            dim=len(self.rest),
            inputs=[
                self.c,
                self.elastic,
                self.affine,
                self.mass,
                self.volume,
                self.mu,
                self.lam,
                self.h,
                self.dt,
                self.errors,
                self.tear,
                self.damage,
                self.bruise_dose,
                self.bruise_stress,
                self.bruise_rate,
            ],
        )
        if self.juice:
            self.juice.affines(self)
        if self.juice and self.juice.refinement > 1:
            self.juice.p2g(self)
        else:
            wp.launch(
                particle_to_grid,
                dim=len(self.rest) * 27,
                inputs=[
                    self.x,
                    self.v,
                    self.affine,
                    self.region_gpu,
                    self.mass,
                    self.h,
                    self.origin,
                    self.res,
                    self.nodes,
                    self.node_mass,
                    self.node_momentum,
                    self.node_gradient,
                    self.node_moment,
                    self.active,
                    self.count,
                    self.visited,
                    self.errors,
                    self.spacing / 2,
                    self.node_lower,
                    self.node_upper,
                    self.fields,
                ],
            )
        if self.juice:
            self.juice.grid_step(self)
        else:
            wp.launch(
                grid_solve,
                dim=self.capacity,
                inputs=[
                    self.active,
                    self.count,
                    self.nodes,
                    self.fields,
                    self.node_mass,
                    self.node_momentum,
                    self.node_velocity,
                    self.node_gradient,
                    self.node_moment,
                    self.released,
                    self.origin,
                    self.res,
                    self.h,
                    self.dt,
                    self.gravity,
                    self.damping,
                    self.friction,
                    self.cube,
                    self.cube_speed,
                    self.use_plane,
                    self.use_cube,
                    self.node_lower,
                    self.node_upper,
                    self.tool_half,
                    self.contact_friction,
                    self.interfield_contact,
                ],
            )
        if self.contact:
            self.contact.grid_step(self)
        if self.juice:
            self.juice.contact_grid(self)
        wp.launch(
            grid_to_particle,
            dim=len(self.rest),
            inputs=[
                self.x,
                self.v,
                self.c,
                self.elastic,
                self.frames,
                self.history,
                self.damage,
                self.interface_gpu,
                self.region_gpu,
                self.node_velocity,
                self.nodes,
                self.h,
                self.dt,
                self.origin,
                self.res,
                self.bulk_yield,
                self.interface_yield,
                self.hardening,
                self.cube,
                self.cube_speed,
                self.use_plane,
                self.use_cube,
                self.friction,
                self.errors,
                self.tear,
                self.softening,
                self.tear_onset,
                self.tear_end,
                self.tool_half,
            ],
        )
        if self.contact:
            self.contact.particle_step(self)
        if self.juice:
            self.juice.project(self)

    def prepare(self, depth=0.85):
        saved = [
            wp.clone(a)
            for a in (
                self.x,
                self.v,
                self.c,
                self.elastic,
                self.frames,
                self.history,
                self.damage,
                self.tear,
                self.clock,
                self.bruise_dose,
                self.ticks,
            )
        ]
        saved_adhesion = [wp.clone(a) for a in self.adhesion.arrays] if self.adhesion else []
        saved_contact = [wp.clone(a) for a in self.contact.state_arrays] if self.contact else []
        saved_juice = [wp.clone(a) for a in self.juice.state_arrays] if self.juice else []
        self.step(depth)
        with wp.ScopedCapture() as capture:
            for _ in range(self.hz // self.frame_hz):
                self.step(depth)
        self.graph = capture.graph
        for target, source in zip(
            (
                self.x,
                self.v,
                self.c,
                self.elastic,
                self.frames,
                self.history,
                self.damage,
                self.tear,
                self.clock,
                self.bruise_dose,
                self.ticks,
            ),
            saved,
        ):
            target.assign(source)
        if self.adhesion:
            for target, source in zip(self.adhesion.arrays, saved_adhesion):
                target.assign(source)
        if self.contact:
            for target, source in zip(self.contact.state_arrays, saved_contact):
                target.assign(source)
        if self.juice:
            for target, source in zip(self.juice.state_arrays, saved_juice):
                target.assign(source)
        self.errors.zero_()
        wp.synchronize()

    def advance(self, depth=0.85):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            for _ in range(self.hz // self.frame_hz):
                self.step(depth)

    def check(self):
        errors = self.errors.numpy()
        if np.any(errors):
            raise RuntimeError(f"Explicit MPM errors [inversion, domain exit, active capacity]: {errors}")
        for name in ("x", "v", "c", "elastic", "frames", "history", "damage", "tear", "bruise_dose"):
            if not np.isfinite(getattr(self, name).numpy()).all():
                raise RuntimeError(f"Nonfinite {name}")
        if self.adhesion:
            for values in (self.adhesion.tool_force, self.adhesion.plane_force):
                if not np.isfinite(values.numpy()).all():
                    raise RuntimeError("Nonfinite adhesive force")
