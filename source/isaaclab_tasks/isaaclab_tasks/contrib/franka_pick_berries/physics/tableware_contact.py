# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic static contact for the table, punnet, reject dish and glass bowl."""

import numpy as np
import warp as wp

from ..scene.tableware import PUNNET, tableware_solids
from .mpm.hand_contact import project_velocity

PUNNET_CORNER = wp.constant(PUNNET[4])
PUNNET_WALL = wp.constant(PUNNET[5])


@wp.func
def rounded_rectangle(p: wp.vec3, hx: float, hy: float, corner: float):
    q = wp.vec2(wp.abs(p[0]) - hx + corner, wp.abs(p[1]) - hy + corner)
    outside = wp.vec2(wp.max(q[0], 0.0), wp.max(q[1], 0.0))
    length = wp.length(outside)
    normal = wp.vec2(0.0, wp.where(p[1] >= 0.0, 1.0, -1.0))
    if q[0] > q[1]:
        normal = wp.vec2(wp.where(p[0] >= 0.0, 1.0, -1.0), 0.0)
    if length > 1.0e-12:
        normal = (
            wp.vec2(wp.where(p[0] >= 0.0, 1.0, -1.0) * outside[0], wp.where(p[1] >= 0.0, 1.0, -1.0) * outside[1])
            / length
        )
    return wp.vec3(normal[0], normal[1], length + wp.min(wp.max(q[0], q[1]), 0.0) - corner)


@wp.func
def vessel_surface(x: wp.vec3, center: wp.vec3, size: wp.vec3):
    # Rounded-rectangle base/wall or annular cylinder; see tableware_solids encoding.
    p = x - center
    if size[0] < 0.0:
        hx, hy = -size[0], wp.abs(size[1])
        planar = rounded_rectangle(p, hx, hy, PUNNET_CORNER)
        if size[1] < 0.0:
            inner = rounded_rectangle(p, hx - PUNNET_WALL, hy - PUNNET_WALL, PUNNET_CORNER - PUNNET_WALL)
            if -inner[2] > planar[2]:
                planar = -inner
        dz = wp.abs(p[2]) - size[2]
        vertical = wp.vec3(0.0, 0.0, wp.where(p[2] >= 0.0, 1.0, -1.0))
        radial = wp.vec3(planar[0], planar[1], 0.0)
        normal = vertical
        if planar[2] > dz:
            normal = radial
        outside = wp.vec2(wp.max(planar[2], 0.0), wp.max(dz, 0.0))
        length = wp.length(outside)
        if length > 1.0e-12:
            normal = (outside[0] * radial + outside[1] * vertical) / length
        return wp.vec4(normal[0], normal[1], normal[2], length + wp.min(wp.max(planar[2], dz), 0.0))
    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    radial = wp.vec3(1.0, 0.0, 0.0)
    if r > 1.0e-12:
        radial = wp.vec3(p[0] / r, p[1] / r, 0.0)
    dr = r - size[1]
    if size[0] > 0.0 and size[0] - r > dr:
        dr = size[0] - r
        radial = -radial
    dz = wp.abs(p[2]) - size[2]
    vertical = wp.vec3(0.0, 0.0, wp.where(p[2] >= 0.0, 1.0, -1.0))
    normal = vertical
    if dr > dz:
        normal = radial
    distance = wp.min(wp.max(dr, dz), 0.0)
    outside = wp.vec2(wp.max(dr, 0.0), wp.max(dz, 0.0))
    length = wp.length(outside)
    if length > 1.0e-12:
        normal = (outside[0] * radial + outside[1] * vertical) / length
    return wp.vec4(normal[0], normal[1], normal[2], distance + length)


@wp.kernel
def tableware_grid_contact(
    active: wp.array[int],
    count: wp.array[int],
    nodes: int,
    fields: int,
    velocity: wp.array[wp.vec3],
    origin: wp.vec3,
    res: wp.vec3i,
    h: float,
    centers: wp.array[wp.vec3],
    sizes: wp.array[wp.vec3],
    floor: float,
):
    tid = wp.tid()
    if tid >= wp.min(count[0], active.shape[0]):
        return
    n = active[tid]
    coord = wp.vec3i(n // (res[1] * res[2]), (n // res[2]) % res[1], n % res[2])
    x = origin + h * wp.vec3(float(coord[0]), float(coord[1]), float(coord[2]))
    for region in range(fields):
        i = region * nodes + n
        v = velocity[i] + wp.vec3(0.0)
        if x[2] <= floor:
            v = project_velocity(v, wp.vec3(0.0, 0.0, 1.0), 0.3)
        for j in range(centers.shape[0]):
            surf = vessel_surface(x, centers[j], sizes[j])
            # Thin bases can lie between grid nodes (notably blackberry's 2.8 mm
            # grid). Support nearby nodes too; particle projection stays exact.
            if surf[3] <= 0.5 * h:
                v = project_velocity(v, wp.vec3(surf[0], surf[1], surf[2]), 0.3)
        velocity[i] = v


@wp.kernel
def tableware_particle_contact(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    centers: wp.array[wp.vec3],
    sizes: wp.array[wp.vec3],
    floor: float,
    radius: float,
):
    i = wp.tid()
    p = x[i] + wp.vec3(0.0)
    speed = v[i] + wp.vec3(0.0)
    if p[2] < floor + radius:
        p[2] = floor + radius
        speed = project_velocity(speed, wp.vec3(0.0, 0.0, 1.0), 0.3)
    for j in range(centers.shape[0]):
        surf = vessel_surface(p, centers[j], sizes[j])
        if surf[3] < radius:
            normal = wp.vec3(surf[0], surf[1], surf[2])
            p += (radius - surf[3]) * normal
            speed = project_velocity(speed, normal, 0.3)
    x[i] = p
    v[i] = speed


class TablewareContact:
    """Support grid velocities and project particles without any adhesive force."""

    def __init__(self, offset: np.ndarray):
        """Express fixed task-frame props relative to the MPM origin offset [m]."""
        solids = tableware_solids()
        self.centers = wp.array(solids[:, :3] - np.asarray(offset), dtype=wp.vec3)
        self.sizes = wp.array(solids[:, 3:], dtype=wp.vec3)
        self.floor = -float(offset[2])

    def grid_step(self, sim):
        wp.launch(
            tableware_grid_contact,
            sim.capacity,
            inputs=[
                sim.active,
                sim.count,
                sim.nodes,
                sim.fields,
                sim.node_velocity,
                sim.origin,
                sim.res,
                sim.h,
                self.centers,
                self.sizes,
                self.floor,
            ],
        )

    def particle_step(self, sim):
        wp.launch(
            tableware_particle_contact,
            len(sim.rest),
            inputs=[sim.x, sim.v, self.centers, self.sizes, self.floor, 0.5 * sim.spacing],
        )
