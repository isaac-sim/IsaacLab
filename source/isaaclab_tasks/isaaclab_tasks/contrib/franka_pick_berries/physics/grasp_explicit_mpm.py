# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit MLS-MPM tissue solver with frictional finger-pad contact, as an entry of Newton's coupled solver.

:class:`SolverGraspExplicitMPM` advances soft tissue with an explicit material point method and grips it with finger
pads attached to rigid bodies. :class:`.grasp_implicit_mpm.SolverGraspImplicitMPM` is the alternative built on
Newton's implicit MPM.

Method:

* MLS-MPM transfers with quadratic B-splines and APIC velocity gradients (Hu et al. 2018, "A Moving Least Squares
  Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling"; Jiang et al. 2015, "The
  Affine Particle-In-Cell Method").
* Fixed-corotated elasticity and a volume-preserving return mapping of the elastic log-strain deviator (Jiang et al.
  2016, "The Material Point Method for Simulating Continuum Materials", SIGGRAPH course).
* Separate velocity fields per body of tissue, with frictional contact between fields at shared grid nodes
  (Bardenhagen et al. 2001, "An Improved Contact Algorithm for the Material Point Method").
* Penalty contact between particles and the finger pads, with an elastic tangential displacement capped by Coulomb
  friction, so a gripped particle can stick.
* An irreversible damage model: plastic strain history weakens the yield limit and, past a threshold, the tensile
  stress (tearing); sustained compression accumulates a bruise dose. Damaged tissue adheres briefly to the pads.

The solver works on all particles of its model, in the world frame. Each particle's stiffness, mass and volume come
from the model (``mpm:young_modulus``, ``mpm:poisson_ratio``, ``particle_mass`` and ``particle_radius``); yield
limits, tearing and the velocity field are set per body of tissue with :meth:`SolverGraspExplicitMPM.set_tissue`.
The grid is fixed and dense, with an active-node list bounding the work. Contact with the ground and with static vessels
(cylinders, annuli and rounded rectangles) is analytic.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp
from isaaclab_newton.physics import MPMSolverCfg, NewtonMPMManager
from newton._src.solvers.coupled.interface import CouplingInterface  # experimental; not yet exported by Newton
from newton.solvers import SolverBase

from isaaclab.utils import configclass

from ..scene.tableware import PUNNET, tableware_solids
from .materials import POISSON_RATIO, YOUNG_MODULUS

# Error counters of :attr:`SolverGraspExplicitMPM.errors`.
_INVERTED, _OUTSIDE_GRID, _ACTIVE_OVERFLOW = 0, 1, 2
_FREE = -1


# --------------------------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------------------------


@wp.func
def spline_weights(t: float):
    """Quadratic B-spline weights of the three nodes around a particle, at offset ``t`` from the first."""
    return wp.vec3(0.5 * (1.5 - t) * (1.5 - t), 0.75 - (t - 1.0) * (t - 1.0), 0.5 * (t - 0.5) * (t - 0.5))


@wp.func
def spline_gradients(t: float):
    return wp.vec3(t - 1.5, -2.0 * (t - 1.0), t - 0.5)


@wp.func
def node_index(node: wp.vec3i, resolution: wp.vec3i):
    return (node[0] * resolution[1] + node[1]) * resolution[2] + node[2]


@wp.func
def node_position(n: int, origin: wp.vec3, resolution: wp.vec3i, spacing: float):
    node = wp.vec3i(n // (resolution[1] * resolution[2]), (n // resolution[2]) % resolution[1], n % resolution[2])
    return origin + spacing * wp.vec3(float(node[0]), float(node[1]), float(node[2]))


@wp.func
def box_surface(local: wp.vec3, half: wp.vec3):
    """Outward normal and signed distance [m] of a point to a box, in the box frame."""
    excess = wp.vec3(wp.abs(local[0]) - half[0], wp.abs(local[1]) - half[1], wp.abs(local[2]) - half[2])
    outside = wp.vec3(wp.max(excess[0], 0.0), wp.max(excess[1], 0.0), wp.max(excess[2], 0.0))
    length = wp.length(outside)
    axis = int(0)
    if excess[1] > excess[axis]:
        axis = 1
    if excess[2] > excess[axis]:
        axis = 2
    normal = wp.vec3(0.0)
    if length > 1.0e-12:
        normal = wp.cw_mul(outside / length, wp.vec3(wp.sign(local[0]), wp.sign(local[1]), wp.sign(local[2])))
    else:
        normal[axis] = wp.where(local[axis] >= 0.0, 1.0, -1.0)
    return wp.vec4(normal[0], normal[1], normal[2], length + wp.min(excess[axis], 0.0))


@wp.func
def rounded_rectangle(p: wp.vec3, hx: float, hy: float, corner: float):
    """Planar outward normal and signed distance [m] to a rounded rectangle in the xy plane."""
    q = wp.vec2(wp.abs(p[0]) - hx + corner, wp.abs(p[1]) - hy + corner)
    outside = wp.vec2(wp.max(q[0], 0.0), wp.max(q[1], 0.0))
    length = wp.length(outside)
    sx = wp.where(p[0] >= 0.0, 1.0, -1.0)
    sy = wp.where(p[1] >= 0.0, 1.0, -1.0)
    normal = wp.vec2(0.0, sy)
    if q[0] > q[1]:
        normal = wp.vec2(sx, 0.0)
    if length > 1.0e-12:
        normal = wp.vec2(sx * outside[0], sy * outside[1]) / length
    return wp.vec3(normal[0], normal[1], length + wp.min(wp.max(q[0], q[1]), 0.0) - corner)


@wp.func
def vessel_surface(x: wp.vec3, center: wp.vec3, size: wp.vec3, corner: float, wall: float):
    """Outward normal and signed distance [m] to a static vessel part.

    ``size`` = (a, b, half height). Nonnegative ``a`` is an annular cylinder with inner radius ``a`` (0 for a disk)
    and outer radius ``b``. Negative ``a`` is a rounded rectangle of half extents ``(|a|, |b|)``, solid for positive
    ``b`` and a wall of thickness ``wall`` for negative ``b``.
    """
    p = x - center
    vertical = wp.vec3(0.0, 0.0, wp.where(p[2] >= 0.0, 1.0, -1.0))
    dz = wp.abs(p[2]) - size[2]
    radial = wp.vec3(1.0, 0.0, 0.0)
    dr = float(0.0)
    if size[0] < 0.0:
        hx, hy = -size[0], wp.abs(size[1])
        planar = rounded_rectangle(p, hx, hy, corner)
        if size[1] < 0.0:
            inner = rounded_rectangle(p, hx - wall, hy - wall, corner - wall)
            if -inner[2] > planar[2]:
                planar = -inner
        radial = wp.vec3(planar[0], planar[1], 0.0)
        dr = planar[2]
    else:
        r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
        if r > 1.0e-12:
            radial = wp.vec3(p[0] / r, p[1] / r, 0.0)
        dr = r - size[1]
        if size[0] > 0.0 and size[0] - r > dr:
            dr = size[0] - r
            radial = -radial
    normal = wp.where(dr > dz, radial, vertical)
    outside = wp.vec2(wp.max(dr, 0.0), wp.max(dz, 0.0))
    length = wp.length(outside)
    if length > 1.0e-12:
        normal = (outside[0] * radial + outside[1] * vertical) / length
    return wp.vec4(normal[0], normal[1], normal[2], length + wp.min(wp.max(dr, dz), 0.0))


@wp.func
def coulomb_velocity(v: wp.vec3, normal: wp.vec3, friction: float):
    """Remove an approaching normal velocity and the tangential velocity it can stop by Coulomb friction."""
    vn = wp.dot(v, normal)
    if vn < 0.0:
        tangent = v - vn * normal
        return tangent * wp.max(0.0, 1.0 + friction * vn / wp.max(wp.length(tangent), 1.0e-12))
    return v


@wp.func
def static_boundaries(
    x: wp.vec3,
    v: wp.vec3,
    ground: float,
    ground_friction: float,
    vessel_center: wp.array[wp.vec3],
    vessel_size: wp.array[wp.vec3],
    vessel_friction: float,
    corner: float,
    wall: float,
    reach: float,
):
    """Apply ground friction to velocity ``v`` at or below the ground, and vessel friction within ``reach`` [m] of a
    vessel surface."""
    result = v
    if x[2] <= ground:
        result = coulomb_velocity(result, wp.vec3(0.0, 0.0, 1.0), ground_friction)
    for j in range(vessel_center.shape[0]):
        surface = vessel_surface(x, vessel_center[j], vessel_size[j], corner, wall)
        if surface[3] <= reach:
            result = coulomb_velocity(result, wp.vec3(surface[0], surface[1], surface[2]), vessel_friction)
    return result


# --------------------------------------------------------------------------------------------------------------------
# Finger pads
# --------------------------------------------------------------------------------------------------------------------


@wp.struct
class PadState:
    """Pad poses and the rigid motion of their bodies at the current substep."""

    pose: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    linear: wp.array[wp.vec3]
    angular: wp.array[wp.vec3]


@wp.func
def pad_point_velocity(pads: PadState, pad: int, x: wp.vec3):
    return pads.linear[pad] + wp.cross(pads.angular[pad], x - pads.body_com[pad])


@wp.kernel
def advance_pads(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    pad_body: wp.array[int],
    pad_offset: wp.array[wp.transform],
    elapsed: float,
    pads: PadState,
):
    """Move each pad with its body's twist from the start of the step, ``elapsed`` [s] earlier."""
    i = wp.tid()
    pose = body_q[pad_body[i]]
    linear = wp.spatial_top(body_qd[pad_body[i]])
    angular = wp.spatial_bottom(body_qd[pad_body[i]])
    angle = wp.length(angular) * elapsed
    turn = wp.quat_identity()
    if angle > 1.0e-12:
        turn = wp.quat_from_axis_angle(wp.normalize(angular), angle)
    # Rotate about the body's center of mass, which moves with the linear velocity.
    com = wp.transform_point(pose, body_com[pad_body[i]])
    moved_com = com + linear * elapsed
    origin = moved_com + wp.quat_rotate(turn, wp.transform_get_translation(pose) - com)
    moved = wp.transform(origin, turn * wp.transform_get_rotation(pose))
    pads.pose[i] = wp.transform_multiply(moved, pad_offset[i])
    pads.body_com[i] = moved_com
    pads.linear[i] = linear
    pads.angular[i] = angular


@wp.func
def adhesion_activation(damage: float):
    t = wp.clamp((damage - 0.1) / 0.5, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.kernel
def pad_adhesion(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    damage: wp.array[float],
    mass: wp.array[float],
    spacing: wp.array[float],
    pads: PadState,
    pad_half: wp.array[wp.vec3],
    strength: float,
    reach: float,
    lifetime: float,
    dt: float,
    attached: wp.array[int],
    anchor: wp.array[wp.vec3],
    anchor_normal: wp.array[wp.vec3],
    peak: wp.array[float],
    age: wp.array[float],
    impulse: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
):
    """Bond damaged (wet) tissue that touches a pad to it, with a traction that softens as the bond opens or ages."""
    p = wp.tid()
    wet = adhesion_activation(damage[p])
    if wet <= 0.0:
        attached[p] = _FREE
        return
    h = spacing[p]
    pad = attached[p]
    if pad == _FREE:
        nearest = float(1.0e10)
        for i in range(pads.pose.shape[0]):
            local = wp.transform_point(wp.transform_inverse(pads.pose[i]), x[p])
            distance = box_surface(local, pad_half[i])[3]
            if distance > -0.1 * h and distance < nearest:
                nearest = distance
                pad = i
        if age[p] >= 1.0 and nearest > 0.2 * h:
            age[p] = 0.0
        if age[p] < 1.0 and pad != _FREE and nearest <= 0.1 * h:
            local = wp.transform_point(wp.transform_inverse(pads.pose[pad]), x[p])
            surface = box_surface(local, pad_half[pad])
            attached[p] = pad
            anchor[p] = local
            anchor_normal[p] = wp.vec3(surface[0], surface[1], surface[2])
            peak[p] = 0.0
            age[p] = 0.0
        else:
            return
    pose = pads.pose[pad]
    normal = wp.transform_vector(pose, anchor_normal[p])
    delta = x[p] - wp.transform_point(pose, anchor[p])
    # Hard contact resists compression; the bond resists opening and sliding only.
    delta -= wp.min(wp.dot(delta, normal), 0.0) * normal
    opening = wp.length(delta)
    distance = wp.max(peak[p], opening)
    if lifetime > 0.0 and (age[p] > 0.0 or opening > 1.0e-6):
        age[p] += dt / lifetime
    if distance >= reach or age[p] >= 1.0:
        attached[p] = _FREE
        age[p] = 1.0
        peak[p] = 0.0
        return
    peak[p] = distance
    # Bilinear traction: peak at 20% of the reach, then irreversible softening.
    s = distance / reach
    envelope = wp.max(wp.min(s / 0.2, (1.0 - s) / 0.8), 0.0)
    strength_p = strength * h * h * wet * (1.0 - age[p]) * (1.0 - age[p])
    force = -strength_p * envelope * delta / wp.max(distance, 1.0e-12)
    v[p] = v[p] + dt / mass[p] * force
    wp.atomic_add(impulse, pad, dt * force)
    wp.atomic_add(moment, pad, wp.cross(x[p] - pads.body_com[pad], dt * force))


@wp.kernel
def pad_contact(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    mass: wp.array[float],
    spacing: wp.array[float],
    young: wp.array[float],
    pads: PadState,
    pad_half: wp.array[wp.vec3],
    friction: float,
    dt: float,
    tangential: wp.array2d[wp.vec3],
    impulse: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
):
    """Push particles out of the pads with a spring-damper and hold them by Coulomb-capped tangential springs."""
    p = wp.tid()
    velocity = v[p]
    # A particle stands for a finite volume of tissue, not a point.
    radius = 0.5 * spacing[p]
    stiffness = young[p] * spacing[p]
    damping = wp.sqrt(mass[p] * stiffness)
    for pad in range(pads.pose.shape[0]):
        pose = pads.pose[pad]
        bound = wp.length(pad_half[pad]) + radius
        if wp.length_sq(x[p] - wp.transform_get_translation(pose)) > bound * bound:
            tangential[p, pad] = wp.vec3(0.0)
            continue
        local = wp.transform_point(wp.transform_inverse(pose), x[p])
        surface = box_surface(local, pad_half[pad])
        depth = radius - surface[3]
        if depth <= 0.0:
            tangential[p, pad] = wp.vec3(0.0)
            continue
        normal = wp.transform_vector(pose, wp.vec3(surface[0], surface[1], surface[2]))
        relative = velocity - pad_point_velocity(pads, pad, x[p])
        vn = wp.dot(relative, normal)
        # Unilateral: no tensile normal force, and no tangential grip without load.
        normal_impulse = dt * wp.max(0.0, stiffness * depth - damping * vn)
        tangent = relative - vn * normal
        # The tangential spring's elongation is kept in the pad frame, so it moves with the pad.
        displacement = wp.transform_vector(pose, tangential[p, pad]) + dt * tangent
        displacement -= wp.dot(displacement, normal) * normal
        tangent_force = -0.5 * stiffness * displacement - 0.5 * damping * tangent
        limit = friction * normal_impulse / dt
        magnitude = wp.length(tangent_force)
        if magnitude > limit:
            # Sliding: cap the force on the friction cone and relax the spring to match.
            tangent_force *= limit / wp.max(magnitude, 1.0e-12)
            displacement = -(tangent_force + 0.5 * damping * tangent) / (0.5 * stiffness)
        if limit <= 0.0:
            displacement = wp.vec3(0.0)
        tangential[p, pad] = wp.transform_vector(wp.transform_inverse(pose), displacement)
        change = normal_impulse * normal + dt * tangent_force
        velocity += change / mass[p]
        wp.atomic_add(impulse, pad, change)
        wp.atomic_add(moment, pad, wp.cross(x[p] - pads.body_com[pad], change))
    v[p] = velocity


# --------------------------------------------------------------------------------------------------------------------
# Material point method
# --------------------------------------------------------------------------------------------------------------------


@wp.kernel
def clear_grid(
    active: wp.array[int],
    count: wp.array[int],
    visited: wp.array[int],
    nodes: int,
    fields: int,
    mass: wp.array[float],
    momentum: wp.array[wp.vec3],
    mass_gradient: wp.array[wp.vec3],
    mass_moment: wp.array[wp.vec3],
    lower: wp.array2d[float],
    upper: wp.array2d[float],
):
    """Zero the nodes the previous substep touched."""
    t = wp.tid()
    if t < wp.min(count[0], active.shape[0]):
        n = active[t]
        visited[n] = 0
        for f in range(fields):
            i = f * nodes + n
            mass[i] = 0.0
            momentum[i] = wp.vec3(0.0)
            if fields > 1:
                mass_gradient[i] = wp.vec3(0.0)
                mass_moment[i] = wp.vec3(0.0)
                for axis in range(3):
                    lower[i, axis] = 1.0e30
                    upper[i, axis] = -1.0e30


@wp.kernel
def reset_count(count: wp.array[int]):
    count[0] = 0


@wp.kernel
def particle_stress(
    c: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    mass: wp.array[float],
    volume: wp.array[float],
    young: wp.array[float],
    poisson: wp.array[float],
    tear: wp.array[float],
    spacing: float,
    dt: float,
    bruise_stress: float,
    bruise_rate: float,
    errors: wp.array[int],
    damage: wp.array[float],
    dose: wp.array[float],
    affine: wp.array[wp.mat33],
):
    """Fixed-corotated Kirchhoff stress, folded with the APIC velocity gradient into the MLS-MPM affine momentum."""
    p = wp.tid()
    mu = young[p] / (2.0 * (1.0 + poisson[p]))
    lam = young[p] * poisson[p] / ((1.0 + poisson[p]) * (1.0 - 2.0 * poisson[p]))
    f = elastic[p]
    u, sigma, w = wp.svd3(f)
    j = wp.determinant(f)
    if j <= 0.0:
        wp.atomic_add(errors, _INVERTED, 1)
    stress = 2.0 * mu * (f - u @ wp.transpose(w)) @ wp.transpose(f) + lam * j * (j - 1.0) * wp.identity(3, dtype=float)
    if bruise_rate > 0.0:
        # Bruising: a dose accumulates while the largest principal (Cauchy) compression exceeds the threshold.
        compression = float(0.0)
        for axis in range(3):
            compression = wp.max(compression, -(2.0 * mu / j * (sigma[axis] - 1.0) * sigma[axis] + lam * (j - 1.0)))
        excess = wp.max(compression / bruise_stress - 1.0, 0.0)
        dose[p] += dt * bruise_rate * excess * excess
        damage[p] = wp.max(damage[p], 1.0 - wp.exp(-dose[p]))
    if tear[p] > 0.0:
        # Torn tissue loses tensile stress but keeps supporting compression.
        principal = wp.vec3(0.0)
        for axis in range(3):
            value = 2.0 * mu * (sigma[axis] - 1.0) * sigma[axis] + lam * j * (j - 1.0)
            principal[axis] = wp.where(value > 0.0, value * (1.0 - 0.98 * tear[p]), value)
        stress = u @ wp.diag(principal) @ wp.transpose(u)
    affine[p] = mass[p] * c[p] - dt * volume[p] * 4.0 / (spacing * spacing) * stress


@wp.kernel
def particle_to_grid(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    affine: wp.array[wp.mat33],
    mass: wp.array[float],
    spacing_p: wp.array[float],
    tissue_field: wp.array[int],
    origin: wp.vec3,
    resolution: wp.vec3i,
    spacing: float,
    fields: int,
    errors: wp.array[int],
    active: wp.array[int],
    count: wp.array[int],
    visited: wp.array[int],
    node_mass: wp.array[float],
    node_momentum: wp.array[wp.vec3],
    mass_gradient: wp.array[wp.vec3],
    mass_moment: wp.array[wp.vec3],
    lower: wp.array2d[float],
    upper: wp.array2d[float],
):
    """Scatter one particle's mass and momentum to one of its 27 nodes, and record the node as active."""
    tid = wp.tid()
    p = tid // 27
    a, b, k = (tid % 27) // 9, (tid % 9) // 3, tid % 3
    q = (x[p] - origin) / spacing
    base = wp.vec3i(int(wp.floor(q[0] - 0.5)), int(wp.floor(q[1] - 0.5)), int(wp.floor(q[2] - 0.5)))
    fx = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    node = base + wp.vec3i(a, b, k)
    if wp.min(node[0], wp.min(node[1], node[2])) < 0 or node[0] >= resolution[0] or node[1] >= resolution[1]:
        wp.atomic_add(errors, _OUTSIDE_GRID, 1)
        return
    if node[2] >= resolution[2]:
        wp.atomic_add(errors, _OUTSIDE_GRID, 1)
        return
    n = node_index(node, resolution)
    if wp.atomic_cas(visited, n, 0, 1) == 0:
        slot = wp.atomic_add(count, 0, 1)
        if slot < active.shape[0]:
            active[slot] = n
        else:
            wp.atomic_add(errors, _ACTIVE_OVERFLOW, 1)
    wx, wy, wz = spline_weights(fx[0]), spline_weights(fx[1]), spline_weights(fx[2])
    weight = wx[a] * wy[b] * wz[k]
    delta = (wp.vec3(float(a), float(b), float(k)) - fx) * spacing
    i = tissue_field[p] * (resolution[0] * resolution[1] * resolution[2]) + n
    wp.atomic_add(node_mass, i, weight * mass[p])
    wp.atomic_add(node_momentum, i, weight * (mass[p] * v[p] + affine[p] @ delta))
    if fields > 1:
        # Field contact needs each field's surface normal (mass gradient), center and extent at the node.
        gx, gy, gz = spline_gradients(fx[0]), spline_gradients(fx[1]), spline_gradients(fx[2])
        gradient = wp.vec3(gx[a] * wy[b] * wz[k], wx[a] * gy[b] * wz[k], wx[a] * wy[b] * gz[k]) / spacing
        wp.atomic_add(mass_gradient, i, mass[p] * gradient)
        wp.atomic_add(mass_moment, i, weight * mass[p] * x[p])
        if weight > 1.0e-5:
            for axis in range(3):
                wp.atomic_min(lower, i, axis, x[p][axis] - 0.5 * spacing_p[p])
                wp.atomic_max(upper, i, axis, x[p][axis] + 0.5 * spacing_p[p])


@wp.kernel
def grid_update(
    active: wp.array[int],
    count: wp.array[int],
    origin: wp.vec3,
    resolution: wp.vec3i,
    spacing: float,
    fields: int,
    dt: float,
    gravity: wp.vec3,
    damping: float,
    field_friction: float,
    ground: float,
    ground_friction: float,
    vessel_center: wp.array[wp.vec3],
    vessel_size: wp.array[wp.vec3],
    vessel_friction: float,
    corner: float,
    wall: float,
    node_mass: wp.array[float],
    node_momentum: wp.array[wp.vec3],
    mass_gradient: wp.array[wp.vec3],
    mass_moment: wp.array[wp.vec3],
    lower: wp.array2d[float],
    upper: wp.array2d[float],
    node_velocity: wp.array[wp.vec3],
):
    """Integrate each active node's velocities, resolve contact between fields, then with the static boundaries."""
    tid = wp.tid()
    if tid >= wp.min(count[0], active.shape[0]):
        return
    n = active[tid]
    nodes = resolution[0] * resolution[1] * resolution[2]
    x = node_position(n, origin, resolution, spacing)
    for f in range(fields):
        i = f * nodes + n
        node_velocity[i] = wp.vec3(0.0)
        if node_mass[i] > 1.0e-14:
            node_velocity[i] = (node_momentum[i] / node_mass[i] + dt * gravity) * wp.exp(-damping * dt)
    # One thread owns all fields of a node. Each pairwise projection removes approaching relative motion and conserves
    # momentum; separating fields move freely.
    for sweep in range(3):
        for f in range(fields):
            i = f * nodes + n
            if node_mass[i] <= 1.0e-14:
                continue
            for g in range(f + 1, fields):
                k = g * nodes + n
                if node_mass[k] <= 1.0e-14:
                    continue
                normal = mass_gradient[i] / node_mass[i] - mass_gradient[k] / node_mass[k]
                offset = mass_moment[k] / node_mass[k] - mass_moment[i] / node_mass[i]
                if wp.length_sq(normal) < 1.0e-10:
                    normal = offset
                if wp.dot(normal, offset) < 0.0:
                    normal = -normal
                normal = wp.normalize(normal)
                relative = node_velocity[k] - node_velocity[i]
                closing = wp.dot(relative, normal)
                # Overlapping kernel support is not contact: estimate the gap from the fields' particle extents, with
                # a 50 micrometer margin.
                gap = float(0.0)
                for axis in range(3):
                    near_i = wp.where(normal[axis] >= 0.0, upper[i, axis], lower[i, axis])
                    near_k = wp.where(normal[axis] >= 0.0, lower[k, axis], upper[k, axis])
                    gap += normal[axis] * (near_k - near_i)
                if closing < 0.0 and gap <= -closing * dt + 0.00005:
                    reduced = node_mass[i] * node_mass[k] / (node_mass[i] + node_mass[k])
                    tangent = relative - closing * normal
                    tangent *= wp.min(1.0, field_friction * (-closing) / wp.max(wp.length(tangent), 1.0e-12))
                    impulse = reduced * (closing * normal + tangent)
                    node_velocity[i] += impulse / node_mass[i]
                    node_velocity[k] -= impulse / node_mass[k]
    for f in range(fields):
        i = f * nodes + n
        # Thin vessel bases can lie between nodes, so nodes within half a cell are supported too; particle projection
        # is exact.
        node_velocity[i] = static_boundaries(
            x,
            node_velocity[i],
            ground,
            ground_friction,
            vessel_center,
            vessel_size,
            vessel_friction,
            corner,
            wall,
            0.5 * spacing,
        )


@wp.kernel
def grid_to_particle(
    node_velocity: wp.array[wp.vec3],
    tissue_field: wp.array[int],
    interface: wp.array[int],
    yield_strain: wp.array[float],
    tear_onset: wp.array[float],
    tear_end: wp.array[float],
    origin: wp.vec3,
    resolution: wp.vec3i,
    spacing: float,
    dt: float,
    hardening: float,
    softening: float,
    damage_onset: wp.vec2,
    damage_interval: wp.vec2,
    errors: wp.array[int],
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    c: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    history: wp.array[float],
    damage: wp.array[float],
    tear: wp.array[float],
):
    """Gather velocity and its gradient, update the elastic deformation with plastic flow, damage and tearing, and
    advect the particle."""
    p = wp.tid()
    q = (x[p] - origin) / spacing
    base = wp.vec3i(int(wp.floor(q[0] - 0.5)), int(wp.floor(q[1] - 0.5)), int(wp.floor(q[2] - 0.5)))
    fx = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    wx, wy, wz = spline_weights(fx[0]), spline_weights(fx[1]), spline_weights(fx[2])
    first = tissue_field[p] * (resolution[0] * resolution[1] * resolution[2])
    velocity = wp.vec3(0.0)
    gradient = wp.mat33(0.0)
    for a in range(3):
        for b in range(3):
            for k in range(3):
                node = base + wp.vec3i(a, b, k)
                if wp.min(node[0], wp.min(node[1], node[2])) < 0 or node[0] >= resolution[0]:
                    continue
                if node[1] >= resolution[1] or node[2] >= resolution[2]:
                    continue
                weight = wx[a] * wy[b] * wz[k]
                delta = (wp.vec3(float(a), float(b), float(k)) - fx) * spacing
                vg = node_velocity[first + node_index(node, resolution)]
                velocity += weight * vg
                gradient += 4.0 * weight / (spacing * spacing) * wp.outer(vg, delta)

    # Plasticity: the deviatoric part of the elastic log-strain is returned to the yield radius, preserving volume.
    trial = (wp.identity(3, dtype=float) + dt * gradient) @ elastic[p]
    u, s, w = wp.svd3(trial)
    if wp.min(s[0], wp.min(s[1], s[2])) <= 0.0:
        wp.atomic_add(errors, _INVERTED, 1)
    log_s = wp.vec3(wp.log(wp.max(s[0], 1.0e-8)), wp.log(wp.max(s[1], 1.0e-8)), wp.log(wp.max(s[2], 1.0e-8)))
    mean = (log_s[0] + log_s[1] + log_s[2]) / 3.0
    deviator = log_s - wp.vec3(mean)
    radius = yield_strain[p] * (1.0 + hardening * history[p]) * (1.0 - softening * damage[p])
    norm = wp.length(deviator)
    kept = wp.min(1.0, radius / wp.max(norm, 1.0e-12))
    projected = wp.vec3(mean) + kept * deviator
    stretch = wp.vec3(wp.exp(projected[0]), wp.exp(projected[1]), wp.exp(projected[2]))
    elastic[p] = u @ wp.diag(stretch) @ wp.transpose(w)

    # Damage follows the accumulated plastic strain; tearing follows it further, with a smooth onset.
    history[p] += (1.0 - kept) * norm
    side = wp.where(interface[p] != 0, 1, 0)
    damage[p] = wp.max(damage[p], wp.clamp((history[p] - damage_onset[side]) / damage_interval[side], 0.0, 1.0))
    if tear_end[p] > tear_onset[p]:
        failure = wp.clamp((history[p] - tear_onset[p]) / (tear_end[p] - tear_onset[p]), 0.0, 1.0)
        tear[p] = wp.max(tear[p], failure * failure * (3.0 - 2.0 * failure))

    c[p] = gradient
    v[p] = velocity
    x[p] = x[p] + dt * velocity


@wp.kernel
def particle_boundaries(
    mass: wp.array[float],
    spacing: wp.array[float],
    pads: PadState,
    pad_half: wp.array[wp.vec3],
    ground: float,
    ground_friction: float,
    vessel_center: wp.array[wp.vec3],
    vessel_size: wp.array[wp.vec3],
    vessel_friction: float,
    corner: float,
    wall: float,
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
):
    """Project particles out of the pads (without friction, already applied at the surface) and the static
    boundaries."""
    p = wp.tid()
    position = x[p]
    velocity = v[p]
    radius = 0.5 * spacing[p]
    for pad in range(pads.pose.shape[0]):
        pose = pads.pose[pad]
        if wp.length_sq(position - wp.transform_get_translation(pose)) > wp.length_sq(pad_half[pad]) * 2.5:
            continue
        local = wp.transform_point(wp.transform_inverse(pose), position)
        surface = box_surface(local, pad_half[pad])
        if surface[3] < 0.0:
            normal = wp.transform_vector(pose, wp.vec3(surface[0], surface[1], surface[2]))
            position -= surface[3] * normal
            speed = pad_point_velocity(pads, pad, position)
            before = velocity
            velocity = speed + coulomb_velocity(velocity - speed, normal, 0.0)
            change = mass[p] * (velocity - before)
            wp.atomic_add(impulse, pad, change)
            wp.atomic_add(moment, pad, wp.cross(position - pads.body_com[pad], change))
    if position[2] < ground + radius:
        position[2] = ground + radius
        velocity = coulomb_velocity(velocity, wp.vec3(0.0, 0.0, 1.0), ground_friction)
    for j in range(vessel_center.shape[0]):
        surface = vessel_surface(position, vessel_center[j], vessel_size[j], corner, wall)
        if surface[3] < radius:
            normal = wp.vec3(surface[0], surface[1], surface[2])
            position += (radius - surface[3]) * normal
            velocity = coulomb_velocity(velocity, normal, vessel_friction)
    x[p] = position
    v[p] = velocity


@wp.kernel
def pad_reaction(
    impulse: wp.array[wp.vec3],
    moment: wp.array[wp.vec3],
    pad_body: wp.array[int],
    body_local_to_proxy_global: wp.array[int],
    dt: float,
    out_body_f: wp.array[wp.spatial_vector],
):
    """Return the momentum the pads gave the tissue as an opposite wrench at each pad's body: force [N] and torque
    [N m] about its center of mass, in the world frame."""
    i = wp.tid()
    target = body_local_to_proxy_global[pad_body[i]]
    if target >= 0:
        wp.atomic_add(out_body_f, target, wp.spatial_vector(-impulse[i] / dt, -moment[i] / dt))


@wp.kernel
def particle_geometry(
    particle_mass: wp.array[float],
    particle_radius: wp.array[float],
    volume: wp.array[float],
    spacing: wp.array[float],
):
    p = wp.tid()
    volume[p] = 4.0 / 3.0 * wp.pi * particle_radius[p] * particle_radius[p] * particle_radius[p]
    spacing[p] = wp.pow(volume[p], 1.0 / 3.0)


@wp.kernel
def set_range(
    start: int,
    tissue_field: int,
    yield_value: float,
    interface_yield_value: float,
    tear_onset_value: float,
    tear_end_value: float,
    interface: wp.array[int],
    field_out: wp.array[int],
    yield_out: wp.array[float],
    tear_onset_out: wp.array[float],
    tear_end_out: wp.array[float],
):
    i = wp.tid()
    p = start + i
    field_out[p] = tissue_field
    yield_out[p] = wp.where(interface[i] != 0, interface_yield_value, yield_value)
    tear_onset_out[p] = tear_onset_value
    tear_end_out[p] = tear_end_value


# --------------------------------------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class PadConfig:
    """A box-shaped finger pad attached to a body."""

    body: str
    """Regular expression fully matching the label of the pad's body."""
    offset: tuple[float, float, float]
    """Pad center in the body frame [m]."""
    half_extents: tuple[float, float, float]
    """Pad half extents in the body frame [m]."""


class SolverGraspExplicitMPM(SolverBase, CouplingInterface):
    """Explicit MPM tissue gripped by frictional finger pads; see the module docstring.

    The coupler steps it at :attr:`Config.coupling_rate`; each step runs ``substep_rate / coupling_rate`` explicit
    substeps. Each step reads the pad bodies' pose and twist from the input state, and returns the momentum the pads
    exchanged with the tissue to the proxy bodies as a force (:meth:`coupling_harvest_proxy_wrenches`).
    """

    @dataclass
    class Config:
        grid_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """World position of the grid's first node [m]."""
        grid_resolution: tuple[int, int, int] = (64, 64, 64)
        """Number of grid nodes along each axis."""
        grid_spacing: float = 0.002
        """Grid spacing [m]."""
        max_active_nodes: int = 32768
        """Capacity of the active-node list; overflows are counted in :attr:`errors`."""
        fields: int = 1
        """Number of separate velocity fields (bodies of tissue in contact)."""
        substep_rate: int = 6000
        """Explicit substep rate [Hz]; it must keep the elastic CFL number below about 0.45."""
        coupling_rate: int = 120
        """Rate at which the coupler steps the solver [Hz]; it must divide :attr:`substep_rate`."""
        damping: float = 1.0
        """Exponential velocity damping rate [1/s]."""
        field_friction: float = 0.4
        """Coulomb friction between fields."""
        ground_height: float = 0.0
        """Height of the ground plane [m]."""
        ground_friction: float = 0.35
        vessels: tuple[tuple[float, float, float, float, float, float], ...] = ()
        """Static vessel parts: center (x, y, z) and size (a, b, half height) as in :func:`vessel_surface` [m]."""
        vessel_friction: float = 0.3
        vessel_corner_radius: float = 0.01
        vessel_wall_thickness: float = 0.001
        pads: tuple[PadConfig, ...] = ()
        pad_friction: float = 1.2
        hardening: float = 0.0
        """Growth of the yield radius with plastic strain history."""
        softening: float = 0.25
        """Fraction of the yield radius that full damage removes."""
        damage_onset: tuple[float, float] = (0.5, 0.08)
        """Plastic strain history at which damage starts, in bulk tissue and at interfaces."""
        damage_interval: tuple[float, float] = (1.0, 0.22)
        """Further history over which damage reaches 1, in bulk tissue and at interfaces."""
        bruise_stress: float = 8500.0
        """Principal compression [Pa] above which a bruise dose accumulates."""
        bruise_rate: float = 8.0
        """Bruise dose rate [1/s] at twice the threshold compression."""
        adhesion_strength: float = 400.0
        """Peak adhesive traction of fully wet tissue on a pad [Pa]."""
        adhesion_reach: float = 0.005
        """Opening [m] at which an adhesive bond breaks."""
        adhesion_lifetime: float = 1.5
        """Time [s] after which a loaded adhesive bond expires."""

    def __init__(self, model, config: Config):
        super().__init__(model)
        if config.substep_rate % config.coupling_rate:
            raise ValueError("The coupling rate must divide the substep rate")
        self.config = config
        device = model.device
        gravity = np.asarray(model.gravity.numpy(), np.float32).reshape(-1, 3)[0]
        self.gravity = wp.vec3(*gravity.tolist())
        count = model.particle_count
        nodes = int(np.prod(config.grid_resolution))
        fields = config.fields
        labels = list(model.body_label)
        pad_bodies = []
        for pad in config.pads:
            matches = [i for i, label in enumerate(labels) if re.fullmatch(pad.body, label)]
            if len(matches) != 1:
                raise ValueError(f"Pad body {pad.body!r} must match exactly one body, found {len(matches)}")
            pad_bodies.append(matches[0])
        with wp.ScopedDevice(device):
            self.pad_body = wp.array(pad_bodies, dtype=int)
            self.pad_offset = wp.array(
                [wp.transform(wp.vec3(*pad.offset), wp.quat_identity()) for pad in config.pads], dtype=wp.transform
            )
            self.pad_half = wp.array([pad.half_extents for pad in config.pads], dtype=wp.vec3)
            self.pads = PadState()
            self.pads.pose = wp.zeros(len(pad_bodies), dtype=wp.transform)
            self.pads.body_com = wp.zeros(len(pad_bodies), dtype=wp.vec3)
            self.pads.linear = wp.zeros(len(pad_bodies), dtype=wp.vec3)
            self.pads.angular = wp.zeros(len(pad_bodies), dtype=wp.vec3)
            self.pad_impulse = wp.zeros(len(pad_bodies), dtype=wp.vec3)
            # Angular impulse about each pad body's center of mass [N m s].
            self.pad_moment = wp.zeros(len(pad_bodies), dtype=wp.vec3)
            vessels = np.asarray(config.vessels, np.float32).reshape(-1, 6)
            self.vessel_center = wp.array(vessels[:, :3], dtype=wp.vec3)
            self.vessel_size = wp.array(vessels[:, 3:], dtype=wp.vec3)

            # Per-particle material: geometry from the model, the rest from set_tissue().
            self.volume = wp.empty(count, dtype=float)
            self.spacing = wp.empty(count, dtype=float)
            wp.launch(
                particle_geometry,
                dim=count,
                inputs=[model.particle_mass, model.particle_radius],
                outputs=[self.volume, self.spacing],
            )
            self.tissue_field = wp.zeros(count, dtype=int)
            self.interface = wp.zeros(count, dtype=int)
            self.yield_strain = wp.full(count, 1.0e6, dtype=float)
            self.tear_onset = wp.zeros(count, dtype=float)
            self.tear_end = wp.zeros(count, dtype=float)

            # Per-particle history.
            self.c = wp.zeros(count, dtype=wp.mat33)
            self.elastic = wp.zeros(count, dtype=wp.mat33)
            self.affine = wp.zeros(count, dtype=wp.mat33)
            self.history = wp.zeros(count, dtype=float)
            self.damage = wp.zeros(count, dtype=float)
            self.tear = wp.zeros(count, dtype=float)
            self.dose = wp.zeros(count, dtype=float)
            self.tangential = wp.zeros((count, max(len(pad_bodies), 1)), dtype=wp.vec3)
            self.attached = wp.zeros(count, dtype=int)
            self.anchor = wp.zeros(count, dtype=wp.vec3)
            self.anchor_normal = wp.zeros(count, dtype=wp.vec3)
            self.peak = wp.zeros(count, dtype=float)
            self.age = wp.zeros(count, dtype=float)

            # Grid: one block of nodes per field.
            self.node_mass = wp.zeros(nodes * fields, dtype=float)
            self.node_momentum = wp.zeros(nodes * fields, dtype=wp.vec3)
            self.node_velocity = wp.zeros(nodes * fields, dtype=wp.vec3)
            extra = nodes * fields if fields > 1 else 1
            self.mass_gradient = wp.zeros(extra, dtype=wp.vec3)
            self.mass_moment = wp.zeros(extra, dtype=wp.vec3)
            self.lower = wp.full((extra, 3), 1.0e30, dtype=float)
            self.upper = wp.full((extra, 3), -1.0e30, dtype=float)
            self.active = wp.zeros(min(nodes, config.max_active_nodes), dtype=int)
            self.count = wp.zeros(1, dtype=int)
            self.visited = wp.zeros(nodes, dtype=int)
            self.errors = wp.zeros(3, dtype=int)
            """Counts of inverted particles, particles leaving the grid, and active-node overflows."""
        self._reset_history()

    def set_tissue(self, start: int, interface: np.ndarray, field: int, material: TissueMaterial) -> None:
        """Set the material of a body of tissue: the particles from ``start`` on, one per ``interface`` flag.

        Args:
            start: Index of the body's first particle in the model.
            interface: Nonzero for particles on a weak internal interface, with a lower yield radius.
            field: Velocity field of the body, below :attr:`Config.fields`.
            material: The body's yield radii and tearing.
        """
        if not 0 <= field < self.config.fields:
            raise ValueError(f"Field {field} is outside [0, {self.config.fields})")
        flags = wp.array(np.asarray(interface, np.int32), dtype=int, device=self.model.device)
        wp.launch(
            set_range,
            dim=len(flags),
            inputs=[
                start,
                field,
                material.yield_strain,
                material.interface_yield_strain,
                material.tear_onset,
                material.tear_end,
                flags,
            ],
            outputs=[self.tissue_field, self.yield_strain, self.tear_onset, self.tear_end],
            device=self.model.device,
        )
        self.interface[start : start + len(flags)].assign(flags)

    def damage_view(self, start: int, count: int) -> dict[str, wp.array]:
        """Damage arrays of ``count`` particles from ``start``: ``damage``, ``tear``, ``history`` and ``dose``."""
        span = slice(start, start + count)
        return {
            "damage": self.damage[span],
            "tear": self.tear[span],
            "history": self.history[span],
            "dose": self.dose[span],
        }

    def elastic_strain(self, state, start: int, count: int) -> wp.array:
        """Elastic deformation gradients of ``count`` particles from ``start``; the solver keeps them itself."""
        del state
        return self.elastic[start : start + count]

    def _reset_history(self):
        identity = np.broadcast_to(np.eye(3, dtype=np.float32), (self.model.particle_count, 3, 3))
        self.elastic.assign(identity)
        for array in (self.c, self.history, self.damage, self.tear, self.dose, self.tangential, self.peak, self.age):
            array.zero_()
        self.attached.fill_(_FREE)
        self.errors.zero_()
        self._reported_errors = (0, 0)

    def reset(self, state, world_mask=None, flags=None):
        # The tissue's position and velocity come from the reset state; its deformation and damage start over.
        self._reset_history()

    def step(self, state_in, state_out, control, contacts, dt):
        config = self.config
        substeps = round(dt * config.substep_rate)
        if abs(substeps - dt * config.substep_rate) > 1.0e-6 or substeps < 1:
            raise ValueError(f"Step {dt} s is not a whole number of {config.substep_rate} Hz substeps")
        h = dt / substeps
        model = self.model
        origin = wp.vec3(*config.grid_origin)
        resolution = wp.vec3i(*config.grid_resolution)
        vessels = [
            config.ground_height,
            config.ground_friction,
            self.vessel_center,
            self.vessel_size,
            config.vessel_friction,
            config.vessel_corner_radius,
            config.vessel_wall_thickness,
        ]
        count = model.particle_count
        x, v = state_out.particle_q, state_out.particle_qd
        if state_out is not state_in:
            wp.copy(x, state_in.particle_q)
            wp.copy(v, state_in.particle_qd)
        young = model.mpm.young_modulus
        with wp.ScopedDevice(model.device):
            self.pad_impulse.zero_()
            self.pad_moment.zero_()
            for substep in range(substeps):
                wp.launch(
                    clear_grid,
                    dim=len(self.active),
                    inputs=[self.active, self.count, self.visited, int(np.prod(config.grid_resolution)), config.fields],
                    outputs=[
                        self.node_mass,
                        self.node_momentum,
                        self.mass_gradient,
                        self.mass_moment,
                        self.lower,
                        self.upper,
                    ],
                )
                wp.launch(reset_count, dim=1, inputs=[self.count])
                if len(self.pad_body):
                    wp.launch(
                        advance_pads,
                        dim=len(self.pad_body),
                        inputs=[
                            state_in.body_q,
                            state_in.body_qd,
                            model.body_com,
                            self.pad_body,
                            self.pad_offset,
                            substep * h,
                        ],
                        outputs=[self.pads],
                    )
                    wp.launch(
                        pad_adhesion,
                        dim=count,
                        inputs=[
                            x,
                            v,
                            self.damage,
                            model.particle_mass,
                            self.spacing,
                            self.pads,
                            self.pad_half,
                            config.adhesion_strength,
                            config.adhesion_reach,
                            config.adhesion_lifetime,
                            h,
                        ],
                        outputs=[
                            self.attached,
                            self.anchor,
                            self.anchor_normal,
                            self.peak,
                            self.age,
                            self.pad_impulse,
                            self.pad_moment,
                        ],
                    )
                    wp.launch(
                        pad_contact,
                        dim=count,
                        inputs=[
                            x,
                            v,
                            model.particle_mass,
                            self.spacing,
                            young,
                            self.pads,
                            self.pad_half,
                            config.pad_friction,
                            h,
                        ],
                        outputs=[self.tangential, self.pad_impulse, self.pad_moment],
                    )
                wp.launch(
                    particle_stress,
                    dim=count,
                    inputs=[
                        self.c,
                        self.elastic,
                        model.particle_mass,
                        self.volume,
                        young,
                        model.mpm.poisson_ratio,
                        self.tear,
                        config.grid_spacing,
                        h,
                        config.bruise_stress,
                        config.bruise_rate,
                    ],
                    outputs=[self.errors, self.damage, self.dose, self.affine],
                )
                wp.launch(
                    particle_to_grid,
                    dim=count * 27,
                    inputs=[
                        x,
                        v,
                        self.affine,
                        model.particle_mass,
                        self.spacing,
                        self.tissue_field,
                        origin,
                        resolution,
                        config.grid_spacing,
                        config.fields,
                    ],
                    outputs=[
                        self.errors,
                        self.active,
                        self.count,
                        self.visited,
                        self.node_mass,
                        self.node_momentum,
                        self.mass_gradient,
                        self.mass_moment,
                        self.lower,
                        self.upper,
                    ],
                )
                wp.launch(
                    grid_update,
                    dim=len(self.active),
                    inputs=[
                        self.active,
                        self.count,
                        origin,
                        resolution,
                        config.grid_spacing,
                        config.fields,
                        h,
                        self.gravity,
                        config.damping,
                        config.field_friction,
                        *vessels,
                        self.node_mass,
                        self.node_momentum,
                        self.mass_gradient,
                        self.mass_moment,
                        self.lower,
                        self.upper,
                    ],
                    outputs=[self.node_velocity],
                )
                wp.launch(
                    grid_to_particle,
                    dim=count,
                    inputs=[
                        self.node_velocity,
                        self.tissue_field,
                        self.interface,
                        self.yield_strain,
                        self.tear_onset,
                        self.tear_end,
                        origin,
                        resolution,
                        config.grid_spacing,
                        h,
                        config.hardening,
                        config.softening,
                        wp.vec2(*config.damage_onset),
                        wp.vec2(*config.damage_interval),
                    ],
                    outputs=[self.errors, x, v, self.c, self.elastic, self.history, self.damage, self.tear],
                )
                wp.launch(
                    particle_boundaries,
                    dim=count,
                    inputs=[model.particle_mass, self.spacing, self.pads, self.pad_half, *vessels],
                    outputs=[x, v, self.pad_impulse, self.pad_moment],
                )

    def coupling_harvest_proxy_wrenches(
        self, body_local_to_proxy_global, out_body_f, *, body_qd_before, state, state_out, contacts, dt
    ):
        out_body_f.zero_()
        if len(self.pad_body):
            wp.launch(
                pad_reaction,
                dim=len(self.pad_body),
                inputs=[self.pad_impulse, self.pad_moment, self.pad_body, body_local_to_proxy_global, dt],
                outputs=[out_body_f],
                device=self.model.device,
            )

    def check(self) -> None:
        """Raise if the active-node list overflowed since the last reset, and warn when more particles inverted or
        left the grid.

        An overflowing node is never queued, so it is never cleared and its grid values accumulate; inverted particles
        and particles outside the grid are clamped or skipped, and only reported. Reading the counters synchronizes
        with the device.
        """
        inverted, outside, overflow = (int(count) for count in self.errors.numpy())
        if overflow:
            raise RuntimeError(
                f"Explicit MPM active-node list overflowed {overflow} times since the last reset; "
                f"increase Config.max_active_nodes (currently {self.config.max_active_nodes})"
            )
        if (inverted, outside) != getattr(self, "_reported_errors", (0, 0)):
            self._reported_errors = (inverted, outside)
            warnings.warn(
                f"Explicit MPM since the last reset: {inverted} inverted particle updates, "
                f"{outside} particle transfers outside the grid",
                RuntimeWarning,
                stacklevel=2,
            )


# --------------------------------------------------------------------------------------------------------------------
# Isaac Lab integration
# --------------------------------------------------------------------------------------------------------------------

# Franka finger pads: the flat inner fingertip faces, in each finger's frame [m].
_PAD_HALF_EXTENTS = (0.009, 0.0054, 0.0089)
_PAD_OFFSETS = {"left": (0.0, 0.0054, 0.045), "right": (0.0, -0.0054, 0.045)}


@dataclass
class TissueMaterial:
    """Explicit-solver material of one berry species."""

    yield_strain: float
    interface_yield_strain: float
    tear_onset: float
    tear_end: float
    grid_spacing: float
    """Grid spacing [m] the berry's tissue sampling was prepared for."""
    substep_rate: int
    """Substep rate [Hz] keeping its elastic CFL number below 0.44, a multiple of 120 Hz."""
    friction: float
    damping: float
    hardening: float = 0.0


def tissue_material(profile: dict) -> TissueMaterial:
    """Return the explicit-solver material of a berry from its asset profile."""
    source = profile["simulation"]
    berry = profile["berry"]
    young = YOUNG_MODULUS[berry]
    density, spacing = float(source["density"]), float(source["grid"])
    wave_speed = math.sqrt(young * (1 - POISSON_RATIO) / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO) * density))
    rate = 5040 if berry == "raspberry" else math.ceil(source["hz"] / 120) * 120
    rate = max(rate, math.ceil(wave_speed / (spacing * 0.44 * 120)) * 120)
    tears = source.get("tear_end", 0) > 0
    return TissueMaterial(
        yield_strain=0.8 if berry == "raspberry" else 0.35,
        interface_yield_strain=0.35,
        tear_onset=2.5 if tears else 0.0,
        tear_end=5.0 if tears else 0.0,
        grid_spacing=spacing,
        substep_rate=rate,
        friction=float(source["friction"]),
        damping=float(source["damping"]),
        hardening=float(source.get("hardening", 0.0)),
    )


def workcell_grid(background: str, spacing: float, center) -> tuple[tuple[float, ...], tuple[int, int, int]]:
    """Return the grid origin [m] and node counts covering the workcell around ``center``, and on the EBC table the
    transfer to the bowl and the reject dish."""
    origin = np.array([-0.12, -0.12, -0.016 if background == "ebc" else -0.008])
    resolution = np.array([121, 191 if background == "ebc" else 121, 121])
    if background == "ebc":
        padding = math.ceil(0.04 / spacing)
        origin[:2] -= padding * spacing
        resolution[:2] += padding
    return tuple((np.asarray(center) + origin).tolist()), tuple(int(r) for r in resolution)


def explicit_solver_config(specs, background: str, fingers: dict[str, str]) -> SolverGraspExplicitMPM.Config:
    """Configure the explicit solver for the task's berries: one field per berry when there are several.

    Args:
        specs: The berries' :class:`.tissue.TissueSpec`.
        background: ``studio`` or ``ebc``; the EBC table adds the punnet, bowl and reject dish.
        fingers: Body label pattern of each finger, keyed ``left`` and ``right``.
    """
    materials = [tissue_material(spec.profile) for spec in specs]
    # One grid serves every berry: the coarsest spacing and the fastest rate keep each stable.
    spacing = max(m.grid_spacing for m in materials)
    rate = max(m.substep_rate for m in materials)
    origin, resolution = workcell_grid(background, spacing, specs[0].offset)
    ebc = background == "ebc"
    return SolverGraspExplicitMPM.Config(
        grid_origin=origin,
        grid_resolution=resolution,
        grid_spacing=spacing,
        fields=len(specs),
        substep_rate=rate,
        damping=materials[0].damping,
        ground_friction=0.3 if ebc else materials[0].friction,
        vessels=tuple(tuple(row) for row in tableware_solids().tolist()) if ebc else (),
        vessel_corner_radius=PUNNET[4],
        vessel_wall_thickness=PUNNET[5],
        pads=tuple(PadConfig(fingers[side], _PAD_OFFSETS[side], _PAD_HALF_EXTENTS) for side in ("left", "right")),
        hardening=materials[0].hardening,
    )


class GraspExplicitMPMManager(NewtonMPMManager):
    """Newton MPM manager that builds a :class:`SolverGraspExplicitMPM`; it keeps the MPM particle attributes."""

    @classmethod
    def _create_solver(cls, model, solver_cfg: GraspExplicitMPMSolverCfg) -> SolverGraspExplicitMPM:
        if solver_cfg.solver_config is None:
            raise ValueError("Set GraspExplicitMPMSolverCfg.solver_config, e.g. with explicit_solver_config()")
        return SolverGraspExplicitMPM(model, solver_cfg.solver_config)


@configclass
class GraspExplicitMPMSolverCfg(MPMSolverCfg):
    """Configuration of :class:`SolverGraspExplicitMPM`. The implicit-MPM settings it inherits are unused."""

    class_type: type = GraspExplicitMPMManager

    solver_config: SolverGraspExplicitMPM.Config | None = None
    """Solver configuration; see :func:`explicit_solver_config`."""
