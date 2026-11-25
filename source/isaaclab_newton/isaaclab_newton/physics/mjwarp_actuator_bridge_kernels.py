# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the MuJoCo Warp actuator bridge's friction publication, load readback, and stiffening."""

from __future__ import annotations

import warp as wp

vec5 = wp.types.vector(length=5, dtype=wp.float32)
"""Solver-impedance vector type, matching MuJoCo Warp's ``dof_solimp`` element type."""


@wp.kernel(enable_backward=False)
def publish_dof_friction_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    newton_dof_to_slot: wp.array[wp.int32],
    friction_budget: wp.array[float],
    dof_frictionloss: wp.array2d[float],
):
    """Scatter an actuator's per-DOF dry-friction budget into the MuJoCo model."""
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return
    slot = newton_dof_to_slot[newton_dof]
    if slot < 0:
        return
    dof_frictionloss[world, mjc_dof] = friction_budget[slot]


@wp.kernel(enable_backward=False)
def gather_external_torque_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    newton_dof_to_slot: wp.array[wp.int32],
    qfrc_bias: wp.array2d[float],
    qfrc_constraint: wp.array2d[float],
    efc_type: wp.array2d[wp.int32],
    efc_id: wp.array2d[wp.int32],
    efc_force: wp.array2d[float],
    nefc: wp.array[wp.int32],
    friction_constraint_type: int,
    external_torque: wp.array[float],
):
    """Gather the external load on each managed DOF from the previous solve.

    The load the gearbox works against is the gravity/Coriolis bias plus the constraint
    forces, **minus** the DOF-friction constraint the actuator itself injected on the previous
    solve: leaving it in would make the load-dependent friction terms feed back on themselves.
    Applied body wrenches and other externally applied force channels are not included.
    """
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return
    slot = newton_dof_to_slot[newton_dof]
    if slot < 0:
        return
    own_friction = float(0.0)
    for row in range(nefc[world]):
        if efc_type[world, row] == friction_constraint_type and efc_id[world, row] == mjc_dof:
            own_friction += efc_force[world, row]
    external_torque[slot] = -qfrc_bias[world, mjc_dof] + qfrc_constraint[world, mjc_dof] - own_friction


@wp.kernel(enable_backward=False)
def stiffen_friction_constraint_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    newton_dof_to_slot: wp.array[wp.int32],
    solref: wp.vec2,
    solimp: vec5,
    dof_solref: wp.array2d[wp.vec2],
    dof_solimp: wp.array2d[vec5],
):
    """Write a stiff friction-constraint solver reference onto the managed DOFs."""
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return
    if newton_dof_to_slot[newton_dof] < 0:
        return
    dof_solref[world, mjc_dof] = solref
    dof_solimp[world, mjc_dof] = solimp


@wp.kernel(enable_backward=False)
def stiffen_model_friction_solver_kernel(
    dof_indices: wp.array[wp.uint32],
    solref: wp.vec2,
    solimp: vec5,
    model_solref: wp.array[wp.vec2],
    model_solimp: wp.array[vec5],
):
    """Mirror the stiffening onto the Newton model so a property re-sync reproduces it."""
    i = wp.tid()
    dof = wp.int32(dof_indices[i])
    model_solref[dof] = solref
    model_solimp[dof] = solimp
