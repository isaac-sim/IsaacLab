# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional fixed-effort inverse dynamics for NeRD sensor estimates.

The constrained projection preserves applied motor effort and unilateral friction cones.
It adjusts diagnostic acceleration, not the learned motion, and does not impose MuJoCo's
soft-contact law. Newton owns coordinate/contact conversion and Newton-Euler evaluation.
"""

import mujoco_warp as mjw
import newton
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.inverse import discrete_acc
from newton._src.solvers.mujoco.kernels import convert_rigid_forces_from_mj_kernel

_Vec64 = wp.types.vector(length=64, dtype=wp.float32)


def _make_projection_kernel(block_size):
    @wp.kernel(enable_backward=False, module="unique")
    def solve_projection_kernel(
        inverse: wp.array3d[wp.float32],
        rhs: wp.array2d[wp.float32],
        rho: wp.array[wp.float32],
        active: wp.array2d[wp.bool],
        contact: wp.array2d[wp.bool],
        normal: wp.array2d[wp.int64],
        tangent1: wp.array2d[wp.int64],
        tangent2: wp.array2d[wp.int64],
        iterations: int,
        tolerance: wp.float32,
        force: wp.array2d[wp.float32],
        primal: wp.array2d[wp.float32],
        dual_residual: wp.array2d[wp.float32],
        iteration_counts: wp.array[int],
    ):
        world, lane = wp.tid()
        count = rhs.shape[1]
        row = _Vec64()
        b = wp.float32(0.0)
        enabled = False
        friction = False
        ni, ti, tj = int(0), int(0), int(0)
        if lane < count:
            enabled = active[world, lane]
            b = wp.where(enabled, rhs[world, lane], wp.float32(0.0))
            friction = contact[world, lane]
            ni = int(normal[world, lane])
            ti = int(tangent1[world, lane])
            tj = int(tangent2[world, lane])
        extra_enabled = int(0)
        if wp.static(block_size == 32):
            if lane + 32 < count:
                extra_enabled = int(active[world, lane + 32])
        active_count = wp.tile_sum(wp.tile(int(enabled) + extra_enabled))[0]
        if wp.static(block_size == 32):
            if active_count > 32:
                return
        else:
            if active_count <= 32:
                return
        if enabled:
            for j in range(64):
                if j < count and j < block_size:
                    row[j] = inverse[world, lane, j]
        penalty = rho[world]
        z, dual, x, previous = wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
        performed = int(0)
        for iteration in range(iterations):
            if active_count == 0:
                break
            vector = wp.tile(b + penalty * (z - dual))
            x = wp.float32(0.0)
            if active_count <= 16:
                for j in range(16):
                    x += row[j] * vector[j]
            elif active_count <= 32:
                for j in range(32):
                    x += row[j] * vector[j]
            else:
                if wp.static(block_size == 64):
                    for j in range(64):
                        x += row[j] * vector[j]
            values = wp.tile(x + dual)
            n, a, c = values[ni], values[ti], values[tj]
            radius = wp.float32(0.0)
            if friction:
                radius = wp.sqrt(a * a + c * c)
            magnitude = wp.max((n + radius) * wp.float32(0.5), wp.float32(0.0))
            factor = wp.float32(1.0)
            if n < radius:
                factor = magnitude / wp.max(radius, wp.float32(1.0e-30))
            previous = z
            z = (x + dual) * factor
            if lane == ni:
                z = n
                if n < radius:
                    z = magnitude
            if not enabled:
                z = wp.float32(0.0)
            dual += x - z
            performed = iteration + 1
            if tolerance > wp.float32(0.0) and performed % 8 == 0:
                residual = wp.tile_max(wp.tile(wp.max(wp.abs(x - z), wp.abs(z - previous))))[0]
                if residual <= tolerance:
                    break
        if lane < count:
            force[world, lane] = z
            primal[world, lane] = x - z
            dual_residual[world, lane] = penalty * (z - previous)
        if lane == 0:
            iteration_counts[world] = performed
        if wp.static(block_size == 32):
            if lane + 32 < count:
                force[world, lane + 32] = wp.float32(0.0)
                primal[world, lane + 32] = wp.float32(0.0)
                dual_residual[world, lane + 32] = wp.float32(0.0)

    wp.set_module_options({"max_unroll": 64}, module=solve_projection_kernel.module)
    return solve_projection_kernel


_PROJECTION_KERNELS = (_make_projection_kernel(32), _make_projection_kernel(64))


@wp.kernel(enable_backward=False)
def _prepare_projection(
    counts: wp.array[int],
    types: wp.array2d[int],
    ids: wp.array2d[int],
    addresses: wp.array2d[int],
    friction: wp.array2d[float],
    source_jacobian: wp.array3d[float],
    source_mass: wp.array2d[float],
    mass_indices: wp.array2d[int],
    source_smooth: wp.array2d[float],
    source_acceleration: wp.array2d[float],
    active: wp.array2d[wp.bool],
    contact: wp.array2d[wp.bool],
    normal: wp.array2d[wp.int64],
    tangent1: wp.array2d[wp.int64],
    tangent2: wp.array2d[wp.int64],
    scale: wp.array2d[wp.float64],
    jacobian: wp.array3d[wp.float64],
    mass: wp.array3d[wp.float64],
    smooth: wp.array2d[wp.float64],
    acceleration: wp.array2d[wp.float64],
):
    world, row = wp.tid()
    enabled = row < counts[world]
    is_contact = enabled and types[world, row] == 7
    ni, t1, t2 = row, row, row
    coefficient = wp.float64(1.0)
    if is_contact:
        contact_id = wp.max(ids[world, row], 0)
        ni = wp.max(addresses[contact_id, 0], 0)
        t1 = wp.max(addresses[contact_id, 1], 0)
        t2 = wp.max(addresses[contact_id, 2], 0)
        if row == t1:
            coefficient = wp.float64(friction[contact_id, 0])
        if row == t2:
            coefficient = wp.float64(friction[contact_id, 1])
    active[world, row] = enabled
    contact[world, row] = is_contact
    normal[world, row] = wp.int64(ni)
    tangent1[world, row] = wp.int64(t1)
    tangent2[world, row] = wp.int64(t2)
    scale[world, row] = coefficient
    for column in range(mass.shape[1]):
        value = wp.float64(0.0)
        if enabled:
            value = wp.float64(source_jacobian[world, row, column])
        jacobian[world, row, column] = value
        if row < mass.shape[1]:
            value = wp.float64(0.0)
            index = mass_indices[row, column]
            if index >= 0:
                value = wp.float64(source_mass[world, index])
            mass[world, row, column] = value
    if row < mass.shape[1]:
        smooth[world, row] = wp.float64(source_smooth[world, row])
        acceleration[world, row] = wp.float64(source_acceleration[world, row])


@wp.func
def _to_float32(value: wp.float64) -> wp.float32:
    return wp.float32(value)


def _make_setup_kernels(num_dofs, rows, lower_bound):
    @wp.kernel(enable_backward=False, module="unique")
    def form_projection(
        mass: wp.array3d[wp.float64],
        jacobian: wp.array3d[wp.float64],
        scale: wp.array2d[wp.float64],
        smooth: wp.array2d[wp.float64],
        acceleration: wp.array2d[wp.float64],
        counts: wp.array[int],
        mass_identity: wp.array2d[wp.float64],
        mass_inverse: wp.array3d[wp.float64],
        hessian: wp.array3d[wp.float64],
        rhs: wp.array2d[wp.float32],
        rho: wp.array[wp.float64],
    ):
        world, lane = wp.tid()
        if counts[world] <= lower_bound or counts[world] > rows:
            return
        m = wp.tile_load(mass[world], shape=(num_dofs, num_dofs))
        eye_m = wp.tile_load(mass_identity, shape=(num_dofs, num_dofs))
        mi = wp.tile_cholesky_solve(wp.tile_cholesky(m), eye_m)
        j = wp.tile_load(jacobian[world], shape=(rows, num_dofs))
        s = wp.tile_reshape(wp.tile_load(scale[world], shape=rows), shape=(rows, 1))
        scaled_j = wp.tile_map(wp.mul, j, wp.tile_broadcast(s, shape=(rows, num_dofs)))
        jm = wp.tile_matmul(scaled_j, mi)
        h = wp.tile_matmul(jm, wp.tile_transpose(scaled_j))
        a = wp.tile_reshape(wp.tile_load(acceleration[world], shape=num_dofs), shape=(num_dofs, 1))
        f = wp.tile_reshape(wp.tile_load(smooth[world], shape=num_dofs), shape=(num_dofs, 1))
        b = wp.tile_matmul(jm, wp.tile_matmul(m, a) - f)
        trace = wp.float64(0.0)
        for i in range(rows):
            trace += h[i, i]
        penalty = wp.max(trace / wp.float64(wp.max(counts[world], 1)), wp.float64(1.0e-8))
        wp.tile_store(mass_inverse[world], mi)
        wp.tile_store(hessian[world], h)
        wp.tile_store(rhs[world], wp.tile_map(_to_float32, wp.tile_reshape(b, shape=(rows,))))
        rho[world] = penalty

    @wp.kernel(enable_backward=False, module="unique")
    def factor_projection(
        hessian: wp.array3d[wp.float64],
        rho: wp.array[wp.float64],
        counts: wp.array[int],
        identity: wp.array2d[wp.float64],
        step_inverse: wp.array3d[wp.float32],
    ):
        world, lane = wp.tid()
        if counts[world] <= lower_bound or counts[world] > rows:
            return
        h = wp.tile_load(hessian[world], shape=(rows, rows), storage="shared")
        diagonal = wp.tile_ones(shape=rows, dtype=wp.float64) * rho[world]
        factor = wp.tile_diag_add(h, diagonal)
        wp.tile_cholesky_inplace(factor)
        inverse = wp.tile_load(identity, shape=(rows, rows), storage="shared")
        wp.tile_cholesky_solve_inplace(factor, inverse)
        wp.tile_store(step_inverse[world], wp.tile_map(_to_float32, inverse))

    return form_projection, factor_projection


@wp.kernel
def _prediction_acceleration(
    previous: wp.array2d[float], predicted: wp.array2d[float], dt: float, acceleration: wp.array2d[float]
):
    world, dof = wp.tid()
    acceleration[world, dof] = (predicted[world, dof] - previous[world, dof]) / dt


@wp.kernel
def _write_reaction(
    force: wp.array2d[float],
    scale: wp.array2d[wp.float64],
    jacobian: wp.array3d[wp.float64],
    smooth: wp.array2d[wp.float64],
    reaction: wp.array2d[float],
    generalized_force: wp.array2d[wp.float64],
):
    world, row = wp.tid()
    reaction[world, row] = float(wp.float64(force[world, row]) * scale[world, row])
    if row < generalized_force.shape[1]:
        total = smooth[world, row]
        for constraint in range(force.shape[1]):
            total += jacobian[world, constraint, row] * wp.float64(force[world, constraint]) * scale[world, constraint]
        generalized_force[world, row] = total


@wp.kernel
def _write_acceleration(
    inverse_mass: wp.array3d[wp.float64], generalized_force: wp.array2d[wp.float64], acceleration: wp.array2d[float]
):
    world, row = wp.tid()
    value = wp.float64(0.0)
    for dof in range(acceleration.shape[1]):
        value += inverse_mass[world, row, dof] * generalized_force[world, dof]
    acceleration[world, row] = float(value)


@wp.kernel
def _cast_penalty(source: wp.array[wp.float64], target: wp.array[float]):
    world = wp.tid()
    target[world] = float(source[world])


@wp.kernel
def _reset_wrenches(worlds: wp.array[int], mask: wp.array[wp.bool], values: wp.array[wp.spatial_vector]):
    body = wp.tid()
    world = worlds[body]
    if world < 0:
        world = mask.shape[0] - 1
    if mask[world]:
        values[body] = wp.spatial_vectorf(0.0)


class _ForceReconstruction:
    """Own inverse-dynamics conversion and fixed-capacity buffers for one NeRD solver.

    Minimize ``(a-a_predicted).T M (a-a_predicted) / 2`` subject to
    ``M a = applied + passive - bias + J.T f`` and unilateral elliptic cones.
    Only the diagnostic acceleration is projected; the caller retains predicted joint state.
    """

    def __init__(self, model, dt):
        self.model, self.dt = model, dt
        self.solver = newton.solvers.SolverMuJoCo(
            model,
            separate_worlds=True,
            njmax=60,
            nconmax=60,
            cone="elliptic",
            jacobian="dense",
            integrator="implicitfast",
            use_mujoco_contacts=False,
            enable_sleeping=False,
        )
        m, d = self.solver.mjw_model, self.solver.mjw_data
        if not 1 <= m.nv <= 60 or d.njmax != 60:
            raise ValueError("NeRD force reconstruction supports 1-60 DOFs and at most 60 constraint rows per world.")
        if m.neq or m.ntendon or np.any(self.solver.mj_model.dof_frictionloss):
            raise ValueError("NeRD force reconstruction does not support equalities, tendons or joint friction loss.")
        if np.any(~np.isin(self.solver.mj_model.geom_condim, (1, 3))):
            raise ValueError("NeRD force reconstruction supports frictionless or 3D elliptic contacts only.")
        if np.any(model.shape_material_ka.numpy()):
            raise ValueError("NeRD force reconstruction does not support adhesive contacts.")
        m.opt.timestep.fill_(dt)
        worlds, nv, rows, device = d.nworld, m.nv, d.njmax, model.device
        self._previous_q = wp.empty_like(d.qpos)
        self._previous_qd = wp.empty_like(d.qvel)
        self._active = wp.zeros((worlds, rows), dtype=wp.bool, device=device)
        self._contact = wp.zeros_like(self._active)
        self._normal = wp.zeros((worlds, rows), dtype=wp.int64, device=device)
        self._tangent1, self._tangent2 = wp.zeros_like(self._normal), wp.zeros_like(self._normal)
        self._scale = wp.zeros((worlds, rows), dtype=wp.float64, device=device)
        self._jacobian = wp.zeros((worlds, rows, nv), dtype=wp.float64, device=device)
        self._mass = wp.zeros((worlds, nv, nv), dtype=wp.float64, device=device)
        self._mass_inverse = wp.zeros_like(self._mass)
        self._smooth = wp.zeros((worlds, nv), dtype=wp.float64, device=device)
        self._acceleration = wp.zeros_like(self._smooth)
        self._generalized_force = wp.zeros_like(self._smooth)
        self._hessian = wp.zeros((worlds, rows, rows), dtype=wp.float64, device=device)
        self._rhs = wp.zeros((worlds, rows), dtype=float, device=device)
        self._rho = wp.zeros(worlds, dtype=wp.float64, device=device)
        self._rho_float = wp.zeros(worlds, dtype=float, device=device)
        self._inverse = wp.zeros((worlds, rows, rows), dtype=float, device=device)
        self._force, self._primal, self._dual = [wp.zeros_like(self._rhs) for _ in range(3)]
        self._iterations = wp.zeros(worlds, dtype=int, device=device)
        self._mass_identity = wp.array(np.eye(nv), dtype=wp.float64, device=device)
        self._identity = wp.array(np.eye(rows), dtype=wp.float64, device=device)
        indices = np.full((nv, nv), -1, dtype=np.int32)
        indices[np.repeat(np.arange(nv), np.diff(m.M_mulm_rowadr.numpy())), m.M_mulm_col.numpy()] = (
            m.M_mulm_madr.numpy()
        )
        self._mass_indices = wp.array(indices, dtype=int, device=device)
        self._setup_kernels = tuple(
            (*_make_setup_kernels(nv, capacity, lower), 32 if capacity <= 32 else 128)
            for capacity, lower in ((16, -1), (32, 16), (rows, 32))
        )
        self._row_ids = torch.arange(rows, device=device.alias)[None]
        self._contact_ids = torch.arange(d.naconmax, device=device.alias)

    def prepare(self, state, control, contacts):
        """Read starting state, applied effort and collision geometry before in-place prediction writes."""
        m, d = self.solver.mjw_model, self.solver.mjw_data
        self.solver._update_mjc_data(d, self.model, state)
        self.solver._apply_mjc_control(self.model, state, control, d)
        torch._assert_async(
            (wp.to_torch(contacts.rigid_contact_count) <= min(contacts.rigid_contact_max, d.naconmax)).all(),
            "NeRD reconstruction collision capacity overflowed.",
        )
        self.solver._convert_contacts_to_mjwarp(self.model, state, contacts)
        wp.copy(self._previous_q, d.qpos)
        wp.copy(self._previous_qd, d.qvel)
        mjw.fwd_position(m, d)
        mjw.fwd_velocity(m, d)
        mjw.fwd_acceleration(m, d)

    def update(self, state):
        """Project the prediction and fill Newton's existing incoming-wrench storage."""
        m, d = self.solver.mjw_model, self.solver.mjw_data
        self.solver._update_mjc_data(d, self.model, state)
        wp.launch(
            _prediction_acceleration,
            d.qvel.shape,
            inputs=[self._previous_qd, d.qvel, self.dt],
            outputs=[d.qacc],
            device=self.model.device,
        )
        wp.copy(d.qpos, self._previous_q)
        wp.copy(d.qvel, self._previous_qd)
        discrete_acc(m, d, d.qacc)
        counts, types = wp.to_torch(d.nefc), wp.to_torch(d.efc.type)
        active = self._row_ids < counts[:, None]
        torch._assert_async((counts <= d.njmax).all(), "NeRD reconstruction constraint capacity overflowed.")
        torch._assert_async(
            (wp.to_torch(d.nacon) <= d.naconmax).all(), "NeRD reconstruction contact capacity overflowed."
        )
        torch._assert_async(
            (~active | (types == 3) | (types == 4) | (types == 5) | (types == 7)).all(),
            "NeRD reconstruction encountered an unsupported constraint type.",
        )
        dimensions = wp.to_torch(d.contact.dim)
        torch._assert_async(
            ((self._contact_ids >= wp.to_torch(d.nacon)[0]) | (dimensions == 1) | (dimensions == 3)).all(),
            "NeRD reconstruction supports frictionless or 3D contacts only.",
        )
        wp.launch(
            _prepare_projection,
            (d.nworld, d.njmax),
            inputs=[
                d.nefc,
                d.efc.type,
                d.efc.id,
                d.contact.efc_address,
                wp.from_torch(wp.to_torch(d.contact.friction)),
                d.efc.J,
                d.M,
                self._mass_indices,
                d.qfrc_smooth,
                d.qacc,
            ],
            outputs=[
                self._active,
                self._contact,
                self._normal,
                self._tangent1,
                self._tangent2,
                self._scale,
                self._jacobian,
                self._mass,
                self._smooth,
                self._acceleration,
            ],
            device=self.model.device,
        )
        for form, factor, block in self._setup_kernels:
            wp.launch_tiled(
                form,
                dim=d.nworld,
                block_dim=block,
                inputs=[
                    self._mass,
                    self._jacobian,
                    self._scale,
                    self._smooth,
                    self._acceleration,
                    d.nefc,
                    self._mass_identity,
                ],
                outputs=[self._mass_inverse, self._hessian, self._rhs, self._rho],
                device=self.model.device,
            )
            wp.launch_tiled(
                factor,
                dim=d.nworld,
                block_dim=block,
                inputs=[self._hessian, self._rho, d.nefc, self._identity],
                outputs=[self._inverse],
                device=self.model.device,
            )
        wp.launch(_cast_penalty, d.nworld, inputs=[self._rho], outputs=[self._rho_float], device=self.model.device)
        for block, kernel in zip((32, 64), _PROJECTION_KERNELS):
            wp.launch(
                kernel,
                dim=(d.nworld, block),
                block_dim=block,
                inputs=[
                    self._inverse,
                    self._rhs,
                    self._rho_float,
                    self._active,
                    self._contact,
                    self._normal,
                    self._tangent1,
                    self._tangent2,
                    1000,
                    1e-4,
                ],
                outputs=[self._force, self._primal, self._dual, self._iterations],
                device=self.model.device,
            )
        wp.launch(
            _write_reaction,
            (d.nworld, d.njmax),
            inputs=[self._force, self._scale, self._jacobian, self._smooth],
            outputs=[d.efc.force, self._generalized_force],
            device=self.model.device,
        )
        wp.launch(
            _write_acceleration,
            (d.nworld, m.nv),
            inputs=[self._mass_inverse, self._generalized_force],
            outputs=[d.qacc],
            device=self.model.device,
        )
        torch._assert_async(
            torch.isfinite(wp.to_torch(d.qacc)).all() & torch.isfinite(wp.to_torch(d.efc.force)).all(),
            "NeRD force reconstruction produced nonfinite values.",
        )
        if state.body_parent_f is not None:
            mjw.rne_postconstraint(m, d)
            wp.launch(
                convert_rigid_forces_from_mj_kernel,
                dim=self.solver.mjc_body_to_newton.shape,
                inputs=[
                    self.solver.mjc_body_to_newton,
                    m.body_rootid,
                    m.opt.gravity,
                    d.xipos,
                    d.subtree_com,
                    d.cacc,
                    d.cvel,
                    d.cfrc_int,
                ],
                outputs=[None, state.body_parent_f],
                device=self.model.device,
            )
            torch._assert_async(
                torch.isfinite(wp.to_torch(state.body_parent_f)).all(),
                "NeRD wrench reconstruction produced nonfinite values.",
            )

    def reset(self, state, world_mask):
        """Discard incoming wrenches for reset worlds; each reconstruction starts from zero dual state."""
        if state.body_parent_f is not None:
            if world_mask is None:
                state.body_parent_f.zero_()
            else:
                wp.launch(
                    _reset_wrenches,
                    self.model.body_count,
                    inputs=[self.model.body_world, world_mask, state.body_parent_f],
                    device=self.model.device,
                )
