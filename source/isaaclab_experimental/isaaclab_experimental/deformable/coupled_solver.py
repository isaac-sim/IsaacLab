# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled solver that alternates a rigid-body solver and VBD (cloth) per substep.

Supports two coupling modes (selected via :attr:`CoupledSolverCfg.coupling_mode`):

- ``"one_way"`` (default): Rigid solver advances first, then VBD reads the updated
  body poses for cloth-body contacts. The rigid solver does not feel particle contact forces.
- ``"two_way"``: Same-substep two-way coupling with normal + Coulomb friction forces.
  Contact detection runs first, reaction forces (normal and tangential friction) are
  injected into ``body_f``, then the rigid solver reads ``body_f`` and feels resistance
  from the deformable object. The friction reaction provides the force needed for the
  actuators to carry the object against gravity during a lift.

The rigid solver can be either :class:`SolverFeatherstone` or :class:`SolverMuJoCo`.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Literal

import warp as wp
from newton import CollisionPipeline, Contacts, Control, Model, State
from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverVBD

if TYPE_CHECKING:
    from .newton_manager_cfg import CoupledSolverCfg

logger = logging.getLogger(__name__)

# Maximum contact slots for the reaction kernel. Threads beyond the actual
# contact count early-exit immediately so over-allocating is cheap.
_MAX_REACTION_CONTACTS: int = 2048


# ---------------------------------------------------------------------------
# Warp kernels for two-way coupling
# ---------------------------------------------------------------------------


@wp.kernel
def _kernel_body_particle_reaction(
    contact_count: wp.array(dtype=wp.int32),
    contact_particle: wp.array(dtype=wp.int32),
    contact_shape: wp.array(dtype=wp.int32),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    shape_material_mu: wp.array(dtype=wp.float32),
    soft_contact_ke: float,
    soft_contact_mu: float,
    friction_epsilon: float,
    dt: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Newton's-third-law reaction (normal + Coulomb friction) from soft particles onto rigid bodies.

    Mirrors the complete contact model from ``evaluate_body_particle_contact()``
    in ``newton/_src/solvers/vbd/rigid_vbd_kernels.py``. One thread per contact
    slot; threads beyond the actual contact count early-exit.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    p_idx = contact_particle[tid]
    s_idx = contact_shape[tid]
    body_idx = shape_body[s_idx]
    if body_idx < 0:
        return

    X_wb = body_q[body_idx]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    n = contact_normal[tid]

    penetration = -(wp.dot(n, particle_q[p_idx] - bx) - particle_radius[p_idx])
    if penetration <= 0.0:
        return

    normal_load = soft_contact_ke * penetration
    f_on_particle = n * normal_load

    com_w = wp.transform_point(X_wb, body_com[body_idx])
    mu = wp.sqrt(soft_contact_mu * shape_material_mu[s_idx])

    if mu > 0.0:
        body_v_s = body_qd[body_idx]
        body_lin_v = wp.spatial_top(body_v_s)
        body_ang_v = wp.spatial_bottom(body_v_s)
        r = bx - com_w
        bv = body_lin_v + wp.cross(body_ang_v, r) + wp.transform_vector(X_wb, contact_body_vel[tid])

        dx = particle_q[p_idx] - particle_q_prev[p_idx]
        relative_translation = dx - bv * dt

        dot_nu = wp.dot(n, relative_translation)
        u_t = relative_translation - n * dot_nu
        u_norm = wp.length(u_t)
        eps_u = friction_epsilon * dt

        if u_norm > 0.0:
            if u_norm > eps_u:
                f1_over_x = 1.0 / u_norm
            else:
                f1_over_x = (-u_norm / eps_u + 2.0) / eps_u
            f_on_particle = f_on_particle - (mu * normal_load * f1_over_x) * u_t

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


# ---------------------------------------------------------------------------
# CoupledSolver
# ---------------------------------------------------------------------------

CouplingMode = Literal["one_way", "two_way"]


class CoupledSolver:
    """Coupled rigid-body + VBD solver for rigid-body/cloth interaction.

    Supports two coupling modes:

    **one_way** (default):

    1. Clear forces.
    2. Rigid step (Featherstone or MuJoCo).
    3. Collision detection.
    4. VBD step (particles only).

    **two_way** (same-substep two-way coupling with normal + friction):

    1. Clear forces.
    2. Collision detection.
    3. Inject contact reaction forces (normal + Coulomb friction) into ``body_f``.
    4. Rigid step (reads ``body_f`` -- fingers feel resistance).
    5. VBD step (uses same contacts).
    """

    def __init__(
        self,
        model: Model,
        cfg: CoupledSolverCfg,
        collision_pipeline: CollisionPipeline,
        contacts: Contacts,
    ):
        """Initialize the coupled solver.

        Args:
            model: The Newton model.
            cfg: Coupled solver configuration containing rigid solver, VBD,
                and coupling mode settings.
            collision_pipeline: Collision pipeline for cloth-body contacts.
            contacts: Contacts buffer for the collision pipeline.
        """
        self._model = model
        self._coupling_mode = cfg.coupling_mode

        # --- Build rigid solver from config ---
        rigid_solver_cfg = cfg.rigid_solver_cfg
        if hasattr(rigid_solver_cfg, "to_dict"):
            rigid_solver_cfg = rigid_solver_cfg.to_dict()
        rigid_solver_type = rigid_solver_cfg.get("solver_type", "mujoco_warp")
        self._rigid_solver_type = rigid_solver_type
        self._is_featherstone = rigid_solver_type == "featherstone"

        if rigid_solver_type == "mujoco_warp":
            valid_keys = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"}
            rigid_kwargs = {k: v for k, v in rigid_solver_cfg.items() if k in valid_keys}
            logger.info("Coupled: Creating SolverMuJoCo with args: %s", rigid_kwargs)
            self.rigid_solver = SolverMuJoCo(model, **rigid_kwargs)
        elif rigid_solver_type == "featherstone":
            valid_keys = set(inspect.signature(SolverFeatherstone.__init__).parameters) - {"self", "model"}
            rigid_kwargs = {k: v for k, v in rigid_solver_cfg.items() if k in valid_keys}
            logger.info("Coupled: Creating SolverFeatherstone with args: %s", rigid_kwargs)
            self.rigid_solver = SolverFeatherstone(model, **rigid_kwargs)
        else:
            raise ValueError(f"Unsupported rigid solver type for coupled solver: {rigid_solver_type}")

        # --- Build VBD solver from config ---
        vbd_cfg = cfg.vbd_cfg
        if hasattr(vbd_cfg, "to_dict"):
            vbd_cfg = vbd_cfg.to_dict()
        valid_keys = set(inspect.signature(SolverVBD.__init__).parameters) - {"self", "model"}
        vbd_kwargs = {k: v for k, v in vbd_cfg.items() if k in valid_keys}
        vbd_kwargs["integrate_with_external_rigid_solver"] = True
        self.vbd = SolverVBD(model, **vbd_kwargs)

        # Collision pipeline and contacts buffer (owned by this solver)
        self.collision_pipeline = collision_pipeline
        self.contacts = contacts

        logger.info(
            "CoupledSolver initialized: %s + VBD(%s), coupling_mode=%s",
            rigid_solver_type,
            {k: v for k, v in vbd_kwargs.items() if k != "integrate_with_external_rigid_solver"},
            cfg.coupling_mode,
        )

    def rebuild_bvh(self, state: State) -> None:
        """Rebuild BVH for VBD collision detection."""
        self.vbd.rebuild_bvh(state)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """One coupled substep.

        Args:
            state_in: Current state (read/write).
            state_out: Next state (write).
            control: Joint-level control inputs.
            contacts: Ignored -- the solver uses its own internal contacts.
            dt: Substep timestep [s].
        """
        if self._coupling_mode == "one_way":
            self._step_one_way(state_in, state_out, control, dt)
        else:
            self._step_two_way(state_in, state_out, control, dt)

    def _step_one_way(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """One-way coupling: rigid step, then collide, then VBD."""
        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Rigid-body step
        self._rigid_step(state_in, state_out, control, dt)

        # 3. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 4. Collision detection (cloth-body contacts)
        self.collision_pipeline.collide(state_in, self.contacts)

        # 5. VBD step -- particles only, reads updated rigid poses
        self.vbd.step(state_in, state_out, control, self.contacts, dt)

    def _step_two_way(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Two-way coupling: collide, inject reactions into body_f, rigid step, VBD step."""
        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Collision detection BEFORE rigid step
        self.collision_pipeline.collide(state_in, self.contacts)

        # 3. Inject contact reaction forces into body_f
        if state_in.body_f is not None:
            self._apply_reactions(state_in, dt)

        # 4. Rigid-body step (reads body_f for soft-contact reactions)
        self._rigid_step(state_in, state_out, control, dt)

        # 5. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 6. VBD step -- uses same contacts detected in step 2
        self.vbd.step(state_in, state_out, control, self.contacts, dt)

    def _rigid_step(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies with the configured sub-solver."""
        model = self._model

        saved_particle_count = model.particle_count
        model.particle_count = 0

        self.rigid_solver.step(state_in, state_out, control, None, dt)

        model.particle_count = saved_particle_count

    def _apply_reactions(self, state: State, dt: float) -> None:
        """Launch the reaction kernel to inject normal + friction forces into body_f."""
        model = self._model
        contacts = self.contacts

        wp.launch(
            _kernel_body_particle_reaction,
            dim=_MAX_REACTION_CONTACTS,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                state.particle_q,
                state.particle_q,
                model.particle_radius,
                state.body_q,
                state.body_qd,
                model.body_com,
                model.shape_body,
                model.shape_material_mu,
                float(model.soft_contact_ke),
                float(model.soft_contact_mu),
                float(self.vbd.friction_epsilon),
                float(dt),
                state.body_f,
            ],
        )

    def notify_model_changed(self, change: int) -> None:
        """Forward model-change notifications to both sub-solvers."""
        self.rigid_solver.notify_model_changed(change)
        self.vbd.notify_model_changed(change)
