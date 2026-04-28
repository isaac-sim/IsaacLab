# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled solver that alternates a rigid-body solver and VBD (cloth) per substep.

Supports two coupling modes (selected via :attr:`CoupledSolverCfg.coupling_mode`):

- ``"one_way"`` (default): Contact detection runs first so VBD can use the contacts,
  then the rigid solver advances independently. The rigid solver does not feel
  particle contact forces.
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
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    evaluate_body_particle_contact as _evaluate_body_particle_contact,
)
from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverVBD

if TYPE_CHECKING:
    from .newton_manager_cfg import CoupledSolverCfg

logger = logging.getLogger(__name__)

# Fixed upper bound on contact slots for the reaction kernel.  The kernel is
# launched with this many threads; threads beyond the actual contact count
# early-exit immediately so over-allocating is cheap.
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
    particle_qd: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    shape_material_mu: wp.array(dtype=wp.float32),
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    friction_epsilon: float,
    dt: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Newton's-third-law reaction from soft particles onto rigid bodies.

    Delegates to Newton's ``evaluate_body_particle_contact()`` for the contact
    force computation (normal + damping + Coulomb friction) so the model stays
    in sync with the VBD solver. The force on the particle is negated and
    applied as a wrench on the rigid body via ``body_f``.

    One thread per contact slot; threads beyond the actual contact count
    early-exit.

    The "previous" particle position required by the contact model is
    reconstructed from the current velocity (``particle_q - particle_qd * dt``)
    rather than read from a stored previous-state array. VBD mutates
    ``particle_q`` in place during its iteration, so the swapped state's
    ``particle_q`` is no longer a reliable snapshot of the prior substep.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    p_idx = contact_particle[tid]
    s_idx = contact_shape[tid]
    body_idx = shape_body[s_idx]
    if body_idx < 0:
        return

    # Reconstruct previous particle position from velocity so that
    # dx = particle_qd * dt regardless of what VBD wrote into stored states.
    p_pos = particle_q[p_idx]
    p_pos_prev = p_pos - particle_qd[p_idx] * dt

    # Delegate to Newton's canonical contact model
    f_on_particle, _ = _evaluate_body_particle_contact(
        p_idx,
        p_pos,
        p_pos_prev,
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
        dt,
    )

    # Newton's third law: negate particle force → rigid body wrench
    X_wb = body_q[body_idx]
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    com_w = wp.transform_point(X_wb, body_com[body_idx])

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
    2. Collision detection.
    3. Rigid step (Featherstone or MuJoCo) -- does not read soft-contact reactions.
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
        """One-way coupling: collide, then rigid step, then VBD."""
        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Collision detection (cloth-body contacts)
        self.collision_pipeline.collide(state_in, self.contacts)

        # 3. Rigid-body step (does not read soft-contact reactions)
        self._rigid_step(state_in, state_out, control, dt)

        # 4. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 5. VBD step -- particles only, reads updated rigid poses
        self.vbd.step(state_in, state_out, control, self.contacts, dt)

    def _step_two_way(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Two-way coupling: collide, inject reactions into body_f, rigid step, VBD step."""
        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Collision detection BEFORE rigid step
        self.collision_pipeline.collide(state_in, self.contacts)

        # 3. Inject contact reaction forces into body_f.
        #    state_out holds the previous substep's body_q (states swap each
        #    substep), used for finite-difference body velocity in friction.
        #    particle_q_prev is reconstructed from particle_qd inside the
        #    kernel because VBD mutates particle_q in place, so the swapped
        #    state's particle_q is not a clean prior-substep snapshot.
        if state_in.body_f is not None:
            self._apply_reactions(state_in, state_out, dt)

        # 4. Rigid-body step (reads body_f for soft-contact reactions)
        self._rigid_step(state_in, state_out, control, dt)

        # 5. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 6. VBD step -- uses same contacts detected in step 2
        self.vbd.step(state_in, state_out, control, self.contacts, dt)

    def _rigid_step(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies with the configured sub-solver."""
        model = self._model

        # set particle_count = 0 to disable particle simulation in robot solver
        saved_particle_count = model.particle_count
        model.particle_count = 0

        self.rigid_solver.step(state_in, state_out, control, None, dt)

        # restore original settings
        model.particle_count = saved_particle_count

    def _apply_reactions(self, state: State, state_prev: State, dt: float) -> None:
        """Launch the reaction kernel to inject normal + friction forces into body_f.

        Args:
            state: Current state with particle positions/velocities and body state.
            state_prev: Previous substep state whose ``body_q`` provides
                the reference poses for finite-difference body velocity.
            dt: Substep timestep [s].
        """
        model = self._model
        contacts = self.contacts

        # The kernel reconstructs particle_q_prev from particle_qd internally:
        # state_prev.particle_q is unreliable because VBD mutates particle_q
        # in place during its iteration, so the swapped state's particle_q is
        # not a clean snapshot of the prior substep.
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
                state.particle_qd,
                model.particle_radius,
                state.body_q,
                state_prev.body_q,
                state.body_qd,
                model.body_com,
                model.shape_body,
                model.shape_material_mu,
                float(model.soft_contact_ke),
                float(model.soft_contact_kd),
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
