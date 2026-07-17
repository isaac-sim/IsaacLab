# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled Featherstone + VBD Newton manager."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverFeatherstone, SolverVBD

from .kernels import _kernel_body_particle_reaction
from .newton_manager_cfg import CoupledFeatherstoneVBDSolverCfg
from .vbd_manager import NewtonVBDManager


class NewtonCoupledFeatherstoneVBDManager(NewtonVBDManager):
    """:class:`NewtonVBDManager` specialization for the coupled Featherstone + VBD
    solver. Reuses the VBD manager's deformable stage handling and adds a
    custom rigid/soft coupling step.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.

    .. deprecated:: 0.5.0
        Replace Featherstone with MJWarp and migrate to
        :class:`isaaclab_contrib.coupling.CouplerProxyCfg`, or retain this
        deprecated path.
    """

    _rigid_solver: SolverFeatherstone | None = None
    _soft_solver: SolverVBD | None = None
    _coupling_mode: str | None = None

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        from isaaclab.physics import PhysicsManager

        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Notify solver of model changes
        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._rigid_solver.notify_model_changed(change)
                    cls._soft_solver.notify_model_changed(change)
                NewtonManager._model_changes = set()
        super().step()

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Clear deprecated coupling state."""
        super()._solver_specific_clear()
        cls._rigid_solver = None
        cls._soft_solver = None
        cls._coupling_mode = None

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledFeatherstoneVBDSolverCfg) -> None:
        """Construct a custom coupling between two solvers and populate the
        base-class slots.

        VBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        if solver_cfg.coupling_mode not in {"kinematic", "one_way", "two_way"}:
            raise ValueError(
                f"Unknown coupling_mode={solver_cfg.coupling_mode!r}; "
                "expected one of {'kinematic', 'one_way', 'two_way'}."
            )
        if not solver_cfg.soft_solver_cfg.integrate_with_external_rigid_solver:
            raise ValueError("The coupled manager requires VBD external rigid-body integration.")
        if NewtonManager._report_contacts:
            raise NotImplementedError("Newton contact sensors are not supported by the coupled manager.")

        cls._coupling_mode = solver_cfg.coupling_mode

        cls._rigid_solver = SolverFeatherstone(
            model, **cls._filter_solver_kwargs(SolverFeatherstone, solver_cfg.rigid_solver_cfg)
        )
        cls._soft_solver = SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg.soft_solver_cfg))

        # Dummy solver for the newtonmanager
        NewtonManager._solver = SolverBase(model)

        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

        if solver_cfg.coupling_mode == "kinematic":
            cls._gravity_zero = wp.zeros(1, dtype=wp.vec3)
            cls._gravity_saved = wp.clone(model.gravity)
            # Save original PD gains and create zeroed versions for kinematic step
            cls._ke_saved = wp.clone(model.joint_target_ke)
            cls._kd_saved = wp.clone(model.joint_target_kd)
            cls._ke_zero = wp.zeros_like(model.joint_target_ke)
            cls._kd_zero = wp.zeros_like(model.joint_target_kd)

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Reset both sub-solvers."""
        if world_mask is None:
            return
        cls._rigid_solver.reset(cls._state_0, world_mask=world_mask, flags=0)
        cls._soft_solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _step_solver(
        cls, state_in: State, state_out: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """One coupled substep.

        Args:
            state_in: Current state (read/write).
            state_out: Next state (write).
            control: Joint-level control inputs.
            contacts: Ignored -- the solver uses its own internal contacts.
            dt: Substep timestep [s].
        """
        if cls._coupling_mode == "kinematic":
            cls._step_kinematic(state_in, state_out, control, substep_dt)
        elif cls._coupling_mode == "one_way":
            cls._step_one_way(state_in, state_out, control, substep_dt)
        elif cls._coupling_mode == "two_way":
            cls._step_two_way(state_in, state_out, control, substep_dt)
        else:
            raise ValueError(
                f"Unknown coupling_mode={cls._coupling_mode!r}; expected one of {{'kinematic', 'one_way', 'two_way'}}."
            )

    @classmethod
    def _simulate_physics_only(cls) -> None:
        # Rebuild BVH once per step for solvers that require it (e.g. VBD cloth).
        if hasattr(cls._soft_solver, "rebuild_bvh"):
            cls._soft_solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()

    @classmethod
    def _step_kinematic(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies kinematically, then solve VBD."""
        model = cls._model
        state_out.clear_forces()

        shape_contact_pair_count = model.shape_contact_pair_count
        model.gravity.assign(cls._gravity_zero)
        model.shape_contact_pair_count = 0
        model.joint_target_ke.assign(cls._ke_zero)
        model.joint_target_kd.assign(cls._kd_zero)
        state_in.joint_qd.assign(control.joint_target_vel)

        cls._rigid_step(state_in, state_out, control, dt)

        model.gravity.assign(cls._gravity_saved)
        model.shape_contact_pair_count = shape_contact_pair_count
        model.joint_target_ke.assign(cls._ke_saved)
        model.joint_target_kd.assign(cls._kd_saved)
        cls._collision_pipeline.collide(state_in, cls._contacts)
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_one_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance Featherstone before VBD without reaction forces."""
        state_out.clear_forces()
        cls._collision_pipeline.collide(state_in, cls._contacts)
        cls._rigid_step(state_in, state_out, control, dt)
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_two_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance Featherstone and VBD with deformable reaction forces."""
        state_out.clear_forces()
        cls._collision_pipeline.collide(state_in, cls._contacts)
        if state_in.body_f is not None:
            cls._apply_reactions(state_in, state_out, dt)
        cls._rigid_step(state_in, state_out, control, dt)
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _rigid_step(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies without integrating VBD particles."""
        model = cls._model
        particle_count = model.particle_count
        if particle_count == 0:
            cls._rigid_solver.step(state_in, state_out, control, None, dt)
            return

        particle_f = state_in.particle_f
        model.particle_count = 0
        state_in.particle_f = state_out.particle_f
        try:
            cls._rigid_solver.step(state_in, state_out, control, None, dt)
        finally:
            state_in.particle_f = particle_f
            model.particle_count = particle_count

    @classmethod
    def _apply_reactions(cls, state: State, state_prev: State, dt: float) -> None:
        """Launch the reaction kernel to inject normal + friction forces into body_f.

        Args:
            state: Current state with particle positions/velocities and body state.
            state_prev: Previous substep state whose ``body_q`` provides
                the reference poses for finite-difference body velocity.
            dt: Substep timestep [s].
        """
        model = cls._model
        contacts = cls._contacts

        if contacts is None:
            return

        contact_capacity = int(contacts.soft_contact_particle.shape[0])
        if contact_capacity == 0:
            return

        # The kernel reconstructs particle_q_prev from particle_qd internally:
        # state_prev.particle_q is unreliable because VBD mutates particle_q
        # in place during its iteration, so the swapped state's particle_q is
        # not a clean snapshot of the prior substep.
        wp.launch(
            _kernel_body_particle_reaction,
            dim=contact_capacity,
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
                model.shape_margin,
                float(model.soft_contact_ke),
                float(model.soft_contact_kd),
                float(model.soft_contact_mu),
                float(cls._soft_solver.friction_epsilon),
                float(dt),
                state.body_f,
            ],
        )
