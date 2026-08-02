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
    """

    _rigid_solver: SolverFeatherstone
    _soft_solver: SolverVBD
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
        """Kinematic coupling: mirrors some Newton examples (e.g. softbody_franka) exactly.

        1. Clear forces.
        2. Assign joint_qd from control targets (velocity = (target - current) / frame_dt).
        3. Disable gravity and rigid contacts for the rigid solver step.
        4. Step rigid solver as kinematic integrator (q += qd * dt).
        5. Restore gravity, collision detect, VBD step.
        """
        model = cls._model

        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Kinematic rigid step: assign qd, disable gravity/contacts/PD gains
        saved_particle_count = model.particle_count
        saved_shape_contact_pair_count = model.shape_contact_pair_count
        model.particle_count = 0
        model.gravity.assign(cls._gravity_zero)
        model.shape_contact_pair_count = 0

        # Zero out PD gains so rigid solver (Featherstone) acts as a pure kinematic integrator
        model.joint_target_ke.assign(cls._ke_zero)
        model.joint_target_kd.assign(cls._kd_zero)

        # Assign joint velocities from control targets
        state_in.joint_qd.assign(control.joint_target_vel)

        cls._rigid_solver.step(state_in, state_out, control, None, dt)

        # 3. Restore everything
        state_in.particle_f.zero_()
        model.particle_count = saved_particle_count
        model.gravity.assign(cls._gravity_saved)
        model.shape_contact_pair_count = saved_shape_contact_pair_count
        model.joint_target_ke.assign(cls._ke_saved)
        model.joint_target_kd.assign(cls._kd_saved)

        # 4. Collision detection
        cls._collision_pipeline.collide(state_in, cls._contacts)

        # 5. VBD step
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_one_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """One-way coupling: collide, then rigid step, then VBD."""
        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Collision detection (cloth-body contacts)
        cls._collision_pipeline.collide(state_in, cls._contacts)

        # 3. Rigid-body step (does not read soft-contact reactions)
        cls._rigid_step(state_in, state_out, control, dt)

        # 4. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 5. VBD step -- particles only, reads updated rigid poses
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_two_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Two-way coupling: collide, inject reactions into body_f, rigid step, VBD step."""
        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Collision detection BEFORE rigid step
        cls._collision_pipeline.collide(state_in, cls._contacts)

        # 3. Inject contact reaction forces into body_f.
        #    state_out holds the previous substep's body_q (states swap each
        #    substep), used for finite-difference body velocity in friction.
        #    particle_q_prev is reconstructed from particle_qd inside the
        #    kernel because VBD mutates particle_q in place, so the swapped
        #    state's particle_q is not a clean prior-substep snapshot.
        if state_in.body_f is not None:
            cls._apply_reactions(state_in, state_out, dt)

        # 4. Rigid-body step (reads body_f for soft-contact reactions)
        cls._rigid_step(state_in, state_out, control, dt)

        # 5. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 6. VBD step -- uses same contacts detected in step 2
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _rigid_step(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies with the configured sub-solver."""
        model = cls._model

        # set particle_count = 0 to disable particle simulation in robot solver
        saved_particle_count = model.particle_count
        model.particle_count = 0

        cls._rigid_solver.step(state_in, state_out, control, None, dt)

        # restore original settings
        model.particle_count = saved_particle_count

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
