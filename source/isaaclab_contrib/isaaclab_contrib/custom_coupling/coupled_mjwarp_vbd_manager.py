# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MJWarp and VBD coupling manager."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo, SolverVBD

from isaaclab_contrib.deformable.kernels import _kernel_body_particle_reaction
from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

from .newton_manager_cfg import CoupledMJWarpVBDSolverCfg


class NewtonCoupledMJWarpVBDManager(NewtonVBDManager):
    """Alternate MJWarp and VBD steps with optional reaction forces."""

    _rigid_solver: SolverMuJoCo | None = None
    _soft_solver: SolverVBD | None = None
    _coupling_mode: str | None = None

    @classmethod
    def step(cls) -> None:
        """Step the simulation."""
        from isaaclab.physics import PhysicsManager

        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        if cls._model_changes:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._rigid_solver.notify_model_changed(change)
                    cls._soft_solver.notify_model_changed(change)
                NewtonManager._model_changes = set()
        super().step()

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledMJWarpVBDSolverCfg) -> None:
        """Construct both solvers."""
        if solver_cfg.coupling_mode not in ("one_way", "two_way"):
            raise ValueError("coupling_mode must be 'one_way' or 'two_way'.")
        if not solver_cfg.rigid_solver_cfg.use_mujoco_contacts:
            raise ValueError("The custom coupling manager requires MJWarp internal contacts.")
        if not solver_cfg.soft_solver_cfg.integrate_with_external_rigid_solver:
            raise ValueError("The custom coupling manager requires VBD external rigid-body integration.")
        if NewtonManager._report_contacts:
            raise NotImplementedError("Newton contact sensors are not supported by the custom coupling manager.")

        cls._coupling_mode = solver_cfg.coupling_mode

        cls._rigid_solver = SolverMuJoCo(model, **cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg.rigid_solver_cfg))
        cls._soft_solver = SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg.soft_solver_cfg))

        NewtonManager._solver = SolverBase(model)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

    @classmethod
    def _step_solver(
        cls, state_in: State, state_out: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one coupled substep."""
        if cls._coupling_mode == "one_way":
            cls._step_one_way(state_in, state_out, control, substep_dt)
        else:
            cls._step_two_way(state_in, state_out, control, substep_dt)

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Reset both sub-solvers."""
        if world_mask is None:
            return
        if cls._rigid_solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        cls._rigid_solver.reset(cls._state_0, world_mask=world_mask, flags=0)
        cls._soft_solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Clear custom coupling state."""
        super()._solver_specific_clear()
        cls._rigid_solver = None
        cls._soft_solver = None
        cls._coupling_mode = None

    @classmethod
    def _simulate_physics_only(cls) -> None:
        if hasattr(cls._soft_solver, "rebuild_bvh"):
            cls._soft_solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()

    @classmethod
    def _step_one_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        state_out.clear_forces()
        cls._collision_pipeline.collide(state_in, cls._contacts)
        cls._rigid_step(state_in, state_out, control, dt)
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_two_way(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        state_out.clear_forces()
        cls._collision_pipeline.collide(state_in, cls._contacts)
        if state_in.body_f is not None:
            cls._apply_reactions(state_in, state_out, dt)
        cls._rigid_step(state_in, state_out, control, dt)
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _rigid_step(cls, state_in: State, state_out: State, control: Control, dt: float) -> None:
        cls._rigid_solver.step(state_in, state_out, control, None, dt)

    @classmethod
    def _apply_reactions(cls, state: State, state_prev: State, dt: float) -> None:
        model = cls._model
        contacts = cls._contacts
        if contacts is None:
            return

        contact_capacity = int(contacts.soft_contact_particle.shape[0])
        if contact_capacity == 0:
            return

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
