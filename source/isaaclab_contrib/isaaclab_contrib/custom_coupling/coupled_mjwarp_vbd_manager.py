# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MJWarp and VBD coupling manager."""

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo, SolverVBD

from .kernels import _kernel_body_particle_reaction
from .newton_manager_cfg import CoupledMJWarpVBDSolverCfg


class NewtonCoupledMJWarpVBDManager(NewtonVBDManager):
    """:class:`NewtonVBDManager` specialization for custom MJWarp and VBD coupling.

    Reuses the VBD manager's deformable stage handling and adds a custom rigid-deformable coupling step.
    Newton's :class:`CollisionPipeline` provides deformable contacts.
    """

    _rigid_solver: SolverMuJoCo | None = None
    _soft_solver: SolverVBD | None = None
    _coupling_mode: str | None = None
    _builder_attribute_solvers = (SolverMuJoCo,)

    def step(self) -> None:
        """Step the physics simulation."""
        sim = self._sim
        if sim is None or not sim.is_playing():
            return

        # Notify both sub-solvers of model changes.
        if self._model_changes:
            with wp.ScopedDevice(self._device):
                for change in self._model_changes:
                    self._rigid_solver.notify_model_changed(change)
                    self._soft_solver.notify_model_changed(change)
                self._model_changes = set()
        super().step()

    def _build_solver(self, model: Model, solver_cfg: CoupledMJWarpVBDSolverCfg) -> None:
        """Construct the coupled solvers and populate the base lifecycle slots.

        VBD uses Newton's collision pipeline and separate input/output states, so the lifecycle flags are fixed.
        """
        if solver_cfg.coupling_mode not in ("one_way", "two_way"):
            raise ValueError("coupling_mode must be 'one_way' or 'two_way'.")
        if not solver_cfg.rigid_solver_cfg.use_mujoco_contacts:
            raise ValueError("The custom coupling manager requires MJWarp internal contacts.")
        if not solver_cfg.soft_solver_cfg.integrate_with_external_rigid_solver:
            raise ValueError("The custom coupling manager requires VBD external rigid-body integration.")
        if self._report_contacts:
            raise NotImplementedError("Newton contact sensors are not supported by the custom coupling manager.")

        self._coupling_mode = solver_cfg.coupling_mode

        self._rigid_solver = solver_cfg.rigid_solver_cfg.class_type._create_solver(
            model, solver_cfg.rigid_solver_cfg, self._deterministic_mode
        )
        self._soft_solver = solver_cfg.soft_solver_cfg.class_type._create_solver(
            model, solver_cfg.soft_solver_cfg, self._deterministic_mode
        )

        # The base lifecycle needs a solver slot; substeps use the two solvers above.
        self._solver = SolverBase(model)
        self._use_single_state = False
        self._supports_contact_sensors = False
        self._needs_collision_pipeline = True
        self._supports_rigid_body_force_input = True

    def _step_solver(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one coupled substep.

        Args:
            state_in: Current read/write state.
            state_out: Next state.
            control: Joint-level control inputs.
            contacts: Unused; the coupling helpers use the manager-owned contact buffer.
            substep_dt: Substep timestep [s].
        """
        if self._coupling_mode == "one_way":
            self._step_one_way(state_in, state_out, control, substep_dt)
        else:
            self._step_two_way(state_in, state_out, control, substep_dt)

    def _reset_solver_internals(self, world_mask: wp.array | None) -> None:
        """Reset both sub-solvers."""
        if world_mask is None:
            return
        if self._rigid_solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        self._rigid_solver.reset(self.backend.state_0, world_mask=world_mask, flags=0)
        self._soft_solver.reset(self.backend.state_0, world_mask=world_mask, flags=0)

    def _solver_specific_clear(self) -> None:
        """Clear custom coupling state."""
        super()._solver_specific_clear()
        self._rigid_solver = None
        self._soft_solver = None
        self._coupling_mode = None

    def _simulate_physics_only(self) -> None:
        # Rebuild the BVH before stepping solvers that require it, such as VBD cloth.
        if hasattr(self._soft_solver, "rebuild_bvh"):
            self._soft_solver.rebuild_bvh(self.backend.state_0)
        super()._simulate_physics_only()

    def _step_one_way(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies and particles without deformable reaction forces."""
        # 1. Clear output forces.
        state_out.clear_forces()

        # 2. Detect deformable-rigid contacts.
        self._collision_pipeline.collide(state_in, self._contacts)

        # 3. Advance rigid bodies without injected deformable reactions.
        self._rigid_step(state_in, state_out, control, dt)

        # 4. Advance particles using the updated rigid poses.
        self._soft_solver.step(state_in, state_out, control, self._contacts, dt)

    def _step_two_way(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies and particles with deformable reaction forces."""
        # 1. Clear output forces.
        state_out.clear_forces()

        # 2. Detect contacts before advancing rigid bodies.
        self._collision_pipeline.collide(state_in, self._contacts)

        # 3. Inject contact reactions before MJWarp consumes body_f.
        # The inactive state buffer supplies reference poses for friction velocity estimation.
        # The kernel reconstructs particle history because VBD mutates particle_q in place.
        if state_in.body_f is not None:
            self._apply_reactions(state_in, state_out, dt)

        # 4. Advance rigid bodies with the injected reactions.
        self._rigid_step(state_in, state_out, control, dt)

        # 5. Advance particles using the contacts detected above.
        self._soft_solver.step(state_in, state_out, control, self._contacts, dt)

    def _rigid_step(self, state_in: State, state_out: State, control: Control, dt: float) -> None:
        """Advance rigid bodies with the configured sub-solver."""
        self._rigid_solver.step(state_in, state_out, control, None, dt)

    def _apply_reactions(self, state: State, state_prev: State, dt: float) -> None:
        """Inject normal and friction reaction forces into body_f.

        Args:
            state: Current particle and body state.
            state_prev: Inactive state buffer providing reference poses for friction velocity estimation.
            dt: Substep timestep [s].
        """
        contacts = self._contacts
        if contacts is None:
            return

        contact_capacity = int(contacts.soft_contact_particle.shape[0])
        if contact_capacity == 0:
            return

        model = self.backend.model
        # VBD mutates particle_q in place, so the kernel reconstructs prior positions from particle_qd.
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
                float(self._soft_solver.friction_epsilon),
                float(dt),
                state.body_f,
            ],
        )
