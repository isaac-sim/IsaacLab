# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MJWarp and VBD coupled solver and its specification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics.solver_specs import SolverSpec, solver_spec
from newton import CollisionPipeline, Contacts, Control, Model, ModelBuilder, State
from newton.solvers import SolverBase, SolverMuJoCo, SolverVBD

from isaaclab.utils import instantiate

from .kernels import _kernel_body_particle_reaction

if TYPE_CHECKING:
    from .coupled_mjwarp_vbd_solver_cfg import CoupledMJWarpVBDSolverCfg


class CoupledMJWarpVBDSolver(SolverBase):
    """Newton solver coupling MJWarp rigid bodies with VBD deformables.

    MJWarp advances rigid bodies with its internal contacts, and VBD advances deformables using Newton's
    :class:`CollisionPipeline`. In two-way mode, deformable contact reactions are injected into ``body_f`` before
    MJWarp consumes it.
    """

    def __init__(self, model: Model, rigid_solver: SolverMuJoCo, soft_solver: SolverVBD, coupling_mode: str):
        super().__init__(model)
        self.rigid_solver = rigid_solver
        self.soft_solver = soft_solver
        self.coupling_mode = coupling_mode
        self.contacts: Contacts | None = None
        """Contacts the coupled substep collides into; bound by the manager once allocated."""
        self.collision_pipeline: CollisionPipeline | None = None

    def rebuild_bvh(self, state: State) -> None:
        """Rebuild the VBD BVH before each physics step."""
        self.soft_solver.rebuild_bvh(state)

    def notify_model_changed(self, flags: int) -> None:
        """Notify both sub-solvers of a model change."""
        self.rigid_solver.notify_model_changed(flags)
        self.soft_solver.notify_model_changed(flags)

    def reset(self, state: State, world_mask: wp.array | None = None, flags: int = 0) -> None:
        """Reset both sub-solvers for masked worlds."""
        if self.rigid_solver.use_mujoco_cpu and world_mask is not None and not world_mask.numpy().any():
            return
        self.rigid_solver.reset(state, world_mask=world_mask, flags=flags)
        self.soft_solver.reset(state, world_mask=world_mask, flags=flags)

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts | None, dt: float) -> None:
        """Run one coupled substep.

        Args:
            state_in: Current read/write state.
            state_out: Next state.
            control: Joint-level control inputs.
            contacts: Unused; the coupled substep collides into the bound contacts itself.
            dt: Substep timestep [s].
        """
        # 1. Clear output forces.
        state_out.clear_forces()
        # 2. Detect deformable-rigid contacts before advancing rigid bodies.
        self.collision_pipeline.collide(state_in, self.contacts)
        # 3. In two-way mode, inject contact reactions before MJWarp consumes body_f. The inactive state buffer
        # supplies reference poses for friction velocity estimation.
        if self.coupling_mode == "two_way" and state_in.body_f is not None:
            self._apply_reactions(state_in, state_out, dt)
        # 4. Advance rigid bodies.
        self.rigid_solver.step(state_in, state_out, control, None, dt)
        # 5. Advance particles using the updated rigid poses and the contacts detected above.
        self.soft_solver.step(state_in, state_out, control, self.contacts, dt)

    def _apply_reactions(self, state: State, state_prev: State, dt: float) -> None:
        """Inject normal and friction reaction forces into body_f.

        Args:
            state: Current particle and body state.
            state_prev: Inactive state buffer providing reference poses for friction velocity estimation.
            dt: Substep timestep [s].
        """
        contacts = self.contacts
        if contacts is None:
            return

        contact_capacity = int(contacts.soft_contact_particle.shape[0])
        if contact_capacity == 0:
            return

        model = self.model
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
                float(self.soft_solver.friction_epsilon),
                float(dt),
                state.body_f,
            ],
        )


def build(
    model: Model, cfg: CoupledMJWarpVBDSolverCfg, deterministic_mode: wp.DeterministicMode
) -> CoupledMJWarpVBDSolver:
    """Construct both sub-solvers through their own specs."""
    if cfg.coupling_mode not in ("one_way", "two_way"):
        raise ValueError("coupling_mode must be 'one_way' or 'two_way'.")
    if not cfg.rigid_solver_cfg.use_mujoco_contacts:
        raise ValueError("The custom coupling manager requires MJWarp internal contacts.")
    if not cfg.soft_solver_cfg.integrate_with_external_rigid_solver:
        raise ValueError("The custom coupling manager requires VBD external rigid-body integration.")
    rigid, soft = cfg.rigid_solver_cfg, cfg.soft_solver_cfg
    return CoupledMJWarpVBDSolver(
        model,
        instantiate(rigid, model, deterministic_mode),
        instantiate(soft, model, deterministic_mode),
        cfg.coupling_mode,
    )


def prepare_builder(builder: ModelBuilder, cfg: CoupledMJWarpVBDSolverCfg) -> None:
    """Prepare the completed builder for both sub-solvers, e.g. color it for VBD."""
    for sub_cfg in (cfg.rigid_solver_cfg, cfg.soft_solver_cfg):
        solver_spec(sub_cfg).prepare_builder(builder, sub_cfg)


def prepare_contacts(
    solver: CoupledMJWarpVBDSolver,
    cfg: CoupledMJWarpVBDSolverCfg,
    contacts: Contacts,
    pipeline: CollisionPipeline | None,
) -> None:
    """Bind the contacts and pipeline used inside each coupled substep."""
    solver.contacts = contacts
    solver.collision_pipeline = pipeline


COUPLED_MJWARP_VBD_SPEC = SolverSpec(
    attribute_solvers=lambda cfg: (SolverMuJoCo,),
    prepare_builder=prepare_builder,
    build=build,
    contact_sensors=False,
    prepare_contacts=prepare_contacts,
    prepare_step=lambda solver, cfg, state: solver.rebuild_bvh(state),
)
"""Specification of the custom MJWarp and VBD coupling (:class:`CoupledMJWarpVBDSolver`)."""
