# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MJWarp and VBD coupled solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import NewtonSolver
from newton import CollisionPipeline, Contacts, Control, Model, ModelBuilder, State
from newton.solvers import SolverBase, SolverMuJoCo, SolverVBD

from .kernels import _kernel_body_particle_reaction

if TYPE_CHECKING:
    from .coupled_mjwarp_vbd_solver_cfg import CoupledMJWarpVBDSolverCfg


class CoupledMJWarpVBDSolver(NewtonSolver):
    """Couple MJWarp rigid bodies with VBD deformables, each built from its own configuration."""

    contact_sensors = False

    @classmethod
    def solver_classes(cls, cfg: CoupledMJWarpVBDSolverCfg) -> tuple[type[SolverBase], ...]:
        return tuple(
            solver_cls
            for sub_cfg in (cfg.rigid_solver_cfg, cfg.soft_solver_cfg)
            for solver_cls in sub_cfg.class_type.solver_classes(sub_cfg)
        )

    @classmethod
    def prepare_builder(cls, cfg: CoupledMJWarpVBDSolverCfg, builder: ModelBuilder) -> None:
        """Prepare the completed builder for both sub-solvers, e.g. color it for VBD."""
        for sub_cfg in (cfg.rigid_solver_cfg, cfg.soft_solver_cfg):
            sub_cfg.class_type.prepare_builder(sub_cfg, builder)

    @classmethod
    def build(
        cls, cfg: CoupledMJWarpVBDSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> SolverCoupledMJWarpVBD:
        """Construct both sub-solvers through their own configurations."""
        if cfg.coupling_mode not in ("one_way", "two_way"):
            raise ValueError("coupling_mode must be 'one_way' or 'two_way'.")
        if not cfg.rigid_solver_cfg.use_mujoco_contacts:
            raise ValueError("The custom coupled solver requires MJWarp internal contacts.")
        if not cfg.soft_solver_cfg.integrate_with_external_rigid_solver:
            raise ValueError("The custom coupled solver requires VBD external rigid-body integration.")
        rigid, soft = cfg.rigid_solver_cfg, cfg.soft_solver_cfg
        return SolverCoupledMJWarpVBD(
            model,
            rigid.class_type.build(rigid, model, mode, env_template),
            soft.class_type.build(soft, model, mode, env_template),
            cfg.coupling_mode,
        )

    @classmethod
    def prepare_contacts(
        cls,
        cfg: CoupledMJWarpVBDSolverCfg,
        solver: SolverCoupledMJWarpVBD,
        contacts: Contacts,
        pipeline: CollisionPipeline | None,
    ) -> None:
        """Bind the collision pipeline used inside each coupled substep."""
        solver.collision_pipeline = pipeline

    @classmethod
    def pre_step(cls, cfg: CoupledMJWarpVBDSolverCfg, solver: SolverCoupledMJWarpVBD, state: State) -> None:
        solver.rebuild_bvh(state)


class SolverCoupledMJWarpVBD(SolverBase):
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
            contacts: Contacts the coupled substep collides into with the bound collision pipeline.
            dt: Substep timestep [s].
        """
        # 1. Clear output forces.
        state_out.clear_forces()
        # 2. Detect deformable-rigid contacts before advancing rigid bodies.
        self.collision_pipeline.collide(state_in, contacts)
        # 3. In two-way mode, inject contact reactions before MJWarp consumes body_f. The inactive state buffer
        # supplies reference poses for friction velocity estimation.
        if self.coupling_mode == "two_way" and state_in.body_f is not None:
            self._apply_reactions(state_in, state_out, contacts, dt)
        # 4. Advance rigid bodies.
        self.rigid_solver.step(state_in, state_out, control, None, dt)
        # 5. Advance particles using the updated rigid poses and the contacts detected above.
        self.soft_solver.step(state_in, state_out, control, contacts, dt)

    def _apply_reactions(self, state: State, state_prev: State, contacts: Contacts, dt: float) -> None:
        """Inject normal and friction reaction forces into body_f.

        Args:
            state: Current particle and body state.
            state_prev: Inactive state buffer providing reference poses for friction velocity estimation.
            contacts: Contacts detected for this substep.
            dt: Substep timestep [s].
        """
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
