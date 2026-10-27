# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom MJWarp and VBD coupling manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonSolverCfg, VBDSolverCfg

from isaaclab.utils import configclass

if TYPE_CHECKING:
    import warp as wp
    from newton import CollisionPipeline, Contacts, Model, ModelBuilder, State
    from newton.solvers import SolverBase

    from .coupled_mjwarp_vbd_solver import CoupledMJWarpVBDSolver


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the custom MJWarp and VBD coupling manager."""

    contact_sensors: ClassVar[bool] = False

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp rigid-body solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD deformable solver configuration."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and deformable solvers."""

    def solver_classes(self) -> tuple[type[SolverBase], ...]:
        return (*self.rigid_solver_cfg.solver_classes(), *self.soft_solver_cfg.solver_classes())

    def prepare_builder(self, builder: ModelBuilder) -> None:
        """Prepare the completed builder for both sub-solvers, e.g. color it for VBD."""
        self.rigid_solver_cfg.prepare_builder(builder)
        self.soft_solver_cfg.prepare_builder(builder)

    def build(
        self, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> CoupledMJWarpVBDSolver:
        """Construct both sub-solvers through their own configurations."""
        from .coupled_mjwarp_vbd_solver import CoupledMJWarpVBDSolver

        if self.coupling_mode not in ("one_way", "two_way"):
            raise ValueError("coupling_mode must be 'one_way' or 'two_way'.")
        if not self.rigid_solver_cfg.use_mujoco_contacts:
            raise ValueError("The custom coupling manager requires MJWarp internal contacts.")
        if not self.soft_solver_cfg.integrate_with_external_rigid_solver:
            raise ValueError("The custom coupling manager requires VBD external rigid-body integration.")
        return CoupledMJWarpVBDSolver(
            model,
            self.rigid_solver_cfg.build(model, mode, env_template),
            self.soft_solver_cfg.build(model, mode, env_template),
            self.coupling_mode,
        )

    def prepare_contacts(
        self, solver: CoupledMJWarpVBDSolver, contacts: Contacts, pipeline: CollisionPipeline | None
    ) -> None:
        """Bind the contacts and pipeline used inside each coupled substep."""
        solver.contacts = contacts
        solver.collision_pipeline = pipeline

    def pre_step(self, solver: CoupledMJWarpVBDSolver, state: State) -> None:
        solver.rebuild_bvh(state)
