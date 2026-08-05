# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from newton import Model, ModelBuilder
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    @classmethod
    def _get_usd_ignore_paths(cls) -> list[str]:
        """Return registered deformable mesh paths excluded from USD import."""
        return [
            path for entry in cls._deformable_registry for path in (entry.sim_mesh_prim_path, entry.vis_mesh_prim_path)
        ]

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Color the builder for VBD."""
        super()._prepare_builder_for_finalize(builder)
        builder.color()

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> SolverVBD:
        """Construct the configured VBD solver."""
        return SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct VBD and configure its base-manager state."""
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = not solver_cfg.integrate_with_external_rigid_solver

    @classmethod
    def _pre_physics_step(cls) -> None:
        """Rebuild the VBD particle BVH before collision detection."""
        super()._pre_physics_step()
        if cls._model.particle_count > 0 and hasattr(cls._solver, "rebuild_bvh"):
            cls._solver.rebuild_bvh(cls._state_0)
