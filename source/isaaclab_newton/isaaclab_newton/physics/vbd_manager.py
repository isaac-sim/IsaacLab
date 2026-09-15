# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import Model
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize VBD deformable integration when contrib is available."""
        try:
            from isaaclab_contrib.deformable.deformable_object import install_deformable_builder_hooks
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            install_deformable_builder_hooks()
        super().initialize(sim_context)

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation and bind registered deformables to Fabric."""
        if cls._builder is not None:
            cls._builder.color()
        super().start_simulation()
        try:
            from isaaclab_contrib.deformable.deformable_object import setup_registered_deformable_fabric_sync
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            setup_registered_deformable_fabric_sync(cls)

    @classmethod
    def instantiate_builder_from_stage(cls) -> None:
        """Create and color the VBD builder from the USD stage."""
        super().instantiate_builder_from_stage()
        if cls._builder is None:
            raise RuntimeError("Newton stage import did not create a builder.")
        cls._builder.color()

    @classmethod
    def _get_usd_import_ignore_paths(cls) -> list[str]:
        """Return registered deformable mesh paths excluded from USD import."""
        return [
            path for entry in cls._deformable_registry for path in (entry.sim_mesh_prim_path, entry.vis_mesh_prim_path)
        ]

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
    def _solver_specific_clear(cls) -> None:
        """Clear contrib deformable integration when available."""
        try:
            from isaaclab_contrib.deformable.deformable_object import clear_deformable_builder_hooks
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            clear_deformable_builder_hooks()

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Rebuild the VBD particle BVH before stepping physics."""
        if cls._model.particle_count > 0 and hasattr(cls._solver, "rebuild_bvh"):
            cls._solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()
