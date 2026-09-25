# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import Model, ModelBuilder
from newton.solvers import SolverVBD

from ..cloner.newton_clone_utils import add_deformable_entry_to_builder
from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    import numpy as np

    from isaaclab.sim.simulation_context import SimulationContext


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Register Newton deformable construction for each cloned world."""
        NewtonManager._deformable_registry = []
        if cls._add_deformables_to_builder not in NewtonManager._per_world_builder_hooks:
            NewtonManager._per_world_builder_hooks.append(cls._add_deformables_to_builder)
        super().initialize(sim_context)

    @classmethod
    def start_simulation(cls) -> None:
        """Color the prebuilt model before starting simulation."""
        if cls._builder is not None:
            cls._builder.color(balance_colors=False)
        super().start_simulation()

    @classmethod
    def instantiate_builder_from_stage(cls) -> None:
        """Create and color the VBD builder from the USD stage."""
        super().instantiate_builder_from_stage()
        if cls._builder is None:
            raise RuntimeError("Newton stage import did not create a builder.")
        # Warp's optional balancing pass can cycle indefinitely for valid graph colorings.
        # The initial assignment is sufficient for VBD correctness.
        cls._builder.color(balance_colors=False)

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
    def _simulate_physics_only(cls) -> None:
        """Rebuild the VBD particle BVH before stepping physics."""
        if cls.backend.model.particle_count > 0 and hasattr(cls._solver, "rebuild_bvh"):
            cls._solver.rebuild_bvh(cls.backend.state_0)
        super()._simulate_physics_only()

    @staticmethod
    def _add_deformables_to_builder(
        builder: ModelBuilder, world_idx: int, env_position: np.ndarray, env_rotation: np.ndarray
    ) -> None:
        """Pass registered prototypes to the cloner for the current world."""
        for entry in NewtonManager._deformable_registry:
            add_deformable_entry_to_builder(builder, entry, world_idx, env_position, env_rotation)
