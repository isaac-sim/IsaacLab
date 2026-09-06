# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FeatherPGS Newton manager."""

from __future__ import annotations

import warp as wp
from newton import Model, ModelBuilder
from newton.solvers import SolverFeatherPGS

from .feather_pgs_manager_cfg import FeatherPGSSolverCfg
from .newton_manager import NewtonManager


class NewtonFeatherPGSManager(NewtonManager):
    """:class:`NewtonManager` specialization for the FeatherPGS solver.

    FeatherPGS uses Newton's :class:`CollisionPipeline` for contact handling and
    steps with separate input/output states.
    """

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register the rigid-body custom attributes required by FeatherPGS."""
        if not builder.has_custom_attribute("rigid_body_max_linear_velocity"):
            SolverFeatherPGS.register_custom_attributes(builder)

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> SolverFeatherPGS:
        """Construct the configured FeatherPGS solver without changing manager state."""
        collision_cfg = NewtonManager._collision_cfg
        if collision_cfg is not None and collision_cfg.rigid_contact_max is not None:
            # FeatherPGS owns per-contact scratch and is constructed before the
            # collision pipeline. Publish the explicit pipeline capacity on the
            # model first so both allocations use the same bound.
            model.rigid_contact_max = int(collision_cfg.rigid_contact_max)
        return SolverFeatherPGS(model, **cls._filter_solver_kwargs(SolverFeatherPGS, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> None:
        """Construct :class:`SolverFeatherPGS` and populate the base-class slots."""
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = True

    @classmethod
    def _prepare_cuda_graph_capture(cls) -> None:
        """Seed FeatherPGS buffer events so captured waits survive graph replay."""
        cls._solver.seed_double_buffer_events()

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array(dtype=wp.bool) | None) -> None:
        """Reset solver history for environment worlds selected by Isaac Lab.

        Isaac Lab reserves the last reset-mask entry for global-world objects,
        while FeatherPGS accepts one entry per model world.

        Args:
            world_mask: Canonical Isaac Lab mask with shape ``(world_count + 1,)``.
        """
        if world_mask is None:
            return
        cls._solver.reset(cls._state_0, world_mask=world_mask[: cls._model.world_count], flags=0)
