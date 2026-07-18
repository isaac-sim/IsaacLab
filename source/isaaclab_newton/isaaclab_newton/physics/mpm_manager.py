# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM Newton manager."""

from __future__ import annotations

import warp as wp
from newton import BodyFlags, Contacts, Control, Model, ModelBuilder, State
from newton.solvers import SolverImplicitMPM
from warp.fem import TemporaryStore

from .mpm_manager_cfg import MPMSolverCfg
from .newton_manager import NewtonManager


def _make_solver_config(solver_cfg: MPMSolverCfg) -> SolverImplicitMPM.Config:
    """Build Newton's implicit MPM solver config from Isaac Lab's cfg."""
    return SolverImplicitMPM.Config(
        max_iterations=solver_cfg.max_iterations,
        tolerance=solver_cfg.tolerance,
        solver=solver_cfg.solver,
        warmstart_mode=solver_cfg.warmstart_mode,
        collider_velocity_mode=solver_cfg.collider_velocity_mode,
        voxel_size=solver_cfg.voxel_size,
        grid_type=solver_cfg.grid_type,
        grid_padding=solver_cfg.grid_padding,
        max_active_cell_count=solver_cfg.max_active_cell_count,
        transfer_scheme=solver_cfg.transfer_scheme,
        integration_scheme=solver_cfg.integration_scheme,
        critical_fraction=solver_cfg.critical_fraction,
        air_drag=solver_cfg.air_drag,
        collider_normal_from_sdf_gradient=solver_cfg.collider_normal_from_sdf_gradient,
        collider_basis=solver_cfg.collider_basis,
        strain_basis=solver_cfg.strain_basis,
        velocity_basis=solver_cfg.velocity_basis,
    )


class NewtonMPMManager(NewtonManager):
    """:class:`NewtonManager` specialization for Newton's implicit MPM solver.

    MPM advances particle materials in-place and treats rigid geometry as
    colliders, so it does not consume Newton's rigid-body collision pipeline
    and steps with a single :class:`State`.
    """

    _project_outside_colliders: bool = False
    """Whether :meth:`_step_solver` projects particles out of colliders each substep.

    Set from :attr:`MPMSolverCfg.project_outside_colliders` in
    :meth:`_build_solver` and read in :meth:`_step_solver`.
    """

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register the particle custom attributes required by :class:`SolverImplicitMPM`.

        Implicit MPM materials are configured per-particle through Newton
        custom attributes (``mpm:young_modulus``, ``mpm:viscosity``, ...).
        These must be present on the builder *before* particles are added so
        that ``add_particles(custom_attributes=...)`` succeeds and so that
        ``builder.finalize()`` allocates the matching model arrays.

        Idempotent: ``has_custom_attribute`` guards against re-registration
        when the hook is invoked multiple times (e.g. once via
        :meth:`create_builder` and again via :meth:`start_simulation`).
        """
        if not builder.has_custom_attribute("mpm:young_modulus"):
            SolverImplicitMPM.register_custom_attributes(builder)

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Normalize kinematic rigid bodies before MPM solver construction.

        Newton's implicit MPM solver treats positive-mass body colliders as
        finite-mass colliders. Isaac Lab kinematic assets can import with a
        computed mass, so clear mass and inertia for kinematic bodies to match
        Newton's direct-builder MPM examples.
        """
        kinematic_flag = int(BodyFlags.KINEMATIC)
        for body_id, flags in enumerate(builder.body_flags):
            if int(flags) & kinematic_flag:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MPMSolverCfg) -> None:
        """Construct :class:`SolverImplicitMPM` and populate the base-class slots.

        MPM steps in-place on a single :class:`State` and runs collision
        handling internally, so it neither double-buffers state nor drives
        Newton's :class:`CollisionPipeline`.

        Args:
            model: Finalized Newton model the solver should run on.
            solver_cfg: Implicit MPM solver configuration.
        """
        NewtonManager._solver = SolverImplicitMPM(
            model,
            _make_solver_config(solver_cfg),
            temporary_store=TemporaryStore(),
        )
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._needs_fk_before_step = True
        cls._project_outside_colliders = solver_cfg.project_outside_colliders

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Return ``True`` only for fixed-grid MPM.

        Sparse and dense grids reallocate as particles move, which is not
        capturable in a CUDA graph; the fixed grid keeps a static topology.
        """
        return cls._solver.grid_type == "fixed"

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one implicit MPM substep, optionally projecting particles out of colliders.

        The implicit solve already resolves colliders at the grid level. When
        :attr:`MPMSolverCfg.project_outside_colliders` is set, the manager also
        runs ``project_outside`` after the step (as in Newton's MPM examples) to
        hard-project particles out of collider interiors. The flag is evaluated
        when the step is first run, so the chosen branch is baked into any
        captured CUDA graph.
        """
        cls._solver.step(state_0, state_1, control, contacts, substep_dt)
        if cls._project_outside_colliders:
            cls._solver.project_outside(state_1, state_1, substep_dt)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Reset MPM-specific class state on teardown.

        :meth:`_build_solver` sets :attr:`_project_outside_colliders` from the
        active config. Resetting it here keeps a teardown-only :meth:`clear`
        (without a follow-up rebuild) from leaving a stale value on the class,
        mirroring how :meth:`NewtonManager.clear` resets the base-class flags.
        """
        cls._project_outside_colliders = False
