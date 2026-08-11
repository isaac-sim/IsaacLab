# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM Newton manager."""

from __future__ import annotations

import numpy as np
import warp as wp
from newton import BodyFlags, Contacts, Control, GeoType, Model, ModelBuilder, State, StateFlags
from newton.solvers import SolverImplicitMPM
from warp.fem import TemporaryStore

from .mpm_manager_cfg import MPMSolverCfg
from .newton_manager import NewtonManager


def adapt_world_mask_for_implicit_mpm(
    solver: SolverImplicitMPM,
    world_mask: wp.array | None,
) -> wp.array | None:
    """Adapt an Isaac Lab world-reset mask to Implicit MPM's mask contract.

    Isaac Lab and most Newton solvers use a per-world mask of shape
    ``(world_count,)``. :meth:`SolverImplicitMPM.reset` instead expects
    ``(world_count + 1,)``, where the trailing entry selects global objects
    whose world index is ``-1``.

    Args:
        solver: Implicit MPM solver whose model defines ``world_count``.
        world_mask: Optional Isaac Lab / Newton mask of shape ``(world_count,)``
            or an already-adapted Implicit MPM mask of shape
            ``(world_count + 1,)``.

    Returns:
        ``None`` when no mask is provided or every world is selected (a full
        reset, including global world ``-1``). Otherwise a boolean Warp array
        of shape ``(world_count + 1,)`` with the global bit left ``False``.

    Raises:
        ValueError: If ``world_mask`` has neither ``(world_count,)`` nor
            ``(world_count + 1,)`` shape.
    """
    if world_mask is None:
        return None

    local_selected = _local_world_selection(solver, world_mask)
    if bool(np.all(local_selected)):
        # Full world selection is equivalent to an unmasked reset and also
        # covers global (world index -1) particle/collider history.
        return None

    world_count = int(solver.model.world_count)
    if tuple(world_mask.shape) == (world_count + 1,):
        return world_mask

    padded = np.zeros(world_count + 1, dtype=bool)
    padded[:-1] = local_selected
    return wp.array(padded, dtype=wp.bool, device=world_mask.device)


def should_skip_implicit_mpm_masked_reset(
    solver: SolverImplicitMPM,
    world_mask: wp.array | None,
) -> bool:
    """Whether a masked Implicit MPM reset must be skipped.

    Shared multi-world Implicit MPM (``separate_worlds=False``) rejects selective
    masks when clearing grid-backed warm starts. Matching
    :meth:`NewtonMPMManager._reset_solver_internals`, Isaac Lab skips those
    resets instead of raising. Full-world selection still proceeds as an
    unmasked reset.
    """
    if world_mask is None:
        return False
    if bool(getattr(solver, "_separate_worlds", False)) or int(solver.model.world_count) <= 1:
        return False
    return not bool(np.all(_local_world_selection(solver, world_mask)))


def _local_world_selection(solver: SolverImplicitMPM, world_mask: wp.array) -> np.ndarray:
    """Return the per-world selection bits from an Isaac Lab or Implicit MPM mask."""
    world_count = int(solver.model.world_count)
    shape = tuple(world_mask.shape)
    selected = world_mask.numpy()
    if shape == (world_count + 1,):
        return selected[:-1]
    if shape == (world_count,):
        return selected
    raise ValueError(
        f"world_mask has shape {shape}, expected ({world_count},) or ({world_count + 1},) "
        "for SolverImplicitMPM.reset."
    )


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
        """Normalize rigid colliders before MPM solver construction.

        Newton's implicit MPM solver treats positive-mass body colliders as
        finite-mass colliders. Isaac Lab kinematic assets can import with a
        computed mass, so clear mass and inertia for kinematic bodies to match
        Newton's direct-builder MPM examples. The solver consumes mesh vertices
        and indices but only accepts the triangle-mesh geometry type, so classify
        convex meshes as meshes without changing their geometry.
        """
        kinematic_flag = int(BodyFlags.KINEMATIC)
        for body_id, flags in enumerate(builder.body_flags):
            if int(flags) & kinematic_flag:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()
        for shape_id, shape_type in enumerate(builder.shape_type):
            if shape_type == GeoType.CONVEX_MESH:
                builder.shape_type[shape_id] = GeoType.MESH

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: MPMSolverCfg) -> SolverImplicitMPM:
        """Construct the configured implicit MPM solver."""
        return SolverImplicitMPM(
            model,
            _make_solver_config(solver_cfg),
            temporary_store=TemporaryStore(),
        )

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
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._supports_rigid_body_force_input = False
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
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Skip the solver-internal reset for implicit MPM.

        :meth:`SolverImplicitMPM.reset` only honors a per-world mask when the
        solver runs one FEM environment per world (``Config.separate_worlds``).
        With the shared topology Isaac Lab builds, a masked reset is rejected,
        and a full reset would clear the particle history of environments that
        were not reset. Leaving MPM history untouched on an environment reset
        matches the solver's behavior before it gained :meth:`reset`.

        Args:
            world_mask: Per-world reset mask, ignored.
        """

    @classmethod
    def reset_solver_state(
        cls,
        state: State | None = None,
        world_mask: wp.array(dtype=wp.bool) | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset Implicit MPM history after simulation state is rewritten.

        Expands Isaac Lab's ``(world_count,)`` mask to the
        ``(world_count + 1,)`` shape required by :meth:`SolverImplicitMPM.reset`
        before delegating to :meth:`NewtonManager.reset_solver_state`. Shared
        multi-world selective masks are skipped; see
        :func:`should_skip_implicit_mpm_masked_reset`.
        """
        if not isinstance(cls._solver, SolverImplicitMPM):
            raise RuntimeError(
                f"{cls.__name__}.reset_solver_state requires an active SolverImplicitMPM; "
                f"got {type(cls._solver).__name__}."
            )
        if should_skip_implicit_mpm_masked_reset(cls._solver, world_mask):
            return
        super().reset_solver_state(
            state=state,
            world_mask=adapt_world_mask_for_implicit_mpm(cls._solver, world_mask),
            flags=flags,
        )

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Reset MPM-specific class state on teardown.

        :meth:`_build_solver` sets :attr:`_project_outside_colliders` from the
        active config. Resetting it here keeps a teardown-only :meth:`clear`
        (without a follow-up rebuild) from leaving a stale value on the class,
        mirroring how :meth:`NewtonManager.clear` resets the base-class flags.
        """
        cls._project_outside_colliders = False
