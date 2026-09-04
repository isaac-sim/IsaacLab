# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM Newton manager."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import warp as wp
from newton import (
    BodyFlags,
    Contacts,
    Control,
    GeoType,
    Model,
    ModelBuilder,
    State,
    StateFlags,
)
from newton.solvers import SolverImplicitMPM
from warp.fem import TemporaryStore

from isaaclab.physics import PhysicsManager

from .mpm_manager_cfg import MPMSolverCfg
from .newton_manager import NewtonManager

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim import SimulationContext


_RHEOLOGY_SOLVER_USD_NAMES = {
    "auto": "auto",
    "gs": "gauss-seidel",
    "gauss-seidel": "gauss-seidel",
    "gs-soa": "gauss-seidel-soa",
    "gauss-seidel-soa": "gauss-seidel-soa",
    "gs-batched": "gauss-seidel-batched",
    "gauss-seidel-batched": "gauss-seidel-batched",
    "jacobi": "jacobi",
    "cg": "conjugate-gradient",
    "conjugate-gradient": "conjugate-gradient",
    "cr": "conjugate-residual",
    "conjugate-residual": "conjugate-residual",
    "gmres": "generalized-minimal-residual",
    "generalized-minimal-residual": "generalized-minimal-residual",
}


def _canonical_collider_velocity_mode(solver_cfg: MPMSolverCfg) -> str:
    """Resolve deprecated collider-velocity aliases used by older configurations."""
    collider_velocity_mode = solver_cfg.collider_velocity_mode
    deprecated_velocity_modes = {
        "instantaneous": "forward",
        "finite_difference": "backward",
    }
    if replacement := deprecated_velocity_modes.get(collider_velocity_mode):
        warnings.warn(
            f"collider_velocity_mode={collider_velocity_mode!r} is deprecated; use {replacement!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        collider_velocity_mode = replacement
    return collider_velocity_mode


def _make_solver_config(solver_cfg: MPMSolverCfg, scene_prim: Usd.Prim | None = None) -> SolverImplicitMPM.Config:
    """Build Newton's implicit MPM config, consuming authored USD when available."""
    collider_velocity_mode = _canonical_collider_velocity_mode(solver_cfg)
    values = {
        "max_iterations": solver_cfg.max_iterations,
        "tolerance": solver_cfg.tolerance,
        "solver": solver_cfg.solver,
        "warmstart_mode": solver_cfg.warmstart_mode,
        "collider_velocity_mode": collider_velocity_mode,
        "voxel_size": solver_cfg.voxel_size,
        "grid_type": solver_cfg.grid_type,
        "grid_padding": solver_cfg.grid_padding,
        "max_active_cell_count": solver_cfg.max_active_cell_count,
        "max_leaf_node_count": solver_cfg.max_leaf_node_count,
        "max_lower_node_count": solver_cfg.max_lower_node_count,
        "max_upper_node_count": solver_cfg.max_upper_node_count,
        "separate_worlds": solver_cfg.separate_worlds,
        "transfer_scheme": solver_cfg.transfer_scheme,
        "integration_scheme": solver_cfg.integration_scheme,
        "critical_fraction": solver_cfg.critical_fraction,
        "air_drag": solver_cfg.air_drag,
        "collider_normal_from_sdf_gradient": solver_cfg.collider_normal_from_sdf_gradient,
        "collider_basis": solver_cfg.collider_basis,
        "strain_basis": solver_cfg.strain_basis,
        "velocity_basis": solver_cfg.velocity_basis,
    }
    if scene_prim is None:
        return SolverImplicitMPM.Config(**values)

    config = SolverImplicitMPM.Config.create_from_usd(scene_prim)
    # Retain the cfg's Python values after validating the authored schema. USD
    # float attributes otherwise quantize grid-sensitive values to float32.
    for name, value in values.items():
        setattr(config, name, value)
    return config


def _author_mpm_scene_config(scene_prim: Usd.Prim, solver_cfg: MPMSolverCfg) -> None:
    """Author schema-representable MPM solver settings on a physics scene."""
    from pxr import Vt  # noqa: PLC0415

    if "NewtonMPMSceneAPI" not in scene_prim.GetAppliedSchemas() and not scene_prim.ApplyAPI("NewtonMPMSceneAPI"):
        raise RuntimeError(f"Failed to apply NewtonMPMSceneAPI to '{scene_prim.GetPath()}'.")

    solvers = (solver_cfg.solver,) if isinstance(solver_cfg.solver, str) else solver_cfg.solver
    try:
        rheology_solvers = [_RHEOLOGY_SOLVER_USD_NAMES[solver] for solver in solvers]
    except KeyError as error:
        raise ValueError(f"Unsupported MPM rheology solver {error.args[0]!r}.") from error

    attributes = {
        "newton:maxSolverIterations": solver_cfg.max_iterations,
        "newton:mpm:tolerance": solver_cfg.tolerance,
        "newton:mpm:rheologySolvers": Vt.TokenArray(rheology_solvers),
        "newton:mpm:voxelSize": solver_cfg.voxel_size,
        "newton:mpm:gridType": solver_cfg.grid_type,
        "newton:mpm:gridPadding": solver_cfg.grid_padding,
        "newton:mpm:maxActiveCellCount": solver_cfg.max_active_cell_count,
        "newton:mpm:transferScheme": solver_cfg.transfer_scheme,
        "newton:mpm:integrationScheme": solver_cfg.integration_scheme,
        "newton:mpm:criticalFraction": solver_cfg.critical_fraction,
        "newton:mpm:airDrag": solver_cfg.air_drag,
    }
    attributes.update(_basis_attributes("collider", solver_cfg.collider_basis))
    attributes.update(_basis_attributes("strain", solver_cfg.strain_basis))
    attributes.update(_basis_attributes("velocity", solver_cfg.velocity_basis))
    for name, value in attributes.items():
        scene_prim.GetAttribute(name).Set(value)


def _basis_attributes(prefix: str, basis: str) -> dict[str, object]:
    """Expand a compact Newton basis name into newton-usd-schemas attributes."""
    if basis.startswith("pic"):
        if prefix == "velocity":
            raise ValueError("The MPM velocity basis cannot use a particle basis.")
        basis_type, order, discontinuous = "particle", 0, False
    else:
        matched = re.fullmatch(r"([PQBS])([0-9]+)(d?)", basis)
        if matched is None:
            raise ValueError(f"Unsupported MPM {prefix} basis {basis!r}.")
        basis_prefix = matched.group(1)
        order = int(matched.group(2))
        discontinuous = order == 0 or bool(matched.group(3))
        if basis_prefix == "P" and order > 0 and not discontinuous:
            raise ValueError(f"Unsupported MPM {prefix} basis {basis!r}: positive-order P bases must be discontinuous.")
        if (prefix == "strain" and basis_prefix in "BS") or (
            prefix == "velocity" and (basis_prefix not in "QB" or not 1 <= order <= 3)
        ):
            return {}
        if basis_prefix in "BS" and not 1 <= order <= 3:
            return {}
        basis_type = {
            "P": "linear",
            "Q": "trilinear",
            "B": "bspline",
            "S": "serendipity",
        }[basis_prefix]

    namespace = f"newton:mpm:{prefix}"
    attributes: dict[str, object] = {
        f"{namespace}BasisType": basis_type,
        f"{namespace}BasisOrder": order,
    }
    if prefix != "velocity":
        attributes[f"{namespace}DiscontinuousBasis"] = discontinuous
    return attributes


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
    _implicit_mpm_solver_root: object | None = None
    _implicit_mpm_solver_cache: tuple[SolverImplicitMPM, ...] = ()

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize Newton and author the MPM solver configuration in USD."""
        super().initialize(sim_context)
        scene_prim = sim_context.stage.GetPrimAtPath(sim_context.cfg.physics_prim_path)
        _author_mpm_scene_config(scene_prim, sim_context.cfg.physics.solver_cfg)

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
        scene_prim = None
        sim = PhysicsManager._sim
        if sim is not None:
            scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
            _author_mpm_scene_config(scene_prim, solver_cfg)
        return SolverImplicitMPM(
            model,
            _make_solver_config(solver_cfg, scene_prim),
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
    def _requires_initial_reset_before_graph_capture(cls) -> bool:
        """Capture MPM only after the task authors its initial particle state."""
        return True

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Return whether the active MPM grid has capture-stable storage."""
        return cls._solver_supports_cuda_graph_capture(cls._solver)

    @staticmethod
    def _solver_supports_cuda_graph_capture(solver: SolverImplicitMPM) -> bool:
        """Return whether an implicit-MPM solver satisfies Newton's capture contract."""
        if solver.grid_type == "fixed":
            return True
        if solver.grid_type != "sparse":
            return False

        strain_rebuild_safe = solver.strain_basis.startswith("pic") or solver.strain_basis in (
            "P0",
            "P1d",
            "Q1d",
            "Q1",
        )
        collider_rebuild_safe = solver.collider_basis.startswith("pic") or solver.collider_basis in (
            "Q1",
            "S2",
            "S3",
        )
        return (
            solver.max_active_cell_count > 0
            and solver.grid_padding == 0
            and solver.velocity_basis == "Q1"
            and strain_rebuild_safe
            and collider_rebuild_safe
        )

    @classmethod
    def _check_solver_status(cls) -> None:
        """Raise asynchronous sparse-grid rebuild failures after graph replay."""
        if NewtonManager._graph is None:
            return
        for solver in cls._implicit_mpm_solvers():
            solver.check_sparse_grid_rebuild_status()

    @classmethod
    def _implicit_mpm_solvers(cls) -> tuple[SolverImplicitMPM, ...]:
        """Return direct or coupled implicit-MPM solvers without importing the coupler."""
        root_solver = NewtonManager._solver
        if root_solver is cls._implicit_mpm_solver_root:
            return cls._implicit_mpm_solver_cache
        if isinstance(root_solver, SolverImplicitMPM):
            solvers = (root_solver,)
        elif root_solver is None or not hasattr(root_solver, "entry_names") or not hasattr(root_solver, "solver"):
            solvers = ()
        else:
            solvers = tuple(
                entry_solver
                for name in root_solver.entry_names()
                if isinstance((entry_solver := root_solver.solver(name)), SolverImplicitMPM)
            )
        cls._implicit_mpm_solver_root = root_solver
        cls._implicit_mpm_solver_cache = solvers
        return solvers

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
        """Preserve the existing implicit-MPM behavior for automatic asset resets.

        Shared-grid MPM configurations cannot reset solver history for only a
        subset of worlds. Tasks that use independent MPM worlds and require an
        exact history reset call :meth:`reset_solver_state`
        explicitly after authoring their complete state.

        Args:
            world_mask: Per-world reset mask, intentionally ignored.
        """

    @classmethod
    def reset_solver_state(
        cls,
        state: State | None = None,
        world_mask: wp.array(dtype=wp.bool) | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset MPM and coupled-solver history after task state is rewritten.

        When :paramref:`state` is omitted, both distinct manager state buffers
        are reset so a later buffer swap cannot restore stale history. A mask
        follows Newton's canonical ``world_count + 1`` contract, where the last
        entry selects global entities in world -1. A selected single local world
        is promoted to a full reset because a one-world MPM grid has no
        environment offsets.

        Args:
            state: State whose solver-owned history should be reset. If omitted,
                reset both manager states.
            world_mask: Canonical per-world mask, including the final global-world entry.
            flags: State components whose solver-owned history should reset.

        Raises:
            RuntimeError: If the MPM solver or a usable state is not initialized.
            ValueError: If :paramref:`world_mask` does not use Newton's canonical shape.
        """
        solver = NewtonManager._solver
        model = NewtonManager._model
        if solver is None or model is None or not cls._implicit_mpm_solvers():
            raise RuntimeError("An implicit MPM solver is not initialized; cannot reset solver state.")

        reset_mask = world_mask
        if world_mask is not None:
            expected_shape = (model.world_count + 1,)
            if world_mask.shape != expected_shape:
                raise ValueError(f"world_mask must have shape {expected_shape}; got {world_mask.shape}.")
            if model.world_count == 1:
                selected = world_mask.numpy()
                if not selected.any():
                    return
                if selected[0] and not selected[-1]:
                    reset_mask = None

        candidates = (state,) if state is not None else (NewtonManager._state_1, NewtonManager._state_0)
        states: list[State] = []
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            states.append(candidate)
        if not states:
            raise RuntimeError("Newton state is not initialized; provide an explicit state to reset.")

        for candidate in states:
            solver.reset(candidate, world_mask=reset_mask, flags=flags)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Reset MPM-specific class state on teardown.

        :meth:`_build_solver` sets :attr:`_project_outside_colliders` from the
        active config. Resetting it here keeps a teardown-only :meth:`clear`
        (without a follow-up rebuild) from leaving a stale value on the class,
        mirroring how :meth:`NewtonManager.clear` resets the base-class flags.
        """
        cls._project_outside_colliders = False
        cls._implicit_mpm_solver_root = None
        cls._implicit_mpm_solver_cache = ()
