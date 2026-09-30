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
from newton.solvers import SolverBase, SolverImplicitMPM
from warp.fem import TemporaryStore

from isaaclab.physics import PhysicsManager

from .mpm_manager_cfg import MPMSolverCfg
from .newton_manager import NewtonManager
from .solver_binding import NewtonSolverBinding

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim import SimulationContext


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
    from pxr import Sdf, Vt  # noqa: PLC0415

    # Preserve the raw API metadata when Kit's registry predates the installed codeless schemas.
    if not scene_prim.AddAppliedSchema("NewtonMPMSceneAPI"):
        raise RuntimeError(f"Failed to apply NewtonMPMSceneAPI to '{scene_prim.GetPath()}'.")

    rheology_solvers = (solver_cfg.solver,) if isinstance(solver_cfg.solver, str) else solver_cfg.solver

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
        if name == "newton:mpm:rheologySolvers":
            type_name = Sdf.ValueTypeNames.TokenArray
        elif isinstance(value, bool):
            type_name = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            type_name = Sdf.ValueTypeNames.Int
        elif isinstance(value, str):
            type_name = Sdf.ValueTypeNames.Token
        else:
            type_name = Sdf.ValueTypeNames.Float
        scene_prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)


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


def implicit_mpm_solvers(solver: SolverBase | None) -> tuple[SolverImplicitMPM, ...]:
    """Return a direct implicit-MPM solver, or the implicit-MPM entries of a coupled solver.

    Args:
        solver: Root solver.

    Returns:
        Implicit-MPM solvers reachable from ``solver``.
    """
    if isinstance(solver, SolverImplicitMPM):
        return (solver,)
    if solver is None or not hasattr(solver, "entry_names") or not hasattr(solver, "solver"):
        return ()
    return tuple(
        entry_solver
        for name in solver.entry_names()
        if isinstance((entry_solver := solver.solver(name)), SolverImplicitMPM)
    )


def mpm_supports_graph_capture(solver: SolverImplicitMPM) -> bool:
    """Return whether an implicit-MPM solver satisfies Newton's capture contract."""
    if solver.grid_type == "fixed":
        # An unbounded active partition reads its cell count back to the CPU on every step.
        return solver.max_active_cell_count > 0
    if solver.grid_type != "sparse":
        return False
    strain_rebuild_safe = solver.strain_basis.startswith("pic") or solver.strain_basis in ("P0", "P1d", "Q1d", "Q1")
    collider_rebuild_safe = solver.collider_basis.startswith("pic") or solver.collider_basis in ("Q1", "S2", "S3")
    return (
        solver.max_active_cell_count > 0
        and solver.grid_padding == 0
        and solver.velocity_basis == "Q1"
        and strain_rebuild_safe
        and collider_rebuild_safe
    )


class MPMSolverBinding(NewtonSolverBinding):
    """Binding for Newton's implicit MPM solver.

    MPM advances particle materials in place on one state and treats rigid geometry as colliders, so it does not use
    Newton's collision pipeline or consume applied rigid-body forces.
    """

    solver: SolverImplicitMPM

    def __init__(self, model: Model, solver_cfg: MPMSolverCfg, deterministic_mode: wp.DeterministicMode):
        super().__init__(model, solver_cfg, deterministic_mode)
        self.single_state = True
        self.needs_collision_pipeline = False
        self.supports_body_forces = False
        self.implicit_mpm_solvers = implicit_mpm_solvers(self.solver)

    @classmethod
    def register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register the per-particle material attributes before particles are added.

        Implicit MPM materials are configured per particle through Newton custom attributes (``mpm:young_modulus``,
        ``mpm:viscosity``, ...), which must exist before ``add_particles(custom_attributes=...)`` and finalization.
        """
        if not builder.has_custom_attribute("mpm:young_modulus"):
            SolverImplicitMPM.register_custom_attributes(builder)

    @classmethod
    def prepare_builder(cls, builder: ModelBuilder) -> None:
        """Normalize rigid colliders before solver construction.

        Newton's implicit MPM treats positive-mass body colliders as finite-mass colliders, so kinematic bodies lose
        their mass and inertia, matching Newton's MPM examples. The solver accepts only the triangle-mesh geometry type,
        so convex meshes are classified as meshes without changing their geometry.
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
    def create(
        cls,
        model: Model,
        solver_cfg: MPMSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverImplicitMPM:
        """Construct the configured implicit MPM solver."""
        scene_prim = None
        sim = PhysicsManager._sim
        if sim is not None:
            scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
            _author_mpm_scene_config(scene_prim, solver_cfg)
        return SolverImplicitMPM(model, _make_solver_config(solver_cfg, scene_prim), temporary_store=TemporaryStore())

    @property
    def supports_graph_capture(self) -> bool:
        """Whether the active MPM grid has capture-stable storage."""
        return mpm_supports_graph_capture(self.solver)

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts | None, dt: float) -> None:
        """Run one implicit MPM substep, optionally projecting particles out of colliders.

        The implicit solve resolves colliders at the grid level. With :attr:`MPMSolverCfg.project_outside_colliders`,
        the substep also hard-projects particles out of collider interiors, as in Newton's MPM examples.
        """
        self.solver.step(state_in, state_out, control, contacts, dt)
        if self.cfg.project_outside_colliders:
            self.solver.project_outside(state_out, state_out, dt)

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Keep implicit-MPM history across automatic asset resets.

        Shared-grid MPM cannot reset history for a subset of worlds. Tasks with independent MPM worlds that need an
        exact history reset call :meth:`NewtonMPMManager.reset_solver_state` after authoring their complete state.
        """

    def check_status(self, captured: bool) -> None:
        """Raise asynchronous sparse-grid rebuild failures after graph replay."""
        if captured:
            for solver in self.implicit_mpm_solvers:
                solver.check_sparse_grid_rebuild_status()


class NewtonMPMManager(NewtonManager):
    """:class:`NewtonManager` running Newton's implicit MPM solver."""

    solver_binding = MPMSolverBinding

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize Newton and author the MPM solver configuration in USD."""
        super().initialize(sim_context)
        scene_prim = sim_context.stage.GetPrimAtPath(sim_context.cfg.physics_prim_path)
        _author_mpm_scene_config(scene_prim, sim_context.cfg.physics.solver_cfg)

    @classmethod
    def reset_solver_state(
        cls,
        state: State | None = None,
        world_mask: wp.array(dtype=wp.bool) | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset MPM and coupled-solver history after task state is rewritten.

        When :paramref:`state` is omitted, both distinct manager state buffers are reset so a later buffer swap cannot
        restore stale history. A mask follows Newton's canonical ``world_count + 1`` contract, where the last entry
        selects global entities in world -1. A selected single local world is promoted to a full reset because a
        one-world MPM grid has no environment offsets.

        Args:
            state: State whose solver-owned history should be reset. If omitted, reset both manager states.
            world_mask: Canonical per-world mask, including the final global-world entry.
            flags: State components whose solver-owned history should reset.

        Raises:
            RuntimeError: If the MPM solver or a usable state is not initialized.
            ValueError: If :paramref:`world_mask` does not use Newton's canonical shape.
        """
        solver, backend = cls.get_solver(), cls.get_newton_backend()
        if solver is None or backend is None or not implicit_mpm_solvers(solver):
            raise RuntimeError("An implicit MPM solver is not initialized; cannot reset solver state.")
        model = backend.model

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

        candidates = (state,) if state is not None else (backend.state_1, backend.state_0)
        states = list({id(candidate): candidate for candidate in candidates if candidate is not None}.values())
        if not states:
            raise RuntimeError("Newton state is not initialized; provide an explicit state to reset.")
        for candidate in states:
            solver.reset(candidate, world_mask=reset_mask, flags=flags)
