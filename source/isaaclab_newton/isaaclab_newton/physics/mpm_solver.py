# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM solver implementation and reset helpers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import warp as wp
from newton import Contacts, Control, GeoType, Model, ModelBuilder, State, StateFlags
from newton.solvers import SolverBase, SolverImplicitMPM
from warp.fem import TemporaryStore

from isaaclab.sim import SimulationContext

from .mpm_solver_cfg import MPMSolverCfg
from .newton_solver import NewtonSolver

if TYPE_CHECKING:
    from .newton_backend import NewtonBackend


def implicit_mpm_solvers(solver: SolverBase | None) -> tuple[SolverImplicitMPM, ...]:
    """Return a direct implicit-MPM solver, or the implicit-MPM entries of a coupled solver."""
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
    """Return whether the configured grid and bases support native graph capture."""
    if solver.enable_timers or solver.max_active_cell_count <= 0:
        return False
    for basis in (solver.strain_basis, solver.collider_basis):
        if basis.startswith("pic") and not basis[3:].isdigit():
            return False
    if solver.grid_type == "fixed":
        return True
    return (
        solver.grid_type == "sparse"
        and solver.grid_padding == 0
        and (solver.strain_basis.startswith("pic") or solver.strain_basis in ("P0", "P1d", "Q1d", "Q1"))
        and (solver.collider_basis.startswith("pic") or solver.collider_basis in ("Q1", "S2", "S3"))
    )


def mpm_solver_config(cfg: MPMSolverCfg) -> SolverImplicitMPM.Config:
    """Return the native configuration and retain legacy active-scene authoring.

    .. deprecated::
        Use :meth:`MPMSolverCfg.to_solver_config` for construction and
        :meth:`MPMSolverCfg.author_usd` when exporting solver settings to USD.

    Args:
        cfg: MPM solver configuration.
    """
    warnings.warn(
        "mpm_solver_config() is deprecated; use cfg.to_solver_config() and explicitly call cfg.author_usd(scene_prim)"
        " when exporting solver settings.",
        DeprecationWarning,
        stacklevel=2,
    )
    sim = SimulationContext.instance()
    if sim is not None:
        cfg.author_usd(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
    return cfg.to_solver_config()


class MPMSolver(NewtonSolver):
    """Newton's implicit MPM solver, which steps particle materials in place on one state."""

    cfg: MPMSolverCfg

    solver_class = SolverImplicitMPM
    single_state = True
    contact_sensors = False

    @classmethod
    def prepare_builder(cls, cfg: MPMSolverCfg, builder: ModelBuilder) -> None:
        """Convert convex colliders to the triangle-mesh type consumed by MPM.

        Newton applies zero effective collider mass to kinematic bodies without modifying their physical model mass,
        which other coupled solvers need.
        """
        for shape_id, shape_type in enumerate(builder.shape_type):
            if shape_type == GeoType.CONVEX_MESH:
                builder.shape_type[shape_id] = GeoType.MESH

    @classmethod
    def coupling_error(cls, cfg: MPMSolverCfg, in_place: bool) -> str | None:
        """MPM entries step in place, and the per-substep collider projection runs outside Newton's coupled step."""
        if cfg.project_outside_colliders:
            return (
                "enables MPMSolverCfg.project_outside_colliders, whose per-substep projection cannot yet run inside a"
                " coupled-solver entry"
            )
        if not in_place:
            return "uses MPMSolverCfg and must set in_place=True"
        return None

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverImplicitMPM:
        """Construct implicit MPM from the configured Python values."""
        return self.solver_class(model, self.cfg.to_solver_config(), temporary_store=TemporaryStore())

    def applies_body_forces(self) -> bool:
        return False

    def supports_capture(self) -> bool:
        """Whether the solver can be captured: on CUDA only, and only for grids with static topology.

        On the CPU, the solver reads its rheology residual on the host and rebuilds sparse grids through
        :meth:`warp.Volume.rebuild`, neither of which Warp's CPU graph capture replays.
        """
        return self.native.model.device.is_cuda and mpm_supports_graph_capture(self.native)

    def uses_collision_pipeline(self) -> bool:
        return False

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Run one implicit MPM substep, optionally projecting particles out of collider interiors."""
        self.native.step(state_in, state_out, control, contacts, dt)
        if self.cfg.project_outside_colliders:
            self.native.project_outside(state_out, state_out, dt)

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Keep implicit-MPM history across automatic asset resets.

        Shared-grid MPM cannot reset history for a subset of worlds. Tasks with independent MPM worlds that need an
        exact history reset call :func:`reset_mpm_solver_state` after authoring their state.
        """

    def check_step(self, captured: bool, debug: bool) -> None:
        """Raise asynchronous sparse-grid rebuild failures after a graph replay."""
        if captured:
            self.native.check_sparse_grid_rebuild_status()


def reset_mpm_solver_state(
    backend: NewtonBackend,
    state: State | None = None,
    world_mask: wp.array(dtype=wp.bool) | None = None,
    flags: StateFlags | int | None = None,
) -> None:
    """Reset MPM and coupled-solver history after task state is rewritten.

    When :paramref:`state` is omitted, both distinct backend state buffers are reset so a later buffer swap cannot
    restore stale history. A mask follows Newton's canonical ``world_count + 1`` contract, where the last entry
    selects global entities in world -1. A selected single local world is promoted to a full reset because a
    one-world MPM grid has no environment offsets.

    Args:
        backend: Backend whose solver to reset, e.g. :attr:`NewtonManager.backend`.
        state: State whose solver-owned history should be reset. If omitted, reset both backend states.
        world_mask: Canonical per-world mask, including the final global-world entry.
        flags: State components whose solver-owned history should reset.

    Raises:
        RuntimeError: If the MPM solver or a usable state is not initialized.
        ValueError: If :paramref:`world_mask` does not use Newton's canonical shape.
    """
    solver = backend.solver
    if not implicit_mpm_solvers(solver):
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
