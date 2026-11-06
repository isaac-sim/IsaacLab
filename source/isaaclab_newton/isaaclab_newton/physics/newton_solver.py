# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How Isaac Lab builds and drives a Newton solver."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import warp as wp
from newton import CollisionPipeline, Contacts, Control, Model, ModelBuilder, ModelFlags, State, eval_fk
from newton.solvers import SolverBase

from isaaclab.utils import to_dict

if TYPE_CHECKING:
    from .newton_manager_cfg import NewtonSolverCfg


def solver_kwargs(solver_cls: type, cfg: Any, mode: wp.DeterministicMode) -> dict:
    """Return the configuration fields matching ``solver_cls``'s constructor, plus ``deterministic`` if accepted.

    A constructor forwarding ``**kwargs`` also accepts its base class's parameters, e.g. a solver subclass.
    """
    valid = set()
    for cls in solver_cls.__mro__:
        parameters = inspect.signature(cls.__init__).parameters.values()
        valid.update(p.name for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
        if not any(p.kind is p.VAR_KEYWORD for p in parameters):
            break
    valid -= {"self", "model"}
    kwargs = {key: value for key, value in to_dict(cfg).items() if key in valid}
    if "deterministic" in valid:
        kwargs["deterministic"] = mode
    return kwargs


class NewtonSolver:
    """How Isaac Lab builds and drives a Newton solver.

    The class is never instantiated and holds no state: every method is a classmethod that takes the solver
    configuration and, once built, Newton's solver. The generic methods below follow Newton's
    :class:`~newton.solvers.SolverBase` API; a subclass overrides only where its solver differs. A solver configuration
    selects the subclass through :attr:`~NewtonSolverCfg.class_type`.
    """

    solver_class: ClassVar[type[SolverBase] | None] = None
    """Newton solver class that :meth:`build` constructs; ``None`` describes geometry only."""

    single_state: ClassVar[bool] = False
    """Whether the solver steps in place on one :class:`newton.State`."""

    deterministic: ClassVar[bool] = False
    """Whether the solver can honor a Warp determinism guarantee."""

    contact_sensors: ClassVar[bool] = True
    """Whether Newton contact sensors can read the solver's contacts."""

    ignored_model_changes: ClassVar[Mapping[ModelFlags, str]] = {}
    """Model changes the solver does not apply after construction, mapped to the warning logged once."""

    # ----- Before the model exists -----

    @classmethod
    def solver_classes(cls, cfg: NewtonSolverCfg) -> tuple[type[SolverBase], ...]:
        """Newton solvers whose custom builder attributes, and USD schemas, the builder needs before import."""
        return () if cls.solver_class is None else (cls.solver_class,)

    @classmethod
    def prepare_builder(cls, cfg: NewtonSolverCfg, builder: ModelBuilder) -> None:
        """Normalize the complete builder before it is finalized."""

    @classmethod
    def applied_deterministic_mode(cls, cfg: NewtonSolverCfg, mode: wp.DeterministicMode) -> wp.DeterministicMode:
        """The Warp determinism mode the solver and collision pipeline run with for a requested ``mode``."""
        return mode

    @classmethod
    def shares_materials(cls, cfg: NewtonSolverCfg) -> bool:
        """Whether contacts read one material table shared by all environments instead of per-shape values."""
        return False

    @classmethod
    def coupling_error(cls, cfg: NewtonSolverCfg, in_place: bool) -> str | None:
        """Why the solver cannot run as a coupled-solver entry (stepping in place or not), or ``None``."""
        return None

    # ----- Construction -----

    @classmethod
    def build(
        cls, cfg: NewtonSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> SolverBase:
        """Construct Newton's solver from the configuration fields matching its constructor.

        Args:
            cfg: Solver configuration.
            model: Model, or coupled-entry model view, the solver advances.
            mode: Warp determinism mode.
            env_template: Environment path template of the clone plan, e.g. ``"/World/envs/env_{}"``.
        """
        if cls.solver_class is None:
            raise NotImplementedError(f"{cls.__name__} describes geometry only and builds no solver.")
        return cls.solver_class(model, **solver_kwargs(cls.solver_class, cfg, mode))

    @classmethod
    def init_state(cls, cfg: NewtonSolverCfg, solver: SolverBase, state: State) -> None:
        """Initialize solver-owned buffers of a double-buffered solver's output state."""

    # ----- Capabilities -----

    @classmethod
    def applies_body_forces(cls, cfg: NewtonSolverCfg, solver: SolverBase) -> bool:
        """Whether the solver consumes applied rigid-body forces from :attr:`newton.State.body_f`."""
        return True

    @classmethod
    def supports_capture(cls, cfg: NewtonSolverCfg, solver: SolverBase) -> bool:
        """Whether the solver can be recorded into a CUDA graph."""
        return True

    # ----- Contacts -----

    @classmethod
    def uses_collision_pipeline(cls, cfg: NewtonSolverCfg, solver: SolverBase) -> bool:
        """Whether contacts come from Newton's :class:`~newton.CollisionPipeline` instead of the solver."""
        return True

    @classmethod
    def create_collision_pipeline(cls, cfg: NewtonSolverCfg, solver: SolverBase, **kwargs) -> CollisionPipeline:
        """Construct the collision pipeline over the solver's model; ``kwargs`` are the pipeline arguments."""
        return CollisionPipeline(solver.model, **kwargs)

    @classmethod
    def create_contacts(cls, cfg: NewtonSolverCfg, solver: SolverBase) -> Contacts | None:
        """Allocate the contacts a solver reports itself when it does not use the collision pipeline."""
        return None

    @classmethod
    def prepare_contacts(
        cls, cfg: NewtonSolverCfg, solver: SolverBase, contacts: Contacts, pipeline: CollisionPipeline | None
    ) -> None:
        """Bind newly allocated contacts to solver-owned buffers before CUDA graph capture."""
        if hasattr(solver, "prepare_contacts"):
            solver.prepare_contacts(contacts)

    # ----- Stepping and resets -----

    @classmethod
    def pre_step(cls, cfg: NewtonSolverCfg, solver: SolverBase, state: State) -> None:
        """Refresh solver structures once per physics step, before collision."""

    @classmethod
    def step(
        cls,
        cfg: NewtonSolverCfg,
        solver: SolverBase,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance one substep of ``dt`` [s]."""
        solver.step(state_in, state_out, control, contacts, dt)

    @classmethod
    def post_step(cls, cfg: NewtonSolverCfg, solver: SolverBase, state: State) -> None:
        """Publish derived state once per physics step, after the substeps."""

    @classmethod
    def reset(cls, cfg: NewtonSolverCfg, solver: SolverBase, state: State, world_mask: wp.array) -> None:
        """Clear solver-owned history for the worlds of a ``(world_count + 1,)`` mask, keeping authored joint state."""
        solver.reset(state, world_mask=world_mask, flags=0)

    @classmethod
    def eval_fk(
        cls,
        cfg: NewtonSolverCfg,
        solver: SolverBase,
        state: State,
        world_mask: wp.array | None,
        fk_mask: wp.array | None,
    ) -> None:
        """Update body state from the joint coordinates of the masked articulations; ``None`` selects everything."""
        eval_fk(solver.model, state.joint_q, state.joint_qd, state, fk_mask)

    @classmethod
    def check_step(cls, cfg: NewtonSolverCfg, solver: SolverBase, captured: bool, debug: bool) -> None:
        """Raise asynchronous solver failures after a step launch (``captured``: a CUDA graph replay), and log
        diagnostics when ``debug`` is set."""

    # ----- Consumers -----

    @classmethod
    def fixed_tendon_control(cls, cfg: NewtonSolverCfg, articulation: Any, model: Model) -> Any:
        """Build the fixed-tendon command adapter of an articulation, or ``None`` when no tendon is driven."""
        raise NotImplementedError(
            "The configured solver does not drive fixed tendons. Fixed-tendon targets require a solver that transmits"
            " to tendons, such as MJWarp."
        )
