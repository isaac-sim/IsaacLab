# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver-specific construction and stepping behind one interface."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar

import warp as wp
from newton import CollisionPipeline, Contacts, Control, Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverBase

from isaaclab.utils import to_dict

if TYPE_CHECKING:
    from .newton_manager_cfg import NewtonCfg, NewtonSolverCfg


class NewtonSolverBinding:
    """Construct and drive one Newton solver for the Newton runtime.

    The binding holds everything that differs between solvers: construction, the capabilities the step program
    composes around, stepping, per-world reset of solver-owned buffers, and forward kinematics. The runtime owns every
    other part of the step, so a new solver implements only this class.

    Subclasses set the capability attributes in :meth:`__init__` after constructing :attr:`solver`.
    """

    builder_attribute_solvers: ClassVar[tuple[type[SolverBase], ...]] = ()
    """Solvers whose custom builder attributes must be registered before import."""

    def __init__(self, model: Model, solver_cfg: NewtonSolverCfg, deterministic_mode: wp.DeterministicMode):
        """Construct the solver.

        Args:
            model: Finalized model the solver runs on.
            solver_cfg: Solver configuration.
            deterministic_mode: Determinism guarantee requested for the solver.
        """
        self.model = model
        self.cfg = solver_cfg
        self.deterministic_mode = deterministic_mode
        self.solver: SolverBase = self.construct()

        self.single_state: bool = False
        """Whether the solver steps in place on one :class:`State`."""
        self.needs_collision_pipeline: bool = True
        """Whether contacts come from Newton's :class:`CollisionPipeline` instead of the solver."""
        self.supports_body_forces: bool = True
        """Whether the solver consumes applied rigid-body forces from :attr:`State.body_f`."""
        self.supports_contact_sensors: bool = True
        """Whether Newton contact sensors can read this solver's contacts."""
        self.ignored_model_changes: dict[int, str] = {}
        """Model changes the solver does not apply after construction, mapped to the warning to log once."""

    # ----- Construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        model: Model,
        solver_cfg: NewtonSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverBase:
        """Construct the configured solver without binding it, so coupled solvers can nest it.

        Args:
            model: Model or model view the solver runs on.
            solver_cfg: Solver configuration.
            deterministic_mode: Determinism guarantee requested for the solver.

        Returns:
            The constructed solver.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement solver construction.")

    def construct(self) -> SolverBase:
        """Construct the solver this binding drives; defaults to :meth:`create` with the bound configuration."""
        return self.create(self.model, self.cfg, self.deterministic_mode)

    @staticmethod
    def filter_kwargs(solver_cls: type, solver_cfg: Any, deterministic_mode: wp.DeterministicMode) -> dict:
        """Return configuration fields that match the solver constructor.

        Args:
            solver_cls: Solver class to construct.
            solver_cfg: Solver configuration.
            deterministic_mode: Forwarded when the constructor accepts ``deterministic``.

        Returns:
            Constructor keyword arguments, excluding ``self`` and ``model``.
        """
        valid = set(inspect.signature(solver_cls.__init__).parameters) - {"self", "model"}
        kwargs = {key: value for key, value in to_dict(solver_cfg).items() if key in valid}
        if "deterministic" in valid:
            kwargs["deterministic"] = deterministic_mode
        return kwargs

    @classmethod
    def register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register custom attributes the solver reads from the model.

        Args:
            builder: Builder that receives imported assets.
        """
        for solver_cls in cls.builder_attribute_solvers:
            solver_cls.register_custom_attributes(builder)

    @classmethod
    def registers_builder_attributes_from(cls, solver_cls: type[SolverBase]) -> bool:
        """Return whether this binding registers ``solver_cls``'s custom builder attributes.

        Args:
            solver_cls: Solver whose attributes, and matching USD schemas, may be imported.
        """
        return solver_cls in cls.builder_attribute_solvers

    @classmethod
    def prepare_builder(cls, builder: ModelBuilder) -> None:
        """Normalize a complete builder before model allocation. The default is a no-op.

        Args:
            builder: Builder about to be finalized.
        """

    @classmethod
    def validate_cfg(cls, cfg: NewtonCfg) -> None:
        """Reject settings that conflict across the physics and solver configurations. The default accepts all.

        Args:
            cfg: Newton physics configuration.
        """

    # ----- Runtime ----------------------------------------------------------

    @property
    def supports_graph_capture(self) -> bool:
        """Whether the configured solver can be recorded into a CUDA graph."""
        return True

    def initialize_output_state(self, state: State) -> None:
        """Initialize solver-owned buffers of the output state of double-buffered solvers before capture.

        The default is a no-op.

        Args:
            state: Output state of the first substep.
        """

    def create_contacts(self) -> Contacts | None:
        """Allocate contacts produced by the solver's internal collision detection.

        Called only when :attr:`needs_collision_pipeline` is ``False``.

        Returns:
            Contacts the solver can report into, or ``None`` when it reports none.
        """
        return None

    def prepare_contacts(self, contacts: Contacts, collision_pipeline: CollisionPipeline | None) -> None:
        """Bind solver-owned buffers to newly allocated contacts. The default is a no-op.

        Args:
            contacts: Contacts used by the step program.
            collision_pipeline: Pipeline that fills ``contacts``, or ``None`` when the solver collides internally.
        """

    def minimum_contact_capacity(self) -> int:
        """Return the rigid-contact capacity the solver requires from the collision pipeline."""
        get_max_contact_count = getattr(self.solver, "get_max_contact_count", None)
        return get_max_contact_count() if get_max_contact_count is not None else 0

    def prepare_step(self, state: State) -> None:
        """Refresh solver acceleration structures once per physics step, before collision.

        Returns without work by default. Subclasses that override it must stay graph-safe.

        Args:
            state: Current state.
        """

    @property
    def prepares_step(self) -> bool:
        """Whether :meth:`prepare_step` performs work."""
        return type(self).prepare_step is not NewtonSolverBinding.prepare_step

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts | None, dt: float) -> None:
        """Advance one solver substep.

        Args:
            state_in: Input state.
            state_out: Output state; the same object as ``state_in`` for single-state solvers.
            control: Control inputs.
            contacts: Contacts from the collision pipeline, or ``None`` when the solver collides internally.
            dt: Substep duration [s].
        """
        self.solver.step(state_in, state_out, control, contacts, dt)

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Clear solver-owned history for masked worlds while keeping authored joint state.

        Args:
            state: State whose solver-owned buffers are reset.
            world_mask: Per-world mask of shape ``(world_count + 1,)``; the final entry selects global entities.
        """
        self.solver.reset(state, world_mask=world_mask, flags=0)

    def eval_fk(self, state: State, world_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body state from joint coordinates for masked articulations.

        Args:
            state: State to update in place.
            world_mask: Per-world mask of reset worlds, or ``None`` for all worlds.
            fk_mask: Per-articulation mask, or ``None`` for all articulations.
        """
        eval_fk(self.model, state.joint_q, state.joint_qd, state, fk_mask)

    def notify_model_changed(self, change: int) -> None:
        """Forward a model change to the solver.

        Args:
            change: :class:`newton.ModelFlags` value.
        """
        self.solver.notify_model_changed(change)

    def check_status(self, captured: bool) -> None:
        """Raise asynchronous solver failures after a step. The default is a no-op.

        Args:
            captured: Whether the step replayed a captured graph.
        """

    def log_debug(self) -> None:
        """Log solver diagnostics after a step when debug mode is enabled. The default is a no-op."""

    @classmethod
    def create_fixed_tendon_control(cls, articulation: Any, model: Model) -> Any:
        """Build the fixed-tendon command adapter for an articulation.

        Only solvers that transmit to tendons implement this; articulations report tendons under those solvers alone.
        Articulations call this while ``PHYSICS_READY`` dispatches, before the solver is constructed, so it depends
        only on the model.

        Args:
            articulation: Newton articulation to drive.
            model: Finalized model.

        Raises:
            NotImplementedError: Always, for solvers without fixed-tendon transmission.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not drive fixed tendons. Fixed-tendon targets require a solver"
            " that transmits to tendons, such as MJWarp."
        )
