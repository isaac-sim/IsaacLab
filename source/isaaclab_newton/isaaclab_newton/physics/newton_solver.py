# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateless solver adapters used by the functional Newton simulation API."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar

import warp as wp
from newton import Contacts, Model, ModelBuilder, ShapeFlags, State, eval_fk
from newton.solvers import SolverBase, SolverMuJoCo
from newton.usd import SchemaResolver, SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

from isaaclab.utils import checked_apply, to_dict

from .newton_manager_cfg import NewtonCfg, NewtonShapeCfg

if TYPE_CHECKING:
    from .newton_backend import NewtonBackend
    from .newton_manager_cfg import NewtonSolverCfg


class NewtonSolver:
    """Construction, stepping, reset and capability hooks over an explicit backend, with no runtime state."""

    @classmethod
    def create_builder(
        cls, up_axis: str | None = None, *, physics_cfg: NewtonCfg | None = None, **kwargs
    ) -> ModelBuilder:
        """Create a :class:`ModelBuilder` configured with default settings.

        Forwards :class:`NewtonShapeCfg` defaults onto Newton's upstream
        ``ModelBuilder.default_shape_cfg`` via :func:`~isaaclab.utils.checked_apply`.
        Falls back to wrapper defaults when no Newton config is active so
        rough-terrain margin/gap still apply during early construction.

        Args:
            up_axis: Override for the up-axis. Defaults to ``"Z"``.
            physics_cfg: Explicit builder settings; None uses the default shape settings.
            **kwargs: Forwarded to :class:`ModelBuilder`.

        Returns:
            New builder with up-axis and per-shape defaults (gap, margin) applied.
        """
        cfg = physics_cfg
        newton_cfg = cfg if isinstance(cfg, NewtonCfg) else None
        builder = ModelBuilder(up_axis=up_axis or "Z", **kwargs)
        builder.default_bvh_cfg = ModelBuilder.BvhConfig(
            mesh_constructor=newton_cfg.bvh_constructor_geometry if newton_cfg else None,
            gaussian_constructor=newton_cfg.bvh_constructor_gaussian if newton_cfg else None,
            shape_constructor=newton_cfg.bvh_constructor_scene if newton_cfg else None,
            shape_flags=ShapeFlags.VISIBLE,
        )
        cls.register_builder_attributes(builder, newton_cfg.solver_cfg if newton_cfg else None)
        checked_apply(newton_cfg.default_shape_cfg if newton_cfg else NewtonShapeCfg(), builder.default_shape_cfg)
        return builder

    @classmethod
    def get_usd_import_schema_resolvers(cls, solver_cfg: NewtonSolverCfg | None) -> list[SchemaResolver]:
        """Return ordered schema resolvers for physics-model USD imports.

        MJC is enabled for adapters that register ``SolverMuJoCo`` attributes. Visualization and articulation-ordering
        builders keep their fixed pair because solver attributes do not affect their outputs.

        Args:
            solver_cfg: Solver configuration of the imported model; ``None`` without Newton physics.
        """
        resolvers: list[SchemaResolver] = [SchemaResolverNewton(), SchemaResolverPhysx()]
        if cls.registers_builder_attributes_from(SolverMuJoCo, solver_cfg):
            resolvers.append(SchemaResolverMjc())
        return resolvers

    # ----- Solver hooks ------------------------------------------------------------------
    # Overridden by solver adapters. They are stateless and take every input explicitly, so a solver adapter can
    # drive several backends at once.

    solver_class: ClassVar[type[SolverBase] | None] = None
    """Newton solver the default :meth:`create_solver` constructs from the matching configuration fields."""

    builder_attribute_solvers: ClassVar[tuple[type[SolverBase], ...]] = ()
    """Solvers whose custom builder attributes are registered before import."""

    single_state: ClassVar[bool] = False
    """Whether the solver steps in place on one :class:`newton.State`."""

    supports_deterministic: ClassVar[bool] = False
    """Whether the solver can honor a Warp determinism guarantee."""

    supports_contact_sensors: ClassVar[bool] = True
    """Whether Newton contact sensors can read the solver's contacts."""

    prepares_step: ClassVar[bool] = False
    """Whether :meth:`prepare_step` does work, so the step graph runs it."""

    supports_heterogeneous_worlds: ClassVar[bool] = False
    """Whether one solver can step worlds with different contents, such as different robots per world."""

    ignored_model_changes: ClassVar[dict[int, str]] = {}
    """Model changes the solver does not apply after construction, mapped to the warning logged once."""

    @classmethod
    def create_solver(
        cls,
        model: Model,
        solver_cfg: NewtonSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverBase:
        """Construct the configured solver. Coupled solvers call this to build their entries.

        The default constructs :attr:`solver_class` from the configuration fields its constructor accepts.

        Args:
            model: Model or model view the solver runs on.
            solver_cfg: Solver configuration.
            deterministic_mode: Determinism guarantee requested for the solver.

        Returns:
            The solver.
        """
        if cls.solver_class is None:
            raise NotImplementedError(f"{cls.__name__} does not implement solver construction.")
        return cls.solver_class(model, **cls.solver_kwargs(cls.solver_class, solver_cfg, deterministic_mode))

    @staticmethod
    def solver_kwargs(solver_cls: type, solver_cfg: Any, deterministic_mode: wp.DeterministicMode) -> dict:
        """Return the configuration fields that match the solver constructor.

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
    def validate_cfg(cls, backend: NewtonBackend) -> None:
        """Reject configurations the solver cannot run, before it is constructed.

        Args:
            backend: Backend whose solver is about to be constructed.
        """
        mode = backend.deterministic_mode
        if mode != wp.DeterministicMode.NOT_GUARANTEED and not cls.supports_deterministic:
            raise ValueError(
                f"Newton deterministic mode {mode.name} is not supported by {type(backend.cfg.solver_cfg).__name__}."
                " Use MJWarp on the GPU, XPBD, or Featherstone, or disable deterministic mode."
            )

    @classmethod
    def register_builder_attributes(cls, builder: ModelBuilder, solver_cfg: NewtonSolverCfg | None) -> None:
        """Register custom attributes the solver reads from the model.

        Args:
            builder: Builder that receives imported assets.
            solver_cfg: Solver configuration of the model.
        """
        for solver_cls in cls.builder_attribute_solvers:
            solver_cls.register_custom_attributes(builder)

    @classmethod
    def registers_builder_attributes_from(
        cls, solver_cls: type[SolverBase], solver_cfg: NewtonSolverCfg | None
    ) -> bool:
        """Return whether this manager registers ``solver_cls``'s custom builder attributes.

        Args:
            solver_cls: Solver whose attributes, and matching USD schemas, may be imported.
            solver_cfg: Solver configuration of the model.
        """
        return solver_cls in cls.builder_attribute_solvers

    @classmethod
    def prepare_solver_builder(cls, builder: ModelBuilder, solver_cfg: NewtonSolverCfg) -> None:
        """Normalize a complete builder for the solver before finalization. The default is a no-op.

        Args:
            builder: Builder about to be finalized.
            solver_cfg: Solver configuration of the model.
        """

    @classmethod
    def uses_collision_pipeline(cls, backend: NewtonBackend) -> bool:
        """Whether contacts come from Newton's :class:`~newton.CollisionPipeline` instead of the solver."""
        return True

    @classmethod
    def supports_body_forces(cls, backend: NewtonBackend) -> bool:
        """Whether the solver consumes applied rigid-body forces from :attr:`newton.State.body_f`."""
        return True

    @classmethod
    def supports_graph_capture(cls, backend: NewtonBackend) -> bool:
        """Whether the configured solver can be recorded into a CUDA graph."""
        return True

    @classmethod
    def create_contacts(cls, backend: NewtonBackend) -> Contacts | None:
        """Allocate contacts the solver reports from internal collision detection.

        Called only when :meth:`uses_collision_pipeline` is ``False``.
        """
        return None

    @classmethod
    def prepare_contacts(cls, backend: NewtonBackend) -> None:
        """Bind solver-owned buffers to newly allocated contacts. The default is a no-op."""

    @classmethod
    def initialize_output_state(cls, backend: NewtonBackend, state: State) -> None:
        """Initialize solver-owned buffers of the output state of double-buffered solvers. The default is a no-op."""

    @classmethod
    def prepare_step(cls, backend: NewtonBackend, state: State) -> None:
        """Refresh solver acceleration structures once per physics step, before collision; set :attr:`prepares_step`.

        Overrides must be graphable.
        """

    @classmethod
    def step_solver(
        cls, backend: NewtonBackend, state_in: State, state_out: State, contacts: Contacts | None, dt: float
    ) -> None:
        """Advance one solver substep.

        Args:
            backend: Backend whose solver to step.
            state_in: Input state.
            state_out: Output state; the same object as ``state_in`` for single-state solvers.
            contacts: Contacts from the collision pipeline, or ``None`` when the solver detects contacts internally.
            dt: Substep duration [s].
        """
        backend.solver.step(state_in, state_out, backend.control, contacts, dt)

    @classmethod
    def reset_solver(cls, backend: NewtonBackend, state: State, world_mask: wp.array) -> None:
        """Clear solver-owned history for masked worlds while keeping authored joint state.

        Args:
            backend: Backend whose solver to reset.
            state: State whose solver-owned buffers are reset.
            world_mask: Per-world mask of shape ``(world_count + 1,)``; the final entry selects global entities.
        """
        backend.solver.reset(state, world_mask=world_mask, flags=0)

    @classmethod
    def eval_fk(
        cls, backend: NewtonBackend, state: State, world_mask: wp.array | None, fk_mask: wp.array | None
    ) -> None:
        """Update body state from joint coordinates for masked articulations.

        Args:
            backend: Backend whose model to evaluate.
            state: State to update in place.
            world_mask: Per-world mask of reset worlds, or ``None`` for all worlds.
            fk_mask: Per-articulation mask, or ``None`` for all articulations.
        """
        eval_fk(backend.model, state.joint_q, state.joint_qd, state, fk_mask)

    @classmethod
    def check_status(cls, backend: NewtonBackend, captured: bool) -> None:
        """Raise asynchronous solver failures after a step. The default is a no-op.

        Args:
            backend: Backend that stepped.
            captured: Whether the step replayed a captured graph.
        """

    @classmethod
    def log_debug(cls, backend: NewtonBackend) -> None:
        """Log solver diagnostics after a step when debug mode is enabled. The default is a no-op."""

    @classmethod
    def create_fixed_tendon_control(cls, articulation: Any, model: Model) -> Any:
        """Build the solver's fixed-tendon command adapter for ``articulation``.

        Tendon state is backend-neutral and lives on the articulation; how a target reaches the solver is not. Only
        solvers that transmit to tendons implement this.

        Args:
            articulation: Newton articulation to drive.
            model: Finalized model of the articulation.

        Raises:
            NotImplementedError: For solvers without fixed-tendon transmission.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not drive fixed tendons. Fixed-tendon targets require a solver"
            " that transmits to tendons, such as MJWarp."
        )
