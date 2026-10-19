# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver specifications: what Isaac Lab needs to know about each Newton solver type.

A :class:`SolverSpec` holds functions and constants, never state. A solver configuration selects one through its
``class_type``, so ``instantiate(solver_cfg, model, deterministic_mode)`` builds Newton's solver, the only solver
instance. The spec functions take that solver and its configuration explicitly, so a coupled solver calls the specs of
its entries with their own solvers and configurations.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import warp as wp
from newton import CollisionPipeline, Contacts, Control, Model, ModelBuilder, ShapeFlags, State, eval_fk
from newton.solvers import SolverBase, SolverMuJoCo
from newton.usd import SchemaResolver, SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

from isaaclab.utils import checked_apply, to_dict
from isaaclab.utils.string import string_to_callable

from .newton_manager_cfg import NewtonCfg, NewtonShapeCfg
from .vbd_solver_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from .newton_manager_cfg import NewtonSolverCfg


# ----- Default spec functions --------------------------------------------------------


def _build_unsupported(model: Model, cfg: Any, deterministic_mode: wp.DeterministicMode) -> SolverBase:
    raise NotImplementedError(f"{type(cfg).__name__} does not select a solver specification that builds a solver.")


def native_step(
    solver: SolverBase,
    cfg: Any,
    state_in: State,
    state_out: State,
    control: Control,
    contacts: Contacts | None,
    dt: float,
) -> None:
    """Advance one substep with Newton's ``solver.step``."""
    solver.step(state_in, state_out, control, contacts, dt)


def native_reset(solver: SolverBase, cfg: Any, state: State, world_mask: wp.array) -> None:
    """Clear solver-owned history for masked worlds with Newton's ``solver.reset``, keeping authored joint state."""
    solver.reset(state, world_mask=world_mask, flags=0)


def native_eval_fk(
    solver: SolverBase, cfg: Any, state: State, world_mask: wp.array | None, fk_mask: wp.array | None
) -> None:
    """Update body state from the joint coordinates of the masked articulations with Newton's ``eval_fk``."""
    eval_fk(solver.model, state.joint_q, state.joint_qd, state, fk_mask)


def native_collision_pipeline(model: Model, solver: SolverBase, cfg: Any, **kwargs) -> CollisionPipeline:
    """Construct Newton's collision pipeline over the whole model."""
    return CollisionPipeline(model, **kwargs)


def _no_fixed_tendons(articulation: Any, model: Model) -> Any:
    raise NotImplementedError(
        "The configured solver does not drive fixed tendons. Fixed-tendon targets require a solver that transmits to"
        " tendons, such as MJWarp."
    )


def _noop(*args, **kwargs) -> None:
    return None


def _true(*args, **kwargs) -> bool:
    return True


# ----- The specification ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolverSpec:
    """Construction, capabilities and lifecycle hooks of one Newton solver type, as functions and constants.

    Functions called before the model exists take the solver configuration (``cfg``); functions called afterwards
    take Newton's solver and its configuration. None of them read the backend, so a coupled solver can call the
    specs of its entries with each entry's solver and configuration. Every field has a default, so a spec sets only
    what differs, e.g. ``SolverSpec(build=build_from_cfg(SolverXPBD), deterministic=True)``; variants derive with
    :func:`dataclasses.replace`.
    """

    # ----- Before the model exists ----------------------------------------------------

    attribute_solvers: Callable[[Any], tuple[type[SolverBase], ...]] = lambda cfg: ()
    """Newton solvers whose custom builder attributes (and USD schemas) the configuration needs before import."""

    prepare_builder: Callable[[ModelBuilder, Any], None] = _noop
    """Normalize the complete builder for the solver before it is finalized."""

    # ----- Construction ------------------------------------------------------------------

    resolve_cfg: Callable[[Model, Any], Any] = lambda model, cfg: cfg
    """Return the configuration with model-dependent defaults resolved. The backend keeps the result as
    :attr:`NewtonBackend.solver_cfg` and passes it to every later function."""

    validate: Callable[[Model, Any, NewtonCfg, wp.DeterministicMode], None] = _noop
    """Reject configurations the solver cannot run, given the model, physics configuration and determinism mode."""

    build: Callable[[Model, Any, wp.DeterministicMode], SolverBase] = _build_unsupported
    """Construct Newton's solver for a model, e.g. :func:`build_from_cfg`."""

    init_state: Callable[[SolverBase, Any, State], None] = _noop
    """Initialize solver-owned buffers of the output state of a double-buffered solver."""

    # ----- Capabilities -------------------------------------------------------------------

    single_state: bool = False
    """Whether the solver steps in place on one :class:`newton.State`."""

    deterministic: bool = False
    """Whether the solver can honor a Warp determinism guarantee."""

    contact_sensors: bool = True
    """Whether Newton contact sensors can read the solver's contacts."""

    body_forces: Callable[[Any], bool] = _true
    """Whether the solver consumes applied rigid-body forces from :attr:`newton.State.body_f`."""

    supports_capture: Callable[[SolverBase, Any], bool] = _true
    """Whether the constructed solver can be recorded into a CUDA graph."""

    captures_model_changes: bool = False
    """Whether ``notify_model_changed`` can be recorded into a conditional CUDA graph; otherwise it runs eagerly."""

    ignored_model_changes: Mapping[int, str] = field(default_factory=lambda: MappingProxyType({}))
    """Model changes the solver does not apply after construction, mapped to the warning logged once."""

    coupling_error: Callable[[Any, bool], str | None] = lambda cfg, in_place: None
    """Return why the solver cannot run as an entry of a coupled solver (stepping in place or not), or ``None``."""

    # ----- Contacts --------------------------------------------------------------------------

    external_contacts: Callable[[SolverBase, Any], bool] = _true
    """Whether contacts come from Newton's :class:`~newton.CollisionPipeline` instead of the solver."""

    create_collision_pipeline: Callable[..., CollisionPipeline] = native_collision_pipeline
    """Construct the collision pipeline ``(model, solver, cfg, **pipeline_args)`` of a solver with external
    contacts."""

    create_contacts: Callable[[SolverBase, Any], Contacts | None] = lambda solver, cfg: None
    """Allocate the contacts that a solver without external contacts reports itself."""

    prepare_contacts: Callable[[SolverBase, Any, Contacts, CollisionPipeline | None], None] = _noop
    """Bind solver-owned buffers to newly allocated contacts and the collision pipeline."""

    # ----- Stepping and resets ---------------------------------------------------------------

    prepare_step: Callable[[SolverBase, Any, State], None] | None = None
    """Refresh solver structures once per physics step, before collision."""

    step: Callable[[SolverBase, Any, State, State, Control, Contacts | None, float], None] = native_step
    """Advance one substep ``(solver, cfg, state_in, state_out, control, contacts, dt)``."""

    finish_step: Callable[[SolverBase, Any, State], None] | None = None
    """Publish derived state once per physics step, after the substeps."""

    reset: Callable[[SolverBase, Any, State, wp.array], None] = native_reset
    """Clear solver-owned history for the worlds of a ``(world_count + 1,)`` mask, keeping authored joint state."""

    eval_fk: Callable[[SolverBase, Any, State, wp.array | None, wp.array | None], None] = native_eval_fk
    """Update body state from joint coordinates ``(solver, cfg, state, world_mask, fk_mask)``; ``None`` masks
    select everything."""

    check_status: Callable[[SolverBase, Any, bool], None] = _noop
    """Raise asynchronous solver failures after a step (``True``: after a CUDA graph replay)."""

    log_debug: Callable[[SolverBase, Any], None] = _noop
    """Log solver diagnostics after a step when :attr:`NewtonCfg.debug_mode` is enabled."""

    # ----- Consumers ---------------------------------------------------------------------------

    fixed_tendon_control: Callable[[Any, Model], Any] = _no_fixed_tendons
    """Build the fixed-tendon command adapter of an articulation, or ``None`` when no tendon is driven."""

    def __call__(self, cfg: Any, model: Model, deterministic_mode: wp.DeterministicMode) -> SolverBase:
        """Build Newton's solver; makes the spec a ``class_type`` for :func:`~isaaclab.utils.instantiate`."""
        return self.build(model, cfg, deterministic_mode)

    def __copy__(self) -> SolverSpec:
        return self

    def __deepcopy__(self, memo: dict) -> SolverSpec:
        # Immutable and stateless: configuration copies share the spec, so specs compare by identity. The slots
        # also keep configuration post-processing from rewriting its fields.
        return self


GEOMETRY_SPEC = SolverSpec()
"""Specification of builders that only describe geometry, e.g. for rendering under another physics backend."""


def solver_spec(cfg: NewtonSolverCfg | None) -> SolverSpec:
    """Return the specification a solver configuration selects; ``None`` gives :data:`GEOMETRY_SPEC`.

    Args:
        cfg: Solver configuration whose ``class_type`` is a :class:`SolverSpec` or a ``"module:ATTRIBUTE"`` reference.
    """
    spec = GEOMETRY_SPEC if cfg is None or cfg.class_type is None else cfg.class_type
    if isinstance(spec, str):
        spec = string_to_callable(spec)
    if not isinstance(spec, SolverSpec):
        raise TypeError(f"{type(cfg).__name__}.class_type must select a SolverSpec, got {type(spec).__name__}.")
    return spec


def build_from_cfg(solver_cls: type[SolverBase]) -> Callable[[Model, Any, wp.DeterministicMode], SolverBase]:
    """Return a :attr:`SolverSpec.build` that constructs ``solver_cls`` from the matching configuration fields."""

    def build(model: Model, cfg: Any, deterministic_mode: wp.DeterministicMode) -> SolverBase:
        return solver_cls(model, **solver_kwargs(solver_cls, cfg, deterministic_mode))

    return build


def solver_kwargs(solver_cls: type, solver_cfg: Any, deterministic_mode: wp.DeterministicMode) -> dict:
    """Return the configuration fields matching ``solver_cls``'s constructor, plus ``deterministic`` if accepted."""
    valid = set(inspect.signature(solver_cls.__init__).parameters) - {"self", "model"}
    kwargs = {key: value for key, value in to_dict(solver_cfg).items() if key in valid}
    if isinstance(solver_cfg, VBDSolverCfg) and solver_cfg.rigid_compliant_alm is None:
        kwargs.pop("rigid_compliant_alm", None)
    if "deterministic" in valid:
        kwargs["deterministic"] = deterministic_mode
    return kwargs


def register_builder_attributes(builder: ModelBuilder, cfg: NewtonSolverCfg | None) -> None:
    """Register the custom attributes the configured solver reads on a builder that receives imported assets."""
    for solver_cls in dict.fromkeys(solver_spec(cfg).attribute_solvers(cfg)):
        solver_cls.register_custom_attributes(builder)


def usd_schema_resolvers(cfg: NewtonSolverCfg | None) -> list[SchemaResolver]:
    """Return ordered schema resolvers for physics-model USD imports.

    MJC is enabled when the configured solver registers ``SolverMuJoCo`` attributes. Visualization and
    articulation-ordering builders keep their fixed pair because solver attributes do not affect their outputs.
    """
    resolvers: list[SchemaResolver] = [SchemaResolverNewton(), SchemaResolverPhysx()]
    if SolverMuJoCo in solver_spec(cfg).attribute_solvers(cfg):
        resolvers.append(SchemaResolverMjc())
    return resolvers


def create_builder(physics_cfg: NewtonCfg | None = None, up_axis: str | None = None, **kwargs) -> ModelBuilder:
    """Create a :class:`ModelBuilder` configured with Isaac Lab's defaults.

    Forwards :class:`NewtonShapeCfg` defaults onto Newton's upstream ``ModelBuilder.default_shape_cfg`` via
    :func:`~isaaclab.utils.checked_apply`, and registers the custom attributes the configured solver reads. Without a
    Newton configuration the default shape settings still apply, so rough-terrain margin and gap hold during early
    construction.

    Args:
        physics_cfg: Newton physics configuration; ``None`` uses the default shape settings and no solver attributes.
        up_axis: Override for the up-axis. Defaults to ``"Z"``.
        **kwargs: Forwarded to :class:`ModelBuilder`.

    Returns:
        New builder with up-axis and per-shape defaults (gap, margin) applied.
    """
    newton_cfg = physics_cfg if isinstance(physics_cfg, NewtonCfg) else None
    builder = ModelBuilder(up_axis=up_axis or "Z", **kwargs)
    builder.default_bvh_cfg = ModelBuilder.BvhConfig(
        mesh_constructor=newton_cfg.bvh_constructor_geometry if newton_cfg else None,
        gaussian_constructor=newton_cfg.bvh_constructor_gaussian if newton_cfg else None,
        shape_constructor=newton_cfg.bvh_constructor_scene if newton_cfg else None,
        shape_flags=ShapeFlags.VISIBLE,
    )
    register_builder_attributes(builder, newton_cfg.solver_cfg if newton_cfg else None)
    checked_apply(newton_cfg.default_shape_cfg if newton_cfg else NewtonShapeCfg(), builder.default_shape_cfg)
    return builder
