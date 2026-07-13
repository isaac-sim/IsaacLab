# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution of symbolic articulation ordering conventions into concrete names.

A public joint or body axis can request a symbolic convention
(:class:`~isaaclab.assets.ArticulationOrderingConvention`) instead of an
explicit name permutation. Resolution walks the following chain and stops at the
first source that yields a complete backend-name permutation:

#. Same-backend native fast path. When the active backend already exposes the
   requested convention -- declared through
   :attr:`~isaaclab.assets.BaseArticulation.__backend_native_orderings__` --
   backend names are returned without any discovery.
#. Per-articulation convention cache. A convention resolved earlier for the same
   articulation is reused (keyed by convention and element kind).
#. Authored robot-schema USD relationships. The ``robot_schema`` convention
   reads the ``isaac:physics:robotJoints``/``isaac:physics:robotLinks``
   relationships on the source asset, expanding nested robot targets.
#. Temporary Newton USD ``ModelBuilder`` emulation. The ``physx`` and ``mjwarp``
   conventions build a throwaway Newton view from the source USD, differing only
   in joint traversal: breadth-first reproduces PhysX order and depth-first
   reproduces MJWarp order. The depth-first emulation is coupled to
   isaaclab_newton's ``NewtonManager``, which calls ``ModelBuilder.add_usd`` with
   Newton's default ordering; if that call ever passes explicit ordering
   arguments, the emulation constants here must be updated in lockstep or MJWarp
   resolution silently diverges from the live backend.

:mod:`isaaclab.assets.articulation.ordering` owns the public name-map type and
its validation; this module owns discovery.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from .ordering import (
    ArticulationOrderingConvention,
    _validate_articulation_names,
    parse_articulation_ordering_convention,
)

if TYPE_CHECKING:
    from pxr import Sdf, Usd

    from .base_articulation import BaseArticulation

logger = logging.getLogger(__name__)


def _backend_matches_ordering_convention(
    articulation: BaseArticulation | None,
    convention: ArticulationOrderingConvention,
) -> bool:
    """Return whether an articulation's backend natively exposes a convention's order.

    A backend declares the conventions its native solver-view order already
    satisfies through
    :attr:`~isaaclab.assets.BaseArticulation.__backend_native_orderings__`, so
    a fourth backend self-declares instead of being added to a hardcoded name
    set here. A convention listed there takes the identity fast path: requesting
    it returns backend names without cross-backend discovery. Returns ``False``
    when no articulation is available to consult.
    """
    if articulation is None:
        return False
    return convention.value in getattr(articulation, "__backend_native_orderings__", ())


def _validate_ordering_kind(kind: object) -> Literal["joint", "body"]:
    """Return a supported articulation element kind."""
    if kind == "joint" or kind == "body":
        return kind
    raise ValueError(f"kind must be 'joint' or 'body'; got {kind!r}.")


def _get_backend_names(articulation: BaseArticulation, kind: Literal["joint", "body"]) -> tuple[str, ...]:
    """Return active backend names from an articulation."""
    attr_name = "backend_joint_names" if kind == "joint" else "backend_body_names"
    return _validate_articulation_names(getattr(articulation, attr_name), parameter_name=f"Articulation {attr_name}")


def _get_cached_convention_names(
    articulation: BaseArticulation,
    convention: ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...] | None:
    """Return cached convention names for an articulation, if present.

    The cache is written only by :func:`_cache_convention_names` with already
    validated tuples, so entries are returned without re-validation.
    """
    return articulation._ordering_convention_name_cache.get((convention, kind))


def _cache_convention_names(
    articulation: BaseArticulation,
    convention: ArticulationOrderingConvention,
    names_by_kind: dict[Literal["joint", "body"], tuple[str, ...]],
) -> None:
    """Cache convention names on the articulation instance."""
    cache = articulation._ordering_convention_name_cache
    for kind, names in names_by_kind.items():
        cache[(convention, kind)] = tuple(names)


def _get_prim_path_string(prim: Usd.Prim) -> str:
    """Return a USD prim's path string."""
    return prim.GetPath().pathString


_ROBOT_SCHEMA_RELATIONSHIP_NAMES: dict[Literal["joint", "body"], str] = {
    "joint": "isaac:physics:robotJoints",
    "body": "isaac:physics:robotLinks",
}

_ROBOT_SCHEMA_NAME_OVERRIDE_ATTRS: dict[Literal["joint", "body"], tuple[str, ...]] = {
    "joint": ("isaac:NameOverride", "isaac:nameOverride"),
    "body": ("isaac:nameOverride", "isaac:NameOverride"),
}


def _get_stage_prim_at_path(stage: Usd.Stage, path: Sdf.Path | str) -> Usd.Prim | None:
    """Return the stage prim at a path, or ``None`` when the path resolves to no prim.

    ``Usd.Stage.GetPrimAtPath`` returns an invalid prim (it never raises) for a path that is not
    present on the stage, so an unresolvable relationship target surfaces here as ``None``.
    """
    prim = stage.GetPrimAtPath(path)
    return prim if prim.IsValid() else None


def _get_prim_authored_string(prim: Usd.Prim, attr_names: Sequence[str]) -> str | None:
    """Return the first non-empty authored string among candidate attributes.

    ``Usd.Prim.GetAttribute`` returns an invalid attribute (it never raises) for an unauthored
    attribute, and ``Usd.Attribute.Get`` returns ``None`` for it.
    """
    for attr_name in attr_names:
        value = prim.GetAttribute(attr_name).Get()
        if value is None or value == "":
            continue
        return value if isinstance(value, str) else str(value)
    return None


def _get_prim_name(prim: Usd.Prim) -> str:
    """Return a prim's name."""
    return prim.GetName()


def _get_robot_schema_target_name(prim: Usd.Prim, kind: Literal["joint", "body"]) -> str:
    """Return the articulation name represented by a robot schema target prim."""
    name_override = _get_prim_authored_string(prim, _ROBOT_SCHEMA_NAME_OVERRIDE_ATTRS[kind])
    if name_override is not None:
        return name_override
    return _get_prim_name(prim)


def _get_relationship_targets(prim: Usd.Prim, relationship_name: str) -> tuple[Sdf.Path, ...]:
    """Return relationship target paths, empty when the relationship is unauthored.

    ``Usd.Prim.GetRelationship`` returns an invalid relationship (it never raises) for an unauthored
    relationship, and its ``GetTargets`` returns an empty list.
    """
    return tuple(prim.GetRelationship(relationship_name).GetTargets())


def _collect_robot_schema_relationship_names(
    robot_prim: Usd.Prim,
    kind: Literal["joint", "body"],
    visited_paths: set[str],
    unresolved_targets: list[str] | None = None,
) -> tuple[str, ...]:
    """Collect names from robot schema relationships, expanding nested robot targets.

    An authored relationship target that cannot be resolved on the stage (for
    example a typo'd path) is logged and skipped rather than silently dropped,
    since an unresolvable target otherwise degrades to a generic incomplete-
    ordering failure with no indication of which target was at fault. A target
    that resolves to a prim path already seen in :paramref:`visited_paths`
    raises instead of being silently deduplicated, since an authored ordering
    should never reference the same element twice.

    Args:
        robot_prim: Prim-like object whose robot schema relationship is read.
        kind: Element kind, either joint or body.
        visited_paths: Resolved target prim paths seen so far in this
            expansion, used to detect duplicate targets.
        unresolved_targets: Optional list collecting unresolvable target path
            strings, for diagnostics when the caller reports a failure.

    Returns:
        Names collected from the relationship, in authored order.

    Raises:
        ValueError: If a relationship target resolves to a prim path already
            present in :paramref:`visited_paths`.
    """
    relationship_name = _ROBOT_SCHEMA_RELATIONSHIP_NAMES[kind]
    target_paths = _get_relationship_targets(robot_prim, relationship_name)
    if not target_paths:
        return ()

    stage = robot_prim.GetStage()
    robot_prim_path = _get_prim_path_string(robot_prim)
    names: list[str] = []
    for target_path in target_paths:
        target_prim = _get_stage_prim_at_path(stage, target_path)
        if target_prim is None:
            target_path_string = str(target_path)
            logger.warning(
                "Ignoring unresolvable '%s' target '%s' authored on '%s'.",
                relationship_name,
                target_path_string,
                robot_prim_path,
            )
            if unresolved_targets is not None:
                unresolved_targets.append(target_path_string)
            continue
        target_prim_path = _get_prim_path_string(target_prim)
        if target_prim_path in visited_paths:
            raise ValueError(
                f"Duplicate '{relationship_name}' target '{target_prim_path}' authored on '{robot_prim_path}'; "
                "each articulation-schema target must appear exactly once."
            )
        visited_paths.add(target_prim_path)
        if _get_relationship_targets(target_prim, relationship_name):
            names.extend(
                _collect_robot_schema_relationship_names(
                    target_prim, kind, visited_paths, unresolved_targets=unresolved_targets
                )
            )
        else:
            names.append(_get_robot_schema_target_name(target_prim, kind))
    return tuple(names)


def _filter_complete_backend_name_order(
    names: Sequence[str],
    backend_names: Sequence[str],
) -> tuple[str, ...] | None:
    """Return names when they form a complete backend-name order, ignoring extras."""
    backend_name_set = set(backend_names)
    filtered_names: list[str] = []
    seen_names: set[str] = set()
    for name in names:
        if name not in backend_name_set:
            continue
        if name in seen_names:
            return None
        filtered_names.append(name)
        seen_names.add(name)
    if seen_names != backend_name_set:
        return None
    return tuple(filtered_names)


def _canonical_joint_dof_name(name: str) -> str:
    """Return a backend-agnostic spelling for per-DoF joint names."""
    return name.replace(":", "_")


def _match_backend_joint_name_spellings(
    names: Sequence[str],
    backend_names: Sequence[str],
) -> tuple[str, ...]:
    """Return convention names rewritten with active-backend joint-name spellings."""
    names = tuple(names)
    backend_names = tuple(backend_names)
    if set(names) == set(backend_names):
        return names

    backend_name_by_canonical: dict[str, str] = {}
    for backend_name in backend_names:
        canonical_name = _canonical_joint_dof_name(backend_name)
        if canonical_name in backend_name_by_canonical:
            return names
        backend_name_by_canonical[canonical_name] = backend_name

    matched_names: list[str] = []
    seen_backend_names: set[str] = set()
    for name in names:
        backend_name = backend_name_by_canonical.get(_canonical_joint_dof_name(name))
        if backend_name is None or backend_name in seen_backend_names:
            return names
        matched_names.append(backend_name)
        seen_backend_names.add(backend_name)

    if seen_backend_names != set(backend_names):
        return names
    return tuple(matched_names)


def _match_backend_name_spellings(
    *,
    kind: Literal["joint", "body"],
    names: Sequence[str],
    backend_names: Sequence[str],
) -> tuple[str, ...]:
    """Return convention names rewritten with active-backend spellings when needed."""
    if kind == "joint":
        return _match_backend_joint_name_spellings(names, backend_names)
    return tuple(names)


def _get_complete_convention_names(
    *,
    kind: Literal["joint", "body"],
    names: tuple[str, ...] | None,
    backend_names: Sequence[str],
) -> tuple[str, ...] | None:
    """Return a convention candidate when it is a complete backend-name permutation.

    Providers contractually return validated name tuples or ``None``; a missing
    candidate short-circuits to ``None`` so resolution falls through to the next
    strategy.
    """
    if names is None:
        return None
    candidate_names = names
    candidate_names = _match_backend_name_spellings(
        kind=kind,
        names=candidate_names,
        backend_names=backend_names,
    )
    backend_names = tuple(backend_names)
    if len(candidate_names) != len(backend_names) or set(candidate_names) != set(backend_names):
        return None
    return candidate_names


def _get_complete_convention_names_by_kind(
    articulation: BaseArticulation,
    names_by_kind: dict[Literal["joint", "body"], tuple[str, ...]],
) -> dict[Literal["joint", "body"], tuple[str, ...]]:
    """Return only complete convention-name candidates from a multi-kind provider."""
    complete_names: dict[Literal["joint", "body"], tuple[str, ...]] = {}
    for candidate_kind in ("joint", "body"):
        backend_names = _get_backend_names(articulation, candidate_kind)
        names = _get_complete_convention_names(
            kind=candidate_kind,
            names=names_by_kind.get(candidate_kind),
            backend_names=backend_names,
        )
        if names is not None:
            complete_names[candidate_kind] = names
    return complete_names


def _get_source_asset_prim(articulation: BaseArticulation) -> Usd.Prim | None:
    """Return the source asset prim for an articulation config when available."""
    prim_path = articulation.cfg.prim_path
    from isaaclab.sim.utils.queries import resolve_matching_prims_from_source  # noqa: PLC0415

    source_asset_matches = resolve_matching_prims_from_source(prim_path, expected_num_matches=1)
    if not source_asset_matches:
        return None
    return source_asset_matches[0][0]


def _get_robot_schema_candidate_prims(articulation: BaseArticulation) -> tuple[Usd.Prim, ...]:
    """Return candidate prims that may author robot schema ordering relationships."""
    source_asset_prim = _get_source_asset_prim(articulation)
    if source_asset_prim is None:
        return ()

    candidate_prims = [source_asset_prim]
    articulation_root_prim_path = articulation.cfg.articulation_root_prim_path
    if articulation_root_prim_path is not None:
        root_path = _get_prim_path_string(source_asset_prim) + articulation_root_prim_path
        articulation_root_prim = _get_stage_prim_at_path(source_asset_prim.GetStage(), root_path)
        if articulation_root_prim is not None:
            candidate_prims.append(articulation_root_prim)
    return tuple(candidate_prims)


def _get_robot_schema_names(
    articulation: BaseArticulation,
    kind: Literal["joint", "body"],
) -> tuple[tuple[str, ...] | None, str]:
    """Return complete articulation names from Isaac Sim robot schema relationships.

    Args:
        articulation: Articulation whose source USD relationships are resolved.
        kind: Element kind, either joint or body.

    Returns:
        A tuple of the resolved names, or ``None`` when no candidate prim
        authored a complete backend-name permutation, and a short diagnostic
        reason describing why resolution stopped (empty when names were
        resolved).
    """
    backend_names = _get_backend_names(articulation, kind)
    relationship_name = _ROBOT_SCHEMA_RELATIONSHIP_NAMES[kind]

    candidate_prims = _get_robot_schema_candidate_prims(articulation)
    if not candidate_prims:
        return None, "robot_schema: source asset prim is unavailable"

    unresolved_targets: list[str] = []
    best_relationship_names: tuple[str, ...] = ()
    for candidate_prim in candidate_prims:
        relationship_names = _collect_robot_schema_relationship_names(
            candidate_prim, kind, set(), unresolved_targets=unresolved_targets
        )
        if len(relationship_names) > len(best_relationship_names):
            best_relationship_names = relationship_names
        names = _filter_complete_backend_name_order(relationship_names, backend_names)
        if names is not None:
            return names, ""

    if unresolved_targets:
        return None, (
            f"robot_schema: {len(unresolved_targets)} relationship target(s) unresolved: "
            f"[{', '.join(unresolved_targets)}]"
        )
    if not best_relationship_names:
        return None, f"robot_schema: no '{relationship_name}' relationship authored"
    missing = sorted(set(backend_names) - set(best_relationship_names))
    extra = sorted(set(best_relationship_names) - set(backend_names))
    return None, f"robot_schema: incomplete permutation (missing={missing}, extra={extra})"


def _get_names_from_newton_usd_builder(
    articulation: BaseArticulation,
    *,
    joint_ordering: Literal["bfs", "dfs"],
    bodies_follow_joint_ordering: bool,
) -> dict[Literal["joint", "body"], tuple[str, ...]] | None:
    """Build a lightweight Newton prototype view and return its articulation names."""
    cfg = articulation.cfg
    prim_path = cfg.prim_path

    try:
        from newton import JointType, ModelBuilder, solvers  # noqa: PLC0415
        from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx  # noqa: PLC0415
        from newton.selection import ArticulationView  # noqa: PLC0415

        from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

        from isaaclab.sim.utils.queries import resolve_matching_prims_from_source  # noqa: PLC0415
        from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        missing_module = exc.name or ""
        if missing_module not in {"newton", "pxr"} and not missing_module.startswith(("newton.", "pxr.")):
            raise
        return None

    stage = get_current_stage()
    if stage is None:
        return None

    source_asset_matches = resolve_matching_prims_from_source(prim_path, expected_num_matches=1)
    if not source_asset_matches:
        return None
    source_asset_path = _get_prim_path_string(source_asset_matches[0][0])

    articulation_root_prim_path = cfg.articulation_root_prim_path
    if articulation_root_prim_path is not None:
        source_articulation_path = source_asset_path + articulation_root_prim_path
    else:

        def has_articulation_root_api(prim) -> bool:
            return bool(prim.HasAPI(UsdPhysics.ArticulationRootAPI))

        source_root_matches = resolve_matching_prims_from_source(
            prim_path,
            predicate=has_articulation_root_api,
            expected_num_matches=1,
        )
        if not source_root_matches:
            return None
        source_articulation_path = _get_prim_path_string(source_root_matches[0][0])

    builder = ModelBuilder(up_axis=UsdGeom.GetStageUpAxis(stage))
    solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(
        stage,
        root_path=source_asset_path,
        load_visual_shapes=False,
        skip_mesh_approximation=True,
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        joint_ordering=joint_ordering,
        bodies_follow_joint_ordering=bodies_follow_joint_ordering,
    )
    model = builder.finalize(device="cpu")
    view = ArticulationView(
        model,
        source_articulation_path,
        verbose=False,
        exclude_joint_types=[JointType.FREE, JointType.FIXED],
    )
    return {"joint": tuple(view.joint_dof_names), "body": tuple(view.link_names)}


def _get_physx_names_from_newton_usd_builder(
    articulation: BaseArticulation,
) -> dict[Literal["joint", "body"], tuple[str, ...]] | None:
    """Build a lightweight Newton prototype view with PhysX-style articulation names."""
    # NOTE: "bfs" assumes Newton's breadth-first USD traversal reproduces
    # PhysX's native articulation-view order. Unlike the MJWarp constants
    # below, this is not coupled to isaaclab_newton's NewtonManager: a live
    # PhysX/OVPhysX backend never goes through Newton's ModelBuilder.add_usd,
    # so there is no analogous "active backend already matches these
    # arguments" identity path to keep in sync.
    return _get_names_from_newton_usd_builder(
        articulation,
        joint_ordering="bfs",
        bodies_follow_joint_ordering=True,
    )


def _get_mjwarp_names_from_newton_usd_builder(
    articulation: BaseArticulation,
) -> dict[Literal["joint", "body"], tuple[str, ...]] | None:
    """Build a lightweight Newton prototype view with MJWarp-style articulation names."""
    # NOTE: "dfs" and bodies_follow_joint_ordering=True mirror the defaults of
    # Newton's ModelBuilder.add_usd. isaaclab_newton's NewtonManager calls
    # add_usd (see instantiate_builder_from_stage) without passing
    # joint_ordering/bodies_follow_joint_ordering, so a live Newton backend's
    # native order matches this emulation only because both sides currently
    # rely on the same Newton library defaults. The "mjwarp" convention's
    # same-backend identity path (active backend "newton") returns that live
    # order directly, assuming it equals what these hardcoded constants would
    # produce. If NewtonManager ever passes explicit ordering arguments to
    # add_usd, these constants must be updated in lockstep or MJWarp
    # resolution will silently diverge from the live backend.
    return _get_names_from_newton_usd_builder(
        articulation,
        joint_ordering="dfs",
        bodies_follow_joint_ordering=True,
    )


def _describe_incomplete_convention_names(
    kind: Literal["joint", "body"],
    names: Sequence[str] | None,
    backend_names: Sequence[str],
) -> str:
    """Return a short reason a convention candidate is not a complete backend-name permutation."""
    if names is None:
        return f"no {kind} names were discovered"
    missing = sorted(set(backend_names) - set(names))
    extra = sorted(set(names) - set(backend_names))
    return f"{kind} names are not a complete permutation (missing={missing}, extra={extra})"


def _describe_newton_usd_builder_unavailability(articulation: BaseArticulation) -> str:
    """Return a short reason the temporary Newton USD builder produced no articulation names.

    Only re-derives the cheap early-exit checks (module availability, stage
    availability, source-prim resolution); it never re-runs the Newton model
    build. Never raises: an articulation missing the USD-backed contract
    properties (for example in unit tests) falls back to a generic reason
    instead of masking the caller's real failure.
    """
    cfg = getattr(articulation, "cfg", None)
    prim_path = getattr(cfg, "prim_path", None)
    if prim_path is None:
        return "the Newton USD builder returned no articulation names"

    try:
        import newton  # noqa: F401, PLC0415

        from pxr import UsdPhysics  # noqa: PLC0415

        from isaaclab.sim.utils.queries import resolve_matching_prims_from_source  # noqa: PLC0415
        from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        return f"'{exc.name or 'unknown'}' module is not installed"

    if get_current_stage() is None:
        return "no current USD stage is available"

    if not resolve_matching_prims_from_source(prim_path, expected_num_matches=1):
        return f"source asset prim matching '{prim_path}' was not found"

    articulation_root_prim_path = getattr(cfg, "articulation_root_prim_path", None)
    if articulation_root_prim_path is None:

        def has_articulation_root_api(prim) -> bool:
            return bool(prim.HasAPI(UsdPhysics.ArticulationRootAPI))

        if not resolve_matching_prims_from_source(
            prim_path, predicate=has_articulation_root_api, expected_num_matches=1
        ):
            return "no prim with ArticulationRootAPI was found under the source asset"

    return "the Newton USD builder returned no articulation names"


def _resolve_articulation_convention_name_ordering(
    *,
    articulation: BaseArticulation,
    convention: str | ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Resolve a symbolic convention to names for the public articulation axis.

    A convention matching the active backend returns backend names without
    discovery. Cross-backend resolution uses a validated per-articulation cache,
    authored robot-schema relationships for robot_schema, or a temporary Newton
    USD view. PhysX discovery uses breadth-first joint ordering and MJWarp
    discovery uses depth-first ordering. Builder results are cached only when
    both joint and body names are complete permutations.

    Args:
        articulation: Articulation whose configured source asset is resolved.
        convention: Convention alias or ArticulationOrderingConvention member.
        kind: Element kind, either joint or body.

    Returns:
        Names to expose on the requested public joint or body axis.

    Raises:
        AttributeError: If required articulation contract properties are absent.
        TypeError: If convention or discovered names are malformed.
        ValueError: If kind or convention is invalid, a provider rejects source
            metadata, or an authored robot-schema relationship targets the
            same prim more than once.
        NotImplementedError: If no supported source provides a complete ordering.
            The message identifies the corresponding configuration field, the
            explicit-name fallback, and a short reason the attempted
            resolution strategy did not apply.
    """
    kind = _validate_ordering_kind(kind)
    parsed_convention = parse_articulation_ordering_convention(convention)
    if parsed_convention is None:
        return _get_backend_names(articulation, kind)

    active_backend_name = articulation.__backend_name__
    if _backend_matches_ordering_convention(articulation, parsed_convention):
        return _get_backend_names(articulation, kind)

    backend_names = _get_backend_names(articulation, kind)
    resolution_failures: list[str] = []

    cached_names = _get_cached_convention_names(articulation, parsed_convention, kind)
    cached_names = _get_complete_convention_names(
        kind=kind,
        names=cached_names,
        backend_names=backend_names,
    )
    if cached_names is not None:
        return cached_names

    if parsed_convention is ArticulationOrderingConvention.ROBOT_SCHEMA:
        raw_robot_schema_names, robot_schema_failure_reason = _get_robot_schema_names(articulation, kind)
        robot_schema_names = _get_complete_convention_names(
            kind=kind,
            names=raw_robot_schema_names,
            backend_names=backend_names,
        )
        if robot_schema_names is not None:
            _cache_convention_names(articulation, parsed_convention, {kind: robot_schema_names})
            return robot_schema_names
        resolution_failures.append(robot_schema_failure_reason)

    # PhysX and MJWarp both resolve through a temporary Newton USD build; they
    # differ only in the provider function (breadth- vs depth-first joint
    # ordering) and the failure-reason label.
    newton_usd_builder_providers = {
        ArticulationOrderingConvention.PHYSX: ("physx_usd_builder", _get_physx_names_from_newton_usd_builder),
        ArticulationOrderingConvention.MJWARP: ("mjwarp_usd_builder", _get_mjwarp_names_from_newton_usd_builder),
    }
    builder_provider = newton_usd_builder_providers.get(parsed_convention)
    if builder_provider is not None:
        provider_label, provider = builder_provider
        builder_names = provider(articulation)
        if builder_names is not None:
            complete_names = _get_complete_convention_names_by_kind(articulation, builder_names)
            if len(complete_names) == 2:
                _cache_convention_names(articulation, parsed_convention, complete_names)
            if kind in complete_names:
                return complete_names[kind]
            reason = _describe_incomplete_convention_names(kind, builder_names.get(kind), backend_names)
            resolution_failures.append(f"{provider_label}: {reason}")
        else:
            reason = _describe_newton_usd_builder_unavailability(articulation)
            resolution_failures.append(f"{provider_label}: {reason}")

    config_field = "joint_ordering" if kind == "joint" else "body_ordering"
    attempted_resolutions = f" Attempted resolutions: {'; '.join(resolution_failures)}." if resolution_failures else ""
    raise NotImplementedError(
        f"Unable to resolve '{parsed_convention.value}' {kind} ordering for active backend "
        f"'{active_backend_name}'. Ensure the source USD and required ordering dependencies are available, "
        f"set ArticulationCfg.{config_field} to an explicit {kind}-name permutation on the articulation's "
        f"configuration, or use None to keep active-backend order.{attempted_resolutions}"
    )


def get_articulation_name_ordering(
    articulation: BaseArticulation,
    convention: str | ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return articulation names in the order defined by a naming convention.

    The supported conventions are:

    * ``"physx"`` -- PhysX or OVPhysX articulation-view order. PhysX and OVPhysX
      articulations return active-backend names without discovery; other backends
      discover the order from a temporary Newton USD view using breadth-first
      joint ordering.
    * ``"mjwarp"`` -- Newton or MJWarp articulation-view order. Newton
      articulations return active-backend names without discovery; other backends
      discover the order from a temporary Newton USD view using depth-first joint
      ordering.
    * ``"robot_schema"`` -- authored robot-schema order. The source asset prim or
      configured articulation-root prim must author ``isaac:physics:robotJoints``
      for joints or ``isaac:physics:robotLinks`` for bodies. Nested robot targets
      are expanded, name overrides are honored, unresolvable targets are logged
      and skipped, and the remaining names must be a complete unique permutation
      of active-backend names.

    Cross-backend discovery through the temporary Newton USD view requires a
    source USD readable by the optional Newton and PXR dependencies, and a
    complete joint-and-body result is cached per articulation.

    The result defines the public axis only; backend views remain in native order.

    Args:
        articulation: Articulation whose names are resolved.
        convention: Convention alias (``"physx"``, ``"mjwarp"``, or
            ``"robot_schema"``, matched case-insensitively) or
            :class:`~isaaclab.assets.ArticulationOrderingConvention` member.
        kind: Element kind, either joint or body.

    Returns:
        Names in the requested convention's order.

    Raises:
        TypeError: If backend or discovered names are malformed.
        ValueError: If kind or convention is invalid, the builder or USD
            resolution rejects the source metadata, or an authored robot-schema
            relationship targets the same prim more than once.
        NotImplementedError: If the source USD, builder dependencies, authored
            relationships, or a complete name permutation is unavailable. The
            message identifies the corresponding configuration field, the
            explicit-name fallback, and a short reason resolution did not produce
            a complete ordering.
    """
    return _resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=convention,
        kind=kind,
    )


def _resolve_articulation_ordering_names(
    *,
    kind: Literal["joint", "body"],
    backend_names: tuple[str, ...],
    ordering: list[str] | tuple[str, ...] | str | ArticulationOrderingConvention | None,
    active_backend_name: str,
    articulation: BaseArticulation | None = None,
) -> tuple[str, ...]:
    """Resolve configured public articulation ordering to concrete names.

    ``None`` and conventions matching :paramref:`active_backend_name` take an
    identity fast path and return :paramref:`backend_names`. Explicit sequences
    are type-checked here; complete-permutation validation is performed later by
    :func:`build_articulation_name_map`.

    Cross-backend conventions delegate to
    :func:`_resolve_articulation_convention_name_ordering` and reuse its
    per-articulation discovery cache. Joint names are normalized to active-backend
    spelling when Newton multi-DoF separators differ.

    The returned tuple defines public order. :paramref:`backend_names` and
    solver-view arrays remain in backend order. Supported discovery failures may
    end in :class:`NotImplementedError`; other provider or builder exceptions
    propagate.

    Args:
        kind: Element kind, either ``"joint"`` or ``"body"``.
        backend_names: Names in active backend solver-view order.
        ordering: Explicit public name sequence, symbolic convention alias or
            enum member, or ``None``.
        active_backend_name: Name of the backend exposing
            :paramref:`backend_names`.
        articulation: Articulation used for cached cross-backend discovery when
            a symbolic convention differs from the active backend.

    Returns:
        Concrete names for the public joint or body axis.

    Raises:
        AttributeError: If a provider or builder raises this error.
        TypeError: If :paramref:`ordering` has an unsupported type, an explicit
            sequence contains a non-string, or a provider raises an unhandled type error.
        ValueError: If :paramref:`kind` is invalid, :paramref:`ordering` is an
            unsupported alias, or a provider or builder rejects the source metadata.
        NotImplementedError: If cross-backend ordering lacks an articulation, or
            all supported convention metadata is absent or incomplete.
    """
    kind = _validate_ordering_kind(kind)
    backend_names = _validate_articulation_names(backend_names, parameter_name="backend_names")
    if ordering is None:
        return backend_names
    if isinstance(ordering, ArticulationOrderingConvention):
        convention = ordering
    elif isinstance(ordering, str):
        convention = parse_articulation_ordering_convention(ordering)
    elif type(ordering) in (list, tuple):
        return _validate_articulation_names(ordering, parameter_name=f"{kind}_ordering")
    else:
        raise TypeError(
            f"{kind}_ordering must be a name list or tuple, convention string/enum, or None;"
            f" got {type(ordering).__name__}."
        )

    if convention is None or _backend_matches_ordering_convention(articulation, convention):
        return backend_names
    if articulation is not None:
        convention_names = _resolve_articulation_convention_name_ordering(
            articulation=articulation,
            convention=convention,
            kind=kind,
        )
        return _match_backend_name_spellings(kind=kind, names=convention_names, backend_names=backend_names)

    config_field = "joint_ordering" if kind == "joint" else "body_ordering"
    raise NotImplementedError(
        f"Unable to resolve '{convention.value}' {kind} ordering for active backend '{active_backend_name}'. "
        f"Set ArticulationCfg.{config_field} to an explicit {kind}-name permutation on the articulation's "
        "configuration, or supply an articulation whose source USD can provide that convention."
    )
