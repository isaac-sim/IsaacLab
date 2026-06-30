# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from .ordering import ArticulationOrderingConvention, _coerce_articulation_names, parse_articulation_ordering_convention

if TYPE_CHECKING:
    from .base_articulation import BaseArticulation


def _backend_matches_ordering_convention(
    active_backend_name: str,
    convention: ArticulationOrderingConvention,
) -> bool:
    """Return whether a backend already exposes names in a convention's order."""
    backend_name = active_backend_name.lower()
    if convention is ArticulationOrderingConvention.PHYSX:
        return backend_name in {"physx", "ovphysx"}
    if convention is ArticulationOrderingConvention.MJWARP:
        return backend_name in {"newton", "mjwarp", "newton_mjwarp"}
    return False


def _coerce_name_sequence(names: object) -> tuple[str, ...] | None:
    """Return names as a tuple when they look like an articulation name sequence."""
    if names is None or isinstance(names, str | bytes | bytearray):
        return None
    try:
        name_tuple = tuple(names)
    except TypeError:
        return None
    if not all(isinstance(name, str) for name in name_tuple):
        return None
    return name_tuple


def _get_attr_or_none(obj: object, name: str) -> object | None:
    """Read an optional attribute without requiring every backend to expose it."""
    try:
        return getattr(obj, name)
    except AttributeError:
        try:
            inspect.getattr_static(obj, name)
        except AttributeError:
            return None
        raise


def _get_backend_names(articulation: object, kind: Literal["joint", "body"]) -> tuple[str, ...]:
    """Return active backend names from an articulation."""
    attr_name = "backend_joint_names" if kind == "joint" else "backend_body_names"
    names = _coerce_name_sequence(_get_attr_or_none(articulation, attr_name))
    if names is None:
        raise AttributeError(f"Articulation does not expose {attr_name}.")
    return names


def _get_precomputed_convention_names(
    articulation: object,
    convention: ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...] | None:
    """Return cached convention names supplied by a backend, if present."""
    prefix = convention.value
    candidate_attrs = (
        f"_{prefix}_{kind}_names",
        f"_{prefix}_{kind}_ordering_names",
        f"{prefix}_{kind}_names",
        f"{prefix}_{kind}_ordering_names",
    )
    for attr_name in candidate_attrs:
        names = _coerce_name_sequence(_get_attr_or_none(articulation, attr_name))
        if names is not None:
            return names
    return None


def _get_cached_convention_names(
    articulation: object,
    convention: ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...] | None:
    """Return cached convention names for an articulation, if present."""
    cache = _get_attr_or_none(articulation, "_ordering_convention_name_cache")
    if isinstance(cache, dict):
        names = cache.get((convention, kind))
        return _coerce_name_sequence(names)
    return None


def _cache_convention_names(
    articulation: object,
    convention: ArticulationOrderingConvention,
    names_by_kind: dict[Literal["joint", "body"], tuple[str, ...]],
) -> None:
    """Cache convention names on mutable articulation instances."""
    cache = _get_attr_or_none(articulation, "_ordering_convention_name_cache")
    if not isinstance(cache, dict):
        cache = {}
    for kind, names in names_by_kind.items():
        cache[(convention, kind)] = tuple(names)
    try:
        setattr(articulation, "_ordering_convention_name_cache", cache)
    except AttributeError:
        return


def _get_prim_path_string(prim: object) -> str:
    """Return a USD prim path string from a prim-like object."""
    path = prim.GetPath()
    return str(getattr(path, "pathString", path))


_ROBOT_SCHEMA_RELATIONSHIP_NAMES: dict[Literal["joint", "body"], str] = {
    "joint": "isaac:physics:robotJoints",
    "body": "isaac:physics:robotLinks",
}

_ROBOT_SCHEMA_NAME_OVERRIDE_ATTRS: dict[Literal["joint", "body"], tuple[str, ...]] = {
    "joint": ("isaac:NameOverride", "isaac:nameOverride"),
    "body": ("isaac:nameOverride", "isaac:NameOverride"),
}


def _is_valid_prim(prim: object | None) -> bool:
    """Return whether a prim-like object is present and valid."""
    if prim is None:
        return False
    is_valid = _get_attr_or_none(prim, "IsValid")
    if callable(is_valid):
        return bool(is_valid())
    return True


def _get_stage_prim_at_path(stage: object | None, path: object) -> object | None:
    """Return a stage prim at a path when the stage can resolve it."""
    if stage is None:
        return None
    get_prim_at_path = _get_attr_or_none(stage, "GetPrimAtPath")
    if not callable(get_prim_at_path):
        return None
    try:
        prim = get_prim_at_path(path)
    except (KeyError, TypeError):
        return None
    if not _is_valid_prim(prim):
        return None
    return prim


def _get_prim_authored_string(prim: object, attr_names: Sequence[str]) -> str | None:
    """Return the first non-empty authored string among candidate attributes."""
    get_attribute = _get_attr_or_none(prim, "GetAttribute")
    if not callable(get_attribute):
        return None
    for attr_name in attr_names:
        attr = get_attribute(attr_name)
        if attr is None:
            continue
        get_value = _get_attr_or_none(attr, "Get")
        if not callable(get_value):
            continue
        value = get_value()
        if value is None or value == "":
            continue
        if isinstance(value, str):
            return value
        return str(value)
    return None


def _get_prim_name(prim: object) -> str:
    """Return a prim-like object's name."""
    get_name = _get_attr_or_none(prim, "GetName")
    if callable(get_name):
        name = get_name()
        if isinstance(name, str) and name:
            return name
    return _get_prim_path_string(prim).rsplit("/", maxsplit=1)[-1]


def _get_robot_schema_target_name(prim: object, kind: Literal["joint", "body"]) -> str:
    """Return the articulation name represented by a robot schema target prim."""
    name_override = _get_prim_authored_string(prim, _ROBOT_SCHEMA_NAME_OVERRIDE_ATTRS[kind])
    if name_override is not None:
        return name_override
    return _get_prim_name(prim)


def _get_relationship_targets(prim: object, relationship_name: str) -> tuple[object, ...]:
    """Return relationship targets from a prim-like object."""
    get_relationship = _get_attr_or_none(prim, "GetRelationship")
    if not callable(get_relationship):
        return ()
    relationship = get_relationship(relationship_name)
    if relationship is None:
        return ()
    get_targets = _get_attr_or_none(relationship, "GetTargets")
    if not callable(get_targets):
        return ()
    targets = get_targets()
    if targets is None:
        return ()
    return tuple(targets)


def _collect_robot_schema_relationship_names(
    robot_prim: object,
    kind: Literal["joint", "body"],
    visited_paths: set[str],
) -> tuple[str, ...]:
    """Collect names from robot schema relationships, expanding nested robot targets."""
    relationship_name = _ROBOT_SCHEMA_RELATIONSHIP_NAMES[kind]
    target_paths = _get_relationship_targets(robot_prim, relationship_name)
    if not target_paths:
        return ()

    stage = None
    get_stage = _get_attr_or_none(robot_prim, "GetStage")
    if callable(get_stage):
        stage = get_stage()

    names: list[str] = []
    for target_path in target_paths:
        target_prim = _get_stage_prim_at_path(stage, target_path)
        if target_prim is None:
            continue
        target_prim_path = _get_prim_path_string(target_prim)
        if target_prim_path in visited_paths:
            continue
        visited_paths.add(target_prim_path)
        if _get_relationship_targets(target_prim, relationship_name):
            names.extend(_collect_robot_schema_relationship_names(target_prim, kind, visited_paths))
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
    names: object,
    backend_names: Sequence[str],
) -> tuple[str, ...] | None:
    """Return a convention candidate when it is a complete backend-name permutation."""
    candidate_names = _coerce_name_sequence(names)
    if candidate_names is None:
        return None
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
    articulation: object,
    names_by_kind: dict[Literal["joint", "body"], tuple[str, ...]],
) -> dict[Literal["joint", "body"], tuple[str, ...]]:
    """Return only complete convention-name candidates from a multi-kind provider."""
    complete_names: dict[Literal["joint", "body"], tuple[str, ...]] = {}
    for candidate_kind in ("joint", "body"):
        attr_name = "backend_joint_names" if candidate_kind == "joint" else "backend_body_names"
        backend_names = _coerce_name_sequence(_get_attr_or_none(articulation, attr_name))
        if backend_names is None:
            continue
        names = _get_complete_convention_names(
            kind=candidate_kind,
            names=names_by_kind.get(candidate_kind),
            backend_names=backend_names,
        )
        if names is not None:
            complete_names[candidate_kind] = names
    return complete_names


def _get_source_asset_prim(articulation: object) -> object | None:
    """Return the source asset prim for an articulation config when available."""
    cfg = _get_attr_or_none(articulation, "cfg")
    prim_path = _get_attr_or_none(cfg, "prim_path")
    if prim_path is None:
        return None
    try:
        from isaaclab.sim.utils.queries import resolve_matching_prims_from_source  # noqa: PLC0415
    except ImportError:
        return None

    source_asset_matches = resolve_matching_prims_from_source(prim_path, expected_num_matches=1)
    if not source_asset_matches:
        return None
    return source_asset_matches[0][0]


def _get_robot_schema_candidate_prims(articulation: object) -> tuple[object, ...]:
    """Return candidate prims that may author robot schema ordering relationships."""
    source_asset_prim = _get_source_asset_prim(articulation)
    if source_asset_prim is None:
        return ()

    candidate_prims = [source_asset_prim]
    cfg = _get_attr_or_none(articulation, "cfg")
    articulation_root_prim_path = _get_attr_or_none(cfg, "articulation_root_prim_path")
    if articulation_root_prim_path is not None:
        get_stage = _get_attr_or_none(source_asset_prim, "GetStage")
        stage = get_stage() if callable(get_stage) else None
        root_path = _get_prim_path_string(source_asset_prim) + articulation_root_prim_path
        articulation_root_prim = _get_stage_prim_at_path(stage, root_path)
        if articulation_root_prim is not None:
            candidate_prims.append(articulation_root_prim)
    return tuple(candidate_prims)


def _get_robot_schema_names(
    articulation: object,
    kind: Literal["joint", "body"],
) -> tuple[str, ...] | None:
    """Return complete articulation names from Isaac Sim robot schema relationships."""
    try:
        backend_names = _get_backend_names(articulation, kind)
    except AttributeError:
        return None

    for candidate_prim in _get_robot_schema_candidate_prims(articulation):
        relationship_names = _collect_robot_schema_relationship_names(candidate_prim, kind, set())
        names = _filter_complete_backend_name_order(relationship_names, backend_names)
        if names is not None:
            return names
    return None


def _get_names_from_newton_usd_builder(
    articulation: object,
    *,
    joint_ordering: Literal["bfs", "dfs"],
    bodies_follow_joint_ordering: bool,
) -> dict[Literal["joint", "body"], tuple[str, ...]] | None:
    """Build a lightweight Newton prototype view and return its articulation names."""
    cfg = _get_attr_or_none(articulation, "cfg")
    prim_path = _get_attr_or_none(cfg, "prim_path")
    if prim_path is None:
        return None

    try:
        from newton import JointType, ModelBuilder, solvers  # noqa: PLC0415
        from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx  # noqa: PLC0415
        from newton.selection import ArticulationView  # noqa: PLC0415

        from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

        from isaaclab.sim.utils.queries import resolve_matching_prims_from_source  # noqa: PLC0415
        from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415
    except ImportError:
        return None

    stage = get_current_stage()
    if stage is None:
        return None

    source_asset_matches = resolve_matching_prims_from_source(prim_path, expected_num_matches=1)
    if not source_asset_matches:
        return None
    source_asset_path = _get_prim_path_string(source_asset_matches[0][0])

    articulation_root_prim_path = _get_attr_or_none(cfg, "articulation_root_prim_path")
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
    articulation: object,
) -> dict[Literal["joint", "body"], tuple[str, ...]] | None:
    """Build a lightweight Newton prototype view with PhysX-style articulation names."""
    return _get_names_from_newton_usd_builder(
        articulation,
        joint_ordering="bfs",
        bodies_follow_joint_ordering=True,
    )


def _get_mjwarp_names_from_newton_usd_builder(
    articulation: object,
) -> dict[Literal["joint", "body"], tuple[str, ...]] | None:
    """Build a lightweight Newton prototype view with MJWarp-style articulation names."""
    return _get_names_from_newton_usd_builder(
        articulation,
        joint_ordering="dfs",
        bodies_follow_joint_ordering=True,
    )


def _get_root_view_convention_names(
    root_view: object,
    convention: ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...] | None:
    """Return convention names from backend-specific root-view metadata."""
    if convention is ArticulationOrderingConvention.PHYSX:
        shared_metatype = _get_attr_or_none(root_view, "shared_metatype")
        if shared_metatype is None:
            return None
        attr_name = "dof_names" if kind == "joint" else "link_names"
        return _coerce_name_sequence(_get_attr_or_none(shared_metatype, attr_name))

    if convention is ArticulationOrderingConvention.MJWARP:
        attr_name = "joint_dof_names" if kind == "joint" else "link_names"
        return _coerce_name_sequence(_get_attr_or_none(root_view, attr_name))

    return None


def resolve_articulation_convention_name_ordering(
    *,
    articulation: BaseArticulation,
    convention: str | ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Resolve a symbolic convention to names for the public articulation axis.

    A convention matching the active backend takes an identity fast path and
    returns backend names without metadata discovery. Cross-backend resolution
    checks cached and precomputed names before inspecting backend metadata.
    Names discovered by robot-schema traversal or a Newton USD build are cached
    on mutable articulation instances; precomputed and ``root_view`` metadata are
    read directly.

    PhysX and MJWarp discovery may parse the source USD with a Newton
    ``ModelBuilder`` and finalize a temporary model on CPU. Robot-schema
    discovery requires ``isaac:physics:robotJoints`` or
    ``isaac:physics:robotLinks`` targets that resolve to a complete, unique
    ordering of the active backend names.

    The returned tuple defines public order only. Any ``root_view`` metadata read
    here, and all solver-view arrays, remain in backend order. Optional probes
    treat missing attributes, values that are not name sequences, unavailable
    imports or stages, and selected lookup failures as absent metadata. Resolution
    continues through the remaining sources and may end in
    :class:`NotImplementedError`. Unsupported convention inputs and provider or
    builder exceptions not explicitly handled as absence propagate to the caller.

    Args:
        articulation: Articulation whose configured source asset is resolved.
        convention: Convention alias or
            :class:`ArticulationOrderingConvention` member.
        kind: Element kind, either ``"joint"`` or ``"body"``.

    Returns:
        Names to expose on the requested public joint or body axis.

    Raises:
        AttributeError: If same-backend names are unavailable, or a provider or
            builder raises this error outside an optional metadata probe.
        TypeError: If :paramref:`convention` has an unsupported type, or a
            provider or builder raises an unhandled type error.
        ValueError: If :paramref:`convention` is an unsupported alias, or a
            provider or builder rejects the source metadata.
        NotImplementedError: If all optional sources are absent or incomplete
            for the requested cross-backend convention.
    """
    parsed_convention = parse_articulation_ordering_convention(convention)
    if parsed_convention is None:
        return _get_backend_names(articulation, kind)

    active_backend_name = getattr(articulation, "__backend_name__", "unknown")
    if _backend_matches_ordering_convention(active_backend_name, parsed_convention):
        return _get_backend_names(articulation, kind)

    backend_names = _get_backend_names(articulation, kind)

    cached_names = _get_cached_convention_names(articulation, parsed_convention, kind)
    cached_names = _get_complete_convention_names(
        kind=kind,
        names=cached_names,
        backend_names=backend_names,
    )
    if cached_names is not None:
        return cached_names

    precomputed_names = _get_precomputed_convention_names(articulation, parsed_convention, kind)
    precomputed_names = _get_complete_convention_names(
        kind=kind,
        names=precomputed_names,
        backend_names=backend_names,
    )
    if precomputed_names is not None:
        return precomputed_names

    if parsed_convention is ArticulationOrderingConvention.ROBOT_SCHEMA:
        robot_schema_names = _get_robot_schema_names(articulation, kind)
        robot_schema_names = _get_complete_convention_names(
            kind=kind,
            names=robot_schema_names,
            backend_names=backend_names,
        )
        if robot_schema_names is not None:
            _cache_convention_names(articulation, parsed_convention, {kind: robot_schema_names})
            return robot_schema_names

    root_view = _get_attr_or_none(articulation, "root_view")
    if root_view is not None:
        root_view_names = _get_root_view_convention_names(root_view, parsed_convention, kind)
        root_view_names = _get_complete_convention_names(
            kind=kind,
            names=root_view_names,
            backend_names=backend_names,
        )
        if root_view_names is not None:
            return root_view_names

    if parsed_convention is ArticulationOrderingConvention.PHYSX:
        physx_names = _get_physx_names_from_newton_usd_builder(articulation)
        if physx_names is not None:
            complete_physx_names = _get_complete_convention_names_by_kind(articulation, physx_names)
            if complete_physx_names:
                _cache_convention_names(articulation, parsed_convention, complete_physx_names)
            if kind in complete_physx_names:
                return complete_physx_names[kind]

    if parsed_convention is ArticulationOrderingConvention.MJWARP:
        mjwarp_names = _get_mjwarp_names_from_newton_usd_builder(articulation)
        if mjwarp_names is not None:
            complete_mjwarp_names = _get_complete_convention_names_by_kind(articulation, mjwarp_names)
            if complete_mjwarp_names:
                _cache_convention_names(articulation, parsed_convention, complete_mjwarp_names)
            if kind in complete_mjwarp_names:
                return complete_mjwarp_names[kind]

    raise NotImplementedError(
        f"Resolving {parsed_convention.value} {kind} ordering from backend '{active_backend_name}' requires "
        f"{parsed_convention.value} name metadata for this articulation."
    )


def get_physx_articulation_name_ordering(
    articulation: BaseArticulation,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return names to expose publicly in PhysX or OVPhysX order.

    PhysX and OVPhysX articulations use their backend names through the
    same-backend identity fast path. On another backend, resolution checks
    cached or precomputed PhysX names and ``root_view.shared_metatype``. If
    needed, it may construct a temporary CPU Newton ``ModelBuilder`` view with
    breadth-first joint order. A successful USD build caches both joint and body
    names, so that cross-backend discovery is one-time per articulation.

    ``root_view`` metadata and solver-view arrays remain in backend order; this
    function only returns the name order for the public axis. Missing or malformed
    optional name attributes and unavailable builder dependencies, stages, or
    source prims are treated as absent metadata. Resolution may then fall through
    to :class:`NotImplementedError`. Provider or builder exceptions outside those
    absence checks propagate.

    Args:
        articulation: Articulation whose PhysX names should be resolved.
        kind: Element kind, either ``"joint"`` or ``"body"``.

    Returns:
        Names in public PhysX or OVPhysX tensor-view order.

    Raises:
        AttributeError: If same-backend names are unavailable, or a provider or
            builder raises this error outside optional attribute probes.
        TypeError: If a provider or builder raises an unhandled type error.
        ValueError: If a provider or builder rejects the source asset.
        NotImplementedError: If optional PhysX name metadata is absent or
            incomplete after all fallbacks.
    """
    return resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=ArticulationOrderingConvention.PHYSX,
        kind=kind,
    )


def get_mjwarp_articulation_name_ordering(
    articulation: BaseArticulation,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return names to expose publicly in Newton or MJWarp order.

    Newton, MJWarp, and ``newton_mjwarp`` articulations use their backend names
    through the same-backend identity fast path. On another backend, resolution
    checks cached or precomputed names and ``root_view.joint_dof_names`` or
    ``root_view.link_names``. If needed, it may construct a temporary CPU Newton
    ``ModelBuilder`` view with depth-first joint order. A successful USD build
    caches both joint and body names, so that cross-backend discovery is one-time
    per articulation.

    ``root_view`` metadata and solver-view arrays remain in backend order; this
    function only returns the name order for the public axis. Missing or malformed
    optional name attributes and unavailable builder dependencies, stages, or
    source prims are treated as absent metadata. Resolution may then fall through
    to :class:`NotImplementedError`. Provider or builder exceptions outside those
    absence checks propagate.

    Args:
        articulation: Articulation whose Newton or MJWarp names are resolved.
        kind: Element kind, either ``"joint"`` or ``"body"``.

    Returns:
        Names in public Newton or MJWarp articulation-view order.

    Raises:
        AttributeError: If same-backend names are unavailable, or a provider or
            builder raises this error outside optional attribute probes.
        TypeError: If a provider or builder raises an unhandled type error.
        ValueError: If a provider or builder rejects the source asset.
        NotImplementedError: If optional Newton or MJWarp name metadata is absent
            or incomplete after all fallbacks.
    """
    return resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=ArticulationOrderingConvention.MJWARP,
        kind=kind,
    )


def get_robot_schema_articulation_name_ordering(
    articulation: BaseArticulation,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return names to expose publicly in authored robot-schema order.

    The source asset prim or configured articulation-root prim must author
    ``isaac:physics:robotJoints`` for joints or
    ``isaac:physics:robotLinks`` for bodies. Nested robot targets are expanded,
    name overrides are honored, unrelated targets are ignored, and the remaining
    names must form a complete unique ordering of active backend names.

    No simulation backend is treated as a robot-schema same-backend identity
    case. The first successful relationship discovery is cached for the requested
    element kind, making traversal one-time per articulation. This path does not
    construct a Newton model. ``root_view`` metadata and solver-view arrays remain
    in backend order.

    Missing relationship APIs or targets, unresolved target prims, and incomplete
    relationship orders are treated as absent metadata and may fall through to
    :class:`NotImplementedError`. USD provider exceptions outside those absence
    checks propagate.

    Args:
        articulation: Articulation whose source USD relationships are resolved.
        kind: Element kind, either ``"joint"`` or ``"body"``.

    Returns:
        Names in public authored robot-schema relationship order.

    Raises:
        AttributeError: If a USD provider raises this error outside optional
            attribute probes.
        TypeError: If a USD provider raises this error outside handled target
            lookups or name-sequence coercion.
        ValueError: If source-asset resolution or another USD provider rejects
            the metadata.
        NotImplementedError: If robot-schema relationship metadata is absent or
            incomplete after all fallbacks.
    """
    return resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=ArticulationOrderingConvention.ROBOT_SCHEMA,
        kind=kind,
    )


def resolve_articulation_ordering_names(
    *,
    kind: Literal["joint", "body"],
    backend_names: Sequence[str],
    ordering: Sequence[str] | str | ArticulationOrderingConvention | None,
    active_backend_name: str,
    articulation: BaseArticulation | None = None,
    convention_name_resolver: Callable[[ArticulationOrderingConvention, Literal["joint", "body"]], Sequence[str]]
    | None = None,
) -> tuple[str, ...]:
    """Resolve configured public articulation ordering to concrete names.

    ``None`` and conventions matching :paramref:`active_backend_name` take an
    identity fast path and return :paramref:`backend_names`. Explicit sequences
    are type-checked here; complete-permutation validation is performed later by
    :func:`build_articulation_name_map`.

    A cross-backend convention first uses :paramref:`convention_name_resolver`
    when supplied, otherwise it delegates to
    :func:`resolve_articulation_convention_name_ordering`. The articulation path
    reuses that resolver's per-articulation discovery cache. Joint names are
    normalized to active-backend spelling when Newton multi-DoF separators
    differ.

    The returned tuple defines public order. :paramref:`backend_names`,
    ``root_view`` metadata, and solver-view arrays remain in backend order.
    Optional metadata failures handled as absence by the delegated resolver may
    end in :class:`NotImplementedError`. Unsupported ordering inputs and custom
    resolver, provider, or builder exceptions not handled as absence propagate.

    Args:
        kind: Element kind, either ``"joint"`` or ``"body"``.
        backend_names: Names in active backend solver-view order.
        ordering: Explicit public name sequence, symbolic convention alias or
            enum member, or ``None``.
        active_backend_name: Name of the backend exposing
            :paramref:`backend_names`.
        articulation: Articulation used for cached cross-backend discovery when
            no explicit resolver is supplied.
        convention_name_resolver: Optional resolver called with the parsed
            convention and :paramref:`kind`; its returned sequence defines the
            requested public order.

    Returns:
        Concrete names for the public joint or body axis.

    Raises:
        AttributeError: If a custom resolver, provider, or builder raises this
            error outside an optional metadata probe.
        TypeError: If :paramref:`ordering` has an unsupported type, an explicit
            sequence contains a non-string, or an unhandled resolver error occurs.
        ValueError: If :paramref:`ordering` is an unsupported alias, or a custom
            resolver, provider, or builder rejects the source metadata.
        NotImplementedError: If cross-backend ordering lacks an articulation or
            resolver, or all optional convention metadata is absent or incomplete.
    """
    backend_names = _coerce_articulation_names(backend_names, parameter_name="backend_names")
    if ordering is None:
        return backend_names
    if isinstance(ordering, ArticulationOrderingConvention):
        convention = ordering
    elif isinstance(ordering, str):
        convention = parse_articulation_ordering_convention(ordering)
    elif isinstance(ordering, Sequence) and not isinstance(ordering, bytes | bytearray):
        return _coerce_articulation_names(ordering, parameter_name=f"{kind}_ordering")
    else:
        raise TypeError(
            f"{kind}_ordering must be a name sequence, convention string/enum, or None; got {type(ordering).__name__}."
        )

    if convention is None or _backend_matches_ordering_convention(active_backend_name, convention):
        return backend_names
    if convention_name_resolver is not None:
        convention_names = _coerce_articulation_names(
            convention_name_resolver(convention, kind),
            parameter_name="convention_name_resolver result",
        )
        return _match_backend_name_spellings(kind=kind, names=convention_names, backend_names=backend_names)
    if articulation is not None:
        convention_names = resolve_articulation_convention_name_ordering(
            articulation=articulation,
            convention=convention,
            kind=kind,
        )
        return _match_backend_name_spellings(kind=kind, names=convention_names, backend_names=backend_names)

    raise NotImplementedError(
        f"Resolving {convention.value} {kind} ordering from backend '{active_backend_name}' requires an "
        "articulation or backend convention name resolver for this asset."
    )
