# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
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
        return backend_name == "newton"
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


def _validate_ordering_kind(kind: object) -> Literal["joint", "body"]:
    """Return a supported articulation element kind."""
    if kind == "joint" or kind == "body":
        return kind
    raise ValueError(f"kind must be 'joint' or 'body'; got {kind!r}.")


def _get_backend_names(articulation: BaseArticulation, kind: Literal["joint", "body"]) -> tuple[str, ...]:
    """Return active backend names from an articulation."""
    attr_name = "backend_joint_names" if kind == "joint" else "backend_body_names"
    names = _coerce_name_sequence(getattr(articulation, attr_name))
    if names is None:
        raise TypeError(f"Articulation {attr_name} must be a sequence of strings.")
    return names


def _get_cached_convention_names(
    articulation: BaseArticulation,
    convention: ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...] | None:
    """Return cached convention names for an articulation, if present."""
    cache = getattr(articulation, "_ordering_convention_name_cache", None)
    if isinstance(cache, dict):
        names = cache.get((convention, kind))
        return _coerce_name_sequence(names)
    return None


def _cache_convention_names(
    articulation: BaseArticulation,
    convention: ArticulationOrderingConvention,
    names_by_kind: dict[Literal["joint", "body"], tuple[str, ...]],
) -> None:
    """Cache convention names on mutable articulation instances."""
    cache = getattr(articulation, "_ordering_convention_name_cache", None)
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
    is_valid = getattr(prim, "IsValid", None)
    if callable(is_valid):
        return bool(is_valid())
    return True


def _get_stage_prim_at_path(stage: object | None, path: object) -> object | None:
    """Return a stage prim at a path when the stage can resolve it."""
    if stage is None:
        return None
    get_prim_at_path = getattr(stage, "GetPrimAtPath", None)
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
    get_attribute = getattr(prim, "GetAttribute", None)
    if not callable(get_attribute):
        return None
    for attr_name in attr_names:
        attr = get_attribute(attr_name)
        if attr is None:
            continue
        get_value = getattr(attr, "Get", None)
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
    get_name = getattr(prim, "GetName", None)
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
    get_relationship = getattr(prim, "GetRelationship", None)
    if not callable(get_relationship):
        return ()
    relationship = get_relationship(relationship_name)
    if relationship is None:
        return ()
    get_targets = getattr(relationship, "GetTargets", None)
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
    get_stage = getattr(robot_prim, "GetStage", None)
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


def _get_source_asset_prim(articulation: BaseArticulation) -> object | None:
    """Return the source asset prim for an articulation config when available."""
    prim_path = articulation.cfg.prim_path
    from isaaclab.sim.utils.queries import resolve_matching_prims_from_source  # noqa: PLC0415

    source_asset_matches = resolve_matching_prims_from_source(prim_path, expected_num_matches=1)
    if not source_asset_matches:
        return None
    return source_asset_matches[0][0]


def _get_robot_schema_candidate_prims(articulation: BaseArticulation) -> tuple[object, ...]:
    """Return candidate prims that may author robot schema ordering relationships."""
    source_asset_prim = _get_source_asset_prim(articulation)
    if source_asset_prim is None:
        return ()

    candidate_prims = [source_asset_prim]
    articulation_root_prim_path = articulation.cfg.articulation_root_prim_path
    if articulation_root_prim_path is not None:
        get_stage = getattr(source_asset_prim, "GetStage", None)
        stage = get_stage() if callable(get_stage) else None
        root_path = _get_prim_path_string(source_asset_prim) + articulation_root_prim_path
        articulation_root_prim = _get_stage_prim_at_path(stage, root_path)
        if articulation_root_prim is not None:
            candidate_prims.append(articulation_root_prim)
    return tuple(candidate_prims)


def _get_robot_schema_names(
    articulation: BaseArticulation,
    kind: Literal["joint", "body"],
) -> tuple[str, ...] | None:
    """Return complete articulation names from Isaac Sim robot schema relationships."""
    backend_names = _get_backend_names(articulation, kind)

    for candidate_prim in _get_robot_schema_candidate_prims(articulation):
        relationship_names = _collect_robot_schema_relationship_names(candidate_prim, kind, set())
        names = _filter_complete_backend_name_order(relationship_names, backend_names)
        if names is not None:
            return names
    return None


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
    return _get_names_from_newton_usd_builder(
        articulation,
        joint_ordering="bfs",
        bodies_follow_joint_ordering=True,
    )


def _get_mjwarp_names_from_newton_usd_builder(
    articulation: BaseArticulation,
) -> dict[Literal["joint", "body"], tuple[str, ...]] | None:
    """Build a lightweight Newton prototype view with MJWarp-style articulation names."""
    return _get_names_from_newton_usd_builder(
        articulation,
        joint_ordering="dfs",
        bodies_follow_joint_ordering=True,
    )


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
        ValueError: If kind or convention is invalid, or a provider rejects source
            metadata.
        NotImplementedError: If no supported source provides a complete ordering.
            The message identifies the corresponding configuration field and
            explicit-name fallback.
    """
    kind = _validate_ordering_kind(kind)
    parsed_convention = parse_articulation_ordering_convention(convention)
    if parsed_convention is None:
        return _get_backend_names(articulation, kind)

    active_backend_name = articulation.__backend_name__
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

    if parsed_convention is ArticulationOrderingConvention.PHYSX:
        physx_names = _get_physx_names_from_newton_usd_builder(articulation)
        if physx_names is not None:
            complete_physx_names = _get_complete_convention_names_by_kind(articulation, physx_names)
            if len(complete_physx_names) == 2:
                _cache_convention_names(articulation, parsed_convention, complete_physx_names)
            if kind in complete_physx_names:
                return complete_physx_names[kind]

    if parsed_convention is ArticulationOrderingConvention.MJWARP:
        mjwarp_names = _get_mjwarp_names_from_newton_usd_builder(articulation)
        if mjwarp_names is not None:
            complete_mjwarp_names = _get_complete_convention_names_by_kind(articulation, mjwarp_names)
            if len(complete_mjwarp_names) == 2:
                _cache_convention_names(articulation, parsed_convention, complete_mjwarp_names)
            if kind in complete_mjwarp_names:
                return complete_mjwarp_names[kind]

    config_field = "joint_ordering" if kind == "joint" else "body_ordering"
    raise NotImplementedError(
        f"Unable to resolve '{parsed_convention.value}' {kind} ordering for active backend "
        f"'{active_backend_name}'. Ensure the source USD and required ordering dependencies are available, "
        f"set env.scene.robot.{config_field} to an explicit {kind}-name permutation, or use None to keep "
        "active-backend order."
    )


def get_physx_articulation_name_ordering(
    articulation: BaseArticulation,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return names in PhysX or OVPhysX articulation-view order.

    PhysX and OVPhysX articulations return active-backend names without discovery.
    Other backends require a source USD readable by the optional Newton and PXR
    dependencies. The temporary Newton view uses breadth-first joint ordering; a
    complete joint-and-body result is cached per articulation.

    The result defines the public axis only; backend views remain in native order.

    Args:
        articulation: Articulation whose PhysX names are resolved.
        kind: Element kind, either joint or body.

    Returns:
        Names in PhysX or OVPhysX articulation-view order.

    Raises:
        TypeError: If backend or discovered names are malformed.
        ValueError: If kind is invalid or the builder rejects the source asset.
        NotImplementedError: If the source USD, builder dependencies, or complete
            name permutation is unavailable. The message identifies the
            corresponding configuration field and explicit-name fallback.
    """
    return _resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=ArticulationOrderingConvention.PHYSX,
        kind=kind,
    )


def get_mjwarp_articulation_name_ordering(
    articulation: BaseArticulation,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return names in Newton or MJWarp articulation-view order.

    Newton articulations return active-backend names without discovery. Other
    backends require a source USD readable by the optional Newton and PXR
    dependencies. The temporary Newton view uses depth-first joint ordering; a
    complete joint-and-body result is cached per articulation.

    The result defines the public axis only; backend views remain in native order.

    Args:
        articulation: Articulation whose Newton or MJWarp names are resolved.
        kind: Element kind, either joint or body.

    Returns:
        Names in Newton or MJWarp articulation-view order.

    Raises:
        TypeError: If backend or discovered names are malformed.
        ValueError: If kind is invalid or the builder rejects the source asset.
        NotImplementedError: If the source USD, builder dependencies, or complete
            name permutation is unavailable. The message identifies the
            corresponding configuration field and explicit-name fallback.
    """
    return _resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=ArticulationOrderingConvention.MJWARP,
        kind=kind,
    )


def get_robot_schema_articulation_name_ordering(
    articulation: BaseArticulation,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return names in authored robot-schema order.

    The source asset prim or configured articulation-root prim must author
    robotJoints for joints or robotLinks for bodies. Nested robot targets are
    expanded, name overrides are honored, unrelated targets are ignored, and the
    remaining names must be a complete unique permutation of active-backend
    names. A successful result is cached for the requested element kind.

    The result defines the public axis only; backend views remain in native order.

    Args:
        articulation: Articulation whose source USD relationships are resolved.
        kind: Element kind, either joint or body.

    Returns:
        Names in authored robot-schema relationship order.

    Raises:
        TypeError: If backend names are not a sequence of strings.
        ValueError: If kind is invalid or USD resolution rejects the source
            metadata.
        NotImplementedError: If the required relationships are unavailable or
            incomplete. The message identifies the corresponding configuration
            field and explicit-name fallback.
    """
    return _resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=ArticulationOrderingConvention.ROBOT_SCHEMA,
        kind=kind,
    )


def _resolve_articulation_ordering_names(
    *,
    kind: Literal["joint", "body"],
    backend_names: Sequence[str],
    ordering: Sequence[str] | str | ArticulationOrderingConvention | None,
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
        f"Set env.scene.robot.{config_field} to an explicit {kind}-name permutation, or supply an articulation "
        "whose source USD can provide that convention."
    )
