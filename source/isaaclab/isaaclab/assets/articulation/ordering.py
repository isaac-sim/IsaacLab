# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg


class ArticulationOrderingConvention(str, Enum):
    """Built-in non-default articulation name ordering conventions."""

    PHYSX = "physx"
    MJWARP = "mjwarp"


@dataclass(frozen=True)
class ArticulationNameMap:
    """Mapping between backend and user articulation name order."""

    kind: Literal["joint", "body"]
    """Mapped articulation element kind."""

    backend_names: tuple[str, ...]
    """Names in the order exposed by the active backend."""

    user_names: tuple[str, ...]
    """Names in the order exposed by the public articulation API."""

    user_to_backend_indices: tuple[int, ...]
    """CPU map from public user index to backend index."""

    backend_to_user_indices: tuple[int, ...]
    """CPU map from backend index to public user index."""

    user_to_backend: wp.array(dtype=wp.int32)
    """Device map from public user index to backend index."""

    backend_to_user: wp.array(dtype=wp.int32)
    """Device map from backend index to public user index."""

    is_identity: bool
    """Whether user and backend name order are identical."""


def parse_articulation_ordering_convention(
    ordering: str | ArticulationOrderingConvention | None,
) -> ArticulationOrderingConvention | None:
    """Parse a symbolic articulation ordering convention.

    Args:
        ordering: Ordering convention alias, enum value, or ``None``.

    Returns:
        Parsed ordering convention, or ``None`` when no convention is requested.

    Raises:
        ValueError: If :paramref:`ordering` is an unsupported string alias.
        TypeError: If :paramref:`ordering` is not a supported type.
    """
    if ordering is None:
        return None
    if isinstance(ordering, ArticulationOrderingConvention):
        return ordering
    if isinstance(ordering, str):
        try:
            return ArticulationOrderingConvention(ordering.lower())
        except ValueError as exc:
            valid_values = ", ".join(convention.value for convention in ArticulationOrderingConvention)
            raise ValueError(
                f"Unsupported articulation ordering convention '{ordering}'. Expected one of: {valid_values}."
            ) from exc
    raise TypeError(
        "Articulation ordering convention must be a string, "
        f"{ArticulationOrderingConvention.__name__}, or None. Got {type(ordering).__name__}."
    )


def apply_articulation_ordering_preset(
    cfg: ArticulationCfg,
    ordering: str | ArticulationOrderingConvention | None,
) -> ArticulationCfg:
    """Return ``cfg`` with one ordering preset applied to joints and bodies.

    Args:
        cfg: Articulation configuration to copy.
        ordering: Ordering convention alias, enum value, or ``None``.

    Returns:
        A copy of :paramref:`cfg` with :attr:`joint_ordering` and
        :attr:`body_ordering` set to the parsed convention. If
        :paramref:`ordering` is ``None``, returns :paramref:`cfg` unchanged.
    """
    parsed_ordering = parse_articulation_ordering_convention(ordering)
    if parsed_ordering is None:
        return cfg
    return cfg.replace(joint_ordering=parsed_ordering, body_ordering=parsed_ordering)


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
    if names is None or isinstance(names, str):
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
        return None


def _get_articulation_root_view(articulation: object) -> object | None:
    """Return an articulation root view when it is already available."""
    root_view = _get_attr_or_none(articulation, "root_view")
    if root_view is not None:
        return root_view
    return _get_attr_or_none(articulation, "_root_view")


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
        if names is not None:
            return tuple(names)
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
            shared_metatype = _get_attr_or_none(root_view, "_shared_metatype")
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
    articulation: object,
    convention: str | ArticulationOrderingConvention,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Resolve a symbolic backend convention to concrete articulation names.

    Args:
        articulation: Articulation instance whose configured asset is being resolved.
        convention: Symbolic backend convention to resolve.
        kind: Articulation element kind.

    Returns:
        Concrete articulation names in the requested convention order.

    Raises:
        NotImplementedError: If the active articulation does not expose metadata for
            the requested convention.
    """
    parsed_convention = parse_articulation_ordering_convention(convention)
    if parsed_convention is None:
        return _get_backend_names(articulation, kind)

    active_backend_name = getattr(articulation, "__backend_name__", "unknown")
    if _backend_matches_ordering_convention(active_backend_name, parsed_convention):
        return _get_backend_names(articulation, kind)

    cached_names = _get_cached_convention_names(articulation, parsed_convention, kind)
    if cached_names is not None:
        return cached_names

    precomputed_names = _get_precomputed_convention_names(articulation, parsed_convention, kind)
    if precomputed_names is not None:
        return precomputed_names

    root_view = _get_articulation_root_view(articulation)
    if root_view is not None:
        root_view_names = _get_root_view_convention_names(root_view, parsed_convention, kind)
        if root_view_names is not None:
            return root_view_names

    if parsed_convention is ArticulationOrderingConvention.PHYSX:
        physx_names = _get_physx_names_from_newton_usd_builder(articulation)
        if physx_names is not None:
            _cache_convention_names(articulation, parsed_convention, physx_names)
            return physx_names[kind]

    if parsed_convention is ArticulationOrderingConvention.MJWARP:
        mjwarp_names = _get_mjwarp_names_from_newton_usd_builder(articulation)
        if mjwarp_names is not None:
            _cache_convention_names(articulation, parsed_convention, mjwarp_names)
            return mjwarp_names[kind]

    raise NotImplementedError(
        f"Resolving {parsed_convention.value} {kind} ordering from backend '{active_backend_name}' requires "
        f"{parsed_convention.value} name metadata for this articulation."
    )


def get_physx_articulation_name_ordering(
    articulation: object,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return articulation names in the PhysX backend convention order.

    Args:
        articulation: Articulation instance whose names should be resolved.
        kind: Articulation element kind.

    Returns:
        Concrete articulation names in PhysX order.
    """
    return resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=ArticulationOrderingConvention.PHYSX,
        kind=kind,
    )


def get_mjwarp_articulation_name_ordering(
    articulation: object,
    kind: Literal["joint", "body"],
) -> tuple[str, ...]:
    """Return articulation names in the MJWarp/Newton backend convention order.

    Args:
        articulation: Articulation instance whose names should be resolved.
        kind: Articulation element kind.

    Returns:
        Concrete articulation names in MJWarp/Newton order.
    """
    return resolve_articulation_convention_name_ordering(
        articulation=articulation,
        convention=ArticulationOrderingConvention.MJWARP,
        kind=kind,
    )


def resolve_articulation_ordering_names(
    *,
    kind: Literal["joint", "body"],
    backend_names: Sequence[str],
    ordering: Sequence[str] | str | ArticulationOrderingConvention | None,
    active_backend_name: str,
    articulation: object | None = None,
    convention_name_resolver: Callable[[ArticulationOrderingConvention, Literal["joint", "body"]], Sequence[str]]
    | None = None,
) -> tuple[str, ...]:
    """Resolve a configured public articulation ordering to concrete names.

    Args:
        kind: Articulation element kind.
        backend_names: Names in the order exposed by the active backend.
        ordering: Explicit name permutation, symbolic convention alias, or ``None``.
        active_backend_name: Name of the backend currently exposing
            :paramref:`backend_names`.
        articulation: Optional articulation instance used to resolve cross-backend
            symbolic conventions.
        convention_name_resolver: Optional resolver for cross-backend symbolic
            conventions. It receives the parsed convention and :paramref:`kind`
            and returns the requested concrete name order.

    Returns:
        Concrete public names requested by :paramref:`ordering`.

    Raises:
        NotImplementedError: If a cross-backend symbolic convention is requested
            without a convention resolver.
    """
    backend_names = tuple(backend_names)
    if ordering is None:
        return backend_names
    if isinstance(ordering, list | tuple):
        return tuple(ordering)

    convention = parse_articulation_ordering_convention(ordering)
    if convention is None or _backend_matches_ordering_convention(active_backend_name, convention):
        return backend_names
    if articulation is not None:
        return resolve_articulation_convention_name_ordering(
            articulation=articulation,
            convention=convention,
            kind=kind,
        )
    if convention_name_resolver is not None:
        return tuple(convention_name_resolver(convention, kind))

    raise NotImplementedError(
        f"Resolving {convention.value} {kind} ordering from backend '{active_backend_name}' requires an "
        "articulation or backend convention name resolver for this asset."
    )


def build_articulation_name_map(
    *,
    kind: Literal["joint", "body"],
    backend_names: Sequence[str],
    user_names: Sequence[str] | None,
    device: str,
) -> ArticulationNameMap:
    """Build maps between backend and public articulation name order.

    Args:
        kind: Articulation element kind.
        backend_names: Names in the order exposed by the active backend.
        user_names: Optional complete public ordering permutation. If ``None``,
            the backend order is used.
        device: Device where Warp map arrays are allocated.

    Returns:
        Mapping metadata and optional Warp arrays for non-identity ordering.

    Raises:
        ValueError: If names are duplicated or :paramref:`user_names` is not a
            complete permutation of :paramref:`backend_names`.
    """
    backend_names = tuple(backend_names)
    user_names = backend_names if user_names is None else tuple(user_names)

    if len(set(backend_names)) != len(backend_names):
        raise ValueError(f"Duplicate backend {kind} names are not supported: {backend_names}.")
    if len(set(user_names)) != len(user_names):
        raise ValueError(f"Duplicate requested {kind} names are not supported: {user_names}.")

    backend_name_set = set(backend_names)
    user_name_set = set(user_names)
    if user_name_set != backend_name_set:
        missing = sorted(backend_name_set - user_name_set)
        extra = sorted(user_name_set - backend_name_set)
        raise ValueError(
            f"Requested {kind} names must be a complete permutation of backend names. Missing={missing}, extra={extra}."
        )

    backend_index_by_name = {name: index for index, name in enumerate(backend_names)}
    user_to_backend_np = np.asarray([backend_index_by_name[name] for name in user_names], dtype=np.int32)
    backend_to_user_np = np.empty_like(user_to_backend_np)
    backend_to_user_np[user_to_backend_np] = np.arange(len(user_names), dtype=np.int32)
    is_identity = user_names == backend_names

    return ArticulationNameMap(
        kind=kind,
        backend_names=backend_names,
        user_names=user_names,
        user_to_backend_indices=tuple(int(index) for index in user_to_backend_np),
        backend_to_user_indices=tuple(int(index) for index in backend_to_user_np),
        user_to_backend=wp.array(user_to_backend_np, dtype=wp.int32, device=device),
        backend_to_user=wp.array(backend_to_user_np, dtype=wp.int32, device=device),
        is_identity=is_identity,
    )
