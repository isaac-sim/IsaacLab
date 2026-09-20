# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public articulation ordering name-map type and its validation.

This module owns the public ordering surface: the
:class:`ArticulationOrderingConvention` enum, the frozen
:class:`ArticulationNameMap` permutation type (built only by the module-level
:func:`build_articulation_name_map` factory, never constructed directly), and
the helpers that parse conventions and build validated maps between backend and
public order.
Discovering a convention's concrete names lives in
:mod:`isaaclab.assets.articulation.ordering_resolvers`, and device-side axis
conversion lives in :mod:`isaaclab.assets.articulation.ordering_kernels`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg


def _validate_articulation_names(names: list[str] | tuple[str, ...], *, parameter_name: str) -> tuple[str, ...]:
    """Return the names as a tuple after a hard list-or-tuple-of-str check.

    Ingestion boundaries accept both plain sequence types because config
    sources such as Hydra produce lists; the canonical stored form is always
    a tuple.
    """
    if type(names) not in (list, tuple):
        raise TypeError(f"{parameter_name} must be a list or tuple of strings; got {type(names).__name__}.")
    for index, name in enumerate(names):
        if type(name) is not str:
            raise TypeError(f"{parameter_name} element {index} must be str; got {name!r} ({type(name).__name__}).")
    return tuple(names)


class ArticulationOrderingConvention(str, Enum):
    """Built-in non-default public articulation name-ordering conventions.

    Attributes:
        PHYSX: Active PhysX or OVPhysX tensor-view order.
        MJWARP: Newton or MJWarp articulation-view order.
        ROBOT_SCHEMA: Authored target order of the ``isaac:physics:robotJoints``
            and ``isaac:physics:robotLinks`` relationships.

    ``None`` selects the active backend order by default and is not a member of
    this enum.
    """

    PHYSX = "physx"
    MJWARP = "mjwarp"
    ROBOT_SCHEMA = "robot_schema"


@dataclass(frozen=True)
class ArticulationNameMap:
    """Frozen permutation between backend and public articulation order.

    ``user`` in the field names means the order exposed by the public API.
    ``user_to_backend`` maps a public index to its backend index, while
    ``backend_to_user`` maps a backend index to its public index. The CPU
    tuples and device arrays are complete inverse permutations of the same
    length; both device maps are one-dimensional ``wp.int32`` arrays on the
    articulation's device. The frozen dataclass prevents field reassignment,
    but the Warp arrays remain mutable objects and callers must treat both
    device maps as read-only.

    Instances are built by the owning articulation during initialization via
    :func:`build_articulation_name_map`; the class is not intended for direct
    construction. Identity orderings are represented as ``None`` rather than a
    map — the :attr:`~isaaclab.assets.Articulation.joint_ordering` and
    :attr:`~isaaclab.assets.Articulation.body_ordering` properties are ``None``
    whenever public and backend orders coincide, so a non-``None`` map always
    denotes an actual permutation.
    """

    user_to_backend_indices: tuple[int, ...]
    """One-dimensional CPU map from public index to backend index."""

    backend_to_user_indices: tuple[int, ...]
    """One-dimensional CPU map from backend index to public index."""

    user_to_backend: wp.array(dtype=wp.int32)
    """Read-only public-to-backend device map, shape ``(num_names,)``, dtype ``wp.int32``."""

    backend_to_user: wp.array(dtype=wp.int32)
    """Read-only backend-to-public device map, shape ``(num_names,)``, dtype ``wp.int32``."""


def parse_articulation_ordering_convention(
    ordering: str | ArticulationOrderingConvention | None,
) -> ArticulationOrderingConvention | None:
    """Parse a symbolic public articulation ordering convention.

    Accepted aliases are ``"physx"``, ``"mjwarp"``, and
    ``"robot_schema"``. String aliases are matched case-insensitively.
    ``None`` keeps the active backend's default order and is not an enum member.

    Args:
        ordering: Convention alias, :class:`ArticulationOrderingConvention`
            member, or ``None``.

    Returns:
        The matching :class:`ArticulationOrderingConvention` member, or
        ``None`` when no non-default convention is requested.

    Raises:
        TypeError: If :paramref:`ordering` is not a supported type.
        ValueError: If :paramref:`ordering` is an unsupported string alias.
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
    """Apply one public ordering preset to both joints and bodies.

    Args:
        cfg: Articulation configuration to copy when a preset is requested.
        ordering: Convention alias, :class:`ArticulationOrderingConvention`
            member, or ``None``.

    Returns:
        A copy of :paramref:`cfg` whose
        :attr:`ArticulationCfg.joint_ordering` and
        :attr:`ArticulationCfg.body_ordering` use the parsed convention.
        When :paramref:`ordering` is ``None``, returns the original
        :paramref:`cfg` object unchanged.

    Raises:
        TypeError: If :paramref:`ordering` is not a supported type.
        ValueError: If :paramref:`ordering` is an unsupported string alias.
    """
    parsed_ordering = parse_articulation_ordering_convention(ordering)
    if parsed_ordering is None:
        return cfg
    return cfg.replace(joint_ordering=parsed_ordering, body_ordering=parsed_ordering)


def build_articulation_name_map(
    *,
    kind: Literal["joint", "body"],
    backend_names: Sequence[str],
    user_names: Sequence[str] | None,
    device: str,
) -> ArticulationNameMap | None:
    """Build a validated map between backend and public articulation order.

    Args:
        kind: Mapped element kind, either ``"joint"`` or ``"body"``; used in
            error messages.
        backend_names: Names in active backend solver-view order.
        user_names: Complete permutation in public API order, or ``None`` to
            keep the backend order.
        device: Warp device on which both device index maps are allocated.

    Returns:
        An :class:`ArticulationNameMap` containing CPU index tuples and
        one-dimensional ``wp.int32`` device maps on :paramref:`device`, or
        ``None`` when :paramref:`user_names` is ``None`` or already in backend
        order (identity).

    Raises:
        ValueError: If either name sequence has duplicates or
            :paramref:`user_names` is not a complete permutation of
            :paramref:`backend_names`.
        TypeError: If either name input is not a list or tuple of strings.
    """
    backend_names = _validate_articulation_names(backend_names, parameter_name="backend_names")
    if user_names is None:
        return None
    user_names = _validate_articulation_names(user_names, parameter_name="user_names")

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

    if user_names == backend_names:
        return None

    backend_index_by_name = {name: index for index, name in enumerate(backend_names)}
    user_to_backend_np = np.asarray([backend_index_by_name[name] for name in user_names], dtype=np.int32)
    backend_to_user_np = np.empty_like(user_to_backend_np)
    backend_to_user_np[user_to_backend_np] = np.arange(len(user_names), dtype=np.int32)

    return ArticulationNameMap(
        user_to_backend_indices=tuple(int(index) for index in user_to_backend_np),
        backend_to_user_indices=tuple(int(index) for index in backend_to_user_np),
        user_to_backend=wp.array(user_to_backend_np, dtype=wp.int32, device=device),
        backend_to_user=wp.array(backend_to_user_np, dtype=wp.int32, device=device),
    )
