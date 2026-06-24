# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
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
    ROBOT_SCHEMA = "robot_schema"


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
