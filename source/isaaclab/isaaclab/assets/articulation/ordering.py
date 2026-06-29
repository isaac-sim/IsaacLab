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

    def __post_init__(self) -> None:
        """Validate CPU-side name-map invariants."""
        if self.kind not in {"joint", "body"}:
            raise ValueError(f"ArticulationNameMap kind must be 'joint' or 'body'. Got {self.kind!r}.")

        object.__setattr__(self, "backend_names", tuple(self.backend_names))
        object.__setattr__(self, "user_names", tuple(self.user_names))
        object.__setattr__(self, "user_to_backend_indices", tuple(int(index) for index in self.user_to_backend_indices))
        object.__setattr__(self, "backend_to_user_indices", tuple(int(index) for index in self.backend_to_user_indices))

        num_names = len(self.backend_names)
        if len(self.user_names) != num_names:
            raise ValueError("ArticulationNameMap user_names and backend_names must have the same length.")
        if len(self.user_to_backend_indices) != num_names or len(self.backend_to_user_indices) != num_names:
            raise ValueError("ArticulationNameMap CPU index maps must match the number of names.")
        if len(set(self.backend_names)) != num_names:
            raise ValueError(f"Duplicate backend {self.kind} names are not supported: {self.backend_names}.")
        if len(set(self.user_names)) != num_names:
            raise ValueError(f"Duplicate user {self.kind} names are not supported: {self.user_names}.")

        expected_indices = set(range(num_names))
        user_to_backend = self.user_to_backend_indices
        backend_to_user = self.backend_to_user_indices
        if set(user_to_backend) != expected_indices:
            raise ValueError("ArticulationNameMap user_to_backend_indices must be a complete permutation.")
        if set(backend_to_user) != expected_indices:
            raise ValueError("ArticulationNameMap backend_to_user_indices must be a complete permutation.")
        for user_index, backend_index in enumerate(user_to_backend):
            if backend_to_user[backend_index] != user_index:
                raise ValueError("ArticulationNameMap CPU index maps must be inverse permutations.")

        for map_name, device_map, cpu_map in (
            ("user_to_backend", self.user_to_backend, user_to_backend),
            ("backend_to_user", self.backend_to_user, backend_to_user),
        ):
            if device_map.dtype != wp.int32 or device_map.ndim != 1 or device_map.shape != (num_names,):
                raise ValueError(
                    f"ArticulationNameMap device {map_name} map must be a one-dimensional int32 array "
                    f"with shape ({num_names},)."
                )
            if tuple(int(index) for index in device_map.numpy()) != cpu_map:
                raise ValueError(f"ArticulationNameMap device {map_name} map must match {map_name}_indices.")

        if self.user_to_backend.device != self.backend_to_user.device:
            raise ValueError("ArticulationNameMap device index maps must be allocated on the same device.")

        identity_indices = tuple(range(num_names))
        expected_identity = (
            self.user_names == self.backend_names
            and user_to_backend == identity_indices
            and backend_to_user == identity_indices
        )
        if self.is_identity != expected_identity:
            raise ValueError("ArticulationNameMap is_identity is inconsistent with names and index maps.")


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
        Mapping metadata and device Warp arrays for index conversion.

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
