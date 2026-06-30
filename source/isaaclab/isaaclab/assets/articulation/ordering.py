# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg


def _coerce_articulation_names(names: object, *, parameter_name: str) -> tuple[str, ...]:
    """Return a validated articulation name sequence."""
    if isinstance(names, str | bytes | bytearray):
        raise TypeError(f"{parameter_name} must be a sequence of strings; got {type(names).__name__}.")
    try:
        name_tuple = tuple(names)
    except TypeError as exc:
        raise TypeError(f"{parameter_name} must be a sequence of strings; got {type(names).__name__}.") from exc
    for index, name in enumerate(name_tuple):
        if not isinstance(name, str):
            raise TypeError(f"{parameter_name} element {index} must be str; got {name!r} ({type(name).__name__}).")
    return name_tuple


def _coerce_articulation_indices(indices: object, *, parameter_name: str) -> tuple[int, ...]:
    """Return a validated integer index sequence."""
    if isinstance(indices, str | bytes | bytearray):
        raise TypeError(f"{parameter_name} must be a sequence of integers; got {type(indices).__name__}.")
    try:
        index_tuple = tuple(indices)
    except TypeError as exc:
        raise TypeError(f"{parameter_name} must be a sequence of integers; got {type(indices).__name__}.") from exc

    normalized_indices = []
    for element_index, value in enumerate(index_tuple):
        if isinstance(value, bool):
            raise TypeError(f"{parameter_name} element {element_index} must be an integer; got {value!r} (bool).")
        try:
            normalized_indices.append(operator.index(value))
        except TypeError as exc:
            raise TypeError(
                f"{parameter_name} element {element_index} must be an integer; got {value!r} ({type(value).__name__})."
            ) from exc
    return tuple(normalized_indices)


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
    """Frozen bidirectional descriptor of backend and public articulation order.

    ``user`` in the field names means the order exposed by the public API.
    ``user_to_backend`` maps a public index to its backend index, while
    ``backend_to_user`` maps a backend index to its public index. The CPU tuples
    and device arrays are complete inverse permutations.

    All name and index sequences have the same length. Each name sequence is
    unique, both device maps are one-dimensional arrays with shape
    ``(num_names,)`` and dtype ``wp.int32``, and the device maps share a Warp
    device. Construction copies each device map to host and compares it with the
    corresponding CPU tuple as part of validation. The frozen dataclass prevents
    field reassignment, but the Warp arrays remain mutable objects and callers
    must treat both device maps as read-only.

    When an articulation constructs its maps, this validation occurs during
    initialization. Hot read and write paths reuse the validated device maps and
    do not synchronize them to the host.

    An explicit ordering equal to backend order still produces a map with
    :attr:`is_identity` set. The default ``None`` ordering is represented by no
    map on the articulation instead.

    Args:
        kind: Mapped articulation element kind.
        backend_names: Names in active backend solver-view order.
        user_names: The same number of names in public API order.
        user_to_backend_indices: CPU permutation from public to backend indices.
        backend_to_user_indices: CPU inverse permutation from backend to public indices.
        user_to_backend: Read-only device permutation from public to backend indices.
        backend_to_user: Read-only device inverse permutation from backend to public indices.
        is_identity: Whether names and both permutations are in identical order.

    Raises:
        TypeError: If either name field is not a sequence of strings, a CPU map
            contains a non-integer value, or is_identity is not a built-in bool.
        ValueError: If a field violates the length, uniqueness, permutation,
            device, or identity invariants.
    """

    kind: Literal["joint", "body"]
    """Mapped articulation element kind."""

    backend_names: tuple[str, ...]
    """Names in active backend solver-view order."""

    user_names: tuple[str, ...]
    """Names in public articulation API order."""

    user_to_backend_indices: tuple[int, ...]
    """One-dimensional CPU map from public index to backend index."""

    backend_to_user_indices: tuple[int, ...]
    """One-dimensional CPU map from backend index to public index."""

    user_to_backend: wp.array(dtype=wp.int32)
    """Read-only public-to-backend device map, shape ``(num_names,)``, dtype ``wp.int32``."""

    backend_to_user: wp.array(dtype=wp.int32)
    """Read-only backend-to-public device map, shape ``(num_names,)``, dtype ``wp.int32``."""

    is_identity: bool
    """Whether names and both index maps are identity permutations."""

    def __post_init__(self) -> None:
        """Validate CPU-side name-map invariants."""
        if self.kind not in {"joint", "body"}:
            raise ValueError(f"ArticulationNameMap kind must be 'joint' or 'body'. Got {self.kind!r}.")
        if type(self.is_identity) is not bool:
            raise TypeError(f"ArticulationNameMap is_identity must be bool; got {type(self.is_identity).__name__}.")

        object.__setattr__(
            self, "backend_names", _coerce_articulation_names(self.backend_names, parameter_name="backend_names")
        )
        object.__setattr__(self, "user_names", _coerce_articulation_names(self.user_names, parameter_name="user_names"))
        object.__setattr__(
            self,
            "user_to_backend_indices",
            _coerce_articulation_indices(self.user_to_backend_indices, parameter_name="user_to_backend_indices"),
        )
        object.__setattr__(
            self,
            "backend_to_user_indices",
            _coerce_articulation_indices(self.backend_to_user_indices, parameter_name="backend_to_user_indices"),
        )

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
            if self.user_names[user_index] != self.backend_names[backend_index]:
                raise ValueError("ArticulationNameMap name and index mappings must agree.")

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
) -> ArticulationNameMap:
    """Build a validated map between backend and public articulation order.

    Args:
        kind: Mapped element kind, either ``"joint"`` or ``"body"``.
        backend_names: Names in active backend solver-view order.
        user_names: Complete permutation in public API order. ``None`` uses
            :paramref:`backend_names` and builds an explicit identity map.
        device: Warp device on which both device index maps are allocated.

    Returns:
        An :class:`ArticulationNameMap` containing CPU index tuples and
        one-dimensional ``wp.int32`` device maps on :paramref:`device`.

    Raises:
        ValueError: If :paramref:`kind` is invalid, either name sequence has
            duplicates, :paramref:`user_names` is not a complete permutation
            of :paramref:`backend_names`, or a name-map invariant is violated.
        TypeError: If either name input is not a sequence of strings.
    """
    backend_names = _coerce_articulation_names(backend_names, parameter_name="backend_names")
    user_names = (
        backend_names if user_names is None else _coerce_articulation_names(user_names, parameter_name="user_names")
    )

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
