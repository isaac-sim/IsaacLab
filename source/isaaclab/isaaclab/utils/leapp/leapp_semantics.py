# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LEAPP semantic metadata helpers for raw tensor-producing functions."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

try:
    from leapp import InputKindEnum, OutputKindEnum
except ImportError:

    class _LeappEnumSentinel:
        """Stand-in when leapp is not installed.

        Any attribute access returns ``None`` so that
        ``@leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE)``
        silently stores ``kind=None`` instead of crashing at import time.
        The real enum values are only needed at export time, when leapp
        *is* guaranteed to be available.
        """

        def __getattr__(self, name: str):
            return None

    InputKindEnum = _LeappEnumSentinel()  # type: ignore[assignment,misc]
    OutputKindEnum = _LeappEnumSentinel()  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class LeappTensorSemantics:
    """Semantic metadata attached directly to a raw tensor-producing function."""

    kind: Any = None
    element_names: list[str] | list[list[str]] | None = None
    element_names_resolver: Callable | None = None
    const: bool = False


XYZ_ELEMENT_NAMES: list[str] = ["x", "y", "z"]
QUAT_XYZW_ELEMENT_NAMES: list[str] = ["qx", "qy", "qz", "qw"]
POSE7_ELEMENT_NAMES: list[str] = ["x", "y", "z", "qx", "qy", "qz", "qw"]
POSE6_ELEMENT_NAMES: list[str] = ["x", "y", "z", "angular_x", "angular_y", "angular_z"]
WRENCH6_ELEMENT_NAMES: list[str] = ["fx", "fy", "fz", "tx", "ty", "tz"]


def select_element_names(names: list[str] | None, indices: Any = None) -> list[str] | None:
    """Select element names using optional runtime indices."""
    if names is None:
        return None
    if indices is None or indices == slice(None):
        return list(names)
    if isinstance(indices, slice):
        return list(names[indices])
    with suppress(AttributeError):
        indices = indices.tolist()
    if isinstance(indices, (list, tuple)):
        return [names[int(index)] for index in indices]
    if isinstance(indices, int):
        return [names[indices]]
    return None


def leapp_tensor_semantics(
    *,
    kind: Any = None,
    element_names: list[str] | list[list[str]] | None = None,
    element_names_resolver: Callable | None = None,
    const: bool = False,
) -> Callable:
    """Attach LEAPP semantic metadata to a raw tensor-producing function."""

    semantics = LeappTensorSemantics(
        kind=kind,
        element_names=element_names,
        element_names_resolver=element_names_resolver,
        const=const,
    )

    def _apply(func: Callable) -> Callable:
        func._leapp_semantics = semantics
        return func

    return _apply


def resolve_leapp_element_names(semantics: LeappTensorSemantics | None, data_self) -> list | None:
    """Resolve element names from attached semantics and a tensor-producing object."""
    if semantics is None:
        return None
    if semantics.element_names is not None:
        return semantics.element_names
    if semantics.element_names_resolver is not None:
        return semantics.element_names_resolver(data_self)
    return None


# ── Predefined element-name resolvers ─────────────────────────────


def joint_names_resolver(data_self) -> list[str] | None:
    """Resolve joint element names from the data object at trace time."""
    return select_element_names(
        getattr(data_self, "joint_names", getattr(data_self, "_joint_names", None)),
        getattr(data_self, "_joint_ids", None),
    )


def body_names_resolver(data_self) -> list[str] | None:
    """Resolve body element names from the data object at trace time."""
    return select_element_names(
        getattr(data_self, "body_names", getattr(data_self, "_body_names", None)),
        getattr(data_self, "_body_ids", None),
    )


def _compound_resolver(outer_fn: Callable, inner_names: list[str]) -> Callable:
    """Build a 2D resolver: ``[outer_names, inner_constant_names]``."""

    def resolver(data_self) -> list | None:
        outer = outer_fn(data_self)
        return [outer, inner_names] if outer else None

    return resolver


def _target_frame_names(data_self) -> list[str] | None:
    names = getattr(data_self, "target_frame_names", None)
    return list(names) if names is not None else None


body_xyz_resolver = _compound_resolver(body_names_resolver, XYZ_ELEMENT_NAMES)
body_pose_resolver = _compound_resolver(body_names_resolver, POSE7_ELEMENT_NAMES)
body_pose6_resolver = _compound_resolver(body_names_resolver, POSE6_ELEMENT_NAMES)
body_quat_resolver = _compound_resolver(body_names_resolver, QUAT_XYZW_ELEMENT_NAMES)
body_wrench_resolver = _compound_resolver(body_names_resolver, WRENCH6_ELEMENT_NAMES)
target_frame_xyz_resolver = _compound_resolver(_target_frame_names, XYZ_ELEMENT_NAMES)
target_frame_quat_resolver = _compound_resolver(_target_frame_names, QUAT_XYZW_ELEMENT_NAMES)
target_frame_pose_resolver = _compound_resolver(_target_frame_names, POSE7_ELEMENT_NAMES)
