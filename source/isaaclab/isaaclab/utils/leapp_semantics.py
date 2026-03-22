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


@dataclass(frozen=True)
class LeappTensorSemantics:
    """Semantic metadata attached directly to a raw tensor-producing function."""

    kind: Any = None
    element_names: list[str] | list[list[str]] | None = None
    element_names_source: str | None = None


XYZ_ELEMENT_NAMES: list[str] = ["x", "y", "z"]
QUAT_WXYZ_ELEMENT_NAMES: list[str] = ["qw", "qx", "qy", "qz"]
POSE7_ELEMENT_NAMES: list[str] = ["x", "y", "z", "qw", "qx", "qy", "qz"]
WRENCH6_ELEMENT_NAMES: list[str] = ["fx", "fy", "fz", "tx", "ty", "tz"]


def leapp_tensor_semantics(
    *,
    kind: Any = None,
    element_names: list[str] | list[list[str]] | None = None,
    element_names_source: str | None = None,
) -> Callable:
    """Attach LEAPP semantic metadata to a raw tensor-producing function."""

    semantics = LeappTensorSemantics(
        kind=kind,
        element_names=element_names,
        element_names_source=element_names_source,
    )

    def _apply(func: Callable) -> Callable:
        func._leapp_semantics = semantics
        return func

    return _apply


def _select_element_names(names: list[str] | None, indices: Any = None) -> list[str] | None:
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


def resolve_leapp_element_names(semantics: LeappTensorSemantics | None, data_self) -> list | None:
    """Resolve element names from attached semantics and a tensor-producing object."""
    if semantics is None:
        return None
    if semantics.element_names is not None:
        return semantics.element_names

    source = semantics.element_names_source
    if source == "joint_names":
        return _select_element_names(
            getattr(data_self, "joint_names", getattr(data_self, "_joint_names", None)),
            getattr(data_self, "_joint_ids", None),
        )
    if source == "body_names":
        return _select_element_names(
            getattr(data_self, "body_names", getattr(data_self, "_body_names", None)),
            getattr(data_self, "_body_ids", None),
        )
    if source == "body_xyz":
        body_names = _select_element_names(
            getattr(data_self, "body_names", getattr(data_self, "_body_names", None)),
            getattr(data_self, "_body_ids", None),
        )
        if body_names is None:
            return None
        return [body_names, XYZ_ELEMENT_NAMES]
    if source == "body_pose":
        body_names = _select_element_names(
            getattr(data_self, "body_names", getattr(data_self, "_body_names", None)),
            getattr(data_self, "_body_ids", None),
        )
        if body_names is None:
            return None
        return [body_names, POSE7_ELEMENT_NAMES]
    if source == "body_quat":
        body_names = _select_element_names(
            getattr(data_self, "body_names", getattr(data_self, "_body_names", None)),
            getattr(data_self, "_body_ids", None),
        )
        if body_names is None:
            return None
        return [body_names, QUAT_WXYZ_ELEMENT_NAMES]
    if source == "body_wrench":
        body_names = _select_element_names(
            getattr(data_self, "body_names", getattr(data_self, "_body_names", None)),
            getattr(data_self, "_body_ids", None),
        )
        if body_names is None:
            return None
        return [body_names, WRENCH6_ELEMENT_NAMES]
    if source == "pose7":
        return POSE7_ELEMENT_NAMES
    if source == "xyz":
        return XYZ_ELEMENT_NAMES
    if source == "quat_wxyz":
        return QUAT_WXYZ_ELEMENT_NAMES
    return None
