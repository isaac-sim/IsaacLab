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

import warp as wp

from ..array import convert_to_torch

try:
    from leapp import Axis, AxisKind, ExpressionFrame, InputKindEnum, Kind, OutputKindEnum
except ImportError:

    class _LeappEnumSentinel:
        """Stand-in for optional LEAPP enums and axis constructors.

        Any attribute access returns ``None`` so that
        ``@leapp_tensor_semantics(kind=Kind.FRAME_POSE)``
        silently stores ``kind=None`` instead of crashing at import time.
        Axis construction also returns ``None``. Real descriptors and enum
        values are only needed at export time, when leapp
        *is* guaranteed to be available.
        """

        def __getattr__(self, name: str):
            return None

        def __call__(self, *args, **kwargs):
            return None

    Axis = _LeappEnumSentinel()  # type: ignore[assignment,misc]
    AxisKind = _LeappEnumSentinel()  # type: ignore[assignment,misc]
    InputKindEnum = _LeappEnumSentinel()  # type: ignore[assignment,misc]
    OutputKindEnum = _LeappEnumSentinel()  # type: ignore[assignment,misc]
    Kind = _LeappEnumSentinel()  # type: ignore[assignment,misc]
    ExpressionFrame = _LeappEnumSentinel()  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class LeappTensorSemantics:
    """Semantic metadata attached directly to a raw tensor-producing function."""

    kind: Any = None
    element_names: list[str] | list[list[str]] | None = None
    element_names_resolver: Callable[[Any], list | None] | None = None
    const: bool = False
    axes: list[Axis | None] | None = None
    axes_resolver: Callable[[Any, Any], list[Axis | None] | None] | None = None
    is_setpoint: bool | None = None
    reference: str | None = None
    expressed_in: Any = None


XYZ_ELEMENT_NAMES = ["x", "y", "z"]
QUAT_XYZW_ELEMENT_NAMES = ["qx", "qy", "qz", "qw"]
POSE7_ELEMENT_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
POSE6_ELEMENT_NAMES = ["x", "y", "z", "angular_x", "angular_y", "angular_z"]
WRENCH6_ELEMENT_NAMES = ["fx", "fy", "fz", "tx", "ty", "tz"]

XYZ_COMPONENT_AXIS: Axis = Axis(AxisKind.COMPONENT, names=XYZ_ELEMENT_NAMES.copy())
QUAT_XYZW_COMPONENT_AXIS: Axis = Axis(AxisKind.COMPONENT, names=QUAT_XYZW_ELEMENT_NAMES.copy())
POSE7_COMPONENT_AXIS: Axis = Axis(AxisKind.COMPONENT, names=POSE7_ELEMENT_NAMES.copy())
POSE6_COMPONENT_AXIS: Axis = Axis(AxisKind.COMPONENT, names=POSE6_ELEMENT_NAMES.copy())
WRENCH6_COMPONENT_AXIS: Axis = Axis(AxisKind.COMPONENT, names=WRENCH6_ELEMENT_NAMES.copy())
LINEAR_VELOCITY_COMPONENT_AXIS: Axis = Axis(AxisKind.COMPONENT, names=["x", "y", "z"])
ANGULAR_VELOCITY_COMPONENT_AXIS: Axis = Axis(AxisKind.COMPONENT, names=["angular_x", "angular_y", "angular_z"])
TWIST_COMPONENT_AXIS: Axis = Axis(AxisKind.COMPONENT, names=["x", "y", "z", "angular_x", "angular_y", "angular_z"])

_VELOCITY_COMPONENT_ALIASES = {
    "lin_vel_x": "x",
    "lin_vel_y": "y",
    "lin_vel_z": "z",
    "ang_vel_x": "angular_x",
    "ang_vel_y": "angular_y",
    "ang_vel_z": "angular_z",
}


def select_element_names(names: list[str] | None, indices: Any = None) -> list[str] | None:
    """Select element names using optional runtime indices."""
    if names is None:
        return None
    if indices is None or indices == slice(None):
        return list(names)
    if isinstance(indices, slice):
        return list(names[indices])
    if isinstance(indices, wp.array):
        indices = convert_to_torch(indices)
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
    element_names_resolver: Callable[[Any], list | None] | None = None,
    axes: list[Axis | None] | None = None,
    axes_resolver: Callable[[Any, Any], list[Axis | None] | None] | None = None,
    const: bool = False,
    is_setpoint: bool | None = None,
    reference: str | None = None,
    expressed_in: Any = None,
) -> Callable:
    """Attach deferred LEAPP metadata without wrapping or evaluating the function.

    ``element_names`` labels non-system dimensions; a flat list labels a vector.
    Frame/vector quantities use a trailing component axis, while other named
    dimensions use element axes. Use explicit ``axes`` for other layouts.
    ``axes`` and ``axes_resolver`` take precedence over element names.
    Name resolvers receive the data object; axis resolvers also receive the Torch
    tensor. Resolution happens during export, which supplies the port name.
    """
    semantics = LeappTensorSemantics(
        kind=kind,
        element_names=element_names,
        element_names_resolver=element_names_resolver,
        axes=axes,
        axes_resolver=axes_resolver,
        const=const,
        is_setpoint=is_setpoint,
        reference=reference,
        expressed_in=expressed_in,
    )

    def _apply(func: Callable) -> Callable:
        func._leapp_semantics = semantics
        return func

    return _apply


def resolve_leapp_element_names(semantics: LeappTensorSemantics | None, data_self) -> list | None:
    """Resolve simple labels without requiring LEAPP or evaluating the data tensor."""
    if semantics is None:
        return None
    if semantics.element_names is not None:
        return semantics.element_names
    if semantics.element_names_resolver is not None:
        return semantics.element_names_resolver(data_self)
    return None


def resolve_semantics_axes(semantics: LeappTensorSemantics, data_self, ref) -> list[Axis | None] | None:
    """Resolve axes for environment-batched data and actuator tensors.

    Without a declared layout, only the environment/system dimension is known.
    Explicit axes retain the caller's layout, including unspecified dimensions.
    """
    axes = semantics.axes
    if axes is None and semantics.axes_resolver is not None:
        axes = semantics.axes_resolver(data_self, ref)
    if axes is None:
        names = resolve_leapp_element_names(semantics, data_self)
        if names is not None:
            names = [names] if not names or isinstance(names[0], str) else names
            if len(names) != ref.ndim - 1:
                raise ValueError("element_names must label each non-system dimension; use axes for other layouts")
            kind = getattr(semantics.kind, "value", semantics.kind)
            components = isinstance(kind, str) and (
                kind.startswith(("frame/", "state/body/", "command/body/", "target/body/"))
                or kind in ("vector3d", "state/vector3d", "state/wrench", "point_cloud")
            )
            if kind in ("frame/twist", "frame/linear_velocity", "frame/angular_velocity"):
                names = [*names[:-1], [_VELOCITY_COMPONENT_ALIASES.get(label, label) for label in names[-1]]]
            axes = [Axis(AxisKind.SYSTEM)] + [
                Axis(AxisKind.COMPONENT if components and index == len(names) - 1 else AxisKind.ELEMENT, names=labels)
                for index, labels in enumerate(names)
            ]
    if axes is None and ref.ndim:
        axes = [Axis(AxisKind.SYSTEM)] + [None] * (ref.ndim - 1)
    return axes


# ── Predefined axis resolvers ─────────────────────────────────────


def joint_axes_resolver(data_self, ref) -> list[Axis | None] | None:
    """Label the joint dimension using the selected runtime joint order."""
    names = joint_names_resolver(data_self)
    return [Axis(AxisKind.SYSTEM), Axis(AxisKind.ELEMENT, names=names)] if names is not None else None


def joint_names_resolver(data_self) -> list[str] | None:
    """Resolve joint labels in the selected runtime order."""
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


def _compound_axes_resolver(outer_fn: Callable, component_axis: Axis, *, element_axis: int = 1) -> Callable:
    """Label systems, the body/frame dimension, and trailing components while retaining other dimensions."""

    def resolver(data_self, ref) -> list[Axis | None] | None:
        axes = [Axis(AxisKind.SYSTEM)] + [None] * (ref.ndim - 2) + [component_axis]
        if ref.ndim > 2:
            axes[element_axis] = Axis(AxisKind.ELEMENT, names=outer_fn(data_self))
        return axes

    return resolver


def _target_frame_names(data_self) -> list[str] | None:
    names = getattr(data_self, "target_frame_names", None)
    return list(names) if names is not None else None


def _compound_resolver(outer_fn: Callable, inner_names: list[str]) -> Callable:
    """Preserve the list-returning resolver interface for simple annotations."""

    def resolver(data_self) -> list | None:
        outer = outer_fn(data_self)
        return [outer, inner_names] if outer else None

    return resolver


body_xyz_resolver = _compound_resolver(body_names_resolver, XYZ_ELEMENT_NAMES)
body_pose_resolver = _compound_resolver(body_names_resolver, POSE7_ELEMENT_NAMES)
body_pose6_resolver = _compound_resolver(body_names_resolver, POSE6_ELEMENT_NAMES)
body_quat_resolver = _compound_resolver(body_names_resolver, QUAT_XYZW_ELEMENT_NAMES)
body_wrench_resolver = _compound_resolver(body_names_resolver, WRENCH6_ELEMENT_NAMES)
target_frame_xyz_resolver = _compound_resolver(_target_frame_names, XYZ_ELEMENT_NAMES)
target_frame_quat_resolver = _compound_resolver(_target_frame_names, QUAT_XYZW_ELEMENT_NAMES)
target_frame_pose_resolver = _compound_resolver(_target_frame_names, POSE7_ELEMENT_NAMES)


body_xyz_axes_resolver = _compound_axes_resolver(body_names_resolver, XYZ_COMPONENT_AXIS)
body_xyz_history_axes_resolver = _compound_axes_resolver(body_names_resolver, XYZ_COMPONENT_AXIS, element_axis=2)
body_pose_axes_resolver = _compound_axes_resolver(body_names_resolver, POSE7_COMPONENT_AXIS)
body_pose6_axes_resolver = _compound_axes_resolver(body_names_resolver, POSE6_COMPONENT_AXIS)
body_linear_velocity_axes_resolver = _compound_axes_resolver(body_names_resolver, LINEAR_VELOCITY_COMPONENT_AXIS)
body_angular_velocity_axes_resolver = _compound_axes_resolver(body_names_resolver, ANGULAR_VELOCITY_COMPONENT_AXIS)
body_twist_axes_resolver = _compound_axes_resolver(body_names_resolver, TWIST_COMPONENT_AXIS)
body_quat_axes_resolver = _compound_axes_resolver(body_names_resolver, QUAT_XYZW_COMPONENT_AXIS)
body_wrench_axes_resolver = _compound_axes_resolver(body_names_resolver, WRENCH6_COMPONENT_AXIS)
target_frame_xyz_axes_resolver = _compound_axes_resolver(_target_frame_names, XYZ_COMPONENT_AXIS)
target_frame_quat_axes_resolver = _compound_axes_resolver(_target_frame_names, QUAT_XYZW_COMPONENT_AXIS)
target_frame_pose_axes_resolver = _compound_axes_resolver(_target_frame_names, POSE7_COMPONENT_AXIS)
