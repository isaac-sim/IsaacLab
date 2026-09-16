# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical property contracts shared by asset initialization and USD export.

Array rows follow the public asset body/joint order, never USD traversal order. Backends
must supply the corresponding prim identities separately. Geometry, materials, filtering,
tendons and other spawn schemas remain owned by the authored USD stage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class UsdAttribute:
    """One exact USD target bound to a data property.

    ``schema`` names a registered schema, optionally followed by ``:{axis}`` for
    multi-apply instances. Unregistered extensions require an explicit ``type_name``.
    ``component`` selects a vector element and ``output`` selects a representation
    transform result. ``axes``
    restricts a binding to matching joint axes. ``require_uniform`` rejects conflicting
    writes to a shared target. ``replaces`` removes obsolete aliases before authoring.
    ``condition`` names a boolean data property that identifies applicable rows.
    """

    attribute: str
    schema: str | None = None
    component: int | slice | None = None
    output: str | None = None
    type_name: str | None = None
    axes: tuple[str, ...] | None = None
    require_uniform: bool = False
    replaces: tuple[str, ...] = ()
    condition: str | None = None


def usd_field(
    *targets: UsdAttribute,
    extend: bool = False,
    scope: str = "joint",
    transform: Callable | None = None,
    inputs: tuple[str, ...] = (),
    angular_conversion: Callable | None = None,
) -> Callable:
    """Bind USD targets to the decorated property's getter without wrapping it.

    Place below ``@property``. Getter overrides inherit the nearest declaration;
    decorated overrides replace it, or append targets with ``extend=True``.
    An empty declaration requires a concrete backend to supply its semantics.
    """

    def bind(getter: Callable) -> Callable:
        getter._usd_field = (targets, extend, scope, transform, inputs, angular_conversion)
        return getter

    return bind


@lru_cache
def usd_fields(data_type: type, scope: str | None = None) -> dict[str, tuple[UsdAttribute, ...]]:
    """Discover inherited declarations statically, without invoking any getter."""
    result = {}
    for base in reversed(data_type.__mro__):
        for name, prop in vars(base).items():
            if not isinstance(prop, property):
                if name in result:
                    raise NotImplementedError(f"USD data property {name} was shadowed by a non-property.")
                continue
            declaration = getattr(prop.fget, "_usd_field", None)
            if declaration is not None:
                targets, extend, group, _, _, _ = declaration
                if scope is not None and scope != group:
                    continue
                result[name] = result.get(name, ()) + targets if extend else targets
    for name, targets in result.items():
        if not targets:
            raise NotImplementedError(f"Missing backend USD declaration for {data_type.__name__}.{name}.")
    return result


def read_usd_array(prim, target: UsdAttribute):
    """Read an optional declared scalar array."""
    import numpy as np

    value = prim.GetAttribute(target.attribute).Get()
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"Expected a scalar array for {prim.GetPath()}.{target.attribute}.")
    return array


class LimitComponent(IntEnum):
    """Named components of a public lower/upper joint-limit pair."""

    LOWER = 0
    UPPER = 1


def radians_to_degrees(value: np.ndarray) -> np.ndarray:
    """Convert angular coordinates or rates from radians to USD degrees."""
    return np.rad2deg(value)


def per_radian_to_per_degree(value: np.ndarray) -> np.ndarray:
    """Convert angular gains from effort per radian to effort per USD degree."""
    return value * (np.pi / 180.0)


def property_metadata(data_type: type, name: str, key: str):
    """Find inherited getter metadata without evaluating a backend property."""
    for owner in data_type.__mro__:
        prop = vars(owner).get(name)
        if isinstance(prop, property) and hasattr(prop.fget, key):
            return getattr(prop.fget, key)
    raise NotImplementedError(f"Missing {key} declaration for {data_type.__name__}.{name}.")


def principal_inertia(value: np.ndarray, frame: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose a link-frame inertia [kg*m²] into principal moments and xyzw axes.

    This is a representation transform with two outputs, not a unit conversion.
    """
    from pxr import Gf

    tensor = np.asarray(value).reshape(3, 3)
    if not np.isfinite(tensor).all() or not np.allclose(tensor, tensor.T, atol=1e-7):
        raise ValueError("Inertia must be finite and symmetric.")
    rotation = Gf.Quatd(float(frame[3]), Gf.Vec3d(*map(float, frame[:3])))
    axes = np.asarray(Gf.Matrix3d(rotation)).T
    principal = axes.T @ tensor @ axes
    # Retain an existing principal frame, including its axis order and repeated eigenvalues.
    if np.allclose(principal, np.diag(np.diag(principal)), atol=1e-7):
        if np.any(np.diag(principal) < -1e-7):
            raise ValueError("Negative inertia.")
        return {"moments": np.diag(principal), "principal_axes": np.asarray(frame)}
    moments, axes = np.linalg.eigh(tensor)
    if np.any(moments < -1e-7):
        raise ValueError("Negative inertia.")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    rotation = Gf.Matrix3d(*map(float, axes.T.flatten())).ExtractRotation().GetQuat()
    return {
        "moments": np.maximum(moments, 0),
        "principal_axes": np.array([*rotation.GetImaginary(), rotation.GetReal()]),
    }
