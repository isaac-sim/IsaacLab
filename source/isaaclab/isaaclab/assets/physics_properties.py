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
from functools import lru_cache


@dataclass(frozen=True)
class UsdAttribute:
    """One exact USD target bound to a data property.

    ``schema`` names a registered schema, optionally followed by ``:{axis}`` for
    multi-apply instances. Unregistered extensions require an explicit ``type_name``.
    ``angular_power`` converts SI angular values by (degrees/radian)**power;
    linear values are unchanged. ``component`` selects a vector element. ``axes``
    restricts a binding to matching joint axes. ``require_uniform`` rejects conflicting
    writes to a shared target. ``replaces`` removes obsolete aliases before authoring.
    """

    attribute: str
    schema: str | None = None
    angular_power: int = 0
    component: int | None = None
    type_name: str | None = None
    axes: tuple[str, ...] | None = None
    require_uniform: bool = False
    replaces: tuple[str, ...] = ()


def usd_field(*targets: UsdAttribute, extend: bool = False) -> Callable:
    """Bind USD targets to the decorated property's getter without wrapping it.

    Place below ``@property``. Getter overrides inherit the nearest declaration;
    decorated overrides replace it, or append targets with ``extend=True``.
    An empty declaration requires a concrete backend to supply its semantics.
    """

    def bind(getter: Callable) -> Callable:
        getter._usd_field = (targets, extend)
        return getter

    return bind


@lru_cache
def usd_fields(data_type: type) -> dict[str, tuple[UsdAttribute, ...]]:
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
                targets, extend = declaration
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
