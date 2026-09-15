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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
    from isaaclab.assets.rigid_object_collection.base_rigid_object_collection import BaseRigidObjectCollection


def _apply_inertia_diagonal_offsets(asset: BaseRigidObject | BaseRigidObjectCollection, offsets: list[float]) -> None:
    """Apply configured isotropic inertia additions in public body order after initialization."""
    import math

    if any(not math.isfinite(value) or value < 0.0 for value in offsets):
        raise ValueError("inertia_diagonal_offset must be finite and nonnegative.")
    if not any(offsets):
        return
    import torch

    inertias = asset.data.body_inertia.torch.clone()
    values = torch.tensor(offsets, dtype=inertias.dtype, device=inertias.device)
    # Adding a multiple of the identity is invariant to the backend's inertial frame.
    inertias[..., (0, 4, 8)] += values[None, :, None]
    asset.set_inertias_index(inertias=inertias)


@dataclass(frozen=True)
class UsdAttribute:
    """One exact USD target bound to a data property.

    ``schema`` names a registered schema, optionally followed by ``:{axis}`` for
    multi-apply instances. Unregistered extensions require an explicit ``type_name``.
    ``angular_power`` converts SI angular values by (degrees/radian)**power;
    linear values are unchanged. ``component`` selects a vector element.
    """

    attribute: str
    schema: str | None = None
    angular_power: int = 0
    component: int | None = None
    type_name: str | None = None


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
