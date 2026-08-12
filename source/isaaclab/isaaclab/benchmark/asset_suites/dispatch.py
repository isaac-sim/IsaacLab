# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy provider lookup for backend-specific asset benchmark adapters."""

from __future__ import annotations

import importlib
from dataclasses import replace

from .types import AssetBenchmarkAdapter, AssetPhysicsFamily, AssetPhysicsVariant

_ASSET_COMPONENTS = frozenset({"articulation", "rigid_object", "rigid_object_collection"})
_PHYSICS_VARIANT_FAMILIES: dict[str, AssetPhysicsFamily] = {
    "physx": "physx",
    "isaacsim_physx": "physx",
    "newton_mjwarp": "newton",
    "newton_kamino": "newton",
    "ovphysx": "ovphysx",
}
_ASSET_ADAPTER_MODULES = {
    "physx": "isaaclab_physx.benchmark.assets",
    "newton": "isaaclab_newton.benchmark.assets",
    "ovphysx": "isaaclab_ovphysx.benchmark.assets",
}


def get_asset_physics_family(physics_variant: AssetPhysicsVariant) -> AssetPhysicsFamily:
    """Resolve an exact asset physics selector to its provider package family."""
    try:
        return _PHYSICS_VARIANT_FAMILIES[physics_variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported asset physics selector: {physics_variant!r}") from exc


def get_asset_benchmark_adapter(physics_variant: AssetPhysicsVariant, component: str) -> AssetBenchmarkAdapter:
    """Load an exact physics variant's adapter for one asset component."""
    if component not in _ASSET_COMPONENTS:
        raise ValueError(f"Unsupported asset component: {component!r}")
    family = get_asset_physics_family(physics_variant)
    module = importlib.import_module(_ASSET_ADAPTER_MODULES[family])
    adapter = module.get_asset_benchmark_adapter(component)
    return replace(adapter, physics_variant=physics_variant)
