# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy provider lookup for asset micro-benchmark adapters."""

import importlib

from .types import AssetBenchmarkAdapter

_ASSET_COMPONENTS = {"articulation", "rigid_object", "rigid_object_collection"}
_ASSET_ADAPTER_MODULES = {
    "physx": "isaaclab_physx.benchmark.assets",
    "newton": "isaaclab_newton.benchmark.assets",
    "ovphysx": "isaaclab_ovphysx.benchmark.assets",
}


def get_asset_benchmark_adapter(physics: str, component: str) -> AssetBenchmarkAdapter:
    """Load the selected backend's adapter for one asset component."""
    if component not in _ASSET_COMPONENTS:
        raise ValueError(f"Unsupported asset component: {component!r}")
    try:
        module_name = _ASSET_ADAPTER_MODULES[physics]
    except KeyError as exc:
        raise ValueError(f"Unsupported physics backend: {physics!r}") from exc
    module = importlib.import_module(module_name)
    return module.get_asset_benchmark_adapter(component)
