# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative shared suites for backend asset micro-benchmarks."""

from .cli import run_asset_benchmark_cli
from .dispatch import get_asset_benchmark_adapter, get_asset_physics_family
from .runner import resolve_method_benchmarks, resolve_property_benchmarks, run_asset_benchmark
from .suites import get_asset_benchmark_suite
from .types import (
    AssetBenchmarkAdapter,
    AssetBenchmarkRequest,
    AssetBenchmarkSuite,
    AssetBenchmarkTargets,
    AssetMethodSpec,
    AssetPropertySpec,
    AssetPhysicsFamily,
    AssetPhysicsVariant,
)

__all__ = [
    "AssetBenchmarkAdapter",
    "AssetBenchmarkRequest",
    "AssetBenchmarkSuite",
    "AssetBenchmarkTargets",
    "AssetMethodSpec",
    "AssetPropertySpec",
    "AssetPhysicsFamily",
    "AssetPhysicsVariant",
    "get_asset_benchmark_adapter",
    "get_asset_physics_family",
    "get_asset_benchmark_suite",
    "resolve_method_benchmarks",
    "resolve_property_benchmarks",
    "run_asset_benchmark",
    "run_asset_benchmark_cli",
]
