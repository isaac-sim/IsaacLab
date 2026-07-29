# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable declaration for lazy package-local asset benchmark adapters."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from ..method_benchmark import MethodBenchmarkRunnerConfig
from .types import (
    AssetBenchmarkRequest,
    AssetBenchmarkTargets,
    AssetComponent,
    AssetPhysicsFamily,
    AssetPhysicsVariant,
    InputGenerator,
    WorkloadKey,
)


@dataclass(frozen=True)
class PackageAssetBenchmarkAdapter:
    """Describe a backend suite while deferring heavyweight runtime imports."""

    physics: AssetPhysicsFamily
    physics_variant: AssetPhysicsVariant
    component: AssetComponent
    artifact_prefix: str
    runtime_module: str
    capabilities: frozenset[str]
    supported_properties: frozenset[str]
    default_num_bodies: int
    default_num_joints: int
    generator_overrides: Mapping[WorkloadKey, InputGenerator] = field(default_factory=dict)
    property_dependency_overrides: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def method_benchmark_name(self) -> str:
        """Return the historical method workflow name."""
        return f"{self.artifact_prefix}{self.component}_benchmark"

    @property
    def data_benchmark_name(self) -> str:
        """Return the historical data workflow name."""
        return f"{self.artifact_prefix}{self.component}_data_benchmark"

    def method_config(self, request: AssetBenchmarkRequest) -> MethodBenchmarkRunnerConfig:
        """Return the combined command's requested method configuration."""
        return request.config

    def data_config(self, request: AssetBenchmarkRequest) -> MethodBenchmarkRunnerConfig:
        """Return the combined command's requested data configuration."""
        return request.config

    def open_targets(
        self,
        request: AssetBenchmarkRequest,
        method_config: MethodBenchmarkRunnerConfig,
        data_config: MethodBenchmarkRunnerConfig,
    ) -> AbstractContextManager[AssetBenchmarkTargets]:
        """Load and invoke this backend's runtime target factory."""
        runtime = importlib.import_module(self.runtime_module)
        return runtime.open_asset_targets(self, request, method_config, data_config)
