# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal contracts for declarative asset micro-benchmark suites."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..method_benchmark import MethodBenchmarkRunnerConfig, PrepareTarget

AssetComponent = Literal["articulation", "rigid_object", "rigid_object_collection"]
AssetPhysicsFamily = Literal["physx", "newton", "ovphysx"]
AssetPhysicsVariant = Literal["physx", "isaacsim_physx", "newton_mjwarp", "newton_kamino", "ovphysx"]
InputGenerator = Callable[[MethodBenchmarkRunnerConfig], dict[str, object]]
WorkloadKey = tuple[str, str]


@dataclass(frozen=True)
class AssetMethodSpec:
    """Declarative definition of one asset method benchmark."""

    name: str
    method_name: str
    input_generators: Mapping[str, InputGenerator]
    category: str
    requires: str | None = None
    prepare_target: PrepareTarget | None = None


@dataclass(frozen=True)
class AssetPropertySpec:
    """Declarative definition of one asset data-property benchmark."""

    name: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssetBenchmarkSuite:
    """Shared method and property definitions for one asset concept."""

    component: AssetComponent
    methods: tuple[AssetMethodSpec, ...]
    properties: tuple[AssetPropertySpec, ...]


@dataclass(frozen=True)
class AssetBenchmarkRequest:
    """One backend-specific execution request for a shared asset suite."""

    config: MethodBenchmarkRunnerConfig
    physics_variant: AssetPhysicsVariant
    formatter_type: str | tuple[str, ...]
    output_path: Path
    check_shapes: bool = True
    launcher_args: object | None = None


@dataclass
class AssetBenchmarkTargets:
    """Backend objects used by the method and data benchmark phases."""

    method_target: object
    data_target: object
    refresh_data: Callable[[MethodBenchmarkRunnerConfig], None]


class AssetBenchmarkAdapter(Protocol):
    """Backend adapter consumed by shared asset-suite orchestration."""

    physics: AssetPhysicsFamily
    physics_variant: AssetPhysicsVariant
    component: AssetComponent
    method_benchmark_name: str
    data_benchmark_name: str
    capabilities: frozenset[str]
    default_num_bodies: int
    default_num_joints: int
    method_name_overrides: Mapping[str, str]
    generator_overrides: Mapping[WorkloadKey, InputGenerator]
    generator_replacements: Mapping[str, Mapping[str, InputGenerator]]
    supported_properties: frozenset[str]
    property_dependency_overrides: Mapping[str, tuple[str, ...]]

    def method_config(self, request: AssetBenchmarkRequest) -> MethodBenchmarkRunnerConfig:
        """Return the method-phase runner configuration."""

    def data_config(self, request: AssetBenchmarkRequest) -> MethodBenchmarkRunnerConfig:
        """Return the data-phase runner configuration."""

    def open_targets(
        self,
        request: AssetBenchmarkRequest,
        method_config: MethodBenchmarkRunnerConfig,
        data_config: MethodBenchmarkRunnerConfig,
    ) -> AbstractContextManager[AssetBenchmarkTargets]:
        """Open backend targets and all required runtime/patch state."""
