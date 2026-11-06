# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared orchestration for backend-adapted asset micro-benchmark suites."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

from ..method_benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunner
from .generators import build_fill_benchmarks
from .suites import get_asset_benchmark_suite
from .types import AssetBenchmarkAdapter, AssetBenchmarkRequest, AssetBenchmarkSuite


def resolve_method_benchmarks(
    suite: AssetBenchmarkSuite,
    adapter: AssetBenchmarkAdapter,
) -> list[MethodBenchmarkDefinition]:
    """Resolve shared method specs against backend capabilities and overrides."""
    definitions: list[MethodBenchmarkDefinition] = []
    method_name_overrides = getattr(adapter, "method_name_overrides", {})
    generator_replacements = getattr(adapter, "generator_replacements", {})
    for spec in suite.methods:
        if spec.requires is not None and spec.requires not in adapter.capabilities:
            continue
        replacement = generator_replacements.get(spec.method_name)
        if replacement is not None:
            generators = dict(replacement)
        else:
            generators = dict(spec.input_generators)
            for mode in tuple(generators):
                override = adapter.generator_overrides.get((spec.method_name, mode))
                if override is not None:
                    generators[mode] = override
        definitions.append(
            MethodBenchmarkDefinition(
                name=spec.name,
                method_name=method_name_overrides.get(spec.method_name, spec.method_name),
                input_generators=generators,
                category=spec.category,
                prepare_target=spec.prepare_target,
            )
        )
    return definitions


def resolve_property_benchmarks(
    suite: AssetBenchmarkSuite,
    adapter: AssetBenchmarkAdapter,
    target_data: object,
) -> tuple[list[str], dict[str, list[str]]]:
    """Resolve supported properties and validate their runtime surface."""
    properties: list[str] = []
    dependencies: dict[str, list[str]] = {}
    for spec in suite.properties:
        if spec.name not in adapter.supported_properties:
            continue
        if inspect.getattr_static(target_data, spec.name, None) is None:
            raise RuntimeError(
                f"{adapter.physics} {suite.component} data target is missing declared property {spec.name!r}"
            )
        properties.append(spec.name)
        resolved = adapter.property_dependency_overrides.get(spec.name, spec.dependencies)
        if resolved:
            dependencies[spec.name] = list(resolved)
    return properties, dependencies


def run_asset_benchmark(
    request: AssetBenchmarkRequest,
    adapter: AssetBenchmarkAdapter,
    *,
    runner_factory: Callable[..., MethodBenchmarkRunner] = MethodBenchmarkRunner,
) -> tuple[Path, ...]:
    """Run one asset concept's method and data suites as two compatible artifacts."""
    if request.physics_variant != adapter.physics_variant:
        raise ValueError(
            f"Request physics variant {request.physics_variant!r} does not match adapter variant "
            f"{adapter.physics_variant!r}"
        )
    suite = get_asset_benchmark_suite(adapter.component)
    method_config = adapter.method_config(request)
    data_config = adapter.data_config(request)
    output_paths: list[Path] = []

    with adapter.open_targets(request, method_config, data_config) as targets:
        definitions = resolve_method_benchmarks(suite, adapter)
        method_runner = runner_factory(
            benchmark_name=adapter.method_benchmark_name,
            config=method_config,
            backend_type=request.formatter_type,
            output_path=str(request.output_path),
            use_recorders=True,
            physics_variant=request.physics_variant,
        )
        method_runner.run_benchmarks(definitions, targets.method_target)
        fill_benchmarks = build_fill_benchmarks(definitions, capabilities=adapter.capabilities)
        if fill_benchmarks:
            print("\n" + "=" * 80)
            print("Fill-Ratio Benchmarks (env_ids at 5%, 95%, 100% fill)")
            print("=" * 80)
            method_runner.run_benchmarks(fill_benchmarks, targets.method_target)
        output_paths.extend(method_runner.finalize())

        properties, dependencies = resolve_property_benchmarks(suite, adapter, targets.data_target)
        data_runner = runner_factory(
            benchmark_name=adapter.data_benchmark_name,
            config=data_config,
            backend_type=request.formatter_type,
            output_path=str(request.output_path),
            use_recorders=True,
            physics_variant=request.physics_variant,
        )
        data_runner.run_property_benchmarks(
            target_data=targets.data_target,
            properties=properties,
            gen_mock_data=targets.refresh_data,
            dependencies=dependencies,
            category="property",
        )
        output_paths.extend(data_runner.finalize())

    return tuple(output_paths)
