# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared argument generators for asset micro-benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import warp as wp

from ..method_benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunnerConfig
from .types import InputGenerator

_DIMENSIONS = {
    "instances": "num_instances",
    "bodies": "num_bodies",
    "joints": "num_joints",
}
FILL_RATIOS = (("5pct", 0.05), ("95pct", 0.95), ("100pct", 1.0))


def _resolve_shape(config: MethodBenchmarkRunnerConfig, shape: Sequence[str | int]) -> tuple[int, ...]:
    return tuple(getattr(config, _DIMENSIONS[value]) if isinstance(value, str) else value for value in shape)


def make_indexed_generators(
    tensor_shapes: Mapping[str, Sequence[str | int]],
    index_dimensions: Mapping[str, str],
) -> dict[str, InputGenerator]:
    """Create list-index and tensor-index generators for a workload."""

    def generate(config: MethodBenchmarkRunnerConfig, *, tensor_indices: bool) -> dict[str, object]:
        inputs: dict[str, object] = {
            name: torch.rand(*_resolve_shape(config, shape), device=config.device, dtype=torch.float32)
            for name, shape in tensor_shapes.items()
        }
        for name, dimension in index_dimensions.items():
            count = getattr(config, _DIMENSIONS[dimension])
            inputs[name] = (
                torch.arange(count, dtype=torch.int32, device=config.device) if tensor_indices else list(range(count))
            )
        return inputs

    return {
        "torch_list": lambda config: generate(config, tensor_indices=False),
        "torch_tensor": lambda config: generate(config, tensor_indices=True),
    }


def make_mask_generator(
    tensor_shapes: Mapping[str, Sequence[str | int]],
    mask_dimensions: Mapping[str, str],
) -> InputGenerator:
    """Create one Warp-mask input generator."""

    def generate(config: MethodBenchmarkRunnerConfig) -> dict[str, object]:
        inputs: dict[str, object] = {
            name: torch.rand(*_resolve_shape(config, shape), device=config.device, dtype=torch.float32)
            for name, shape in tensor_shapes.items()
        }
        for name, dimension in mask_dimensions.items():
            count = getattr(config, _DIMENSIONS[dimension])
            inputs[name] = wp.ones((count,), dtype=wp.bool, device=config.device)
        return inputs

    return generate


def make_scaled_tensor_generator(base_generator: InputGenerator, field_name: str, scale: float) -> InputGenerator:
    """Wrap a generator so one tensor field uses the requested scale."""

    def generate(config: MethodBenchmarkRunnerConfig) -> dict[str, object]:
        inputs = base_generator(config)
        value = inputs[field_name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Scaled field {field_name!r} must be generated as a torch.Tensor")
        inputs[field_name] = value * scale
        return inputs

    return generate


def make_signed_joint_limits_generator(base_generator: InputGenerator) -> InputGenerator:
    """Wrap a generator so joint position limits have ordered lower and upper bounds."""

    def generate(config: MethodBenchmarkRunnerConfig) -> dict[str, object]:
        inputs = base_generator(config)
        limits = inputs["limits"]
        if not isinstance(limits, torch.Tensor):
            raise TypeError("Joint position limits must be generated as a torch.Tensor")
        lower = -torch.rand_like(limits[..., :1]) * 3.14
        upper = torch.rand_like(limits[..., 1:]) * 3.14
        inputs["limits"] = torch.cat((lower, upper), dim=-1)
        return inputs

    return generate


def _make_tensor_fill_generator(base_generator: InputGenerator, fill_ratio: float) -> InputGenerator:
    def generate(config: MethodBenchmarkRunnerConfig) -> dict[str, object]:
        count = max(1, int(config.num_instances * fill_ratio))
        inputs = base_generator(config)
        result: dict[str, object] = {}
        for name, value in inputs.items():
            if name == "env_ids":
                result[name] = (
                    torch.randperm(config.num_instances, device=config.device)[:count].sort().values.to(torch.int32)
                )
            elif isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == config.num_instances:
                result[name] = value[:count]
            else:
                result[name] = value
        return result

    return generate


def _make_mask_fill_generator(base_generator: InputGenerator, fill_ratio: float) -> InputGenerator:
    def generate(config: MethodBenchmarkRunnerConfig) -> dict[str, object]:
        inputs = base_generator(config)
        count = max(1, int(config.num_instances * fill_ratio))
        permutation = torch.randperm(config.num_instances, device=config.device)
        mask = torch.zeros(config.num_instances, dtype=torch.bool, device=config.device)
        mask[permutation[:count]] = True
        inputs["env_mask"] = wp.from_torch(mask, dtype=wp.bool)
        return inputs

    return generate


def build_fill_benchmarks(
    benchmarks: list[MethodBenchmarkDefinition],
    *,
    capabilities: frozenset[str],
) -> list[MethodBenchmarkDefinition]:
    """Build the tensor and mask fill-ratio workloads."""
    fill_benchmarks: list[MethodBenchmarkDefinition] = []
    for benchmark in benchmarks:
        generators: dict[str, InputGenerator] = {}
        if "tensor_fill" in capabilities and "torch_tensor" in benchmark.input_generators:
            base_generator = benchmark.input_generators["torch_tensor"]
            for suffix, ratio in FILL_RATIOS:
                generators[f"tensor_{suffix}"] = _make_tensor_fill_generator(base_generator, ratio)
        if "mask_fill" in capabilities and "warp_mask" in benchmark.input_generators:
            base_generator = benchmark.input_generators["warp_mask"]
            for suffix, ratio in FILL_RATIOS:
                generators[f"mask_{suffix}"] = _make_mask_fill_generator(base_generator, ratio)
        if generators:
            fill_benchmarks.append(
                MethodBenchmarkDefinition(
                    name=benchmark.name,
                    method_name=benchmark.method_name,
                    input_generators=generators,
                    category=f"{benchmark.category}_fill",
                )
            )
    return fill_benchmarks
