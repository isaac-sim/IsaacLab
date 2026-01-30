# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core benchmarking utilities shared across test modules.

This module provides dataclasses, enums, and helper functions used by the
benchmark scripts for both Articulation and RigidObject classes.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmarking framework."""

    num_iterations: int = 1000
    """Number of iterations to run each function."""

    warmup_steps: int = 10
    """Number of warmup steps before timing."""

    num_instances: int = 4096
    """Number of instances (articulations or rigid objects)."""

    num_bodies: int = 12
    """Number of bodies per instance."""

    num_joints: int = 11
    """Number of joints per instance (only applicable for articulations)."""

    device: str = "cuda:0"
    """Device to run benchmarks on."""

    mode: str | list[str] = "all"
    """Benchmark mode(s) to run. Can be a single string or list of strings. 'all' runs all available modes."""


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    """Name of the benchmarked method/property."""

    mean_time_us: float
    """Mean execution time in microseconds."""

    std_time_us: float
    """Standard deviation of execution time in microseconds."""

    num_iterations: int
    """Number of iterations run."""

    mode: str | None = None
    """Input mode used (e.g., 'warp', 'torch_list', etc.). None for property benchmarks."""

    skipped: bool = False
    """Whether the benchmark was skipped."""

    skip_reason: str = ""
    """Reason for skipping the benchmark."""

    dependencies: list[str] | None = None
    """List of parent properties that were pre-computed before timing."""


@dataclass
class MethodBenchmark:
    """Definition of a method to benchmark."""

    name: str
    """Name of the method."""

    method_name: str
    """Actual method name on the class."""

    input_generators: dict[str, Callable]
    """Dictionary mapping mode names to input generator functions."""

    category: str = "general"
    """Category of the method (e.g., 'root_state', 'joint_state', 'joint_params')."""


# =============================================================================
# Common Input Generator Helpers
# =============================================================================

import torch


def make_tensor_env_ids(num_instances: int, device: str) -> torch.Tensor:
    """Create a tensor of environment IDs.

    Args:
        num_instances: Number of environment instances.
        device: Device to create the tensor on.

    Returns:
        Tensor of environment IDs [0, 1, ..., num_instances-1].
    """
    return torch.arange(num_instances, dtype=torch.long, device=device)


def make_tensor_joint_ids(num_joints: int, device: str) -> torch.Tensor:
    """Create a tensor of joint IDs.

    Args:
        num_joints: Number of joints.
        device: Device to create the tensor on.

    Returns:
        Tensor of joint IDs [0, 1, ..., num_joints-1].
    """
    return torch.arange(num_joints, dtype=torch.long, device=device)


def make_tensor_body_ids(num_bodies: int, device: str) -> torch.Tensor:
    """Create a tensor of body IDs.

    Args:
        num_bodies: Number of bodies.
        device: Device to create the tensor on.

    Returns:
        Tensor of body IDs [0, 1, ..., num_bodies-1].
    """
    return torch.arange(num_bodies, dtype=torch.long, device=device)


def make_warp_env_mask(num_instances: int, device: str) -> wp.array:
    """Create an all-true environment mask.

    Args:
        num_instances: Number of environment instances.
        device: Device to create the mask on.

    Returns:
        Warp array of booleans, all set to True.
    """
    return wp.ones((num_instances,), dtype=wp.bool, device=device)


def make_warp_joint_mask(num_joints: int, device: str) -> wp.array:
    """Create an all-true joint mask.

    Args:
        num_joints: Number of joints.
        device: Device to create the mask on.

    Returns:
        Warp array of booleans, all set to True.
    """
    return wp.ones((num_joints,), dtype=wp.bool, device=device)


def make_warp_body_mask(num_bodies: int, device: str) -> wp.array:
    """Create an all-true body mask.

    Args:
        num_bodies: Number of bodies.
        device: Device to create the mask on.

    Returns:
        Warp array of booleans, all set to True.
    """
    return wp.ones((num_bodies,), dtype=wp.bool, device=device)


# =============================================================================
# Benchmark Method Helper Functions
# =============================================================================


def benchmark_method(
    method: Callable | None,
    method_name: str,
    generator: Callable,
    config: BenchmarkConfig,
    dependencies: dict[str, list[str]] = {},
) -> BenchmarkResult:
    """Benchmarks a method.

    Args:
        method: The method to benchmark.
        method_name: The name of the method.
        generator: The input generator to use for the method.
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    # Check if method exists
    if method is None:
        return BenchmarkResult(
            name=method_name,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason="Method not found",
        )

    # Try to access the property once to check if it raises NotImplementedError
    try:
        _ = method(**generator(config))
    except NotImplementedError as e:
        return BenchmarkResult(
            name=method_name,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason=f"NotImplementedError: {e}",
        )
    except Exception as e:
        return BenchmarkResult(
            name=method_name,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason=f"Error: {type(e).__name__}: {e}",
        )

    # Get dependencies for this property (if any)
    dependencies_ = dependencies.get(method_name, [])

    # Warmup phase with random data
    for _ in range(config.warmup_steps):
        try:
            # Warm up dependencies first
            inputs = generator(config)
            for dep in dependencies_:
                _ = method(**inputs)
            # Then warm up the method
            _ = method(**inputs)
        except Exception:
            pass
        # Sync GPU
        if config.device.startswith("cuda"):
            wp.synchronize()

    # Timing phase
    times = []
    for _ in range(config.num_iterations):
        # Call dependencies first to populate their caches (not timed)
        # This ensures we only measure the overhead of the derived property
        inputs = generator(config)
        with contextlib.suppress(Exception):
            for dep in dependencies_:
                _ = method(**inputs)

        # Sync before timing
        if config.device.startswith("cuda"):
            wp.synchronize()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Time only the target property access
        start_time = time.perf_counter()
        try:
            _ = method(**inputs)
        except Exception:
            continue

        # Sync after to ensure kernel completion
        if config.device.startswith("cuda"):
            wp.synchronize()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1e6)  # Convert to microseconds

    if not times:
        return BenchmarkResult(
            name=method_name,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason="No successful iterations",
        )

    return BenchmarkResult(
        name=method_name,
        mean_time_us=float(np.mean(times)),
        std_time_us=float(np.std(times)),
        num_iterations=len(times),
        dependencies=dependencies_ if dependencies_ else None,
    )
