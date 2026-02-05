# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Method-level benchmarking framework for IsaacLab.

This module provides a framework for benchmarking individual methods with support for:
- Multiple input modes (torch_list, torch_tensor, etc.)
- Automatic hardware/version info collection via recorders
- Multiple output backends (JSON, Osmo, OmniPerf)
- Statistical measurements (mean, std, n)

Example usage:

.. code-block:: python

    from isaaclab.test.benchmark import (
        MethodBenchmarkRunner,
        MethodBenchmarkRunnerConfig,
        MethodBenchmarkDefinition,
    )

    # Define benchmarks
    BENCHMARKS = [
        MethodBenchmarkDefinition(
            name="write_root_state_to_sim",
            method_name="write_root_state_to_sim",
            input_generators={
                "torch_list": gen_root_state_torch_list,
                "torch_tensor": gen_root_state_torch_tensor,
            },
            category="root_state",
        ),
    ]

    # Configure and run
    config = MethodBenchmarkRunnerConfig(
        num_iterations=1000,
        warmup_steps=10,
        num_instances=4096,
        device="cuda:0",
    )
    runner = MethodBenchmarkRunner("articulation_benchmark", config, "json", ".")
    runner.run_benchmarks(BENCHMARKS, articulation)
    runner.finalize()
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from .benchmark_core import BaseIsaacLabBenchmark
from .measurements import StatisticalMeasurement

logger = logging.getLogger(__name__)


@dataclass
class MethodBenchmarkRunnerConfig:
    """Configuration for MethodBenchmarkRunner.

    Attributes:
        num_iterations: Number of timing iterations per method.
        warmup_steps: Number of warmup iterations before timing.
        num_instances: Number of environment instances.
        num_bodies: Number of bodies per instance.
        num_joints: Number of joints per instance.
        device: Device to run benchmarks on.
        mode: Which input modes to run ("all" or specific mode name).
    """

    num_iterations: int = 1000
    warmup_steps: int = 10
    num_instances: int = 4096
    num_bodies: int = 12
    num_joints: int = 11
    device: str = "cuda:0"
    mode: str | list[str] = "all"


@dataclass
class MethodBenchmarkDefinition:
    """Definition of a method benchmark.

    Attributes:
        name: Display name for the benchmark.
        method_name: Name of the method to benchmark on the target object.
        input_generators: Dict mapping mode names to input generator functions.
        category: Category for grouping results into phases.
        dependencies: List of method names that must be called first.
    """

    name: str
    method_name: str
    input_generators: dict[str, Callable]
    category: str = "default"
    dependencies: list[str] = field(default_factory=list)


class MethodBenchmarkRunner(BaseIsaacLabBenchmark):
    """Runner for method-level benchmarks using the new benchmark tooling.

    This class extends BaseIsaacLabBenchmark to provide method-level benchmarking
    with automatic hardware/version info collection, multiple backend support,
    and organized output by category phases.
    """

    def __init__(
        self,
        benchmark_name: str,
        config: MethodBenchmarkRunnerConfig,
        backend_type: str = "json",
        output_path: str = ".",
        use_recorders: bool = True,
    ):
        """Initialize the method benchmark runner.

        Args:
            benchmark_name: Name of the benchmark (used in output files).
            config: Benchmark configuration.
            backend_type: Output backend type ("json", "osmo", "omni_perf").
            output_path: Directory to write output files.
            use_recorders: Whether to collect hardware/version info.
        """
        self._config = config

        # Build workflow metadata from config
        workflow_metadata = {
            "metadata": [
                {"name": "num_iterations", "data": config.num_iterations},
                {"name": "warmup_steps", "data": config.warmup_steps},
                {"name": "num_instances", "data": config.num_instances},
                {"name": "num_bodies", "data": config.num_bodies},
                {"name": "num_joints", "data": config.num_joints},
                {"name": "device", "data": config.device},
            ]
        }

        super().__init__(
            benchmark_name=benchmark_name,
            backend_type=backend_type,
            output_path=output_path,
            use_recorders=use_recorders,
            output_prefix=benchmark_name,
            workflow_metadata=workflow_metadata,
        )

        # Determine which modes to run
        if isinstance(config.mode, str):
            if config.mode == "all":
                self._modes_to_run = None  # Run all available
            else:
                self._modes_to_run = [config.mode]
        else:
            self._modes_to_run = config.mode

    @property
    def config(self) -> MethodBenchmarkRunnerConfig:
        """Return the benchmark configuration."""
        return self._config

    def run_benchmarks(
        self,
        benchmarks: list[MethodBenchmarkDefinition],
        target_object: object,
        dependencies: dict[str, list[str]] | None = None,
    ) -> None:
        """Run all defined benchmarks on the target object.

        Args:
            benchmarks: List of benchmark definitions to run.
            target_object: Object containing the methods to benchmark.
            dependencies: Optional dict mapping method names to their dependencies.
        """
        if dependencies is None:
            dependencies = {}

        print(f"\nBenchmarking {len(benchmarks)} methods...")
        print(f"Config: {self._config.num_iterations} iterations, {self._config.warmup_steps} warmup steps")
        print(
            f"        {self._config.num_instances} instances, {self._config.num_bodies} bodies, "
            f"{self._config.num_joints} joints"
        )
        print(f"Device: {self._config.device}")
        print(f"Modes: {self._modes_to_run if self._modes_to_run else 'All available'}")
        print("-" * 80)

        for i, benchmark in enumerate(benchmarks):
            method = getattr(target_object, benchmark.method_name, None)

            # Determine which modes to run for this benchmark
            available_modes = list(benchmark.input_generators.keys())
            current_modes = self._modes_to_run if self._modes_to_run is not None else available_modes
            current_modes = [m for m in current_modes if m in available_modes]

            # Get dependencies for this method
            method_deps = benchmark.dependencies or dependencies.get(benchmark.method_name, [])

            for mode in current_modes:
                # Update manual recorders
                self.update_manual_recorders()

                generator = benchmark.input_generators[mode]
                bench_name = f"{benchmark.name}_{mode}"
                print(f"[{i + 1}/{len(benchmarks)}] [{mode.upper()}] {benchmark.name}...", end=" ", flush=True)

                result = self._benchmark_method(
                    method=method,
                    method_name=bench_name,
                    generator=generator,
                    dependencies=method_deps,
                )

                if result is None:
                    print("SKIPPED (method not found)")
                elif result.get("skipped"):
                    print(f"SKIPPED ({result.get('skip_reason', 'unknown')})")
                else:
                    mean = result["mean"]
                    std = result["std"]
                    print(f"{mean:.2f} +/- {std:.2f} us")

                    # Add measurement to mode-based phase (torch_list, torch_tensor, etc.)
                    measurement = StatisticalMeasurement(
                        name=benchmark.name,
                        mean=mean,
                        std=std,
                        n=result["n"],
                        unit="us",
                    )
                    self.add_measurement(mode, measurement=measurement)

    def _benchmark_method(
        self,
        method: Callable | None,
        method_name: str,
        generator: Callable,
        dependencies: list[str],
    ) -> dict | None:
        """Benchmark a single method.

        Args:
            method: The method to benchmark (or None if not found).
            method_name: Name of the method for reporting.
            generator: Function that generates input arguments.
            dependencies: List of dependency method names to call first.

        Returns:
            Dict with timing results, or None if method not found.
        """
        if method is None:
            return None

        # Try to call the method once to check for NotImplementedError
        try:
            inputs = generator(self._config)
            _ = method(**inputs)
        except NotImplementedError as e:
            return {"skipped": True, "skip_reason": f"NotImplementedError: {e}"}
        except Exception as e:
            return {"skipped": True, "skip_reason": f"Error: {type(e).__name__}: {e}"}

        # Warmup phase
        for _ in range(self._config.warmup_steps):
            try:
                inputs = generator(self._config)
                _ = method(**inputs)
            except Exception:
                pass
            # Sync GPU
            if self._config.device.startswith("cuda"):
                self._sync_device()

        # Timing phase
        times = []
        for _ in range(self._config.num_iterations):
            inputs = generator(self._config)

            # Sync before timing
            if self._config.device.startswith("cuda"):
                self._sync_device()

            # Time the method
            start_time = time.perf_counter()
            try:
                _ = method(**inputs)
            except Exception:
                continue

            # Sync after to ensure kernel completion
            if self._config.device.startswith("cuda"):
                self._sync_device()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1e6)  # Convert to microseconds

        if not times:
            return {"skipped": True, "skip_reason": "No successful iterations"}

        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "n": len(times),
        }

    def _sync_device(self) -> None:
        """Synchronize GPU device."""
        try:
            import warp as wp

            wp.synchronize()
        except ImportError:
            pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_property_benchmarks(
        self,
        target_data: object,
        properties: list[str],
        gen_mock_data: Callable,
        dependencies: dict[str, list[str]] | None = None,
        category: str = "property",
    ) -> None:
        """Run benchmarks for data class properties.

        This is a convenience method for benchmarking properties on data classes
        where the test involves generating mock data and accessing properties.

        Args:
            target_data: Data object containing the properties to benchmark.
            properties: List of property names to benchmark.
            gen_mock_data: Function that generates/updates mock data.
            dependencies: Optional dict mapping property names to their dependencies.
            category: Category name for grouping results.
        """
        if dependencies is None:
            dependencies = {}

        # Update manual recorders at start
        self.update_manual_recorders()

        print(f"\nBenchmarking {len(properties)} properties...")
        print(f"Config: {self._config.num_iterations} iterations, {self._config.warmup_steps} warmup steps")
        print(
            f"        {self._config.num_instances} instances, {self._config.num_bodies} bodies, "
            f"{self._config.num_joints} joints"
        )
        print("-" * 80)

        for i, prop_name in enumerate(properties):
            print(f"[{i + 1}/{len(properties)}] [DEFAULT] {prop_name}...", end=" ", flush=True)

            # Get dependencies for this property
            prop_deps = dependencies.get(prop_name, [])

            result = self._benchmark_property(
                target_data=target_data,
                prop_name=prop_name,
                gen_mock_data=gen_mock_data,
                dependencies=prop_deps,
            )

            if result is None:
                print("SKIPPED (property not found)")
            elif result.get("skipped"):
                print(f"SKIPPED ({result.get('skip_reason', 'unknown')})")
            else:
                mean = result["mean"]
                std = result["std"]
                print(f"{mean:.2f} +/- {std:.2f} us")

                # Add measurement
                measurement = StatisticalMeasurement(
                    name=prop_name,
                    mean=mean,
                    std=std,
                    n=result["n"],
                    unit="us",
                )
                self.add_measurement(category, measurement=measurement)

    def _benchmark_property(
        self,
        target_data: object,
        prop_name: str,
        gen_mock_data: Callable,
        dependencies: list[str],
    ) -> dict | None:
        """Benchmark a single property access.

        Args:
            target_data: Data object containing the property.
            prop_name: Name of the property to benchmark.
            gen_mock_data: Function that generates/updates mock data.
            dependencies: List of property names to access first.

        Returns:
            Dict with timing results, or None if property not found.
        """
        # Check if property exists
        if not hasattr(target_data, prop_name):
            return None

        # Try to access the property once to check for errors
        try:
            gen_mock_data(self._config)
            _ = getattr(target_data, prop_name)
        except NotImplementedError as e:
            return {"skipped": True, "skip_reason": f"NotImplementedError: {e}"}
        except Exception as e:
            return {"skipped": True, "skip_reason": f"Error: {type(e).__name__}: {e}"}

        # Warmup phase
        for _ in range(self._config.warmup_steps):
            try:
                gen_mock_data(self._config)
                # Access dependencies first
                for dep in dependencies:
                    with contextlib.suppress(Exception):
                        _ = getattr(target_data, dep)
                _ = getattr(target_data, prop_name)
            except Exception:
                pass
            # Sync GPU
            if self._config.device.startswith("cuda"):
                self._sync_device()

        # Timing phase
        times = []
        for _ in range(self._config.num_iterations):
            gen_mock_data(self._config)

            # Access dependencies first (not timed)
            for dep in dependencies:
                with contextlib.suppress(Exception):
                    _ = getattr(target_data, dep)

            # Sync before timing
            if self._config.device.startswith("cuda"):
                self._sync_device()

            # Time the property access
            start_time = time.perf_counter()
            try:
                _ = getattr(target_data, prop_name)
            except Exception:
                continue

            # Sync after
            if self._config.device.startswith("cuda"):
                self._sync_device()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1e6)  # microseconds

        if not times:
            return {"skipped": True, "skip_reason": "No successful iterations"}

        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "n": len(times),
        }

    def finalize(self) -> None:
        """Finalize the benchmark and write results."""
        self._finalize_impl()
