# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObjectData class.

This module provides a benchmarking framework to measure the performance of all functions
in the RigidObjectData class. Each function is run multiple times with randomized mock data,
and timing statistics (mean and standard deviation) are reported.

Usage:
    python benchmark_rigid_object_data.py [--num_iterations N] [--warmup_steps W] [--num_instances I] [--num_bodies B]

Example:
    python benchmark_rigid_object_data.py --num_iterations 10000 --warmup_steps 10
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import warp as wp
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData

# Add test directory to path for common module imports
_TEST_DIR = Path(__file__).resolve().parents[2]
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

# Import shared utilities from common module
from common.benchmark_core import BenchmarkConfig, BenchmarkResult, MethodBenchmark, benchmark_method
from common.benchmark_io import (
    export_results_json,
    get_default_output_filename,
    get_hardware_info,
    print_hardware_info,
    print_results,
)

# Import mock classes from common test utilities
from common.mock_newton import MockNewtonArticulationView, MockNewtonModel

# Initialize Warp
wp.init()

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# List of deprecated properties (for backward compatibility) - skip these
DEPRECATED_PROPERTIES = {
    "default_root_state",
    "root_pose_w",
    "root_pos_w",
    "root_quat_w",
    "root_vel_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "root_lin_vel_b",
    "root_ang_vel_b",
    "body_pose_w",
    "body_pos_w",
    "body_quat_w",
    "body_vel_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_acc_w",
    "body_lin_acc_w",
    "body_ang_acc_w",
    "com_pos_b",
    "com_quat_b",
    # Also skip the combined state properties marked as deprecated
    "root_state_w",
    "root_link_state_w",
    "root_com_state_w",
    "body_state_w",
    "body_link_state_w",
    "body_com_state_w",
}

# List of properties that raise NotImplementedError - skip these
NOT_IMPLEMENTED_PROPERTIES = {}

# Private/internal properties and methods to skip
INTERNAL_PROPERTIES = {
    "_create_simulation_bindings",
    "_create_buffers",
    "update",
    "is_primed",
    "device",
    "body_names",
    "GRAVITY_VEC_W",
    "GRAVITY_VEC_W_TORCH",
    "FORWARD_VEC_B",
    "FORWARD_VEC_B_TORCH",
    "ALL_ENV_MASK",
    "ENV_MASK",
}

# Dependency mapping: derived properties and their parent dependencies.
# Before benchmarking a derived property, we first call the parent to populate
# its cache, so we measure only the overhead of the derived property extraction.
PROPERTY_DEPENDENCIES = {
    # Root link velocity slices (depend on root_link_vel_w)
    "root_link_lin_vel_w": ["root_link_vel_w"],
    "root_link_ang_vel_w": ["root_link_vel_w"],
    # Root link velocity in body frame slices (depend on root_link_vel_b)
    "root_link_lin_vel_b": ["root_link_vel_b"],
    "root_link_ang_vel_b": ["root_link_vel_b"],
    # Root COM pose slices (depend on root_com_pose_w)
    "root_com_pos_w": ["root_com_pose_w"],
    "root_com_quat_w": ["root_com_pose_w"],
    # Root COM velocity slices (depend on root_com_vel_b)
    "root_com_lin_vel_b": ["root_com_vel_b"],
    "root_com_ang_vel_b": ["root_com_vel_b"],
    # Root COM velocity in world frame slices (no lazy dependency, direct binding)
    "root_com_lin_vel_w": ["root_com_vel_w"],
    "root_com_ang_vel_w": ["root_com_vel_w"],
    # Root link pose slices (no lazy dependency, direct binding)
    "root_link_pos_w": ["root_link_pose_w"],
    "root_link_quat_w": ["root_link_pose_w"],
    # Body link velocity slices (depend on body_link_vel_w)
    "body_link_lin_vel_w": ["body_link_vel_w"],
    "body_link_ang_vel_w": ["body_link_vel_w"],
    # Body link pose slices (no lazy dependency, direct binding)
    "body_link_pos_w": ["body_link_pose_w"],
    "body_link_quat_w": ["body_link_pose_w"],
    # Body COM pose slices (depend on body_com_pose_w)
    "body_com_pos_w": ["body_com_pose_w"],
    "body_com_quat_w": ["body_com_pose_w"],
    # Body COM velocity slices (no lazy dependency, direct binding)
    "body_com_lin_vel_w": ["body_com_vel_w"],
    "body_com_ang_vel_w": ["body_com_vel_w"],
    # Body COM acceleration slices (depend on body_com_acc_w)
    "body_com_lin_acc_w": ["body_com_acc_w"],
    "body_com_ang_acc_w": ["body_com_acc_w"],
    # Body COM pose/quat in body frame (depend on body_com_pose_b)
    "body_com_quat_b": ["body_com_pose_b"],
}


def get_benchmarkable_properties(rigid_object_data: RigidObjectData) -> list[str]:
    """Get list of properties that can be benchmarked.

    Args:
        rigid_object_data: The RigidObjectData instance to inspect.

    Returns:
        List of property names that can be benchmarked.
    """
    all_properties = []

    # Get all properties from the class
    for name in dir(rigid_object_data):
        # Skip private/dunder methods
        if name.startswith("_"):
            continue

        # Skip deprecated properties
        if name in DEPRECATED_PROPERTIES:
            continue

        # Skip not implemented properties
        if name in NOT_IMPLEMENTED_PROPERTIES:
            continue

        # Skip internal properties
        if name in INTERNAL_PROPERTIES:
            continue

        # Check if it's a property (not a method that needs arguments)
        try:
            attr = getattr(type(rigid_object_data), name, None)
            if isinstance(attr, property):
                all_properties.append(name)
        except Exception:
            pass

    return sorted(all_properties)


def setup_mock_environment(
    config: BenchmarkConfig,
) -> tuple[MockNewtonArticulationView, MockNewtonModel, MagicMock, MagicMock]:
    """Set up the mock environment for benchmarking.

    Args:
        config: Benchmark configuration.

    Returns:
        Tuple of (mock_view, mock_model, mock_state, mock_control).
    """
    # Create mock Newton model
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()

    # Create mock view
    mock_view = MockNewtonArticulationView(
        num_instances=config.num_instances,
        num_bodies=1,
        num_joints=0,
        device=config.device,
    )

    return mock_view, mock_model, mock_state, mock_control


# We need a way to pass the instance and view to the generator, but gen_mock_data
# only takes config. We can use a class or closure, or rely on global state set up in run_benchmarks.
# For simplicity, we'll assume `_rigid_object_data` and `_mock_view` are available in the scope
# or passed via a partial.
# Since benchmark_method expects generator(config) -> dict, we can't easily pass the instance.
# However, we can create a closure inside run_benchmarks.


def run_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run all benchmarks for RigidObjectData.

    Args:
        config: Benchmark configuration.

    Returns:
        Tuple of (List of BenchmarkResults, hardware_info dict).
    """
    results = []

    # Setup mock environment
    mock_view, mock_model, mock_state, mock_control = setup_mock_environment(config)

    # Patch NewtonManager
    with patch("isaaclab_newton.assets.rigid_object.rigid_object_data.NewtonManager") as MockManager:
        MockManager.get_model.return_value = mock_model
        MockManager.get_state_0.return_value = mock_state
        MockManager.get_control.return_value = mock_control
        MockManager.get_dt.return_value = 0.01

        # Initialize mock data
        mock_view.set_random_mock_data()

        # Create RigidObjectData instance
        rigid_object_data = RigidObjectData(mock_view, config.device)

        # Get list of properties to benchmark
        properties = get_benchmarkable_properties(rigid_object_data)

        # Generator that updates mock data and invalidates timestamp
        def gen_mock_data(config: BenchmarkConfig) -> dict:
            mock_view.set_random_mock_data()
            rigid_object_data._sim_timestamp += 1.0
            return {}

        # Create benchmarks dynamically
        benchmarks = []
        for prop_name in properties:
            benchmarks.append(
                MethodBenchmark(
                    name=prop_name,
                    method_name=prop_name,
                    input_generators={"default": gen_mock_data},
                    category="property",
                )
            )

        print(f"\nBenchmarking {len(benchmarks)} properties...")
        print(f"Config: {config.num_iterations} iterations, {config.warmup_steps} warmup steps")
        print(f"        {config.num_instances} instances, {config.num_bodies} bodies")
        print("-" * 80)

        for i, benchmark in enumerate(benchmarks):
            # For properties, we need a wrapper that accesses the property
            # We can't bind a property to an instance easily like a method
            # So we create a lambda that takes **kwargs (which will be empty)
            # and accesses the property on the instance.
            # We must bind prop_name to avoid closure issues
            def prop_accessor(prop=benchmark.method_name, **kwargs):
                return getattr(rigid_object_data, prop)

            print(f"[{i + 1}/{len(benchmarks)}] [DEFAULT] {benchmark.name}...", end=" ", flush=True)

            result = benchmark_method(
                method=prop_accessor,
                method_name=benchmark.name,
                generator=gen_mock_data,
                config=config,
                dependencies=PROPERTY_DEPENDENCIES,
            )
            # Property benchmarks only have one "mode" (default/access)
            result.mode = "default"
            results.append(result)

            if result.skipped:
                print(f"SKIPPED ({result.skip_reason})")
            else:
                print(f"{result.mean_time_us:.2f} ± {result.std_time_us:.2f} µs")

    return results


def main():
    """Main entry point for the benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Micro-benchmarking framework for RigidObjectData class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filename")
    parser.add_argument("--no_csv", action="store_true", help="Disable CSV output")

    args = parser.parse_args()

    config = BenchmarkConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=1,
        num_joints=0,
        device=args.device,
    )

    results = run_benchmark(config)

    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)
    print_results(results, include_mode=False)

    if args.output:
        json_filename = args.output
    else:
        json_filename = get_default_output_filename("rigid_object_data_benchmark")

    export_results_json(results, config, hardware_info, json_filename)

    if not args.no_csv:
        csv_filename = json_filename.replace(".json", ".csv")
        from common.benchmark_io import export_results_csv

        export_results_csv(results, csv_filename)


if __name__ == "__main__":
    main()
