# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for ArticulationData class.

This module provides a benchmarking framework to measure the performance of all functions
in the ArticulationData class. Each function is run multiple times with randomized mock data,
and timing statistics (mean and standard deviation) are reported.

Usage:
    python benchmark_articulation_data.py [--num_iterations N] [--warmup_steps W] [--num_instances I] [--num_bodies B] [--num_joints J]

Example:
    python benchmark_articulation_data.py --num_iterations 10000 --warmup_steps 10
"""

from __future__ import annotations

import argparse
import contextlib
import numpy as np
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import warp as wp
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData

# Add test directory to path for common module imports
_TEST_DIR = Path(__file__).resolve().parents[2]
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

# Import shared utilities from common module
from common.benchmark_core import BenchmarkConfig, BenchmarkResult
from common.benchmark_io import (
    export_results_csv,
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
    "joint_limits",
    "joint_friction",
    "fixed_tendon_limit",
    "applied_torque",
    "computed_torque",
    "joint_dynamic_friction",
    "joint_effort_target",
    "joint_viscous_friction",
    "joint_velocity_limits",
    # Also skip the combined state properties marked as deprecated
    "root_state_w",
    "root_link_state_w",
    "root_com_state_w",
    "body_state_w",
    "body_link_state_w",
    "body_com_state_w",
}

# List of properties that raise NotImplementedError - skip these
NOT_IMPLEMENTED_PROPERTIES = {
    "fixed_tendon_stiffness",
    "fixed_tendon_damping",
    "fixed_tendon_limit_stiffness",
    "fixed_tendon_rest_length",
    "fixed_tendon_offset",
    "fixed_tendon_pos_limits",
    "spatial_tendon_stiffness",
    "spatial_tendon_damping",
    "spatial_tendon_limit_stiffness",
    "spatial_tendon_offset",
    "body_incoming_joint_wrench_b",
}

# Private/internal properties and methods to skip
INTERNAL_PROPERTIES = {
    "_create_simulation_bindings",
    "_create_buffers",
    "update",
    "is_primed",
    "device",
    "body_names",
    "joint_names",
    "fixed_tendon_names",
    "spatial_tendon_names",
    "GRAVITY_VEC_W",
    "GRAVITY_VEC_W_TORCH",
    "FORWARD_VEC_B",
    "FORWARD_VEC_B_TORCH",
    "ALL_ENV_MASK",
    "ALL_BODY_MASK",
    "ALL_JOINT_MASK",
    "ENV_MASK",
    "BODY_MASK",
    "JOINT_MASK",
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


def get_benchmarkable_properties(articulation_data: ArticulationData) -> list[str]:
    """Get list of properties that can be benchmarked.

    Args:
        articulation_data: The ArticulationData instance to inspect.

    Returns:
        List of property names that can be benchmarked.
    """
    all_properties = []

    # Get all properties from the class
    for name in dir(articulation_data):
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
            attr = getattr(type(articulation_data), name, None)
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
        num_bodies=config.num_bodies,
        num_joints=config.num_joints,
        device=config.device,
    )

    return mock_view, mock_model, mock_state, mock_control


def benchmark_property(
    articulation_data: ArticulationData,
    mock_view: MockNewtonArticulationView,
    property_name: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark a single property of ArticulationData.

    Args:
        articulation_data: The ArticulationData instance.
        mock_view: The mock view for setting random data.
        property_name: Name of the property to benchmark.
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    # Check if property exists
    if not hasattr(articulation_data, property_name):
        return BenchmarkResult(
            name=property_name,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason="Property not found",
        )

    # Try to access the property once to check if it raises NotImplementedError
    try:
        _ = getattr(articulation_data, property_name)
    except NotImplementedError as e:
        return BenchmarkResult(
            name=property_name,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason=f"NotImplementedError: {e}",
        )
    except Exception as e:
        return BenchmarkResult(
            name=property_name,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason=f"Error: {type(e).__name__}: {e}",
        )

    # Get dependencies for this property (if any)
    dependencies = PROPERTY_DEPENDENCIES.get(property_name, [])

    # Warmup phase with random data
    for _ in range(config.warmup_steps):
        mock_view.set_random_mock_data()
        articulation_data._sim_timestamp += 1.0  # Invalidate cached data
        try:
            # Warm up dependencies first
            for dep in dependencies:
                _ = getattr(articulation_data, dep)
            # Then warm up the target property
            _ = getattr(articulation_data, property_name)
        except Exception:
            pass
        # Sync GPU
        if config.device.startswith("cuda"):
            wp.synchronize()

    # Timing phase
    times = []
    for _ in range(config.num_iterations):
        # Randomize mock data each iteration
        mock_view.set_random_mock_data()
        articulation_data._sim_timestamp += 1.0  # Invalidate cached data

        # Call dependencies first to populate their caches (not timed)
        # This ensures we only measure the overhead of the derived property
        with contextlib.suppress(Exception):
            for dep in dependencies:
                _ = getattr(articulation_data, dep)

        # Sync before timing
        if config.device.startswith("cuda"):
            wp.synchronize()

        # Time only the target property access
        start_time = time.perf_counter()
        try:
            _ = getattr(articulation_data, property_name)
        except Exception:
            continue

        # Sync after to ensure kernel completion
        if config.device.startswith("cuda"):
            wp.synchronize()

        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1e6)  # Convert to microseconds

    if not times:
        return BenchmarkResult(
            name=property_name,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason="No successful iterations",
        )

    return BenchmarkResult(
        name=property_name,
        mean_time_us=float(np.mean(times)),
        std_time_us=float(np.std(times)),
        num_iterations=len(times),
        dependencies=dependencies if dependencies else None,
    )


def run_benchmarks(config: BenchmarkConfig) -> tuple[list[BenchmarkResult], dict]:
    """Run all benchmarks for ArticulationData.

    Args:
        config: Benchmark configuration.

    Returns:
        Tuple of (List of BenchmarkResults, hardware_info dict).
    """
    results = []

    # Gather and print hardware info
    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)

    # Setup mock environment
    mock_view, mock_model, mock_state, mock_control = setup_mock_environment(config)

    # Patch NewtonManager
    with patch("isaaclab_newton.assets.articulation.articulation_data.NewtonManager") as MockManager:
        MockManager.get_model.return_value = mock_model
        MockManager.get_state_0.return_value = mock_state
        MockManager.get_control.return_value = mock_control
        MockManager.get_dt.return_value = 0.01

        # Initialize mock data
        mock_view.set_random_mock_data()

        # Create ArticulationData instance
        articulation_data = ArticulationData(mock_view, config.device)

        # Get list of properties to benchmark
        properties = get_benchmarkable_properties(articulation_data)

        print(f"\nBenchmarking {len(properties)} properties...")
        print(f"Config: {config.num_iterations} iterations, {config.warmup_steps} warmup steps")
        print(f"        {config.num_instances} instances, {config.num_bodies} bodies, {config.num_joints} joints")
        print("-" * 80)

        for i, prop_name in enumerate(properties):
            print(f"[{i + 1}/{len(properties)}] Benchmarking {prop_name}...", end=" ", flush=True)

            result = benchmark_property(articulation_data, mock_view, prop_name, config)
            results.append(result)

            if result.skipped:
                print(f"SKIPPED ({result.skip_reason})")
            else:
                print(f"{result.mean_time_us:.2f} ± {result.std_time_us:.2f} µs")

    return results, hardware_info


def main():
    """Main entry point for the benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Micro-benchmarking framework for ArticulationData class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10000,
        help="Number of iterations to run each function.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps before timing.",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=16384,
        help="Number of articulation instances.",
    )
    parser.add_argument(
        "--num_bodies",
        type=int,
        default=12,
        help="Number of bodies per articulation.",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=11,
        help="Number of joints per articulation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run benchmarks on.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file for benchmark results. Default: articulation_data_DATE.json",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default=None,
        help="Additionally export results to CSV file.",
    )
    parser.add_argument(
        "--no_json",
        action="store_true",
        help="Disable JSON output.",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=args.num_bodies,
        num_joints=args.num_joints,
        device=args.device,
    )

    # Run benchmarks
    results, hardware_info = run_benchmarks(config)

    # Print results
    print_results(results, include_mode=False)

    # Export to JSON (default)
    if not args.no_json:
        output_filename = args.output if args.output else get_default_output_filename("articulation_data")
        export_results_json(results, config, hardware_info, output_filename, include_mode=False)

    # Export to CSV if requested
    if args.export_csv:
        export_results_csv(results, args.export_csv)


if __name__ == "__main__":
    main()
