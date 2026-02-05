# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObjectCollectionData class.

This module provides a benchmarking framework to measure the performance of all functions
in the RigidObjectCollectionData class. Each function is run multiple times with randomized mock data,
and timing statistics (mean and standard deviation) are reported.

Usage:
    python benchmark_rigid_object_collection_data.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B]

Example:
    python benchmark_rigid_object_collection_data.py --num_iterations 10000 --warmup_steps 10
"""

from __future__ import annotations

import argparse
import sys
import warnings
from types import ModuleType
from unittest.mock import MagicMock

import torch
import warp as wp

# Initialize Warp first
wp.init()


# =============================================================================
# Mock Setup - Must happen BEFORE importing RigidObjectCollectionData
# =============================================================================


# Mock BaseRigidObjectCollectionData - this is just an abstract class
class BaseRigidObjectCollectionData:
    """Mock base class to avoid importing isaaclab.assets (which has many dependencies)."""

    def __init__(self, root_view, num_bodies: int, device: str):
        self.device = device


# Create mock module for isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data
mock_base_module = ModuleType("isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data")
mock_base_module.BaseRigidObjectCollectionData = BaseRigidObjectCollectionData
sys.modules["isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data"] = mock_base_module

# Mock pxr (USD library - not available in headless docker, used by isaaclab.utils.mesh)
sys.modules["pxr"] = MagicMock()
sys.modules["pxr.Usd"] = MagicMock()
sys.modules["pxr.UsdGeom"] = MagicMock()


class MockPhysicsSimView:
    """Simple mock for the physics simulation view."""

    def get_gravity(self):
        """Return gravity as a tuple of 3 floats."""
        return (0.0, 0.0, -9.81)


class MockSimulationManager:
    """Simple mock for SimulationManager."""

    @staticmethod
    def get_physics_sim_view():
        return MockPhysicsSimView()


# Mock isaacsim.core.simulation_manager
mock_sim_manager_module = ModuleType("isaacsim.core.simulation_manager")
mock_sim_manager_module.SimulationManager = MockSimulationManager
sys.modules["isaacsim"] = ModuleType("isaacsim")
sys.modules["isaacsim.core"] = ModuleType("isaacsim.core")
sys.modules["isaacsim.core.simulation_manager"] = mock_sim_manager_module

# Now we can directly import RigidObjectCollectionData
import importlib.util
from pathlib import Path

benchmark_dir = Path(__file__).resolve().parent
data_path = (
    benchmark_dir.parents[1]
    / "isaaclab_physx"
    / "assets"
    / "rigid_object_collection"
    / "rigid_object_collection_data.py"
)

spec = importlib.util.spec_from_file_location(
    "isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data", data_path
)
data_module = importlib.util.module_from_spec(spec)
sys.modules["isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data"] = data_module
spec.loader.exec_module(data_module)
RigidObjectCollectionData = data_module.RigidObjectCollectionData

# Import shared utilities from common module
# Import mock classes from PhysX test utilities
from isaaclab_physx.test.mock_interfaces.views import MockRigidBodyView

from isaaclab.test.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    MethodBenchmark,
    benchmark_method,
    export_results_csv,
    export_results_json,
    get_default_output_filename,
    get_hardware_info,
    print_hardware_info,
    print_results,
)

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Skip Lists
# =============================================================================

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
    "object_link_state_w",
    "object_com_state_w",
}

# List of properties that raise NotImplementedError - skip these
NOT_IMPLEMENTED_PROPERTIES = {}

# Removed default_* properties that raise RuntimeError
REMOVED_PROPERTIES = {
    "default_inertia",
    "default_mass",
}

# Private/internal properties and methods to skip
INTERNAL_PROPERTIES = {
    "_create_simulation_bindings",
    "_create_buffers",
    "update",
    "is_primed",
    "device",
    "object_names",
    "GRAVITY_VEC_W",
    "GRAVITY_VEC_W_TORCH",
    "FORWARD_VEC_B",
    "FORWARD_VEC_B_TORCH",
    "ALL_ENV_MASK",
    "ENV_MASK",
    "ALL_OBJECT_MASK",
    "OBJECT_MASK",
    "num_bodies",
    "num_instances",
}

# Dependency mapping for properties
PROPERTY_DEPENDENCIES = {
    "object_link_lin_vel_w": ["object_link_vel_w"],
    "object_link_ang_vel_w": ["object_link_vel_w"],
    "object_link_lin_vel_b": ["object_link_vel_b"],
    "object_link_ang_vel_b": ["object_link_vel_b"],
    "object_com_pos_w": ["object_com_pose_w"],
    "object_com_quat_w": ["object_com_pose_w"],
    "object_com_lin_vel_b": ["object_com_vel_b"],
    "object_com_ang_vel_b": ["object_com_vel_b"],
    "object_com_lin_vel_w": ["object_com_vel_w"],
    "object_com_ang_vel_w": ["object_com_vel_w"],
    "object_link_pos_w": ["object_link_pose_w"],
    "object_link_quat_w": ["object_link_pose_w"],
    "object_com_lin_acc_w": ["object_com_acc_w"],
    "object_com_ang_acc_w": ["object_com_acc_w"],
    "object_com_quat_b": ["object_com_pose_b"],
}


# =============================================================================
# Benchmark Functions
# =============================================================================


def get_benchmarkable_properties(data: RigidObjectCollectionData) -> list[str]:
    """Get list of properties that can be benchmarked."""
    all_properties = []

    for name in dir(data):
        if name.startswith("_"):
            continue
        if name in DEPRECATED_PROPERTIES:
            continue
        if name in NOT_IMPLEMENTED_PROPERTIES:
            continue
        if name in REMOVED_PROPERTIES:
            continue
        if name in INTERNAL_PROPERTIES:
            continue

        try:
            attr = getattr(type(data), name, None)
            if isinstance(attr, property):
                all_properties.append(name)
        except Exception:
            pass

    return sorted(all_properties)


def setup_mock_environment(config: BenchmarkConfig) -> MockRigidBodyView:
    """Set up the mock environment for benchmarking."""
    # For collection, total count = num_instances * num_bodies
    total_count = config.num_instances * config.num_bodies
    mock_view = MockRigidBodyView(
        count=total_count,
        device=config.device,
    )
    return mock_view


def run_benchmarks(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run all benchmarks for RigidObjectCollectionData."""
    results = []

    # Setup mock environment
    mock_view = setup_mock_environment(config)
    mock_view.set_random_mock_data()

    # Create RigidObjectCollectionData instance
    data = RigidObjectCollectionData(mock_view, config.num_bodies, config.device)

    # Get list of properties to benchmark
    properties = get_benchmarkable_properties(data)

    total_count = config.num_instances * config.num_bodies

    # Generator that updates mock data and invalidates timestamp
    def gen_mock_data(cfg: BenchmarkConfig) -> dict:
        mock_view.set_mock_transforms(torch.randn(total_count, 7, device=cfg.device))
        mock_view.set_mock_velocities(torch.randn(total_count, 6, device=cfg.device))
        mock_view.set_mock_accelerations(torch.randn(total_count, 6, device=cfg.device))
        mock_view.set_mock_coms(torch.randn(total_count, 7, device=cfg.device))
        data._sim_timestamp += 1.0
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

        def prop_accessor(prop=benchmark.method_name, **kwargs):
            return getattr(data, prop)

        print(f"[{i + 1}/{len(benchmarks)}] [DEFAULT] {benchmark.name}...", end=" ", flush=True)

        result = benchmark_method(
            method=prop_accessor,
            method_name=benchmark.name,
            generator=gen_mock_data,
            config=config,
            dependencies=PROPERTY_DEPENDENCIES,
        )
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
        description="Micro-benchmarking framework for RigidObjectCollectionData class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
    parser.add_argument("--num_bodies", type=int, default=4, help="Number of bodies per instance")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filename")
    parser.add_argument("--no_csv", action="store_true", help="Disable CSV output")

    args = parser.parse_args()

    config = BenchmarkConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=args.num_bodies,
        num_joints=0,
        device=args.device,
    )

    results = run_benchmarks(config)

    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)
    print_results(results, include_mode=False)

    if args.output:
        json_filename = args.output
    else:
        json_filename = get_default_output_filename("rigid_object_collection_data_benchmark")

    export_results_json(results, config, hardware_info, json_filename)

    if not args.no_csv:
        csv_filename = json_filename.replace(".json", ".csv")
        export_results_csv(results, csv_filename)


if __name__ == "__main__":
    main()
