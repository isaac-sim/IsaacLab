# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
import time
import warnings
from dataclasses import dataclass
from typing import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import warp as wp

from isaaclab_newton.assets.articulation.articulation_data import ArticulationData

# Import mock classes from shared module
from mock_interface import MockNewtonArticulationView, MockNewtonModel

# Initialize Warp
wp.init()

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_git_info() -> dict:
    """Get git repository information.

    Returns:
        Dictionary containing git commit hash, branch, and other info.
    """
    import subprocess
    import os

    git_info = {
        "commit_hash": "Unknown",
        "commit_hash_short": "Unknown",
        "branch": "Unknown",
        "commit_date": "Unknown",
    }

    # Get the directory of this file to find the repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Get full commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
            git_info["commit_hash_short"] = result.stdout.strip()[:8]

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Get commit date
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_date"] = result.stdout.strip()

    except Exception:
        pass

    return git_info


def get_hardware_info() -> dict:
    """Gather hardware information for the benchmark.

    Returns:
        Dictionary containing CPU, GPU, and memory information.
    """
    import platform
    import os

    hardware_info = {
        "cpu": {},
        "gpu": {},
        "memory": {},
        "system": {},
    }

    # System info
    hardware_info["system"] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }

    # CPU info
    hardware_info["cpu"]["name"] = platform.processor() or "Unknown"
    hardware_info["cpu"]["physical_cores"] = os.cpu_count()

    # Try to get more detailed CPU info on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    hardware_info["cpu"]["name"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    # Memory info
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
            for line in meminfo.split("\n"):
                if "MemTotal" in line:
                    # Convert from KB to GB
                    mem_kb = int(line.split()[1])
                    hardware_info["memory"]["total_gb"] = round(mem_kb / (1024 * 1024), 2)
                    break
    except Exception:
        # Fallback using psutil if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            hardware_info["memory"]["total_gb"] = round(mem.total / (1024**3), 2)
        except ImportError:
            hardware_info["memory"]["total_gb"] = "Unknown"

    # GPU info using PyTorch
    if torch.cuda.is_available():
        hardware_info["gpu"]["available"] = True
        hardware_info["gpu"]["count"] = torch.cuda.device_count()
        hardware_info["gpu"]["devices"] = []

        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            hardware_info["gpu"]["devices"].append({
                "index": i,
                "name": gpu_props.name,
                "total_memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "multi_processor_count": gpu_props.multi_processor_count,
            })

        # Current device info
        current_device = torch.cuda.current_device()
        hardware_info["gpu"]["current_device"] = current_device
        hardware_info["gpu"]["current_device_name"] = torch.cuda.get_device_name(current_device)
    else:
        hardware_info["gpu"]["available"] = False

    # PyTorch and CUDA versions
    hardware_info["gpu"]["pytorch_version"] = torch.__version__
    if torch.cuda.is_available():
        try:
            import torch.version as torch_version
            cuda_version = getattr(torch_version, "cuda", None)
            hardware_info["gpu"]["cuda_version"] = cuda_version if cuda_version else "Unknown"
        except Exception:
            hardware_info["gpu"]["cuda_version"] = "Unknown"
    else:
        hardware_info["gpu"]["cuda_version"] = "N/A"

    # Warp info
    try:
        warp_version = getattr(wp.config, "version", None)
        hardware_info["warp"] = {"version": warp_version if warp_version else "Unknown"}
    except Exception:
        hardware_info["warp"] = {"version": "Unknown"}

    return hardware_info


def print_hardware_info(hardware_info: dict):
    """Print hardware information to console.

    Args:
        hardware_info: Dictionary containing hardware information.
    """
    print("\n" + "=" * 80)
    print("HARDWARE INFORMATION")
    print("=" * 80)

    # System
    sys_info = hardware_info.get("system", {})
    print(f"\nSystem: {sys_info.get('platform', 'Unknown')} {sys_info.get('platform_release', '')}")
    print(f"Python: {sys_info.get('python_version', 'Unknown')}")

    # CPU
    cpu_info = hardware_info.get("cpu", {})
    print(f"\nCPU: {cpu_info.get('name', 'Unknown')}")
    print(f"     Cores: {cpu_info.get('physical_cores', 'Unknown')}")

    # Memory
    mem_info = hardware_info.get("memory", {})
    print(f"\nMemory: {mem_info.get('total_gb', 'Unknown')} GB")

    # GPU
    gpu_info = hardware_info.get("gpu", {})
    if gpu_info.get("available", False):
        print(f"\nGPU: {gpu_info.get('current_device_name', 'Unknown')}")
        for device in gpu_info.get("devices", []):
            print(f"     [{device['index']}] {device['name']}")
            print(f"         Memory: {device['total_memory_gb']} GB")
            print(f"         Compute Capability: {device['compute_capability']}")
            print(f"         SM Count: {device['multi_processor_count']}")
        print(f"\nPyTorch: {gpu_info.get('pytorch_version', 'Unknown')}")
        print(f"CUDA: {gpu_info.get('cuda_version', 'Unknown')}")
    else:
        print("\nGPU: Not available")

    warp_info = hardware_info.get("warp", {})
    print(f"Warp: {warp_info.get('version', 'Unknown')}")

    # Repository info (get separately since it's not part of hardware)
    repo_info = get_git_info()
    print(f"\nRepository:")
    print(f"     Commit: {repo_info.get('commit_hash_short', 'Unknown')}")
    print(f"     Branch: {repo_info.get('branch', 'Unknown')}")
    print(f"     Date:   {repo_info.get('commit_date', 'Unknown')}")
    print("=" * 80)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmarking framework."""

    num_iterations: int = 10000
    """Number of iterations to run each function."""

    warmup_steps: int = 10
    """Number of warmup steps before timing."""

    num_instances: int = 16384
    """Number of articulation instances."""

    num_bodies: int = 12
    """Number of bodies per articulation."""

    num_joints: int = 11
    """Number of joints per articulation."""

    device: str = "cuda:0"
    """Device to run benchmarks on."""


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    """Name of the benchmarked function/property."""

    mean_time_us: float
    """Mean execution time in microseconds."""

    std_time_us: float
    """Standard deviation of execution time in microseconds."""

    num_iterations: int
    """Number of iterations run."""

    skipped: bool = False
    """Whether the benchmark was skipped."""

    skip_reason: str = ""
    """Reason for skipping the benchmark."""

    dependencies: list[str] | None = None
    """List of parent properties that were pre-computed before timing."""


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
        try:
            for dep in dependencies:
                _ = getattr(articulation_data, dep)
        except Exception:
            pass

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


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a formatted table.

    Args:
        results: List of BenchmarkResults to print.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Separate skipped and completed results
    completed = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    # Sort completed by mean time (descending)
    completed.sort(key=lambda x: x.mean_time_us, reverse=True)

    # Print header
    print(f"\n{'Property Name':<45} {'Mean (µs)':<15} {'Std (µs)':<15} {'Iterations':<12}")
    print("-" * 87)

    # Print completed results
    for result in completed:
        # Add marker for properties with pre-computed dependencies
        name_display = result.name
        if result.dependencies:
            name_display = f"{result.name} *"
        print(f"{name_display:<45} {result.mean_time_us:>12.2f}   {result.std_time_us:>12.2f}   {result.num_iterations:>10}")

    # Print summary statistics
    if completed:
        print("-" * 87)
        mean_times = [r.mean_time_us for r in completed]
        print(f"\nSummary Statistics:")
        print(f"  Total properties benchmarked: {len(completed)}")
        print(f"  Fastest: {min(mean_times):.2f} µs ({completed[-1].name})")
        print(f"  Slowest: {max(mean_times):.2f} µs ({completed[0].name})")
        print(f"  Average: {np.mean(mean_times):.2f} µs")
        print(f"  Median:  {np.median(mean_times):.2f} µs")

        # Print note about derived properties
        derived_count = sum(1 for r in completed if r.dependencies)
        if derived_count > 0:
            print(f"\n  * = Derived property ({derived_count} total). Dependencies were pre-computed")
            print(f"      before timing to measure isolated overhead.")

    # Print skipped results
    if skipped:
        print(f"\nSkipped Properties ({len(skipped)}):")
        for result in skipped:
            print(f"  - {result.name}: {result.skip_reason}")


def export_results_csv(results: list[BenchmarkResult], filename: str):
    """Export benchmark results to a CSV file.

    Args:
        results: List of BenchmarkResults to export.
        filename: Output CSV filename.
    """
    import csv

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Property", "Mean (µs)", "Std (µs)", "Iterations", "Dependencies", "Skipped", "Skip Reason"])

        for result in results:
            deps_str = ", ".join(result.dependencies) if result.dependencies else ""
            writer.writerow([
                result.name,
                f"{result.mean_time_us:.4f}" if not result.skipped else "",
                f"{result.std_time_us:.4f}" if not result.skipped else "",
                result.num_iterations,
                deps_str,
                result.skipped,
                result.skip_reason,
            ])

    print(f"\nResults exported to {filename}")


def export_results_json(
    results: list[BenchmarkResult], config: BenchmarkConfig, hardware_info: dict, filename: str
):
    """Export benchmark results to a JSON file.

    Args:
        results: List of BenchmarkResults to export.
        config: Benchmark configuration used.
        hardware_info: Hardware information dictionary.
        filename: Output JSON filename.
    """
    import json
    from datetime import datetime

    # Separate completed and skipped results
    completed = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    # Get git repository info
    git_info = get_git_info()

    # Build the JSON structure
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "repository": git_info,
            "config": {
                "num_iterations": config.num_iterations,
                "warmup_steps": config.warmup_steps,
                "num_instances": config.num_instances,
                "num_bodies": config.num_bodies,
                "num_joints": config.num_joints,
                "device": config.device,
            },
            "hardware": hardware_info,
            "total_properties": len(results),
            "benchmarked_properties": len(completed),
            "skipped_properties": len(skipped),
        },
        "results": [],
        "skipped": [],
    }

    # Add individual results
    for result in completed:
        result_entry = {
            "name": result.name,
            "mean_time_us": result.mean_time_us,
            "std_time_us": result.std_time_us,
            "num_iterations": result.num_iterations,
        }
        if result.dependencies:
            result_entry["dependencies"] = result.dependencies
            result_entry["note"] = "Timing excludes dependency computation (dependencies were pre-computed)"
        output["results"].append(result_entry)

    # Add skipped properties
    for result in skipped:
        output["skipped"].append({
            "name": result.name,
            "reason": result.skip_reason,
        })

    # Write JSON file
    with open(filename, "w") as jsonfile:
        json.dump(output, jsonfile, indent=2)

    print(f"Results exported to {filename}")


def get_default_output_filename() -> str:
    """Generate default output filename with current date and time."""
    from datetime import datetime

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"articulation_data_{datetime_str}.json"


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
    print_results(results)

    # Export to JSON (default)
    if not args.no_json:
        output_filename = args.output if args.output else get_default_output_filename()
        export_results_json(results, config, hardware_info, output_filename)

    # Export to CSV if requested
    if args.export_csv:
        export_results_csv(results, args.export_csv)


if __name__ == "__main__":
    main()

