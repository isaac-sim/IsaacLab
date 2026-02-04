# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark I/O utilities for gathering system info and exporting results.

This module provides functions for:
- Gathering git repository information
- Gathering hardware information (CPU, GPU, memory)
- Printing hardware information to console
- Exporting benchmark results to JSON/CSV
- Printing formatted benchmark result tables
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

if TYPE_CHECKING:
    from .benchmark_core import BenchmarkConfig, BenchmarkResult


def get_git_info() -> dict:
    """Get git repository information.

    Returns:
        Dictionary containing git commit hash, branch, and other info.
    """
    git_info = {
        "commit_hash": "Unknown",
        "commit_hash_short": "Unknown",
        "branch": "Unknown",
        "commit_date": "Unknown",
    }

    # Get the directory of the caller to find the repo root
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
    hardware_info = {
        "cpu": {
            "name": platform.processor() or "Unknown",
            "physical_cores": os.cpu_count(),
        },
        "gpu": {},
        "memory": {},
        "system": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        },
    }

    # Try to get more detailed CPU info on Linux
    with contextlib.suppress(Exception):
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    hardware_info["cpu"]["name"] = line.split(":")[1].strip()
                    break

    # Memory info
    try:
        with open("/proc/meminfo") as f:
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
            hardware_info["gpu"]["devices"].append(
                {
                    "index": i,
                    "name": gpu_props.name,
                    "total_memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                    "multi_processor_count": gpu_props.multi_processor_count,
                }
            )

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

    # Repository info
    repo_info = get_git_info()
    print("\nRepository:")
    print(f"     Commit: {repo_info.get('commit_hash_short', 'Unknown')}")
    print(f"     Branch: {repo_info.get('branch', 'Unknown')}")
    print(f"     Date:   {repo_info.get('commit_date', 'Unknown')}")
    print("=" * 80)


def print_results(results: list[BenchmarkResult], include_mode: bool = True):
    """Print benchmark results in a formatted table.

    Args:
        results: List of BenchmarkResults to print.
        include_mode: Whether to separate and compare results by input mode.
    """
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Separate skipped and completed results
    completed = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    # Get unique modes present in the results
    modes = sorted(list({r.mode for r in completed if r.mode is not None}))

    if include_mode and modes:
        # Separate by mode
        results_by_mode = {mode: [r for r in completed if r.mode == mode] for mode in modes}

        # Print comparison table if multiple modes
        if len(modes) > 1:
            print("\n" + "-" * 120)
            mode_headers = " vs ".join([m.capitalize() for m in modes])
            print(f"COMPARISON: {mode_headers}")
            print("-" * 120)

            # Use the first mode as baseline (usually 'warp')
            baseline_mode = modes[0]

            header = f"{'Method Name':<35}"
            for mode in modes:
                header += f" {mode.capitalize()[:10]:<12}"

            # Add slowdown columns for non-baseline modes
            for mode in modes[1:]:
                header += f" {mode.capitalize()[:6]} Slow.<14"

            print(header)
            print("-" * 120)

            results_by_name = {}
            for mode in modes:
                for r in results_by_mode[mode]:
                    if r.name not in results_by_name:
                        results_by_name[r.name] = {}
                    results_by_name[r.name][mode] = r

            for name in sorted(results_by_name.keys()):
                row_results = results_by_name[name]
                if baseline_mode not in row_results:
                    continue  # Skip if baseline missing

                baseline_time = row_results[baseline_mode].mean_time_us

                row_str = f"{name:<35} {baseline_time:>9.2f}"

                # Time columns
                for mode in modes[1:]:
                    if mode in row_results:
                        row_str += f"   {row_results[mode].mean_time_us:>9.2f}"
                    else:
                        row_str += "   " + "N/A".rjust(9)

                # Slowdown columns
                for mode in modes[1:]:
                    if mode in row_results and baseline_time > 0:
                        slowdown = row_results[mode].mean_time_us / baseline_time
                        row_str += f"   {slowdown:>10.2f}x"
                    else:
                        row_str += "   " + "N/A".rjust(11)

                print(row_str)

        # Print individual results by mode
        for mode in modes:
            mode_results = results_by_mode[mode]
            if mode_results:
                print("\n" + "-" * 100)
                print(f"{mode.upper()}")
                print("-" * 100)

                # Sort by mean time (descending)
                mode_results_sorted = sorted(mode_results, key=lambda x: x.mean_time_us, reverse=True)

                print(f"{'Method Name':<45} {'Mean (µs)':<15} {'Std (µs)':<15} {'Iterations':<12}")
                print("-" * 87)

                for result in mode_results_sorted:
                    print(
                        f"{result.name:<45} {result.mean_time_us:>12.2f}   {result.std_time_us:>12.2f}  "
                        f" {result.num_iterations:>10}"
                    )

                # Summary stats
                mean_times = [r.mean_time_us for r in mode_results_sorted]
                print("-" * 87)
                if mean_times:
                    print(f"  Fastest: {min(mean_times):.2f} µs ({mode_results_sorted[-1].name})")
                    print(f"  Slowest: {max(mean_times):.2f} µs ({mode_results_sorted[0].name})")
                    print(f"  Average: {np.mean(mean_times):.2f} µs")
    else:
        # No mode separation - just print all results
        completed_sorted = sorted(completed, key=lambda x: x.mean_time_us, reverse=True)

        print(f"\n{'Property Name':<45} {'Mean (µs)':<15} {'Std (µs)':<15} {'Iterations':<12}")
        print("-" * 87)

        for result in completed_sorted:
            name_display = result.name
            if result.dependencies:
                name_display = f"{result.name} *"
            print(
                f"{name_display:<45} {result.mean_time_us:>12.2f}   {result.std_time_us:>12.2f}  "
                f" {result.num_iterations:>10}"
            )

        # Print summary statistics
        if completed_sorted:
            print("-" * 87)
            mean_times = [r.mean_time_us for r in completed_sorted]
            print("\nSummary Statistics:")
            print(f"  Total benchmarked: {len(completed_sorted)}")
            print(f"  Fastest: {min(mean_times):.2f} µs ({completed_sorted[-1].name})")
            print(f"  Slowest: {max(mean_times):.2f} µs ({completed_sorted[0].name})")
            print(f"  Average: {np.mean(mean_times):.2f} µs")
            print(f"  Median:  {np.median(mean_times):.2f} µs")

            # Print note about derived properties
            derived_count = sum(1 for r in completed_sorted if r.dependencies)
            if derived_count > 0:
                print(f"\n  * = Derived property ({derived_count} total). Dependencies were pre-computed")
                print("      before timing to measure isolated overhead.")

    # Print skipped results
    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for result in skipped:
            mode_str = f" [{result.mode}]" if result.mode else ""
            print(f"  - {result.name}{mode_str}: {result.skip_reason}")


def export_results_json(
    results: list[BenchmarkResult],
    config: BenchmarkConfig,
    hardware_info: dict,
    filename: str,
    include_mode: bool = True,
):
    """Export benchmark results to a JSON file.

    Args:
        results: List of BenchmarkResults to export.
        config: Benchmark configuration used.
        hardware_info: Hardware information dictionary.
        filename: Output JSON filename.
        include_mode: Whether to include mode separation in output.
    """
    completed = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    git_info = get_git_info()

    # Build config dict - only include relevant fields
    config_dict = {
        "num_iterations": config.num_iterations,
        "warmup_steps": config.warmup_steps,
        "num_instances": config.num_instances,
        "num_bodies": config.num_bodies,
        "device": config.device,
    }
    if hasattr(config, "num_joints") and config.num_joints > 0:
        config_dict["num_joints"] = config.num_joints
    if hasattr(config, "mode"):
        config_dict["mode"] = config.mode

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "repository": git_info,
            "config": config_dict,
            "hardware": hardware_info,
            "total_benchmarks": len(results),
            "completed_benchmarks": len(completed),
            "skipped_benchmarks": len(skipped),
        },
        "results": {},
        "comparison": [],
        "skipped": [],
    }

    # Get unique modes present in the results
    modes = sorted(list({r.mode for r in completed if r.mode is not None}))

    if include_mode and modes:
        for mode in modes:
            output["results"][mode] = []

        for result in completed:
            result_entry = {
                "name": result.name,
                "mean_time_us": result.mean_time_us,
                "std_time_us": result.std_time_us,
                "num_iterations": result.num_iterations,
            }
            if result.dependencies:
                result_entry["dependencies"] = result.dependencies

            if result.mode in output["results"]:
                output["results"][result.mode].append(result_entry)

        # Add comparison data
        results_by_name = {}
        for r in completed:
            if r.mode is not None:
                if r.name not in results_by_name:
                    results_by_name[r.name] = {}
                results_by_name[r.name][r.mode] = r

        baseline_mode = modes[0]  # Usually 'warp' or fastest

        for name in results_by_name:
            if baseline_mode in results_by_name[name]:
                baseline_time = results_by_name[name][baseline_mode].mean_time_us
                comparison_entry = {
                    "name": name,
                    f"{baseline_mode}_time_us": baseline_time,
                }

                for mode in modes:
                    if mode == baseline_mode:
                        continue
                    if mode in results_by_name[name]:
                        mode_time = results_by_name[name][mode].mean_time_us
                        comparison_entry[f"{mode}_time_us"] = mode_time
                        comparison_entry[f"{mode}_slowdown"] = mode_time / baseline_time if baseline_time > 0 else None

                output["comparison"].append(comparison_entry)
    else:
        output["results"] = []
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

    # Add skipped
    for result in skipped:
        entry = {"name": result.name, "reason": result.skip_reason}
        if result.mode:
            entry["mode"] = result.mode
        output["skipped"].append(entry)

    with open(filename, "w") as jsonfile:
        json.dump(output, jsonfile, indent=2)

    print(f"\nResults exported to {filename}")


def export_results_csv(results: list[BenchmarkResult], filename: str):
    """Export benchmark results to a CSV file.

    Args:
        results: List of BenchmarkResults to export.
        filename: Output CSV filename.
    """
    import csv

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Name",
                "Mode",
                "Mean (µs)",
                "Std (µs)",
                "Iterations",
                "Dependencies",
                "Skipped",
                "Skip Reason",
            ]
        )

        for result in results:
            deps_str = ", ".join(result.dependencies) if result.dependencies else ""
            mode_str = result.mode if result.mode else ""
            writer.writerow(
                [
                    result.name,
                    mode_str,
                    f"{result.mean_time_us:.4f}" if not result.skipped else "",
                    f"{result.std_time_us:.4f}" if not result.skipped else "",
                    result.num_iterations,
                    deps_str,
                    result.skipped,
                    result.skip_reason,
                ]
            )

    print(f"\nResults exported to {filename}")


def get_default_output_filename(prefix: str = "benchmark") -> str:
    """Generate default output filename with current date and time.

    Args:
        prefix: Prefix for the filename (e.g., "articulation_benchmark").

    Returns:
        Filename string with timestamp.
    """
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{datetime_str}.json"
