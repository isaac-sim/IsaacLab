#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script comparing Isaac Lab's XFormPrimView against Isaac Sim's XFormPrimView.

This script tests the performance of batched transform operations using either
Isaac Lab's implementation or Isaac Sim's implementation.

Usage:
    # Basic benchmark
    ./isaaclab.sh -p scripts/benchmarks/benchmark_xform_prim_view.py --num_envs 1024 --device cuda:0 --headless

    # With profiling enabled (for snakeviz visualization)
    ./isaaclab.sh -p scripts/benchmarks/benchmark_xform_prim_view.py --num_envs 1024 --profile --headless

    # Then visualize with snakeviz:
    snakeviz profile_results/isaaclab_xformprimview.prof
    snakeviz profile_results/isaacsim_xformprimview.prof
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# parse the arguments
args_cli = argparse.Namespace()

parser = argparse.ArgumentParser(description="This script can help you benchmark the performance of XFormPrimView.")

parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
parser.add_argument("--num_iterations", type=int, default=50, help="Number of iterations for each test.")
parser.add_argument(
    "--profile",
    action="store_true",
    help="Enable profiling with cProfile. Results saved as .prof files for snakeviz visualization.",
)
parser.add_argument(
    "--profile-dir",
    type=str,
    default="./profile_results",
    help="Directory to save profile results. Default: ./profile_results",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cProfile
import time
import torch
from typing import Literal

from isaacsim.core.utils.extensions import enable_extension

# compare against latest Isaac Sim implementation
enable_extension("isaacsim.core.experimental.prims")
from isaacsim.core.experimental.prims import XformPrim as IsaacSimXformPrimView

import isaaclab.sim as sim_utils
from isaaclab.sim.views import XformPrimView as IsaacLabXformPrimView


@torch.no_grad()
def benchmark_xform_prim_view(
    api: Literal["isaaclab", "isaacsim"], num_iterations: int
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Benchmark the Xform view class from either Isaac Lab or Isaac Sim.

    Args:
        api: Which API to benchmark ("isaaclab" or "isaacsim").
        num_iterations: Number of iterations to run.

    Returns:
        A tuple of (timing_results, computed_results) where:
        - timing_results: Dictionary containing timing results for various operations
        - computed_results: Dictionary containing the computed values for validation
    """
    timing_results = {}
    computed_results = {}

    # Setup scene
    print("  Setting up scene")
    # Clear stage
    sim_utils.create_new_stage()
    # Create simulation context
    start_time = time.perf_counter()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    stage = sim_utils.get_current_stage()

    print(f"  Time taken to create simulation context: {time.perf_counter() - start_time} seconds")

    # Create prims
    prim_paths = []
    for i in range(args_cli.num_envs):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", stage=stage, translation=(i * 2.0, 0.0, 1.0))
        sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", stage=stage, translation=(0.0, 0.0, 0.0))
        prim_paths.append(f"/World/Env_{i}/Object")
    # Play simulation
    sim.reset()

    # Pattern to match all prims
    pattern = "/World/Env_.*/Object"
    print(f"  Pattern: {pattern}")

    # Create view
    start_time = time.perf_counter()
    if api == "isaaclab":
        xform_view = IsaacLabXFormPrimView(pattern, device=args_cli.device)
    elif api == "isaacsim":
        xform_view = IsaacSimXFormPrimView(pattern)
    else:
        raise ValueError(f"Invalid API: {api}")
    timing_results["init"] = time.perf_counter() - start_time

    if api == "isaaclab":
        num_prims = xform_view.count
    elif api == "isaacsim":
        num_prims = len(xform_view.prims)
    print(f"  XformPrimView managing {num_prims} prims")

    # Benchmark get_world_poses
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        positions, orientations = xform_view.get_world_poses()
        # Ensure tensors are torch tensors
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)
        if not isinstance(orientations, torch.Tensor):
            orientations = torch.tensor(orientations, dtype=torch.float32)

    timing_results["get_world_poses"] = (time.perf_counter() - start_time) / num_iterations

    # Store initial world poses
    computed_results["initial_world_positions"] = positions.clone()
    computed_results["initial_world_orientations"] = orientations.clone()

    # Benchmark set_world_poses
    new_positions = positions.clone()
    new_positions[:, 2] += 0.1
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        if api == "isaaclab":
            xform_view.set_world_poses(new_positions, orientations)
        elif api == "isaacsim":
            xform_view.set_world_poses(new_positions.cpu().numpy(), orientations.cpu().numpy())
    timing_results["set_world_poses"] = (time.perf_counter() - start_time) / num_iterations

    # Get world poses after setting to verify
    positions_after_set, orientations_after_set = xform_view.get_world_poses()
    if not isinstance(positions_after_set, torch.Tensor):
        positions_after_set = torch.tensor(positions_after_set, dtype=torch.float32)
    if not isinstance(orientations_after_set, torch.Tensor):
        orientations_after_set = torch.tensor(orientations_after_set, dtype=torch.float32)
    computed_results["world_positions_after_set"] = positions_after_set.clone()
    computed_results["world_orientations_after_set"] = orientations_after_set.clone()

    # Benchmark get_local_poses
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        translations, orientations_local = xform_view.get_local_poses()
        # Ensure tensors are torch tensors
        if not isinstance(translations, torch.Tensor):
            translations = torch.tensor(translations, dtype=torch.float32, device=args_cli.device)
        if not isinstance(orientations_local, torch.Tensor):
            orientations_local = torch.tensor(orientations_local, dtype=torch.float32, device=args_cli.device)

    timing_results["get_local_poses"] = (time.perf_counter() - start_time) / num_iterations

    # Store initial local poses
    computed_results["initial_local_translations"] = translations.clone()
    computed_results["initial_local_orientations"] = orientations_local.clone()

    # Benchmark set_local_poses
    new_translations = translations.clone()
    new_translations[:, 2] += 0.1
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        if api == "isaaclab":
            xform_view.set_local_poses(new_translations, orientations_local)
        elif api == "isaacsim":
            xform_view.set_local_poses(new_translations.cpu().numpy(), orientations_local.cpu().numpy())
    timing_results["set_local_poses"] = (time.perf_counter() - start_time) / num_iterations

    # Get local poses after setting to verify
    translations_after_set, orientations_local_after_set = xform_view.get_local_poses()
    if not isinstance(translations_after_set, torch.Tensor):
        translations_after_set = torch.tensor(translations_after_set, dtype=torch.float32)
    if not isinstance(orientations_local_after_set, torch.Tensor):
        orientations_local_after_set = torch.tensor(orientations_local_after_set, dtype=torch.float32)
    computed_results["local_translations_after_set"] = translations_after_set.clone()
    computed_results["local_orientations_after_set"] = orientations_local_after_set.clone()

    # Benchmark combined get operation
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        positions, orientations = xform_view.get_world_poses()
        translations, local_orientations = xform_view.get_local_poses()
    timing_results["get_both"] = (time.perf_counter() - start_time) / num_iterations

    # close simulation
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()

    return timing_results, computed_results


def compare_results(
    isaaclab_computed: dict[str, torch.Tensor], isaacsim_computed: dict[str, torch.Tensor], tolerance: float = 1e-4
) -> dict[str, dict[str, float]]:
    """Compare computed results between Isaac Lab and Isaac Sim implementations.

    Args:
        isaaclab_computed: Computed values from Isaac Lab's XformPrimView.
        isaacsim_computed: Computed values from Isaac Sim's XformPrimView.
        tolerance: Tolerance for numerical comparison.

    Returns:
        Dictionary containing comparison statistics (max difference, mean difference, etc.) for each result.
    """
    comparison_stats = {}

    for key in isaaclab_computed.keys():
        if key not in isaacsim_computed:
            print(f"  Warning: Key '{key}' not found in Isaac Sim results")
            continue

        isaaclab_val = isaaclab_computed[key]
        isaacsim_val = isaacsim_computed[key]

        # Compute differences
        diff = torch.abs(isaaclab_val - isaacsim_val)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        # Check if within tolerance
        all_close = torch.allclose(isaaclab_val, isaacsim_val, atol=tolerance, rtol=0)

        comparison_stats[key] = {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "all_close": all_close,
        }

    return comparison_stats


def print_comparison_results(comparison_stats: dict[str, dict[str, float]], tolerance: float):
    """Print comparison results between implementations.

    Args:
        comparison_stats: Dictionary containing comparison statistics.
        tolerance: Tolerance used for comparison.
    """
    # Check if all results match
    all_match = all(stats["all_close"] for stats in comparison_stats.values())

    if all_match:
        # Compact output when everything matches
        print("\n" + "=" * 100)
        print("RESULT COMPARISON: Isaac Lab vs Isaac Sim")
        print("=" * 100)
        print(f"✓ All computed values match within tolerance ({tolerance})")
        print("=" * 100)
        print()
    else:
        # Detailed output when there are mismatches
        print("\n" + "=" * 100)
        print("RESULT COMPARISON: Isaac Lab vs Isaac Sim")
        print("=" * 100)
        print(f"{'Computed Value':<40} {'Max Diff':<15} {'Mean Diff':<15} {'Match':<10}")
        print("-" * 100)

        for key, stats in comparison_stats.items():
            # Format the key for display
            display_key = key.replace("_", " ").title()
            match_str = "✓ Yes" if stats["all_close"] else "✗ No"

            print(f"{display_key:<40} {stats['max_diff']:<15.6e} {stats['mean_diff']:<15.6e} {match_str:<10}")

        print("=" * 100)
        print(f"\n✗ Some results differ beyond tolerance ({tolerance})")
        print("  This may indicate implementation differences between Isaac Lab and Isaac Sim")
        print()


def print_results(
    isaaclab_results: dict[str, float], isaacsim_results: dict[str, float], num_prims: int, num_iterations: int
):
    """Print benchmark results in a formatted table.

    Args:
        isaaclab_results: Results from Isaac Lab's XformPrimView benchmark.
        isaacsim_results: Results from Isaac Sim's XformPrimView benchmark.
        num_prims: Number of prims tested.
        num_iterations: Number of iterations run.
    """
    print("\n" + "=" * 100)
    print(f"BENCHMARK RESULTS: {num_prims} prims, {num_iterations} iterations")
    print("=" * 100)

    # Print header
    print(f"{'Operation':<25} {'Isaac Lab (ms)':<25} {'Isaac Sim (ms)':<25} {'Speedup':<15}")
    print("-" * 100)

    # Print each operation
    operations = [
        ("Initialization", "init"),
        ("Get World Poses", "get_world_poses"),
        ("Set World Poses", "set_world_poses"),
        ("Get Local Poses", "get_local_poses"),
        ("Set Local Poses", "set_local_poses"),
        ("Get Both (World+Local)", "get_both"),
    ]

    for op_name, op_key in operations:
        isaaclab_time = isaaclab_results.get(op_key, 0) * 1000  # Convert to ms
        isaacsim_time = isaacsim_results.get(op_key, 0) * 1000  # Convert to ms

        if isaaclab_time > 0 and isaacsim_time > 0:
            speedup = isaacsim_time / isaaclab_time
            print(f"{op_name:<25} {isaaclab_time:>20.4f}    {isaacsim_time:>20.4f}    {speedup:>10.2f}x")
        else:
            print(f"{op_name:<25} {isaaclab_time:>20.4f}    {'N/A':<20}    {'N/A':<15}")

    print("=" * 100)

    # Calculate and print total time
    if isaaclab_results and isaacsim_results:
        total_isaaclab = sum(isaaclab_results.values()) * 1000
        total_isaacsim = sum(isaacsim_results.values()) * 1000
        overall_speedup = total_isaacsim / total_isaaclab if total_isaaclab > 0 else 0
        print(f"\n{'Total Time':<25} {total_isaaclab:>20.4f}    {total_isaacsim:>20.4f}    {overall_speedup:>10.2f}x")
    else:
        total_isaaclab = sum(isaaclab_results.values()) * 1000
        print(f"\n{'Total Time':<25} {total_isaaclab:>20.4f}    {'N/A':<20}    {'N/A':<15}")

    print("\nNotes:")
    print("  - Times are averaged over all iterations")
    print("  - Speedup = (Isaac Sim time) / (Isaac Lab time)")
    print("  - Speedup > 1.0 means Isaac Lab is faster")
    print("  - Speedup < 1.0 means Isaac Sim is faster")
    print()


def main():
    """Main benchmark function."""
    print("=" * 100)
    print("XformPrimView Benchmark")
    print("=" * 100)
    print("Configuration:")
    print(f"  Number of environments: {args_cli.num_envs}")
    print(f"  Iterations per test: {args_cli.num_iterations}")
    print(f"  Device: {args_cli.device}")
    print(f"  Profiling: {'Enabled' if args_cli.profile else 'Disabled'}")
    if args_cli.profile:
        print(f"  Profile directory: {args_cli.profile_dir}")
    print()

    # Create profile directory if profiling is enabled
    if args_cli.profile:
        import os

        os.makedirs(args_cli.profile_dir, exist_ok=True)

    # Benchmark Isaac Lab XformPrimView
    print("Benchmarking XformPrimView from Isaac Lab...")
    if args_cli.profile:
        profiler_isaaclab = cProfile.Profile()
        profiler_isaaclab.enable()

    isaaclab_timing, isaaclab_computed = benchmark_xform_prim_view(
        api="isaaclab", num_iterations=args_cli.num_iterations
    )

    if args_cli.profile:
        profiler_isaaclab.disable()
        profile_file_isaaclab = f"{args_cli.profile_dir}/isaaclab_XformPrimView.prof"
        profiler_isaaclab.dump_stats(profile_file_isaaclab)
        print(f"  Profile saved to: {profile_file_isaaclab}")

    print("  Done!")
    print()

    # Benchmark Isaac Sim XformPrimView
    print("Benchmarking Isaac Sim XformPrimView...")
    if args_cli.profile:
        profiler_isaacsim = cProfile.Profile()
        profiler_isaacsim.enable()

    isaacsim_timing, isaacsim_computed = benchmark_xform_prim_view(
        api="isaacsim", num_iterations=args_cli.num_iterations
    )

    if args_cli.profile:
        profiler_isaacsim.disable()
        profile_file_isaacsim = f"{args_cli.profile_dir}/isaacsim_XformPrimView.prof"
        profiler_isaacsim.dump_stats(profile_file_isaacsim)
        print(f"  Profile saved to: {profile_file_isaacsim}")

    print("  Done!")
    print()

    # Print timing results
    print_results(isaaclab_timing, isaacsim_timing, args_cli.num_envs, args_cli.num_iterations)

    # Compare computed results
    print("\nComparing computed results...")
    comparison_stats = compare_results(isaaclab_computed, isaacsim_computed, tolerance=1e-6)
    print_comparison_results(comparison_stats, tolerance=1e-4)

    # Print profiling instructions if enabled
    if args_cli.profile:
        print("\n" + "=" * 100)
        print("PROFILING RESULTS")
        print("=" * 100)
        print("Profile files have been saved. To visualize with snakeviz, run:")
        print(f"  snakeviz {profile_file_isaaclab}")
        print(f"  snakeviz {profile_file_isaacsim}")
        print("\nAlternatively, use pstats to analyze in terminal:")
        print(f"  python -m pstats {profile_file_isaaclab}")
        print("=" * 100)
        print()

    # Clean up
    sim_utils.SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
