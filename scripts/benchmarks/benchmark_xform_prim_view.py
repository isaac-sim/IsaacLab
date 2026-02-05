# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script comparing XformPrimView implementations across different APIs.

This script tests the performance of batched transform operations using:
- Isaac Lab's XformPrimView implementation with USD backend
- Isaac Lab's XformPrimView implementation with Fabric backend
- Isaac Sim's XformPrimView implementation (legacy)
- Isaac Sim Experimental's XformPrim implementation (latest)

Usage:
    # Basic benchmark (all APIs)
    ./isaaclab.sh -p scripts/benchmarks/benchmark_xform_prim_view.py --num_envs 1024 --device cuda:0 --headless

    # With profiling enabled (for snakeviz visualization)
    ./isaaclab.sh -p scripts/benchmarks/benchmark_xform_prim_view.py --num_envs 1024 --profile --headless

    # Then visualize with snakeviz:
    snakeviz profile_results/isaaclab_usd_benchmark.prof
    snakeviz profile_results/isaaclab_fabric_benchmark.prof
    snakeviz profile_results/isaacsim_benchmark.prof
    snakeviz profile_results/isaacsim_exp_benchmark.prof
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# parse the arguments
args_cli = argparse.Namespace()

parser = argparse.ArgumentParser(description="This script can help you benchmark the performance of XformPrimView.")

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
from typing import Literal

import torch

from isaacsim.core.prims import XFormPrim as IsaacSimXformPrimView
from isaacsim.core.utils.extensions import enable_extension

# compare against latest Isaac Sim implementation
enable_extension("isaacsim.core.experimental.prims")
from isaacsim.core.experimental.prims import XformPrim as IsaacSimExperimentalXformPrimView

import isaaclab.sim as sim_utils
from isaaclab.sim.views import XformPrimView as IsaacLabXformPrimView


@torch.no_grad()
def benchmark_xform_prim_view(  # noqa: C901
    api: Literal["isaaclab-usd", "isaaclab-fabric", "isaacsim-usd", "isaacsim-fabric", "isaacsim-exp"],
    num_iterations: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Benchmark the Xform view class from Isaac Lab, Isaac Sim, or Isaac Sim Experimental.

    Args:
        api: Which API to benchmark:
            - "isaaclab-usd": Isaac Lab XformPrimView with USD backend
            - "isaaclab-fabric": Isaac Lab XformPrimView with Fabric backend
            - "isaacsim-usd": Isaac Sim legacy XformPrimView with USD (usd=True)
            - "isaacsim-fabric": Isaac Sim legacy XformPrimView with Fabric (usd=False)
            - "isaacsim-exp": Isaac Sim Experimental XformPrim
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
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        use_fabric=api in ("isaaclab-fabric", "isaacsim-fabric"),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
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
    if api == "isaaclab-usd" or api == "isaaclab-fabric":
        xform_view = IsaacLabXformPrimView(pattern, device=args_cli.device, validate_xform_ops=False)
    elif api == "isaacsim-usd":
        xform_view = IsaacSimXformPrimView(pattern, reset_xform_properties=False, usd=True)
    elif api == "isaacsim-fabric":
        xform_view = IsaacSimXformPrimView(pattern, reset_xform_properties=False, usd=False)
    elif api == "isaacsim-exp":
        xform_view = IsaacSimExperimentalXformPrimView(pattern)
    else:
        raise ValueError(f"Invalid API: {api}")
    timing_results["init"] = time.perf_counter() - start_time

    if api in ("isaaclab-usd", "isaaclab-fabric", "isaacsim-usd", "isaacsim-fabric"):
        num_prims = xform_view.count
    elif api == "isaacsim-exp":
        num_prims = len(xform_view.prims)
    print(f"  XformView managing {num_prims} prims")

    # Benchmark get_world_poses
    # Warmup call to initialize Fabric (if needed) - excluded from timing
    positions, orientations = xform_view.get_world_poses()

    # Now time the actual iterations (steady-state performance)
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        positions, orientations = xform_view.get_world_poses()

    # Ensure tensors are torch tensors (do this AFTER timing)
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
        if api in ("isaaclab-usd", "isaaclab-fabric", "isaacsim-usd", "isaacsim-fabric"):
            xform_view.set_world_poses(new_positions, orientations)
        elif api == "isaacsim-exp":
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
    # Warmup call (though local poses use USD, so minimal overhead)
    translations, orientations_local = xform_view.get_local_poses()

    # Now time the actual iterations
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        translations, orientations_local = xform_view.get_local_poses()
    # Ensure tensors are torch tensors (do this AFTER timing)
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
        if api in ("isaaclab-usd", "isaaclab-fabric", "isaacsim-usd", "isaacsim-fabric"):
            xform_view.set_local_poses(new_translations, orientations_local)
        elif api == "isaacsim-exp":
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
    # Warmup call (Fabric should already be initialized by now, but for consistency)
    positions, orientations = xform_view.get_world_poses()
    translations, local_orientations = xform_view.get_local_poses()

    # Now time the actual iterations
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        positions, orientations = xform_view.get_world_poses()
        translations, local_orientations = xform_view.get_local_poses()
    timing_results["get_both"] = (time.perf_counter() - start_time) / num_iterations

    # Benchmark interleaved set/get (realistic workflow pattern)
    # Pre-convert tensors for experimental API to avoid conversion overhead in loop
    if api == "isaacsim-exp":
        new_positions_np = new_positions.cpu().numpy()
        orientations_np = orientations

    # Warmup
    if api in ("isaaclab-usd", "isaaclab-fabric", "isaacsim-usd", "isaacsim-fabric"):
        xform_view.set_world_poses(new_positions, orientations)
        positions, orientations = xform_view.get_world_poses()
    elif api == "isaacsim-exp":
        xform_view.set_world_poses(new_positions_np, orientations_np)
        positions, orientations = xform_view.get_world_poses()
        positions = torch.tensor(positions, dtype=torch.float32)
        orientations = torch.tensor(orientations, dtype=torch.float32)

    # Now time the actual interleaved iterations
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        # Write then immediately read (common pattern: set pose, verify/query result)
        if api in ("isaaclab-usd", "isaaclab-fabric", "isaacsim-usd", "isaacsim-fabric"):
            xform_view.set_world_poses(new_positions, orientations)
            positions, orientations = xform_view.get_world_poses()
        elif api == "isaacsim-exp":
            xform_view.set_world_poses(new_positions_np, orientations_np)
            positions, orientations = xform_view.get_world_poses()

    timing_results["interleaved_world_set_get"] = (time.perf_counter() - start_time) / num_iterations

    # close simulation
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()

    return timing_results, computed_results


def compare_results(
    results_dict: dict[str, dict[str, torch.Tensor]], tolerance: float = 1e-4
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare computed results across multiple implementations.

    Only compares implementations using the same data path:
    - USD implementations (isaaclab-usd, isaacsim-usd) are compared with each other
    - Fabric implementations (isaaclab-fabric, isaacsim-fabric) are compared with each other

    This is because Fabric is designed for write-first workflows and may not match
    USD reads on initialization.

    Args:
        results_dict: Dictionary mapping API names to their computed values.
        tolerance: Tolerance for numerical comparison.

    Returns:
        Nested dictionary: {comparison_pair: {metric: {stats}}}, e.g.,
        {"isaaclab-usd_vs_isaacsim-usd": {"initial_world_positions": {"max_diff": 0.001, ...}}}
    """
    comparison_stats = {}

    # Group APIs by their data path (USD vs Fabric)
    usd_apis = [api for api in results_dict.keys() if "usd" in api and "fabric" not in api]
    fabric_apis = [api for api in results_dict.keys() if "fabric" in api]

    # Compare within USD group
    for i, api1 in enumerate(usd_apis):
        for api2 in usd_apis[i + 1 :]:
            pair_key = f"{api1}_vs_{api2}"
            comparison_stats[pair_key] = {}

            computed1 = results_dict[api1]
            computed2 = results_dict[api2]

            for key in computed1.keys():
                if key not in computed2:
                    print(f"  Warning: Key '{key}' not found in {api2} results")
                    continue

                val1 = computed1[key]
                val2 = computed2[key]

                # Compute differences
                diff = torch.abs(val1 - val2)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()

                # Check if within tolerance
                all_close = torch.allclose(val1, val2, atol=tolerance, rtol=0)

                comparison_stats[pair_key][key] = {
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "all_close": all_close,
                }

    # Compare within Fabric group
    for i, api1 in enumerate(fabric_apis):
        for api2 in fabric_apis[i + 1 :]:
            pair_key = f"{api1}_vs_{api2}"
            comparison_stats[pair_key] = {}

            computed1 = results_dict[api1]
            computed2 = results_dict[api2]

            for key in computed1.keys():
                if key not in computed2:
                    print(f"  Warning: Key '{key}' not found in {api2} results")
                    continue

                val1 = computed1[key]
                val2 = computed2[key]

                # Compute differences
                diff = torch.abs(val1 - val2)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()

                # Check if within tolerance
                all_close = torch.allclose(val1, val2, atol=tolerance, rtol=0)

                comparison_stats[pair_key][key] = {
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "all_close": all_close,
                }

    return comparison_stats


def print_comparison_results(comparison_stats: dict[str, dict[str, dict[str, float]]], tolerance: float):
    """Print comparison results across implementations.

    Args:
        comparison_stats: Nested dictionary containing comparison statistics for each API pair.
        tolerance: Tolerance used for comparison.
    """
    if not comparison_stats:
        print("\n" + "=" * 100)
        print("RESULT COMPARISON")
        print("=" * 100)
        print("ℹ️  No comparisons performed.")
        print("   USD and Fabric implementations are not compared because Fabric uses a")
        print("   write-first workflow and may not match USD reads on initialization.")
        print("=" * 100)
        print()
        return

    for pair_key, pair_stats in comparison_stats.items():
        # Format the pair key for display (e.g., "isaaclab_vs_isaacsim" -> "Isaac Lab vs Isaac Sim")
        api1, api2 = pair_key.split("_vs_")
        display_api1 = api1.replace("-", " ").title()
        display_api2 = api2.replace("-", " ").title()
        comparison_title = f"{display_api1} vs {display_api2}"

        # Check if all results match
        all_match = all(stats["all_close"] for stats in pair_stats.values())

        if all_match:
            # Compact output when everything matches
            print("\n" + "=" * 100)
            print(f"RESULT COMPARISON: {comparison_title}")
            print("=" * 100)
            print(f"✓ All computed values match within tolerance ({tolerance})")
            print("=" * 100)
        else:
            # Detailed output when there are mismatches
            print("\n" + "=" * 100)
            print(f"RESULT COMPARISON: {comparison_title}")
            print("=" * 100)
            print(f"{'Computed Value':<40} {'Max Diff':<15} {'Mean Diff':<15} {'Match':<10}")
            print("-" * 100)

            for key, stats in pair_stats.items():
                # Format the key for display
                display_key = key.replace("_", " ").title()
                match_str = "✓ Yes" if stats["all_close"] else "✗ No"

                print(f"{display_key:<40} {stats['max_diff']:<15.6e} {stats['mean_diff']:<15.6e} {match_str:<10}")

            print("=" * 100)
            print(f"\n✗ Some results differ beyond tolerance ({tolerance})")

            # Special note for Isaac Sim Fabric local pose bug
            if "isaacsim-fabric" in pair_key and any("local_translations_after_set" in k for k in pair_stats.keys()):
                if not pair_stats.get("local_translations_after_set", {}).get("all_close", True):
                    print("\n  ⚠️  Known Issue: Isaac Sim Fabric has a bug where get_local_poses() returns stale")
                    print("     values after set_local_poses(). Isaac Lab Fabric correctly returns updated values.")
                    print("     This is a correctness issue in Isaac Sim's implementation, not Isaac Lab's.")
            else:
                print(f"  This may indicate implementation differences between {display_api1} and {display_api2}")

    print()


def print_results(results_dict: dict[str, dict[str, float]], num_prims: int, num_iterations: int):
    """Print benchmark results in a formatted table.

    Args:
        results_dict: Dictionary mapping API names to their timing results.
        num_prims: Number of prims tested.
        num_iterations: Number of iterations run.
    """
    print("\n" + "=" * 100)
    print(f"BENCHMARK RESULTS: {num_prims} prims, {num_iterations} iterations")
    print("=" * 100)

    api_names = list(results_dict.keys())
    # Format API names for display
    display_names = [name.replace("-", " ").replace("_", " ").title() for name in api_names]

    # Calculate column width based on number of APIs
    col_width = 20

    # Print header
    header = f"{'Operation':<25}"
    for display_name in display_names:
        header += f" {display_name + ' (ms)':<{col_width}}"
    print(header)
    print("-" * 100)

    # Print each operation
    operations = [
        ("Initialization", "init"),
        ("Get World Poses", "get_world_poses"),
        ("Set World Poses", "set_world_poses"),
        ("Get Local Poses", "get_local_poses"),
        ("Set Local Poses", "set_local_poses"),
        ("Get Both (World+Local)", "get_both"),
        ("Interleaved World Set→Get", "interleaved_world_set_get"),
    ]

    for op_name, op_key in operations:
        row = f"{op_name:<25}"
        for api_name in api_names:
            api_time = results_dict[api_name].get(op_key, 0) * 1000  # Convert to ms
            row += f" {api_time:>{col_width - 1}.4f}"
        print(row)

    print("=" * 100)

    # Calculate and print total time
    total_row = f"{'Total Time':<25}"
    for api_name in api_names:
        total_time = sum(results_dict[api_name].values()) * 1000
        total_row += f" {total_time:>{col_width - 1}.4f}"
    print(f"\n{total_row}")

    # Calculate speedups relative to Isaac Lab USD (baseline)
    if "isaaclab-usd" in api_names:
        print("\n" + "=" * 100)
        print("SPEEDUP vs Isaac Lab USD (Baseline)")
        print("=" * 100)
        print(f"{'Operation':<25}", end="")
        for api_name, display_name in zip(api_names, display_names):
            if api_name != "isaaclab-usd":
                print(f" {display_name:<{col_width}}", end="")
        print()
        print("-" * 100)

        isaaclab_usd_results = results_dict["isaaclab-usd"]
        for op_name, op_key in operations:
            print(f"{op_name:<25}", end="")
            isaaclab_usd_time = isaaclab_usd_results.get(op_key, 0)
            for api_name, display_name in zip(api_names, display_names):
                if api_name != "isaaclab-usd":
                    api_time = results_dict[api_name].get(op_key, 0)
                    if isaaclab_usd_time > 0 and api_time > 0:
                        speedup = isaaclab_usd_time / api_time
                        print(f" {speedup:>{col_width - 1}.2f}x", end="")
                    else:
                        print(f" {'N/A':>{col_width}}", end="")
            print()

        # Overall speedup
        print("=" * 100)
        print(f"{'Overall Speedup':<25}", end="")
        total_isaaclab_usd = sum(isaaclab_usd_results.values())
        for api_name, display_name in zip(api_names, display_names):
            if api_name != "isaaclab-usd":
                total_api = sum(results_dict[api_name].values())
                if total_isaaclab_usd > 0 and total_api > 0:
                    overall_speedup = total_isaaclab_usd / total_api
                    print(f" {overall_speedup:>{col_width - 1}.2f}x", end="")
                else:
                    print(f" {'N/A':>{col_width}}", end="")
        print()

    print("\n" + "=" * 100)
    print("\nNotes:")
    print("  - Times are averaged over all iterations")
    print("  - Speedup = (Isaac Lab USD time) / (Other API time)")
    print("  - Speedup > 1.0 means the other API is faster than Isaac Lab USD")
    print("  - Speedup < 1.0 means the other API is slower than Isaac Lab USD")
    print()


def main():
    """Main benchmark function."""
    print("=" * 100)
    print("XformPrimView Benchmark - Comparing Multiple APIs")
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

    # Dictionary to store all results
    all_timing_results = {}
    all_computed_results = {}
    profile_files = {}

    # APIs to benchmark
    apis_to_test = [
        ("isaaclab-usd", "Isaac Lab XformPrimView (USD)"),
        ("isaaclab-fabric", "Isaac Lab XformPrimView (Fabric)"),
        ("isaacsim-usd", "Isaac Sim XformPrimView (USD)"),
        ("isaacsim-fabric", "Isaac Sim XformPrimView (Fabric)"),
        ("isaacsim-exp", "Isaac Sim Experimental XformPrim"),
    ]

    # Benchmark each API
    for api_key, api_name in apis_to_test:
        print(f"Benchmarking {api_name}...")

        if args_cli.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        # Cast api_key to Literal type for type checker
        timing, computed = benchmark_xform_prim_view(
            api=api_key,  # type: ignore[arg-type]
            num_iterations=args_cli.num_iterations,
        )

        if args_cli.profile:
            profiler.disable()
            profile_file = f"{args_cli.profile_dir}/{api_key.replace('-', '_')}_benchmark.prof"
            profiler.dump_stats(profile_file)
            profile_files[api_key] = profile_file
            print(f"  Profile saved to: {profile_file}")

        all_timing_results[api_key] = timing
        all_computed_results[api_key] = computed

        print("  Done!")
        print()

    # Print timing results
    print_results(all_timing_results, args_cli.num_envs, args_cli.num_iterations)

    # Compare computed results
    print("\nComparing computed results across APIs...")
    comparison_stats = compare_results(all_computed_results, tolerance=1e-6)
    print_comparison_results(comparison_stats, tolerance=1e-4)

    # Print profiling instructions if enabled
    if args_cli.profile:
        print("\n" + "=" * 100)
        print("PROFILING RESULTS")
        print("=" * 100)
        print("Profile files have been saved. To visualize with snakeviz, run:")
        for api_key, profile_file in profile_files.items():
            api_display = api_key.replace("-", " ").title()
            print(f"  # {api_display}")
            print(f"  snakeviz {profile_file}")
        print("\nAlternatively, use pstats to analyze in terminal:")
        print("  python -m pstats <profile_file>")
        print("=" * 100)
        print()

    # Clean up
    sim_utils.SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
