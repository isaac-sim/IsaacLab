# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script comparing XformPrimView vs PhysX RigidBodyView for transform operations.

This script tests the performance of batched transform operations using:

- Isaac Lab's XformPrimView (USD-based)
- Isaac Lab's XformPrimView (Fabric-based)
- PhysX RigidBodyView (PhysX tensors-based, as used in RigidObject)

Note:
    XformPrimView operates on USD attributes directly (useful for non-physics prims),
    or on Fabric attributes when Fabric is enabled.
    while RigidBodyView requires rigid body physics components and operates on PhysX tensors.
    This benchmark helps understand the performance trade-offs between the two approaches.

Usage:
    # Basic benchmark
    ./isaaclab.sh -p scripts/benchmarks/benchmark_view_comparison.py --num_envs 1024 --device cuda:0 --headless

    # With profiling enabled (for snakeviz visualization)
    ./isaaclab.sh -p scripts/benchmarks/benchmark_view_comparison.py --num_envs 1024 --profile --headless

    # Then visualize with snakeviz:
    snakeviz profile_results/xform_view_benchmark.prof
    snakeviz profile_results/physx_view_benchmark.prof
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# parse the arguments
args_cli = argparse.Namespace()

parser = argparse.ArgumentParser(description="Benchmark XformPrimView vs PhysX RigidBodyView performance.")

parser.add_argument("--num_envs", type=int, default=1000, help="Number of environments to simulate.")
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

from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim.views import XformPrimView


@torch.no_grad()
def benchmark_view(view_type: str, num_iterations: int) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Benchmark the specified view class.

    Args:
        view_type: Type of view to benchmark ("xform", "xform_fabric", or "physx").
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
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, use_fabric=(view_type == "xform_fabric"))
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim_utils.get_current_stage()

    print(f"  Time taken to create simulation context: {time.perf_counter() - start_time:.4f} seconds")

    # create a rigid object
    object_cfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    # Create prims
    for i in range(args_cli.num_envs):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", stage=stage, translation=(i * 2.0, 0.0, 0.0))
        object_cfg.func(f"/World/Env_{i}/Object", object_cfg, translation=(0.0, 0.0, 1.0))

    # Play simulation
    sim.reset()

    # Pattern to match all prims
    pattern = "/World/Env_.*/Object" if view_type == "xform" else "/World/Env_*/Object"
    print(f"  Pattern: {pattern}")

    # Create view based on type
    start_time = time.perf_counter()
    if view_type == "xform":
        view = XformPrimView(pattern, device=args_cli.device, validate_xform_ops=False)
        num_prims = view.count
        view_name = "XformPrimView (USD)"
    elif view_type == "xform_fabric":
        if "cuda" not in args_cli.device:
            raise ValueError("Fabric backend requires CUDA. Please use --device cuda:0 for this benchmark.")
        view = XformPrimView(pattern, device=args_cli.device, validate_xform_ops=False)
        num_prims = view.count
        view_name = "XformPrimView (Fabric)"
    else:  # physx
        physics_sim_view = SimulationManager.get_physics_sim_view()
        view = physics_sim_view.create_rigid_body_view(pattern)
        num_prims = view.count
        view_name = "PhysX RigidBodyView"
    timing_results["init"] = time.perf_counter() - start_time
    # prepare indices for benchmarking
    all_indices = torch.arange(num_prims, device=args_cli.device)

    print(f"  {view_name} managing {num_prims} prims")

    # Fabric is write-first: initialize it to match USD before benchmarking reads.
    if view_type == "xform_fabric" and num_prims > 0:
        init_positions = torch.zeros((num_prims, 3), dtype=torch.float32, device=args_cli.device)
        init_positions[:, 0] = 2.0 * torch.arange(num_prims, device=args_cli.device, dtype=torch.float32)
        init_positions[:, 2] = 1.0
        init_orientations = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * num_prims, dtype=torch.float32, device=args_cli.device
        )
        view.set_world_poses(init_positions, init_orientations)

    # Benchmark get_world_poses
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        if view_type in ("xform", "xform_fabric"):
            positions, orientations = view.get_world_poses()
        else:  # physx
            transforms = view.get_transforms()
            positions = transforms[:, :3]
            orientations = transforms[:, 3:7]
            # Convert quaternion from xyzw to wxyz
            orientations = math_utils.convert_quat(orientations, to="wxyz")
    timing_results["get_world_poses"] = (time.perf_counter() - start_time) / num_iterations

    # Store initial world poses
    computed_results["initial_world_positions"] = positions.clone()
    computed_results["initial_world_orientations"] = orientations.clone()

    # Benchmark set_world_poses
    new_positions = positions.clone()
    new_positions[:, 2] += 0.5
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        if view_type in ("xform", "xform_fabric"):
            view.set_world_poses(new_positions, orientations)
        else:  # physx
            # Convert quaternion from wxyz to xyzw for PhysX
            orientations_xyzw = math_utils.convert_quat(orientations, to="xyzw")
            new_transforms = torch.cat([new_positions, orientations_xyzw], dim=-1)
            view.set_transforms(new_transforms, indices=all_indices)
    timing_results["set_world_poses"] = (time.perf_counter() - start_time) / num_iterations

    # Get world poses after setting to verify
    if view_type in ("xform", "xform_fabric"):
        positions_after_set, orientations_after_set = view.get_world_poses()
    else:  # physx
        transforms_after = view.get_transforms()
        positions_after_set = transforms_after[:, :3]
        orientations_after_set = math_utils.convert_quat(transforms_after[:, 3:7], to="wxyz")
    computed_results["world_positions_after_set"] = positions_after_set.clone()
    computed_results["world_orientations_after_set"] = orientations_after_set.clone()

    # close simulation
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()

    return timing_results, computed_results


def compare_results(
    results_dict: dict[str, dict[str, torch.Tensor]], tolerance: float = 1e-4
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare computed results across implementations.

    Args:
        results_dict: Dictionary mapping implementation names to their computed values.
        tolerance: Tolerance for numerical comparison.

    Returns:
        Nested dictionary: {comparison_pair: {metric: {stats}}}
    """
    comparison_stats = {}
    impl_names = list(results_dict.keys())

    # Compare each pair of implementations
    for i, impl1 in enumerate(impl_names):
        for impl2 in impl_names[i + 1 :]:
            pair_key = f"{impl1}_vs_{impl2}"
            comparison_stats[pair_key] = {}

            computed1 = results_dict[impl1]
            computed2 = results_dict[impl2]

            for key in computed1.keys():
                if key not in computed2:
                    continue

                val1 = computed1[key]
                val2 = computed2[key]

                # Skip zero tensors (not applicable tests)
                if torch.all(val1 == 0) or torch.all(val2 == 0):
                    continue

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
    """Print comparison results.

    Args:
        comparison_stats: Nested dictionary containing comparison statistics.
        tolerance: Tolerance used for comparison.
    """
    for pair_key, pair_stats in comparison_stats.items():
        if not pair_stats:  # Skip if no comparable results
            continue

        # Format the pair key for display
        impl1, impl2 = pair_key.split("_vs_")
        display_impl1 = impl1.replace("_", " ").title()
        display_impl2 = impl2.replace("_", " ").title()
        comparison_title = f"{display_impl1} vs {display_impl2}"

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
            print(f"  This may indicate implementation differences between {display_impl1} and {display_impl2}")

    print()


def print_results(results_dict: dict[str, dict[str, float]], num_prims: int, num_iterations: int):
    """Print benchmark results in a formatted table.

    Args:
        results_dict: Dictionary mapping implementation names to their timing results.
        num_prims: Number of prims tested.
        num_iterations: Number of iterations run.
    """
    print("\n" + "=" * 100)
    print(f"BENCHMARK RESULTS: {num_prims} prims, {num_iterations} iterations")
    print("=" * 100)

    impl_names = list(results_dict.keys())
    # Format names for display
    display_names = [name.replace("_", " ").title() for name in impl_names]

    # Calculate column width
    col_width = 20

    # Print header
    header = f"{'Operation':<30}"
    for display_name in display_names:
        header += f" {display_name + ' (ms)':<{col_width}}"
    print(header)
    print("-" * 100)

    # Print each operation
    operations = [
        ("Initialization", "init"),
        ("Get World Poses", "get_world_poses"),
        ("Set World Poses", "set_world_poses"),
    ]

    for op_name, op_key in operations:
        row = f"{op_name:<30}"
        for impl_name in impl_names:
            impl_time = results_dict[impl_name].get(op_key, 0) * 1000  # Convert to ms
            row += f" {impl_time:>{col_width - 1}.4f}"
        print(row)

    print("=" * 100)

    # Calculate and print total time (excluding N/A operations)
    total_row = f"{'Total Time':<30}"
    for impl_name in impl_names:
        if impl_name == "physx_view":
            # Exclude local pose operations for PhysX
            total_time = (
                results_dict[impl_name].get("init", 0) * 1000
                + results_dict[impl_name].get("get_world_poses", 0) * 1000
                + results_dict[impl_name].get("set_world_poses", 0) * 1000
            )
        else:
            total_time = sum(results_dict[impl_name].values()) * 1000
        total_row += f" {total_time:>{col_width - 1}.4f}"
    print(f"\n{total_row}")

    # Calculate speedups relative to XformPrimView (USD baseline)
    if "xform_view" in impl_names:
        print("\n" + "=" * 100)
        print("SPEEDUP vs XformPrimView (USD)")
        print("=" * 100)
        print(f"{'Operation':<30}", end="")
        for impl_name, display_name in zip(impl_names, display_names):
            if impl_name != "xform_view":
                print(f" {display_name + ' Speedup':<{col_width}}", end="")
        print()
        print("-" * 100)

        xform_results = results_dict["xform_view"]
        for op_name, op_key in operations:
            print(f"{op_name:<30}", end="")
            xform_time = xform_results.get(op_key, 0)
            for impl_name, display_name in zip(impl_names, display_names):
                if impl_name != "xform_view":
                    impl_time = results_dict[impl_name].get(op_key, 0)
                    if xform_time > 0 and impl_time > 0:
                        speedup = impl_time / xform_time
                        print(f" {speedup:>{col_width - 1}.2f}x", end="")
                    else:
                        print(f" {'N/A':>{col_width}}", end="")
            print()

        # Overall speedup (only world pose operations)
        print("=" * 100)
        print(f"{'Overall Speedup (World Ops)':<30}", end="")
        total_xform = (
            xform_results.get("init", 0)
            + xform_results.get("get_world_poses", 0)
            + xform_results.get("set_world_poses", 0)
        )
        for impl_name, display_name in zip(impl_names, display_names):
            if impl_name != "xform_view":
                total_impl = (
                    results_dict[impl_name].get("init", 0)
                    + results_dict[impl_name].get("get_world_poses", 0)
                    + results_dict[impl_name].get("set_world_poses", 0)
                )
                if total_xform > 0 and total_impl > 0:
                    overall_speedup = total_impl / total_xform
                    print(f" {overall_speedup:>{col_width - 1}.2f}x", end="")
                else:
                    print(f" {'N/A':>{col_width}}", end="")
        print()

    print("\n" + "=" * 100)
    print("\nNotes:")
    print("  - Times are averaged over all iterations")
    print("  - Speedup = (Implementation time) / (XformPrimView USD time)")
    print("  - Speedup > 1.0 means USD XformPrimView is faster")
    print("  - Speedup < 1.0 means the implementation is faster than USD")
    print("  - PhysX View requires rigid body physics components")
    print("  - XformPrimView works with any Xform prim (physics or non-physics)")
    print("  - PhysX View does not support local pose operations directly")
    print()


def main():
    """Main benchmark function."""
    print("=" * 100)
    print("View Comparison Benchmark - XformPrimView vs PhysX RigidBodyView")
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

    # Implementations to benchmark
    implementations = [
        ("xform_view", "XformPrimView (USD)", "xform"),
        ("xform_fabric_view", "XformPrimView (Fabric)", "xform_fabric"),
        ("physx_view", "PhysX RigidBodyView", "physx"),
    ]

    # Benchmark each implementation
    for impl_key, impl_name, view_type in implementations:
        print(f"Benchmarking {impl_name}...")

        if args_cli.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        timing, computed = benchmark_view(view_type=view_type, num_iterations=args_cli.num_iterations)

        if args_cli.profile:
            profiler.disable()
            profile_file = f"{args_cli.profile_dir}/{impl_key}_benchmark.prof"
            profiler.dump_stats(profile_file)
            profile_files[impl_key] = profile_file
            print(f"  Profile saved to: {profile_file}")

        all_timing_results[impl_key] = timing
        all_computed_results[impl_key] = computed

        print("  Done!")
        print()

    # Print timing results
    print_results(all_timing_results, args_cli.num_envs, args_cli.num_iterations)

    # Compare computed results
    print("\nComparing computed results across implementations...")
    comparison_stats = compare_results(all_computed_results, tolerance=1e-4)
    print_comparison_results(comparison_stats, tolerance=1e-4)

    # Print profiling instructions if enabled
    if args_cli.profile:
        print("\n" + "=" * 100)
        print("PROFILING RESULTS")
        print("=" * 100)
        print("Profile files have been saved. To visualize with snakeviz, run:")
        for impl_key, profile_file in profile_files.items():
            impl_display = impl_key.replace("_", " ").title()
            print(f"  # {impl_display}")
            print(f"  snakeviz {profile_file}")
        print("\nAlternatively, use pstats to analyze in terminal:")
        print("  python -m pstats <profile_file>")
        print("=" * 100)
        print()

    # Clean up
    sim_utils.SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
