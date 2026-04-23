# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script comparing FrameView implementations across backends.

Compares batched transform operation performance across:
- Isaac Lab FrameView (USD backend) -- baseline
- Isaac Lab FrameView (Fabric backend)
- Isaac Lab FrameView (Newton backend)

Usage:
    ./isaaclab.sh -p scripts/benchmarks/benchmark_xform_prim_view.py --num_envs 1024 --device cuda:0 --headless

    # With profiling
    ./isaaclab.sh -p scripts/benchmarks/benchmark_xform_prim_view.py --num_envs 1024 --profile --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark FrameView performance across backends.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
parser.add_argument("--num_iterations", type=int, default=50, help="Number of iterations for each test.")
parser.add_argument("--profile", action="store_true", help="Enable cProfile profiling.")
parser.add_argument("--profile_dir", type=str, default="./profile_results", help="Directory for .prof files.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cProfile
import time
from typing import Literal

import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.views import NewtonSiteFrameView
from isaaclab_physx.sim.views import FabricFrameView

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.sim.views import UsdFrameView
from isaaclab.utils import configclass


@configclass
class _NewtonSceneCfg(InteractiveSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


# ------------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------------


@torch.no_grad()
def benchmark_frame_view(  # noqa: C901
    api: Literal["isaaclab-usd", "isaaclab-fabric", "isaaclab-newton-site"],
    num_iterations: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Benchmark get/set world/local poses for the given FrameView backend."""
    timing_results: dict[str, float] = {}
    computed_results: dict[str, torch.Tensor] = {}
    device = args_cli.device
    num_envs = args_cli.num_envs

    # -- Scene setup (backend-specific) --------------------------------

    print("  Setting up scene")
    cleanup = None

    if api == "isaaclab-newton-site":
        newton_cfg = SimulationCfg(device=device, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
        ctx = build_simulation_context(device=device, sim_cfg=newton_cfg, add_ground_plane=True)
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None
        InteractiveScene(_NewtonSceneCfg(num_envs=num_envs, env_spacing=2.0))

        stage = sim_utils.get_current_stage()
        for i in range(num_envs):
            prim = stage.DefinePrim(f"/World/envs/env_{i}/Object/Sensor", "Xform")
            sim_utils.standardize_xform_ops(prim)
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.1, 0.0, 0.05))
            prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        sim.reset()

        start_time = time.perf_counter()
        xform_view = NewtonSiteFrameView("/World/envs/env_.*/Object/Sensor", device=device)
        timing_results["init"] = time.perf_counter() - start_time
        cleanup = lambda: ctx.__exit__(None, None, None)  # noqa: E731

    else:
        sim_utils.create_new_stage()
        start_time = time.perf_counter()
        use_fabric = api == "isaaclab-fabric"
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=device, use_fabric=use_fabric))
        stage = sim_utils.get_current_stage()

        for i in range(num_envs):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", stage=stage, translation=(i * 2.0, 0.0, 1.0))
            sim_utils.create_prim(f"/World/Env_{i}/Object", "Xform", stage=stage, translation=(0.0, 0.0, 0.0))

        sim.reset()

        pattern = "/World/Env_.*/Object"
        start_time = time.perf_counter()
        ViewClass = FabricFrameView if use_fabric else UsdFrameView
        xform_view = ViewClass(pattern, device=device, validate_xform_ops=False)
        timing_results["init"] = time.perf_counter() - start_time
        cleanup = lambda: sim.clear_instance()  # noqa: E731

    num_prims = xform_view.count
    print(f"  {api} managing {num_prims} prims")

    is_newton = api == "isaaclab-newton-site"

    def to_torch(a):
        return wp.to_torch(a) if isinstance(a, wp.array) else a

    try:
        # -- Warmup --------------------------------------------------------
        xform_view.get_world_poses()

        # -- get_world_poses -----------------------------------------------
        if is_newton:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            positions, orientations = xform_view.get_world_poses()
        if is_newton:
            torch.cuda.synchronize()
        timing_results["get_world_poses"] = (time.perf_counter() - start_time) / num_iterations

        positions_t = to_torch(positions)
        orientations_t = to_torch(orientations)
        computed_results["initial_world_positions"] = positions_t.clone()
        computed_results["initial_world_orientations"] = orientations_t.clone()

        # -- set_world_poses -----------------------------------------------
        if is_newton:
            new_positions = wp.clone(positions)
            wp.to_torch(new_positions)[:, 2] += 0.1
        else:
            new_positions = positions_t.clone()
            new_positions[:, 2] += 0.1

        if is_newton:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            xform_view.set_world_poses(new_positions, orientations)
        if is_newton:
            torch.cuda.synchronize()
        timing_results["set_world_poses"] = (time.perf_counter() - start_time) / num_iterations

        pa, oa = xform_view.get_world_poses()
        computed_results["world_positions_after_set"] = to_torch(pa).clone()
        computed_results["world_orientations_after_set"] = to_torch(oa).clone()

        # -- get_local_poses -----------------------------------------------
        if is_newton:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            translations, orientations_local = xform_view.get_local_poses()
        if is_newton:
            torch.cuda.synchronize()
        timing_results["get_local_poses"] = (time.perf_counter() - start_time) / num_iterations

        translations_t = to_torch(translations)
        orientations_local_t = to_torch(orientations_local)
        computed_results["initial_local_translations"] = translations_t.clone()
        computed_results["initial_local_orientations"] = orientations_local_t.clone()

        # -- set_local_poses -----------------------------------------------
        if is_newton:
            new_translations = wp.clone(translations)
            wp.to_torch(new_translations)[:, 2] += 0.1
        else:
            new_translations = translations_t.clone()
            new_translations[:, 2] += 0.1

        if is_newton:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            xform_view.set_local_poses(new_translations, orientations_local)
        if is_newton:
            torch.cuda.synchronize()
        timing_results["set_local_poses"] = (time.perf_counter() - start_time) / num_iterations

        ta, ola = xform_view.get_local_poses()
        computed_results["local_translations_after_set"] = to_torch(ta).clone()
        computed_results["local_orientations_after_set"] = to_torch(ola).clone()

        # -- get_both (world + local) --------------------------------------
        if is_newton:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            xform_view.get_world_poses()
            xform_view.get_local_poses()
        if is_newton:
            torch.cuda.synchronize()
        timing_results["get_both"] = (time.perf_counter() - start_time) / num_iterations

        # -- interleaved set -> get ----------------------------------------
        if is_newton:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            xform_view.set_world_poses(new_positions, orientations)
            xform_view.get_world_poses()
        if is_newton:
            torch.cuda.synchronize()
        timing_results["interleaved_world_set_get"] = (time.perf_counter() - start_time) / num_iterations

    finally:
        if cleanup:
            cleanup()

    return timing_results, computed_results


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def print_results(results_dict: dict[str, dict[str, float]], num_prims: int, num_iterations: int):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 120)
    print(f"BENCHMARK RESULTS: {num_prims} prims, {num_iterations} iterations")
    print("=" * 120)

    api_names = list(results_dict.keys())
    display_names = [name.replace("-", " ").title() for name in api_names]
    col_width = 22

    header = f"{'Operation':<28}"
    for dn in display_names:
        header += f" {dn + ' (ms)':>{col_width}}"
    print(header)
    print("-" * 120)

    operations = [
        ("Initialization", "init"),
        ("Get World Poses", "get_world_poses"),
        ("Set World Poses", "set_world_poses"),
        ("Get Local Poses", "get_local_poses"),
        ("Set Local Poses", "set_local_poses"),
        ("Get Both (World+Local)", "get_both"),
        ("Interleaved World Set->Get", "interleaved_world_set_get"),
    ]

    for op_name, op_key in operations:
        row = f"{op_name:<28}"
        for name in api_names:
            val = results_dict[name].get(op_key, 0) * 1000
            row += f" {val:>{col_width}.4f}"
        print(row)

    print("=" * 120)

    total_row = f"{'Total':<28}"
    for name in api_names:
        total_row += f" {sum(results_dict[name].values()) * 1000:>{col_width}.4f}"
    print(f"\n{total_row}")

    baseline = "isaaclab-usd"
    if baseline in results_dict and len(api_names) > 1:
        print("\n" + "=" * 120)
        print(f"SPEEDUP vs {baseline.replace('-', ' ').title()}")
        print("=" * 120)
        header = f"{'Operation':<28}"
        for name in api_names:
            if name != baseline:
                header += f" {name.replace('-', ' ').title():>{col_width}}"
        print(header)
        print("-" * 120)

        base = results_dict[baseline]
        for op_name, op_key in operations:
            row = f"{op_name:<28}"
            base_t = base.get(op_key, 0)
            for name in api_names:
                if name != baseline:
                    impl_t = results_dict[name].get(op_key, 0)
                    if base_t > 0 and impl_t > 0:
                        row += f" {base_t / impl_t:>{col_width}.2f}x"
                    else:
                        row += f" {'N/A':>{col_width}}"
            print(row)

        print("=" * 120)
        print(f"{'Overall':>28}", end="")
        total_base = sum(base.values())
        for name in api_names:
            if name != baseline:
                total_impl = sum(results_dict[name].values())
                if total_base > 0 and total_impl > 0:
                    print(f" {total_base / total_impl:>{col_width}.2f}x", end="")
                else:
                    print(f" {'N/A':>{col_width}}", end="")
        print()

    print("\n" + "=" * 120)
    print("\nNotes:")
    print("  - Times are averaged over all iterations")
    print("  - Speedup > 1.0 means faster than USD baseline")
    print()


def main():
    print("=" * 120)
    print("FrameView Benchmark")
    print("=" * 120)
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Iterations:   {args_cli.num_iterations}")
    print(f"  Device:       {args_cli.device}")
    print()

    if args_cli.profile:
        import os

        os.makedirs(args_cli.profile_dir, exist_ok=True)

    all_timing = {}
    all_computed = {}
    profile_files = {}

    apis = [
        ("isaaclab-usd", "Isaac Lab FrameView (USD)"),
        ("isaaclab-fabric", "Isaac Lab FrameView (Fabric)"),
        ("isaaclab-newton-site", "Isaac Lab FrameView (Newton Site)"),
    ]

    for api_key, api_name in apis:
        print(f"Benchmarking {api_name}...")

        if args_cli.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        timing, computed = benchmark_frame_view(api=api_key, num_iterations=args_cli.num_iterations)

        if args_cli.profile:
            profiler.disable()
            pf = f"{args_cli.profile_dir}/{api_key.replace('-', '_')}_benchmark.prof"
            profiler.dump_stats(pf)
            profile_files[api_key] = pf
            print(f"  Profile saved to: {pf}")

        all_timing[api_key] = timing
        all_computed[api_key] = computed
        print("  Done!\n")

    print_results(all_timing, args_cli.num_envs, args_cli.num_iterations)

    if args_cli.profile:
        print("\nProfile files:")
        for key, pf in profile_files.items():
            print(f"  snakeviz {pf}")
        print()

    sim_utils.SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
