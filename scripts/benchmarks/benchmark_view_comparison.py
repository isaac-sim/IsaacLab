# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script comparing FrameView backends and PhysX RigidBodyView.

Compares batched transform operation performance across:

- **USD** (baseline): Isaac Lab's FrameView via USD XformCache
- **Fabric**: Isaac Lab's FrameView via Fabric GPU arrays
- **Newton**: Isaac Lab's Newton FrameView via Warp site kernels
- **PhysX**: PhysX RigidBodyView via PhysX tensor API (reference)

Usage:
    # All backends
    ./isaaclab.sh -p scripts/benchmarks/benchmark_view_comparison.py --num_envs 1024 --device cuda:0 --headless

    # Select specific backends
    ./isaaclab.sh -p scripts/benchmarks/benchmark_view_comparison.py --backends usd fabric newton --headless

    # With profiling
    ./isaaclab.sh -p scripts/benchmarks/benchmark_view_comparison.py --num_envs 1024 --profile --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark FrameView backends and PhysX RigidBodyView.")

parser.add_argument("--num_envs", type=int, default=1000, help="Number of environments to simulate.")
parser.add_argument("--num_iterations", type=int, default=50, help="Number of iterations for each test.")
parser.add_argument(
    "--backends",
    nargs="+",
    default=["usd", "fabric", "newton", "physx"],
    choices=["usd", "fabric", "newton", "physx"],
    help="Backends to benchmark. Default: all four.",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="Enable profiling with cProfile. Results saved as .prof files for snakeviz visualization.",
)
parser.add_argument(
    "--profile_dir",
    type=str,
    default="./profile_results",
    help="Directory to save profile results. Default: ./profile_results",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cProfile
import time

import torch
import warp as wp

from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.sim.views import FrameView

try:
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_newton.sim.views import NewtonSiteFrameView

    HAS_NEWTON = True
except ImportError:
    HAS_NEWTON = False


# ------------------------------------------------------------------
# Benchmark functions
# ------------------------------------------------------------------


@torch.no_grad()
def benchmark_usd_or_fabric(view_type: str, num_iterations: int) -> dict[str, float]:
    """Benchmark USD or Fabric FrameView."""
    timing_results = {}

    print("  Setting up scene")
    sim_utils.create_new_stage()
    start_time = time.perf_counter()
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, use_fabric=(view_type == "fabric"))
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim_utils.get_current_stage()
    print(f"  SimulationContext: {time.perf_counter() - start_time:.4f}s")

    object_cfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    for i in range(args_cli.num_envs):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", stage=stage, translation=(i * 2.0, 0.0, 0.0))
        object_cfg.func(f"/World/Env_{i}/Object", object_cfg, translation=(0.0, 0.0, 1.0))
        prim = stage.DefinePrim(f"/World/Env_{i}/Object/Sensor", "Xform")
        sim_utils.standardize_xform_ops(prim)
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.1, 0.0, 0.05))
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    sim.reset()

    pattern = "/World/Env_.*/Object/Sensor"

    start_time = time.perf_counter()
    if view_type == "fabric" and "cuda" not in args_cli.device:
        raise ValueError("Fabric backend requires CUDA.")
    view = FrameView(pattern, device=args_cli.device, validate_xform_ops=False)
    num_prims = view.count
    timing_results["init"] = time.perf_counter() - start_time

    print(f"  FrameView ({view_type.upper()}) managing {num_prims} prims")

    positions, orientations = view.get_world_poses()

    _run_pose_benchmarks(view, num_prims, num_iterations, timing_results, positions, orientations)

    sim.clear_instance()
    return timing_results


@torch.no_grad()
def benchmark_newton(num_iterations: int) -> dict[str, float]:
    """Benchmark Newton FrameView."""
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.utils import configclass

    timing_results = {}

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        cube: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )

    print("  Setting up Newton scene")
    newton_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()), device=args_cli.device)
    start_time = time.perf_counter()
    ctx = build_simulation_context(device=args_cli.device, sim_cfg=newton_cfg, add_ground_plane=True)
    sim = ctx.__enter__()
    sim._app_control_on_stop_handle = None
    InteractiveScene(_SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))

    stage = sim_utils.get_current_stage()
    for i in range(args_cli.num_envs):
        prim = stage.DefinePrim(f"/World/envs/env_{i}/Cube/Sensor", "Xform")
        sim_utils.standardize_xform_ops(prim)
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.1, 0.0, 0.05))
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    sim.reset()
    print(f"  Newton scene setup: {time.perf_counter() - start_time:.4f}s")

    start_time = time.perf_counter()
    view = NewtonSiteFrameView("/World/envs/env_.*/Cube/Sensor", device=args_cli.device)
    num_prims = view.count
    timing_results["init"] = time.perf_counter() - start_time

    print(f"  Newton FrameView managing {num_prims} prims")

    positions, orientations = view.get_world_poses()

    _run_pose_benchmarks(view, num_prims, num_iterations, timing_results, positions, orientations)

    ctx.__exit__(None, None, None)
    return timing_results


@torch.no_grad()
def benchmark_physx(num_iterations: int) -> dict[str, float]:
    """Benchmark PhysX RigidBodyView."""
    timing_results = {}

    print("  Setting up scene")
    sim_utils.create_new_stage()
    start_time = time.perf_counter()
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, use_fabric=False)
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim_utils.get_current_stage()
    print(f"  SimulationContext: {time.perf_counter() - start_time:.4f}s")

    object_cfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    for i in range(args_cli.num_envs):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", stage=stage, translation=(i * 2.0, 0.0, 0.0))
        object_cfg.func(f"/World/Env_{i}/Object", object_cfg, translation=(0.0, 0.0, 1.0))

    sim.reset()

    pattern = "/World/Env_*/Object"
    start_time = time.perf_counter()
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    view = physics_sim_view.create_rigid_body_view(pattern)
    num_prims = view.count
    timing_results["init"] = time.perf_counter() - start_time

    print(f"  PhysX RigidBodyView managing {num_prims} prims")

    all_indices = wp.from_torch(torch.arange(num_prims, dtype=torch.int32, device=args_cli.device))

    transforms = view.get_transforms()
    transforms_t = wp.to_torch(transforms) if isinstance(transforms, wp.array) else transforms
    positions_t = transforms_t[:, :3]
    orientations_t = transforms_t[:, 3:7]

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        transforms = view.get_transforms()
    timing_results["get_world_poses"] = (time.perf_counter() - start_time) / num_iterations

    new_positions = positions_t.clone()
    new_positions[:, 2] += 0.5
    expected_positions = new_positions.clone()
    new_transforms = wp.from_torch(torch.cat([new_positions, orientations_t], dim=-1).contiguous())
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        view.set_transforms(new_transforms, indices=all_indices)
    timing_results["set_world_poses"] = (time.perf_counter() - start_time) / num_iterations

    transforms_after = view.get_transforms()
    ta = wp.to_torch(transforms_after) if isinstance(transforms_after, wp.array) else transforms_after
    pos_ok = torch.allclose(ta[:, :3], expected_positions, atol=1e-4, rtol=0)
    quat_ok = torch.allclose(ta[:, 3:7], orientations_t, atol=1e-4, rtol=0)
    if pos_ok and quat_ok:
        print("  Round-trip verification: PASS")
    else:
        pos_diff = (ta[:, :3] - expected_positions).abs().max().item()
        quat_diff = (ta[:, 3:7] - orientations_t).abs().max().item()
        print(f"  Round-trip verification: FAIL (pos max_diff={pos_diff:.6e}, quat max_diff={quat_diff:.6e})")

    sim.clear_instance()
    return timing_results


def _run_pose_benchmarks(
    view,
    num_prims: int,
    num_iterations: int,
    timing_results: dict,
    positions: wp.array,
    orientations: wp.array,
):
    """Shared benchmark loop for get/set world poses on any FrameView."""
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        view.get_world_poses()
    timing_results["get_world_poses"] = (time.perf_counter() - start_time) / num_iterations

    new_positions = wp.clone(positions)
    new_positions_t = wp.to_torch(new_positions)
    new_positions_t[:, 2] += 0.5
    expected_positions = new_positions_t.clone()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        view.set_world_poses(new_positions, orientations)
    timing_results["set_world_poses"] = (time.perf_counter() - start_time) / num_iterations

    ret_pos, ret_quat = view.get_world_poses()
    ret_pos_t = wp.to_torch(ret_pos)
    ret_quat_t = wp.to_torch(ret_quat)
    ori_t = wp.to_torch(orientations)

    pos_ok = torch.allclose(ret_pos_t, expected_positions, atol=1e-4, rtol=0)
    quat_ok = torch.allclose(ret_quat_t, ori_t, atol=1e-4, rtol=0)
    if pos_ok and quat_ok:
        print("  Round-trip verification: PASS")
    else:
        pos_diff = (ret_pos_t - expected_positions).abs().max().item()
        quat_diff = (ret_quat_t - ori_t).abs().max().item()
        print(f"  Round-trip verification: FAIL (pos max_diff={pos_diff:.6e}, quat max_diff={quat_diff:.6e})")


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def print_results(results_dict: dict[str, dict[str, float]], num_prims: int, num_iterations: int):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 120)
    print(f"BENCHMARK RESULTS: {num_prims} prims, {num_iterations} iterations")
    print("=" * 120)

    impl_names = list(results_dict.keys())
    display_names = {n: n.replace("_", " ").title() for n in impl_names}
    col_width = 22

    header = f"{'Operation':<25}"
    for name in impl_names:
        header += f" {display_names[name] + ' (ms)':>{col_width}}"
    print(header)
    print("-" * 120)

    operations = [
        ("Initialization", "init"),
        ("Get World Poses", "get_world_poses"),
        ("Set World Poses", "set_world_poses"),
    ]

    for op_name, op_key in operations:
        row = f"{op_name:<25}"
        for name in impl_names:
            val = results_dict[name].get(op_key, 0) * 1000
            row += f" {val:>{col_width}.4f}"
        print(row)

    print("=" * 120)

    total_row = f"{'Total':<25}"
    for name in impl_names:
        total = sum(results_dict[name].values()) * 1000
        total_row += f" {total:>{col_width}.4f}"
    print(total_row)

    baseline = "usd"
    if baseline in results_dict and len(impl_names) > 1:
        print("\n" + "=" * 120)
        print(f"SPEEDUP vs {display_names[baseline]}")
        print("=" * 120)
        header = f"{'Operation':<25}"
        for name in impl_names:
            if name != baseline:
                header += f" {display_names[name]:>{col_width}}"
        print(header)
        print("-" * 120)

        base = results_dict[baseline]
        for op_name, op_key in operations:
            row = f"{op_name:<25}"
            base_t = base.get(op_key, 0)
            for name in impl_names:
                if name != baseline:
                    impl_t = results_dict[name].get(op_key, 0)
                    if base_t > 0 and impl_t > 0:
                        row += f" {base_t / impl_t:>{col_width}.2f}x"
                    else:
                        row += f" {'N/A':>{col_width}}"
            print(row)
        print("=" * 120)

    print("\nNotes:")
    print("  - Times are averaged over all iterations")
    print("  - Speedup > 1.0 means faster than USD baseline")
    print("  - PhysX RigidBodyView requires rigid body physics; FrameView works with any Xformable prim")
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    print("=" * 120)
    print("FrameView Benchmark: USD vs Fabric vs Newton vs PhysX")
    print("=" * 120)
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Iterations:   {args_cli.num_iterations}")
    print(f"  Device:       {args_cli.device}")
    print(f"  Backends:     {', '.join(args_cli.backends)}")
    print()

    if args_cli.profile:
        import os

        os.makedirs(args_cli.profile_dir, exist_ok=True)

    all_timing = {}
    profile_files = {}

    dispatch = {
        "usd": ("usd", "FrameView (USD)", lambda n: benchmark_usd_or_fabric("usd", n)),
        "fabric": ("fabric", "FrameView (Fabric)", lambda n: benchmark_usd_or_fabric("fabric", n)),
        "newton": ("newton", "FrameView (Newton)", lambda n: benchmark_newton(n)),
        "physx": ("physx", "PhysX RigidBodyView", lambda n: benchmark_physx(n)),
    }

    for backend in args_cli.backends:
        if backend == "newton" and not HAS_NEWTON:
            print(f"Skipping {backend}: isaaclab_newton not installed")
            continue

        key, display_name, bench_fn = dispatch[backend]
        print(f"Benchmarking {display_name}...")

        if args_cli.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        timing = bench_fn(args_cli.num_iterations)

        if args_cli.profile:
            profiler.disable()
            pf = f"{args_cli.profile_dir}/{key}_benchmark.prof"
            profiler.dump_stats(pf)
            profile_files[key] = pf
            print(f"  Profile saved to: {pf}")

        all_timing[key] = timing
        print("  Done!\n")

    print_results(all_timing, args_cli.num_envs, args_cli.num_iterations)

    if args_cli.profile:
        print("\n" + "=" * 100)
        print("PROFILING RESULTS")
        print("=" * 100)
        for key, pf in profile_files.items():
            print(f"  snakeviz {pf}")
        print()

    sim_utils.SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
