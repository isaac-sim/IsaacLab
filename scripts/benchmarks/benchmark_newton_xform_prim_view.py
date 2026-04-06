# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark script for Newton XformPrimView performance.

Tests the performance of batched transform operations using Newton's GPU-backed
XformPrimView (sites / body_q) without requiring Isaac Sim Kit.

Usage:
    VIRTUAL_ENV=env_isaaclab ./isaaclab.sh -p scripts/benchmarks/benchmark_newton_xform_prim_view.py --num_envs 4096
    VIRTUAL_ENV=env_isaaclab ./isaaclab.sh -p scripts/benchmarks/benchmark_newton_xform_prim_view.py \
        --num_envs 4096 --num_iterations 200
"""

from __future__ import annotations

import argparse
import time

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.views import XformPrimView

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass

NEWTON_SIM_CFG = SimulationCfg(
    physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()),
)


@configclass
class BenchSceneCfg(InteractiveSceneCfg):
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


@torch.no_grad()
def benchmark(num_envs: int, num_iterations: int, device: str) -> dict[str, float]:
    """Run the benchmark and return timing results in seconds."""
    NEWTON_SIM_CFG.device = device
    results: dict[str, float] = {}

    with build_simulation_context(device=device, sim_cfg=NEWTON_SIM_CFG, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(BenchSceneCfg(num_envs=num_envs, env_spacing=2.0))
        sim.reset()

        view = XformPrimView("/World/envs/env_.*/Cube", device=device)
        print(f"  View count: {view.count}")

        # -- warmup (compile Warp kernel, first torch allocs) --
        for _ in range(5):
            view.get_world_poses()

        # -- get_world_poses (full) --
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            pos, quat = view.get_world_poses()
        torch.cuda.synchronize()
        results["get_world_poses"] = (time.perf_counter() - t0) / num_iterations

        # -- get_world_poses (indexed, 50 %) --
        half = list(range(0, num_envs, 2))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            pos, quat = view.get_world_poses(half)
        torch.cuda.synchronize()
        results["get_world_poses_indexed_50pct"] = (time.perf_counter() - t0) / num_iterations

        # -- set_world_poses (full) --
        new_pos = torch.rand((num_envs, 3), device=device)
        new_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_envs, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            view.set_world_poses(new_pos, new_quat)
        torch.cuda.synchronize()
        results["set_world_poses"] = (time.perf_counter() - t0) / num_iterations

        # -- interleaved set -> get --
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            view.set_world_poses(new_pos, new_quat)
            pos, quat = view.get_world_poses()
        torch.cuda.synchronize()
        results["interleaved_set_get"] = (time.perf_counter() - t0) / num_iterations

    return results


def print_results(results: dict[str, float], num_envs: int, num_iterations: int):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"Newton XformPrimView Benchmark: {num_envs} envs, {num_iterations} iters")
    print("=" * 70)
    print(f"{'Operation':<40} {'Time (ms)':>12} {'us/env':>12}")
    print("-" * 70)
    for op, t in results.items():
        ms = t * 1000
        us_per_env = t * 1e6 / num_envs
        print(f"{op:<40} {ms:>12.4f} {us_per_env:>12.4f}")
    total = sum(results.values()) * 1000
    print("-" * 70)
    print(f"{'Total':<40} {total:>12.4f}")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Newton XformPrimView performance.")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 70)
    print("Newton XformPrimView Benchmark")
    print("=" * 70)
    print(f"  Envs: {args.num_envs}  Iterations: {args.num_iterations}  Device: {args.device}")
    print()

    results = benchmark(args.num_envs, args.num_iterations, args.device)
    print_results(results, args.num_envs, args.num_iterations)


if __name__ == "__main__":
    main()
