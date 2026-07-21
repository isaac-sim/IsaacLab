# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the standard PhysX RayCaster update path.

Each environment contains one kinematic body carrying a downward-facing grid
sensor. All sensors cast against one shared ground plane.

Usage:
    ./isaaclab.sh -p source/isaaclab_physx/benchmark/sensors/benchmark_ray_caster.py --num_envs 4096
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark the standard PhysX RayCaster update path.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--grid_size", type=float, default=1.0, help="Width and length [m] of each ray grid.")
parser.add_argument("--grid_resolution", type=float, default=0.25, help="Ray-grid resolution [m].")
parser.add_argument("--num_steps", type=int, default=500, help="Number of timed updates.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Number of untimed warm-up updates.")
parser.add_argument("--label", type=str, default="current", help="Label printed with the benchmark results.")
parser.add_argument(
    "--disable_graph",
    action="store_true",
    help="Use the cached PhysX view with ordinary eager Warp launches.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below follows application launch."""

import statistics
import time

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass


@configclass
class RayCasterBenchmarkSceneCfg(InteractiveSceneCfg):
    """One kinematic sensor body per environment above a shared ground plane."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sensor_body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    ray_caster: RayCasterCfg = None


def _percentile(samples: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for sorted scalar samples."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def main() -> None:
    """Run the benchmark and print latency statistics."""
    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = RayCasterBenchmarkSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
        lazy_sensor_update=True,
    )
    scene_cfg.ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        mesh_prim_paths=["/World/ground"],
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=args_cli.grid_resolution,
            size=(args_cli.grid_size, args_cli.grid_size),
            direction=(0.0, 0.0, -1.0),
        ),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    sensor = scene["ray_caster"]
    if args_cli.disable_graph:
        sensor._use_graph = False

    for _ in range(args_cli.warmup_steps):
        sim.step(render=False)
        sensor.update(sim_dt, force_recompute=True)
    wp.synchronize_device(sim.device)

    synchronized_ms: list[float] = []
    submission_ms: list[float] = []
    for _ in range(args_cli.num_steps):
        sim.step(render=False)
        wp.synchronize_device(sim.device)
        start = time.perf_counter()
        sensor.update(sim_dt, force_recompute=True)
        submitted = time.perf_counter()
        wp.synchronize_device(sim.device)
        finished = time.perf_counter()
        synchronized_ms.append((finished - start) * 1000.0)
        submission_ms.append((submitted - start) * 1000.0)

    ray_hits = sensor.data.ray_hits_w.torch
    finite_hits = int(torch.isfinite(ray_hits).all(dim=-1).sum().item())
    expected_hits = args_cli.num_envs * sensor.num_rays
    if finite_hits != expected_hits:
        raise RuntimeError(f"Expected {expected_hits} finite ray hits, received {finite_hits}.")
    max_hit_height = float(torch.abs(ray_hits[..., 2]).max().item())
    hit_height_tolerance = 1.0e-4
    if max_hit_height > hit_height_tolerance:
        raise RuntimeError(
            f"Expected ray hits within {hit_height_tolerance} m of the z=0 plane, "
            f"received maximum |z| {max_hit_height} m."
        )

    print("-" * 80)
    print("RayCaster update benchmark (PhysX)")
    print(f"  label                  : {args_cli.label}")
    print(f"  device                 : {sim.device}")
    print(f"  num_envs               : {args_cli.num_envs}")
    print(f"  rays_per_env           : {sensor.num_rays}")
    print(f"  num_steps              : {args_cli.num_steps}")
    print(f"  synchronized mean      : {statistics.mean(synchronized_ms):.3f} ms")
    print(f"  synchronized p50       : {_percentile(synchronized_ms, 0.50):.3f} ms")
    print(f"  synchronized p95       : {_percentile(synchronized_ms, 0.95):.3f} ms")
    print(f"  submission mean        : {statistics.mean(submission_ms):.3f} ms")
    print(f"  submission p50         : {_percentile(submission_ms, 0.50):.3f} ms")
    print(f"  submission p95         : {_percentile(submission_ms, 0.95):.3f} ms")
    print("-" * 80)
    print(f"  finite ray hits        : {finite_hits} / {expected_hits}")
    print(f"  maximum |hit z|        : {max_hit_height:.6f} m")


if __name__ == "__main__":
    main()
    simulation_app.close()
