# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark ray-caster sensors (single-mesh and multi-mesh).

This script creates a simple scene with:
- a ground plane
- an ANYmal robot per environment
- an optional set of cubes per environment

It places ray-caster sensors under the robot base and benchmarks update
times while varying the number of rays via the grid pattern resolution.

Examples:

  - Single-mesh ray caster against ground only:
      ./isaaclab.sh -p scripts/benchmarks/benchmark_ray_caster.py \\
        --num_envs_list 512 --mode single --resolutions 0.2,0.1,0.05 --headless

  - Multi-mesh ray caster against ground + cubes:
      ./isaaclab.sh -p scripts/benchmarks/benchmark_ray_caster.py \\
        --num_envs 512 --mode multi --resolutions 0.2,0.1,0.05 --headless

  - Run both benchmarks:
      ./isaaclab.sh -p scripts/benchmarks/benchmark_ray_caster.py \\
        --num_envs 512 --mode both --resolutions 0.2,0.1,0.05 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import time

import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark ray caster sensors.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=[256, 512, 1024],
    nargs="+",
    help="List of environment counts to benchmark (e.g., 256 512 1024).",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["single", "multi", "both"],
    default="both",
    help="Which benchmark to run: single (RayCaster), multi (MultiMeshRayCaster), or both.",
)
parser.add_argument(
    "--resolutions",
    type=float,
    default=[0.2, 0.1, 0.05],
    nargs="+",
    help="List of grid resolutions to benchmark (meters).",
)
parser.add_argument("--steps", type=int, default=1000, help="Steps per resolution for timing.")
parser.add_argument("--warmup", type=int, default=150, help="Warmup steps before timing.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.stage as stage_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Robot config
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


@configclass
class RayCasterBenchmarkSceneCfg(InteractiveSceneCfg):
    """Scene config with ground, robot, and optional cubes per env."""

    # ground plane (rough)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore[attr-defined]

    # cubes collection (optionally set at runtime)
    cubes: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    # sensors (set at runtime)
    height_scanner: RayCasterCfg | None = None
    height_scanner_multi: MultiMeshRayCasterCfg | None = None


def _make_scene_cfg_single(num_envs: int, resolution: float, debug_vis: bool) -> RayCasterBenchmarkSceneCfg:
    scene_cfg = RayCasterBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=resolution, size=(5.0, 5.0)),
        ray_alignment="yaw",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        debug_vis=debug_vis,
    )
    return scene_cfg


def _make_scene_cfg_multi(
    num_envs: int, resolution: float, debug_vis: bool, track_mesh_transforms: bool
) -> RayCasterBenchmarkSceneCfg:
    scene_cfg = RayCasterBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.height_scanner_multi = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground", "/World/envs/env_.*/cube"],
        pattern_cfg=patterns.GridPatternCfg(resolution=resolution, size=(5.0, 5.0)),
        attach_yaw_only=True,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        debug_vis=debug_vis,
        track_mesh_transforms=track_mesh_transforms,
    )
    return scene_cfg


def main():
    """Main function."""
    # Prepare benchmark
    results: list[dict[str, object]] = []

    def _setup_scene(scene_cfg: RayCasterBenchmarkSceneCfg) -> tuple[SimulationContext, InteractiveScene, float]:
        # Create a new stage
        stage_utils.create_new_stage()
        # New simulation per run
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))
        # Create scene with sensors
        setup_time_begin = time.perf_counter_ns()
        scene = InteractiveScene(scene_cfg)
        setup_time_end = time.perf_counter_ns()
        print(f"[INFO]: Scene creation time: {(setup_time_end - setup_time_begin) / 1e6:.2f} ms")
        # Reset sim
        reset_time_begin = time.perf_counter_ns()
        sim.reset()
        reset_time_end = time.perf_counter_ns()
        print(f"[INFO]: Sim start time: {(reset_time_end - reset_time_begin) / 1e6:.2f} ms")
        return sim, scene, sim.get_physics_dt()

    def _run_benchmark_single(num_envs: int):
        print(f"\n[INFO]: Benchmarking RayCaster (ground) with {num_envs} envs")
        for res in args_cli.resolutions:
            scene_cfg = _make_scene_cfg_single(num_envs=num_envs, resolution=res, debug_vis=not args_cli.headless)
            sim, scene, sim_dt = _setup_scene(scene_cfg)
            sensor: RayCaster = scene["height_scanner"]
            # Warmup
            for _ in range(args_cli.warmup):
                sim.step()
                sensor.update(dt=sim_dt, force_recompute=True)
            
            used_memory = 0.0
            
            # Timing
            t0 = time.perf_counter_ns()
            for _ in range(args_cli.steps):
                sim.step()
                sensor.update(dt=sim_dt, force_recompute=True)
                free, total = torch.cuda.mem_get_info(args_cli.device)
                used_memory += (total - free) / 1024**2  # Convert to MB
            t1 = time.perf_counter_ns()
            per_step_ms = (t1 - t0) / args_cli.steps / 1e6
            avg_memory = used_memory / args_cli.steps
            print(
                f"[INFO]: RayCaster (ground): res={res:.4f}, rays/sensor={sensor.num_rays}, "
                f"total rays={sensor.num_rays * sensor.num_instances}, per-step={per_step_ms:.3f} ms"
                f", avg_memory={avg_memory:.2f} MB"
            )
            results.append({
                "mode": "single",
                "num_envs": num_envs,
                "resolution": res,
                "rays_per_sensor": int(sensor.num_rays),
                "total_rays": int(sensor.num_rays * sensor.num_instances),
                "per_step_ms": float(per_step_ms),
                "avg_memory": float(avg_memory),
            })
            # Cleanup
            # stop simulation
            # note: cannot use self.sim.stop() since it does one render step after stopping!! This doesn't make sense :(
            sim._timeline.stop()
            # clear the stage
            sim.clear_all_callbacks()
            sim.clear_instance()
            # stop the simulation
            sim.stop()  # FIXME: this should not be necessary as the sim is stopped by the _timeline.stop()

    def _run_benchmark_multi(num_envs: int):
        print(f"\n[INFO]: Benchmarking MultiMeshRayCaster (ground + cubes) with {num_envs} envs")
        for res in args_cli.resolutions:
            scene_cfg = _make_scene_cfg_multi(
                num_envs=num_envs, resolution=res, debug_vis=not args_cli.headless, track_mesh_transforms=False
            )
            sim, scene, sim_dt = _setup_scene(scene_cfg)
            sensor: MultiMeshRayCaster = scene["height_scanner_multi"]
            # Warmup
            for _ in range(args_cli.warmup):
                sim.step()
                sensor.update(dt=sim_dt, force_recompute=True)
            # Timing
            t0 = time.perf_counter_ns()
            used_memory = 0.0
            for _ in range(args_cli.steps):
                sim.step()
                sensor.update(dt=sim_dt, force_recompute=True)
                free, total = torch.cuda.mem_get_info(args_cli.device)
                used_memory += (total - free) / 1024**2  # Convert to MB
            t1 = time.perf_counter_ns()
            per_step_ms = (t1 - t0) / args_cli.steps / 1e6
            avg_memory = used_memory / args_cli.steps
            print(
                f"[INFO]: MultiMeshRayCaster (ground + cubes): res={res:.4f}, rays/sensor={sensor.num_rays}, "
                f"total rays={sensor.num_rays * sensor.num_instances}, per-step={per_step_ms:.3f} ms"
                f", avg_memory={avg_memory:.2f} MB"
            )
            results.append({
                "mode": "multi",
                "num_envs": num_envs,
                "resolution": res,
                "rays_per_sensor": int(sensor.num_rays),
                "total_rays": int(sensor.num_rays * sensor.num_instances),
                "per_step_ms": float(per_step_ms),
            })
            # Cleanup
            # stop simulation
            # note: cannot use self.sim.stop() since it does one render step after stopping!! This doesn't make sense :(
            sim._timeline.stop()
            # clear the stage
            sim.clear_all_callbacks()
            sim.clear_instance()

    # Run selected benchmarks for each env count
    for num_envs in args_cli.num_envs:
        if args_cli.mode in ("single", "both"):
            _run_benchmark_single(num_envs)
        if args_cli.mode in ("multi", "both"):
            _run_benchmark_multi(num_envs)

    # Save results to CSV and Markdown for documentation
    os.makedirs("outputs/benchmarks", exist_ok=True)
    csv_path = os.path.join("outputs/benchmarks", "ray_caster_benchmark.csv")
    md_path = os.path.join("outputs/benchmarks", "ray_caster_benchmark.md")

    fieldnames = [
        "mode",
        "num_envs",
        "resolution",
        "rays_per_sensor",
        "total_rays",
        "per_step_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    # Markdown table
    with open(md_path, "w") as f:
        f.write("| mode | num_envs  | resolution | rays_per_sensor | total_rays | per_step_ms |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['mode']} | {r['num_envs']} | {r['resolution']:.4f} | "
                f"{r['rays_per_sensor']} | {r['total_rays']} | {r['per_step_ms']:.3f} |\n"
            )
    print(f"[INFO]: Saved benchmark results to {csv_path} and {md_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
