# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark ray-caster sensors (single-mesh and multi-mesh).

This script creates a simple scene with:
- a ground plane
- an ANYmal robot per environment
- an optional set of spheres per environment

It places ray-caster sensors under the robot base and benchmarks update
times while varying the number of rays via the grid pattern resolution.

Examples:

  - Single-mesh ray caster against ground only:
      ./isaaclab.sh -p scripts/benchmarks/benchmark_ray_caster.py \\
        --num_envs_list 512 --mode single --resolutions 0.2,0.1,0.05 --headless

  - Multi-mesh ray caster against ground + spheres:
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
    default=[512],
    nargs="+",
    help="List of environment counts to benchmark (e.g., 256 512 1024).",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["multi"],
    default="multi",
    help="Which benchmark to run: single (RayCaster), multi (MultiMeshRayCaster), or both.",
)


parser.add_argument("--num_assets", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128], nargs="+", help="List of asset counts to benchmark.")

parser.add_argument(
    "--decimation", type=int, default=[2, 3, 4, 5, 6, 7], nargs="+", help="List of decimation levels to benchmark."
)

parser.add_argument(
    "--resolutions",
    type=float,
    default=[0.05],
    nargs="+",
    help="List of grid resolutions to benchmark (meters).",
)
parser.add_argument("--steps", type=int, default=200, help="Steps per resolution for timing.")
parser.add_argument("--warmup", type=int, default=10, help="Warmup steps before timing.")

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
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.mesh import _MESH_CONVERTERS_CALLBACKS, _create_sphere_trimesh


@configclass
class RayCasterBenchmarkSceneCfg(InteractiveSceneCfg):
    """Scene config with ground, robot, and optional spheres per env."""

    # ground plane (rough)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ray_caster_origin",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.6, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 4.0)),
    )
    # # robot
    # robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore[attr-defined]

    # spheres collection (optionally set at runtime)
    spheres: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    # sensors (set at runtime)
    height_scanner: RayCasterCfg | None = None
    height_scanner_multi: MultiMeshRayCasterCfg | None = None


def _make_scene_cfg_single(num_envs: int, resolution: float, debug_vis: bool) -> RayCasterBenchmarkSceneCfg:
    scene_cfg = RayCasterBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/ray_caster_origin",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=resolution, size=(5.0, 5.0)),
        ray_alignment="world",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        debug_vis=debug_vis,
    )
    return scene_cfg


def _make_scene_cfg_multi(
    num_envs: int, resolution: float, debug_vis: bool, track_mesh_transforms: bool, num_assets: int = 1
) -> RayCasterBenchmarkSceneCfg:
    scene_cfg = RayCasterBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)

    obj_cfg = scene_cfg.spheres
    if track_mesh_transforms:
        # Enable gravity
        obj_cfg.spawn.rigid_props.disable_gravity = False

    for i in range(num_assets):
        new_obj_cfg = obj_cfg.replace(prim_path=f"{{ENV_REGEX_NS}}/sphere_{i}")
        ratio = i / num_assets
        new_obj_cfg.init_state.pos = (ratio - 0.5, ratio - 0.5, 1.0)
        setattr(scene_cfg, f"sphere_{i}", new_obj_cfg)
    del scene_cfg.spheres

    scene_cfg.height_scanner_multi = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/ray_caster_origin",
        mesh_prim_paths=["/World/ground"]
        + [
            MultiMeshRayCasterCfg.RaycastTargetCfg(target_prim_expr=f"/World/envs/env_.*/sphere_{i}")
            for i in range(num_assets)
        ],
        pattern_cfg=patterns.GridPatternCfg(resolution=resolution, size=(5.0, 5.0)),
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        debug_vis=debug_vis,
        ray_alignment="world",
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

    def _run_benchmark_single(num_envs: int, resolution: float):

        print(f"\n[INFO]: Benchmarking RayCaster (ground) with {num_envs} envs and resolution {resolution}")
        scene_cfg = _make_scene_cfg_single(num_envs=num_envs, resolution=resolution, debug_vis=not args_cli.headless)
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
            f"[INFO]: RayCaster (ground): res={resolution:.4f}, rays/sensor={sensor.num_rays}, "
            f"total rays={sensor.num_rays * sensor.num_instances}, per-step={per_step_ms:.3f} ms"
            f", avg_memory={avg_memory:.2f} MB"
        )
        # Cleanup
        # stop simulation
        # note: cannot use self.sim.stop() since it does one render step after stopping!! This doesn't make sense :(
        sim._timeline.stop()
        # clear the stage
        sim.clear_all_callbacks()
        sim.clear_instance()
        SimulationManager._simulation_manager_interface.reset()
        SimulationManager._callbacks.clear()

        return {
            "mode": "single",
            "num_envs": num_envs,
            "resolution": resolution,
            "rays_per_sensor": int(sensor.num_rays),
            "total_rays": int(sensor.num_rays * sensor.num_instances),
            "per_step_ms": float(per_step_ms),
            "avg_memory": float(avg_memory),
        }

    def _run_benchmark_multi(num_envs: int, resolution: float, num_assets: int, track_mesh_transforms: bool = False):
        print(
            f"\n[INFO]: Benchmarking MultiMeshRayCaster (ground + spheres) with {num_envs} envs and resolution"
            f" {resolution}"
        )
        scene_cfg = _make_scene_cfg_multi(
            num_envs=num_envs,
            resolution=resolution,
            debug_vis=not args_cli.headless,
            track_mesh_transforms=track_mesh_transforms,
            num_assets=num_assets,
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
        print(resolution, sensor.num_rays, sensor.num_instances)
        print(
            f"[INFO]: MultiMeshRayCaster (ground + spheres): res={resolution:.4f}, rays/sensor={sensor.num_rays}, "
            f"total rays={sensor.num_rays * sensor.num_instances}, per-step={per_step_ms:.3f} ms"
            f", avg_memory={avg_memory:.2f} MB"
        )
        # Cleanup
        # stop simulation
        # note: cannot use self.sim.stop() since it does one render step after stopping!! This doesn't make sense :(
        sim._timeline.stop()
        # clear the stage
        sim.clear_all_callbacks()
        sim.clear_instance()

        SimulationManager._simulation_manager_interface.reset()
        SimulationManager._callbacks.clear()

        return {
            "mode": "multi",
            "num_envs": num_envs,
            "resolution": resolution,
            "rays_per_sensor": int(sensor.num_rays),
            "total_rays": int(sensor.num_rays * sensor.num_instances),
            "per_step_ms": float(per_step_ms),
            "avg_memory": float(avg_memory),
            "num_assets": num_assets,
            "track_mesh_transforms": track_mesh_transforms,
        }

    # Run selected benchmarks for each env count
    for num_envs in args_cli.num_envs:
        if args_cli.mode in ("multi", "both"):
            _MESH_CONVERTERS_CALLBACKS["Sphere"] = lambda p: _create_sphere_trimesh(p, subdivisions=2)

            # for res in args_cli.resolutions:
            #     res = _run_benchmark_multi(num_envs, res, num_assets=0, track_mesh_transforms=True)
            #     res["num_faces"] = -1
            #     results.append(res)

            # for decimation in args_cli.decimation:
            #     num_faces = 20 * 4**decimation
            #     _MESH_CONVERTERS_CALLBACKS["Sphere"] = lambda p: _create_sphere_trimesh(p, subdivisions=decimation)
            #     res = _run_benchmark_multi(num_envs, 0.05, num_assets=1, track_mesh_transforms=False)
            #     res["num_faces"] = num_faces
            #     results.append(res)

            for n_assets in args_cli.num_assets:
                res = _run_benchmark_multi(num_envs, 0.05, num_assets=n_assets, track_mesh_transforms=True)
                res["num_faces"] = 20 * 4**2
                results.append(res)

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
        "avg_memory",
        "num_faces",
        "track_mesh_transforms",
        "num_assets",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for key in fieldnames:
                if key not in row:
                    row[key] = "-1"
            writer.writerow(row)

    # Markdown table
    with open(md_path, "w") as f:
        f.write(
            "| mode | num_envs  | resolution | rays_per_sensor | total_rays | per_step_ms | avg_memory | num_faces |"
            " track_mesh_transforms | num_assets |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['mode']} | {r['num_envs']} | {r['resolution']:.4f} | "
                f"{r['rays_per_sensor']} | {r['total_rays']} | {r['per_step_ms']:.3f} | "
                f"{r['avg_memory']:.3f} | {r['num_faces']} | {r['track_mesh_transforms']} | {r['num_assets']} |\n"
            )
    print(f"[INFO]: Saved benchmark results to {csv_path} and {md_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
