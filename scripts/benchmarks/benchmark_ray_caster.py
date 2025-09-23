# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark ray-caster sensors (single-mesh and multi-mesh).

This script creates a simple scene with:
- a ground plane
- 0 to N moving spheres as rigid bodies

It then runs three different benchmarks:
1. Single vs Multi mesh ray caster against ground only
2. Multi-mesh ray caster against ground + various number of spheres
3. Multi-mesh ray caster against spheres with different numbers of vertices
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os
import platform
import time
import torch

import pandas as pd

from local_utils import dataframe_to_markdown

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark ray caster sensors.")
parser.add_argument("--steps", type=int, default=2000, help="Steps per resolution for timing.")
parser.add_argument("--warmup", type=int, default=50, help="Warmup steps before timing.")

# Num assets for benchmarking memory usage with and without caching.
NUM_ASSETS_MEMORY = [1, 2, 4, 8, 16, 32]
# Num assets for benchmarking scaling performance of multi-mesh ray caster.
NUM_ASSETS = [0, 1, 2, 4, 8, 16, 32]
# Num envs for benchmarking single vs multi mesh ray caster.
NUM_ENVS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# Num subdivisions for benchmarking mesh complexity.
MESH_SUBDIVISIONS = [0, 1, 2, 3, 4, 5]
# Different ray caster resolutions to benchmark. Num rays will be (5 / res)^2, e.g. 625, 2500, 10000, 11111
RESOLUTIONS: list[float] = [0.2, 0.1, 0.05, 0.015]

# Output directory for benchmark artifacts (can be overridden via env var BENCHMARK_OUTPUT_DIR)
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
OUTPUT_DIR = "outputs/benchmarks/" + timestamp

# # TINY for debugging
# NUM_ASSETS_MEMORY = [1, 2]  # Num assets for benchmarking memory usage with and without caching.
# NUM_ASSETS = [0, 1]  # Num assets for benchmarking scaling performance. of multi-mesh ray caster.
# NUM_ENVS = [32, 64]  # Num envs for benchmarking single vs multi mesh ray caster.
# MESH_SUBDIVISIONS = [0, 1]  # Num subdivisions for benchmarking mesh complexity.
# RESOLUTIONS: list[float] = [0.2, 0.1]  # Different ray caster resolutions to benchmark. Num rays will be (5 / res)^2, e.g. 625, 2500, 10000, 11111

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
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, RayCaster, RayCasterCfg, patterns
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
    num_envs: int,
    resolution: float,
    debug_vis: bool,
    track_mesh_transforms: bool,
    num_assets: int = 1,
    reference_meshes: bool = True,
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
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                target_prim_expr=f"/World/envs/env_.*/sphere_{i}", track_mesh_transforms=track_mesh_transforms
            )
            for i in range(num_assets)
        ],
        pattern_cfg=patterns.GridPatternCfg(resolution=resolution, size=(5.0, 5.0)),
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        debug_vis=debug_vis,
        ray_alignment="world",
        reference_meshes=reference_meshes,
    )
    return scene_cfg


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


def _run_benchmark(scene_cfg: RayCasterBenchmarkSceneCfg, sensor_name: str):
    sim, scene, sim_dt = _setup_scene(scene_cfg)
    sensor: RayCaster = scene[sensor_name]
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
        "rays_per_sensor": int(sensor.num_rays),
        "total_rays": int(sensor.num_rays * sensor.num_instances),
        "per_step_ms": float(per_step_ms),
        "avg_memory": float(avg_memory),
        "num_meshes": len(np.unique([m.id for m in sensor.meshes.values()])),
    }


def main():
    """Main function."""
    # Prepare benchmark

    # BENCHMARK 1 - Compare Single VS Multi

    print("=== Benchmarking Multi vs Single Raycaster ===")
    results: list[dict[str, object]] = []
    device_name = (
        torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else platform.processor()
    )

    _MESH_CONVERTERS_CALLBACKS["Sphere"] = lambda p: _create_sphere_trimesh(p, subdivisions=5)
    # Compare multi mesh performance over different number of assets.
    # More specifically, compare reference vs non-reference meshes and their respective memory usage.
    for idx, num_assets in enumerate(NUM_ASSETS_MEMORY):
        for reference_meshes in [True]:
            if num_assets > 16 and not reference_meshes:
                continue  # Skip this, otherwise we run out of memory

            print(f"\n[INFO]: Benchmarking with {num_assets} assets. {idx} / {len(NUM_ASSETS_MEMORY)}")
            num_envs = 1024
            resolution = 0.1
            multi_scene_cfg = _make_scene_cfg_multi(
                num_envs=num_envs,
                resolution=resolution,
                debug_vis=not args_cli.headless,
                track_mesh_transforms=True,
                num_assets=num_assets,
                reference_meshes=reference_meshes,
            )
            result = _run_benchmark(multi_scene_cfg, "height_scanner_multi")
            result["num_envs"] = num_envs
            result["resolution"] = resolution
            result["mode"] = "multi"
            result["reference_meshes"] = reference_meshes
            result["num_assets"] = num_assets

            print(result)
            results.append(result)
            del multi_scene_cfg

            df_num_assets = pd.DataFrame(results)
            df_num_assets["device"] = device_name
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            df_num_assets.to_csv(os.path.join(OUTPUT_DIR, "ray_caster_benchmark_num_assets_reference.csv"), index=False)

    results: list[dict[str, object]] = []

    for idx, num_envs in enumerate(NUM_ENVS):
        print(f"\n[INFO]: Benchmarking with {num_envs} envs. {idx + 1} / {len(NUM_ENVS)}")

        # Default Raycaster
        for resolution in RESOLUTIONS:
            single_scene_cfg = _make_scene_cfg_single(
                num_envs=num_envs,
                resolution=resolution,
                debug_vis=not args_cli.headless,
            )
            result = _run_benchmark(single_scene_cfg, "height_scanner")
            result["num_envs"] = num_envs
            result["resolution"] = resolution
            result["mode"] = "single"
            result["num_assets"] = 0
            results.append(result)
            del single_scene_cfg

        # Multi Raycaster
        for resolution in RESOLUTIONS:
            multi_scene_cfg = _make_scene_cfg_multi(
                num_envs=num_envs,
                resolution=resolution,
                debug_vis=not args_cli.headless,
                track_mesh_transforms=False,
                num_assets=0,
            )
            result = _run_benchmark(multi_scene_cfg, "height_scanner_multi")
            result["num_envs"] = num_envs
            result["resolution"] = resolution
            result["mode"] = "multi"
            result["num_assets"] = 0
            results.append(result)
            del multi_scene_cfg

        df_single_vs_multi = pd.DataFrame(results)
        df_single_vs_multi["device"] = device_name
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_single_vs_multi.to_csv(os.path.join(OUTPUT_DIR, "ray_caster_benchmark_single_vs_multi.csv"), index=False)

    print("\n=== Benchmarking Multi Raycaster with different number of assets and faces ===")
    results: list[dict[str, object]] = []

    # Keep fixed resolution for the subdivision and num assets benchmarks
    resolution = 0.05

    # Compare multi mesh performance over different number of assets
    for idx, num_assets in enumerate(NUM_ASSETS):
        print(f"\n[INFO]: Benchmarking with {num_assets} assets. {idx} / {len(NUM_ASSETS)}")
        multi_scene_cfg = _make_scene_cfg_multi(
            num_envs=num_envs,
            resolution=resolution,
            debug_vis=not args_cli.headless,
            track_mesh_transforms=True,
            num_assets=num_assets,
        )
        result = _run_benchmark(multi_scene_cfg, "height_scanner_multi")
        result["num_envs"] = num_envs
        result["resolution"] = resolution
        result["mode"] = "multi"
        result["num_assets"] = num_assets
        results.append(result)
        del multi_scene_cfg

        df_num_assets = pd.DataFrame(results)
        df_num_assets["device"] = device_name
    df_num_assets.to_csv(os.path.join(OUTPUT_DIR, "ray_caster_benchmark_num_assets.csv"), index=False)

    print("\n=== Benchmarking Multi Raycaster with different number of faces ===")
    results: list[dict[str, object]] = []
    # Compare multi mesh performance over different number of vertices
    for idx, subdivision in enumerate(MESH_SUBDIVISIONS):
        print(f"\n[INFO]: Benchmarking with {subdivision} subdivisions. {idx} / {len(MESH_SUBDIVISIONS)}")
        _MESH_CONVERTERS_CALLBACKS["Sphere"] = lambda p: _create_sphere_trimesh(p, subdivisions=subdivision)
        multi_scene_cfg = _make_scene_cfg_multi(
            num_envs=num_envs,
            resolution=resolution,
            debug_vis=not args_cli.headless,
            track_mesh_transforms=False,  # Only static ground
            num_assets=1,
        )
        result = _run_benchmark(multi_scene_cfg, "height_scanner_multi")
        result["num_envs"] = num_envs
        result["resolution"] = resolution
        result["mode"] = "multi"
        result["num_assets"] = 1
        result["num_faces"] = 20 * (4**subdivision)
        results.append(result)
        del multi_scene_cfg

    df_num_faces = pd.DataFrame(results)
    df_num_faces["device"] = device_name
    df_num_faces.to_csv(os.path.join(OUTPUT_DIR, "ray_caster_benchmark_num_faces.csv"), index=False)

    # Create .md file with all three tables
    for df, title in zip(
        [df_single_vs_multi, df_num_assets, df_num_faces], ["Single vs Multi", "Num Assets", "Num Faces"]
    ):
        with open(os.path.join(OUTPUT_DIR, f"ray_caster_benchmark_{title}.md"), "w") as f:
            f.write(f"# {title}\n\n")
            f.write(dataframe_to_markdown(df, floatfmt=".3f"))
            f.write("\n\n")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
