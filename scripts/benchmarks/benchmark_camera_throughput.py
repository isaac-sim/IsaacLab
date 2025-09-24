# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark throughput of different camera implementations.

This script benchmarks per-step time while varying:
- camera implementation: standard camera, tiled camera, warp ray-caster camera
- image resolutions (height x width)
- number of environments

Sensors are added to the scene config before `InteractiveScene` is constructed.
Each benchmark run initializes a fresh simulation and scene and tears it down.

Examples:

  - Benchmark all camera types across resolutions:
      ./isaaclab.sh -p scripts/benchmarks/benchmark_camera_throughput.py \\
        --num_envs 256 512 \\
        --resolutions 240x320,480x640 --steps 200 --warmup 20 --headless

  - Only standard camera at 720p:
      ./isaaclab.sh -p scripts/benchmarks/benchmark_camera_throughput.py \\
        --num_envs 256 --resolutions 720x1280 --steps 200 --warmup 20 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time

# local imports
from local_utils import dataframe_to_markdown

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark throughput of different camera implementations.")
parser.add_argument(
    "--num_envs",
    type=int,
    nargs="+",
    default=[12, 24, 48],   # [256, 512, 1024],
    help="List of environment counts to benchmark (e.g., 256 512 1024).",
)
parser.add_argument(
    "--usd_camera",
    action="store_true",
    default=False,
    help="Whether to benchmark the USD camera.",
)
parser.add_argument(
    "--tiled_camera",
    action="store_true",
    default=False,
    help="Whether to benchmark the tiled camera.",
)
parser.add_argument(
    "--ray_caster_camera",
    action="store_true",
    default=False,
    help="Whether to benchmark the ray caster camera.",
)
parser.add_argument(
    "--resolutions",
    type=str,
    default="240x320,480x640",
    help="Comma-separated list of HxW resolutions, e.g., 240x320,480x640",
)
parser.add_argument(
    "--data_type",
    type=str,
    default="distance_to_image_plane",
    help="Data type, e.g., distance_to_image_plane,rgb",
)
parser.add_argument("--steps", type=int, default=500, help="Steps per run to time.")
parser.add_argument("--warmup", type=int, default=50, help="Warmup steps per run before timing.")

# Append AppLauncher CLI args and parse
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.tiled_camera or args_cli.usd_camera:
    args_cli.enable_cameras = True
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import pandas as pd

import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCameraCfg, TiledCameraCfg, patterns, Camera, TiledCamera, MultiMeshRayCasterCamera
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Robot config to attach sensors under a valid prim
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


def _parse_resolutions(res_str: str) -> list[tuple[int, int]]:
    resolutions: list[tuple[int, int]] = []
    for token in [s for s in res_str.split(",") if s]:
        h, w = token.lower().split("x")
        resolutions.append((int(h), int(w)))
    print("[INFO]: Resolutions: ", resolutions)
    return resolutions


@configclass
class CameraBenchmarkSceneCfg(InteractiveSceneCfg):
    """Scene config with ground, light, robot, and one camera sensor per env."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore[attr-defined]

    # one cube per environment (optional target for ray-caster camera)
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    usd_camera: CameraCfg | None = None
    tiled_camera: TiledCameraCfg | None = None
    ray_caster_camera: MultiMeshRayCasterCameraCfg | None = None


def _make_scene_cfg_usd(num_envs: int, height: int, width: int, data_types: list[str], debug_vis: bool) -> CameraBenchmarkSceneCfg:
    scene_cfg = CameraBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.usd_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=height,
        width=width,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        debug_vis=debug_vis,
    )
    return scene_cfg


def _make_scene_cfg_tiled(num_envs: int, height: int, width: int, data_types: list[str], debug_vis: bool) -> CameraBenchmarkSceneCfg:
    scene_cfg = CameraBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/TiledCamera",
        height=height,
        width=width,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        debug_vis=debug_vis,
    )
    return scene_cfg


def _make_scene_cfg_ray_caster(num_envs: int, height: int, width: int, data_types: list[str], debug_vis: bool) -> CameraBenchmarkSceneCfg:
    scene_cfg = CameraBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.ray_caster_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",  # attach to existing prim
        mesh_prim_paths=["/World/ground", "/World/envs/env_.*/cube"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0, horizontal_aperture=20.955, height=height, width=width
        ),
        data_types=data_types,
        debug_vis=debug_vis,
    )
    return scene_cfg


def _setup_scene(scene_cfg: CameraBenchmarkSceneCfg) -> tuple[SimulationContext, InteractiveScene, float]:
    # Create a new stage to avoid residue across runs
    stage_utils.create_new_stage()
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))
    setup_time_begin = time.perf_counter_ns()
    scene = InteractiveScene(scene_cfg)
    setup_time_end = time.perf_counter_ns()
    print(f"[INFO]: Scene creation time: {(setup_time_end - setup_time_begin) / 1e6:.2f} ms")
    reset_time_begin = time.perf_counter_ns()
    sim.reset()
    reset_time_end = time.perf_counter_ns()
    print(f"[INFO]: Sim start time: {(reset_time_end - reset_time_begin) / 1e6:.2f} ms")
    return sim, scene, sim.get_physics_dt()


def _run_benchmark(scene_cfg: CameraBenchmarkSceneCfg, sensor_name: str):
    sim, scene, sim_dt = _setup_scene(scene_cfg)
    sensor: Camera | TiledCamera | MultiMeshRayCasterCamera = scene[sensor_name]
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
        "num_envs": scene.num_envs,
        "resolution": sensor.image_shape,
        "per_step_ms": float(per_step_ms),
        "avg_memory": float(avg_memory),
    }


def main():
    """Main function."""
    # Prepare benchmark
    resolutions = _parse_resolutions(args_cli.resolutions)
    cameras = []
    if args_cli.usd_camera:
        cameras.append("usd_camera")
    if args_cli.tiled_camera:
        cameras.append("tiled_camera")
    if args_cli.ray_caster_camera:
        cameras.append("ray_caster_camera")
    data_types = [args_cli.data_type]
    device_name = (
        torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else platform.processor()
    )

    # BENCHMARK 1 - Compare Depth Camera
    print(f"=== Benchmarking {args_cli.data_type} CAMERA ===")
    results: list[dict[str, object]] = []

    for idx, num_envs in enumerate(args_cli.num_envs):
        print(f"\n[INFO]: Benchmarking with {num_envs} envs. {idx + 1} / {len(args_cli.num_envs)}")
        for resolution in resolutions:
            
            for camera in cameras: 
                # USD Camera              
                if camera == "usd_camera":
                    single_scene_cfg = _make_scene_cfg_usd(
                        num_envs=num_envs,
                        height=resolution[0],
                        width=resolution[1],
                        data_types=data_types,
                        debug_vis=not args_cli.headless,
                    )
                    result = _run_benchmark(single_scene_cfg, "usd_camera")
            
                # Tiled Camera
                elif camera == "tiled_camera":
                    single_scene_cfg = _make_scene_cfg_tiled(
                        num_envs=num_envs,
                        height=resolution[0],
                        width=resolution[1],
                        data_types=data_types,
                        debug_vis=not args_cli.headless,
                    )
                    result = _run_benchmark(single_scene_cfg, "tiled_camera")
                
                # Multi-Mesh RayCaster Camera
                elif camera == "ray_caster_camera":

                    if args_cli.data_type == "rgb":
                        continue
                    
                    single_scene_cfg = _make_scene_cfg_ray_caster(
                        num_envs=num_envs,
                        height=resolution[0],
                        width=resolution[1],
                        data_types=data_types,
                        debug_vis=not args_cli.headless,
                    )
                    result = _run_benchmark(single_scene_cfg, "ray_caster_camera")
            
                result["num_envs"] = num_envs
                result["resolution"] = resolution
                result["mode"] = camera
                result["data_types"] = data_types
                results.append(result)
                del single_scene_cfg

    df_camera = pd.DataFrame(results)
    df_camera["device"] = device_name
    os.makedirs("outputs/benchmarks", exist_ok=True)
    df_camera.to_csv(f"outputs/benchmarks/camera_{args_cli.data_type}_USD_{args_cli.usd_camera}_Tiled_{args_cli.tiled_camera}_RayCaster_{args_cli.ray_caster_camera}_Resolution_{args_cli.resolutions}.csv", index=False)

    # Create .md file with all three tables
    for df, title in zip(
        [df_camera], [args_cli.data_type]
    ):
        with open(f"outputs/benchmarks/camera_benchmark_USD_{args_cli.usd_camera}_Tiled_{args_cli.tiled_camera}_RayCaster_{args_cli.ray_caster_camera}_Resolution_{args_cli.resolutions}_{title}.md", "w") as f:
            f.write(f"# {title}\n\n")
            f.write(dataframe_to_markdown(df, floatfmt=".3f"))
            f.write("\n\n")

if __name__ == "__main__":
    main()
    simulation_app.close()
