# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script might help you determine how many cameras your system can realistically run
at different desired settings.

You can supply different task environments to inject cameras into, or just test a sample scene.
Additionally, you can automatically find the maximum amount of cameras you can run a task with
through the auto-tune functionality.

.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p scripts/benchmarks/benchmark_cameras.py -h

    # Usage with headless
    ./isaaclab.sh -p scripts/benchmarks/benchmark_cameras.py -h --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# parse the arguments
args_cli = argparse.Namespace()

parser = argparse.ArgumentParser(description="This script can help you benchmark how many cameras you could run.")

"""
The following arguments only need to be supplied for when one wishes
to try injecting cameras into their environment, and automatically determining
the maximum camera count.
"""
parser.add_argument(
    "--task",
    type=str,
    default=None,
    required=False,
    help="Supply this argument to spawn cameras within an known manager-based task environment.",
)

parser.add_argument(
    "--autotune",
    default=False,
    action="store_true",
    help=(
        "Autotuning is only supported for provided task environments."
        " Supply this argument to increase the number of environments until a desired threshold is reached."
        "Install pynvml in your environment; ./isaaclab.sh -m pip install pynvml"
    ),
)

parser.add_argument(
    "--task_num_cameras_per_env",
    type=int,
    default=1,
    help="The number of cameras per environment to use when using a known task.",
)

parser.add_argument(
    "--use_fabric", action="store_true", default=False, help="Enable fabric and use USD I/O operations."
)

parser.add_argument(
    "--autotune_max_percentage_util",
    nargs="+",
    type=float,
    default=[100.0, 80.0, 80.0, 80.0],
    required=False,
    help=(
        "The system utilization percentage thresholds to reach before an autotune is finished. "
        "If any one of these limits are hit, the autotune stops."
        "Thresholds are, in order, maximum CPU percentage utilization,"
        "maximum RAM percentage utilization, maximum GPU compute percent utilization, "
        "amd maximum GPU memory utilization."
    ),
)

parser.add_argument(
    "--autotune_max_camera_count", type=int, default=4096, help="The maximum amount of cameras allowed in an autotune."
)

parser.add_argument(
    "--autotune_camera_count_interval",
    type=int,
    default=25,
    help=(
        "The number of cameras to try to add to the environment if the current camera count"
        " falls within permitted system resource utilization limits."
    ),
)

"""
The following arguments are shared for when injecting cameras into a task environment,
as well as when creating cameras independent of a task environment.
"""

parser.add_argument(
    "--num_tiled_cameras",
    type=int,
    default=0,
    required=False,
    help="Number of tiled cameras to create. For autotuning, this is how many cameras to start with.",
)

parser.add_argument(
    "--num_standard_cameras",
    type=int,
    default=0,
    required=False,
    help="Number of standard cameras to create. For autotuning, this is how many cameras to start with.",
)

parser.add_argument(
    "--num_ray_caster_cameras",
    type=int,
    default=0,
    required=False,
    help="Number of ray caster cameras to create. For autotuning, this is how many cameras to start with.",
)

parser.add_argument(
    "--tiled_camera_data_types",
    nargs="+",
    type=str,
    default=["rgb", "depth"],
    help="The data types rendered by the tiled camera",
)

parser.add_argument(
    "--standard_camera_data_types",
    nargs="+",
    type=str,
    default=["rgb", "distance_to_image_plane", "distance_to_camera"],
    help="The data types rendered by the standard camera",
)

parser.add_argument(
    "--ray_caster_camera_data_types",
    nargs="+",
    type=str,
    default=["distance_to_image_plane"],
    help="The data types rendered by the ray caster camera.",
)

parser.add_argument(
    "--ray_caster_visible_mesh_prim_paths",
    nargs="+",
    type=str,
    default=["/World/ground"],
    help="WARNING: Ray Caster can currently only cast against a single, static, object",
)

parser.add_argument(
    "--convert_depth_to_camera_to_image_plane",
    action="store_true",
    default=True,
    help=(
        "Enable undistorting from perspective view (distance to camera data_type)"
        "to orthogonal view (distance to plane data_type) for depth."
        "This is currently needed to create undisorted depth images/point cloud."
    ),
)

parser.add_argument(
    "--keep_raw_depth",
    dest="convert_depth_to_camera_to_image_plane",
    action="store_false",
    help=(
        "Disable undistorting from perspective view (distance to camera)"
        "to orthogonal view (distance to plane data_type) for depth."
    ),
)

parser.add_argument(
    "--height",
    type=int,
    default=120,
    required=False,
    help="Height in pixels of cameras",
)

parser.add_argument(
    "--width",
    type=int,
    default=140,
    required=False,
    help="Width in pixels of cameras",
)

parser.add_argument(
    "--warm_start_length",
    type=int,
    default=3,
    required=False,
    help=(
        "Number of steps to run the sim before starting benchmark."
        "Needed to avoid blank images at the start of the simulation."
    ),
)

parser.add_argument(
    "--experiment_length",
    type=int,
    default=15,
    required=False,
    help="Number of steps to average over",
)

# This argument is only used when a task is not provided.
parser.add_argument(
    "--num_objects",
    type=int,
    default=10,
    required=False,
    help="Number of objects to spawn into the scene when not using a known task.",
)


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.autotune:
    import pynvml

if len(args_cli.ray_caster_visible_mesh_prim_paths) > 1:
    print("[WARNING]: Ray Casting is only currently supported for a single, static object")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import random
import time
import torch

import isaacsim.core.utils.prims as prim_utils
import psutil
from isaacsim.core.utils.stage import create_new_stage

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.sensors import (
    Camera,
    CameraCfg,
    RayCasterCamera,
    RayCasterCameraCfg,
    TiledCamera,
    TiledCameraCfg,
    patterns,
)
from isaaclab.utils.math import orthogonalize_perspective_depth, unproject_depth

from isaaclab_tasks.utils import load_cfg_from_registry

"""
Camera Creation
"""


def create_camera_base(
    camera_cfg: type[CameraCfg | TiledCameraCfg],
    num_cams: int,
    data_types: list[str],
    height: int,
    width: int,
    prim_path: str | None = None,
    instantiate: bool = True,
) -> Camera | TiledCamera | CameraCfg | TiledCameraCfg | None:
    """Generalized function to create a camera or tiled camera sensor."""
    # Determine prim prefix based on the camera class
    name = camera_cfg.class_type.__name__

    if instantiate:
        # Create the necessary prims
        for idx in range(num_cams):
            prim_utils.create_prim(f"/World/{name}_{idx:02d}", "Xform")
    if prim_path is None:
        prim_path = f"/World/{name}_.*/{name}"
    # If valid camera settings are provided, create the camera
    if num_cams > 0 and len(data_types) > 0 and height > 0 and width > 0:
        cfg = camera_cfg(
            prim_path=prim_path,
            update_period=0,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
            ),
        )
        if instantiate:
            return camera_cfg.class_type(cfg=cfg)
        else:
            return cfg
    else:
        return None


def create_tiled_cameras(
    num_cams: int = 2, data_types: list[str] | None = None, height: int = 100, width: int = 120
) -> TiledCamera | None:
    if data_types is None:
        data_types = ["rgb", "depth"]
    """Defines the tiled camera sensor to add to the scene."""
    return create_camera_base(
        camera_cfg=TiledCameraCfg,
        num_cams=num_cams,
        data_types=data_types,
        height=height,
        width=width,
    )


def create_cameras(
    num_cams: int = 2, data_types: list[str] | None = None, height: int = 100, width: int = 120
) -> Camera | None:
    """Defines the Standard cameras."""
    if data_types is None:
        data_types = ["rgb", "depth"]
    return create_camera_base(
        camera_cfg=CameraCfg, num_cams=num_cams, data_types=data_types, height=height, width=width
    )


def create_ray_caster_cameras(
    num_cams: int = 2,
    data_types: list[str] = ["distance_to_image_plane"],
    mesh_prim_paths: list[str] = ["/World/ground"],
    height: int = 100,
    width: int = 120,
    prim_path: str = "/World/RayCasterCamera_.*/RayCaster",
    instantiate: bool = True,
) -> RayCasterCamera | RayCasterCameraCfg | None:
    """Create the raycaster cameras; different configuration than Standard/Tiled camera"""
    for idx in range(num_cams):
        prim_utils.create_prim(f"/World/RayCasterCamera_{idx:02d}/RayCaster", "Xform")

    if num_cams > 0 and len(data_types) > 0 and height > 0 and width > 0:
        cam_cfg = RayCasterCameraCfg(
            prim_path=prim_path,
            mesh_prim_paths=mesh_prim_paths,
            update_period=0,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            data_types=data_types,
            debug_vis=False,
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                height=480,
                width=640,
            ),
        )
        if instantiate:
            return RayCasterCamera(cfg=cam_cfg)
        else:
            return cam_cfg

    else:
        return None


def create_tiled_camera_cfg(prim_path: str) -> TiledCameraCfg:
    """Grab a simple tiled camera config for injecting into task environments."""
    return create_camera_base(
        TiledCameraCfg,
        num_cams=args_cli.num_tiled_cameras,
        data_types=args_cli.tiled_camera_data_types,
        width=args_cli.width,
        height=args_cli.height,
        prim_path="{ENV_REGEX_NS}/" + prim_path,
        instantiate=False,
    )


def create_standard_camera_cfg(prim_path: str) -> CameraCfg:
    """Grab a simple standard camera config for injecting into task environments."""
    return create_camera_base(
        CameraCfg,
        num_cams=args_cli.num_standard_cameras,
        data_types=args_cli.standard_camera_data_types,
        width=args_cli.width,
        height=args_cli.height,
        prim_path="{ENV_REGEX_NS}/" + prim_path,
        instantiate=False,
    )


def create_ray_caster_camera_cfg(prim_path: str) -> RayCasterCameraCfg:
    """Grab a simple ray caster config for injecting into task environments."""
    return create_ray_caster_cameras(
        num_cams=args_cli.num_ray_caster_cameras,
        data_types=args_cli.ray_caster_camera_data_types,
        width=args_cli.width,
        height=args_cli.height,
        prim_path="{ENV_REGEX_NS}/" + prim_path,
    )


"""
Scene Creation
"""


def design_scene(
    num_tiled_cams: int = 2,
    num_standard_cams: int = 0,
    num_ray_caster_cams: int = 0,
    tiled_camera_data_types: list[str] | None = None,
    standard_camera_data_types: list[str] | None = None,
    ray_caster_camera_data_types: list[str] | None = None,
    height: int = 100,
    width: int = 200,
    num_objects: int = 20,
    mesh_prim_paths: list[str] = ["/World/ground"],
) -> dict:
    """Design the scene."""
    if tiled_camera_data_types is None:
        tiled_camera_data_types = ["rgb"]
    if standard_camera_data_types is None:
        standard_camera_data_types = ["rgb"]
    if ray_caster_camera_data_types is None:
        ray_caster_camera_data_types = ["distance_to_image_plane"]

    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create a dictionary for the scene entities
    scene_entities = {}

    # Xform to hold objects
    prim_utils.create_prim("/World/Objects", "Xform")
    # Random objects
    for i in range(num_objects):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # sample random color
        color = (random.random(), random.random(), random.random())
        # choose random prim type
        prim_type = random.choice(["Cube", "Cone", "Cylinder"])
        common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=5.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5),
            "semantic_tags": [("class", prim_type)],
        }
        if prim_type == "Cube":
            shape_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **common_properties)
        elif prim_type == "Cone":
            shape_cfg = sim_utils.ConeCfg(radius=0.1, height=0.25, **common_properties)
        elif prim_type == "Cylinder":
            shape_cfg = sim_utils.CylinderCfg(radius=0.25, height=0.25, **common_properties)
        # Rigid Object
        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}",
            spawn=shape_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)

    # Sensors
    standard_camera = create_cameras(
        num_cams=num_standard_cams, data_types=standard_camera_data_types, height=height, width=width
    )
    tiled_camera = create_tiled_cameras(
        num_cams=num_tiled_cams, data_types=tiled_camera_data_types, height=height, width=width
    )
    ray_caster_camera = create_ray_caster_cameras(
        num_cams=num_ray_caster_cams,
        data_types=ray_caster_camera_data_types,
        mesh_prim_paths=mesh_prim_paths,
        height=height,
        width=width,
    )
    # return the scene information
    if tiled_camera is not None:
        scene_entities["tiled_camera"] = tiled_camera
    if standard_camera is not None:
        scene_entities["standard_camera"] = standard_camera
    if ray_caster_camera is not None:
        scene_entities["ray_caster_camera"] = ray_caster_camera
    return scene_entities


def inject_cameras_into_task(
    task: str,
    num_cams: int,
    camera_name_prefix: str,
    camera_creation_callable: Callable,
    num_cameras_per_env: int = 1,
) -> gym.Env:
    """Loads the task, sticks cameras into the config, and creates the environment."""
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.sim.device = args_cli.device
    cfg.sim.use_fabric = args_cli.use_fabric
    scene_cfg = cfg.scene

    num_envs = int(num_cams / num_cameras_per_env)
    scene_cfg.num_envs = num_envs

    for idx in range(num_cameras_per_env):
        suffix = "" if idx == 0 else str(idx)
        name = camera_name_prefix + suffix
        setattr(scene_cfg, name, camera_creation_callable(name))
    cfg.scene = scene_cfg
    env = gym.make(task, cfg=cfg)
    return env


"""
System diagnosis
"""


def get_utilization_percentages(reset: bool = False, max_values: list[float] = [0.0, 0.0, 0.0, 0.0]) -> list[float]:
    """Get the maximum CPU, RAM, GPU utilization (processing), and
    GPU memory usage percentages since the last time reset was true."""
    if reset:
        max_values[:] = [0, 0, 0, 0]  # Reset the max values

    # CPU utilization
    cpu_usage = psutil.cpu_percent(interval=0.1)
    max_values[0] = max(max_values[0], cpu_usage)

    # RAM utilization
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    max_values[1] = max(max_values[1], ram_usage)

    # GPU utilization using pynvml
    if torch.cuda.is_available():

        if args_cli.autotune:
            pynvml.nvmlInit()  # Initialize NVML
            for i in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # GPU Utilization
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_processing_utilization_percent = gpu_utilization.gpu  # GPU core utilization
                max_values[2] = max(max_values[2], gpu_processing_utilization_percent)

                # GPU Memory Usage
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_total = memory_info.total
                gpu_memory_used = memory_info.used
                gpu_memory_utilization_percent = (gpu_memory_used / gpu_memory_total) * 100
                max_values[3] = max(max_values[3], gpu_memory_utilization_percent)

            pynvml.nvmlShutdown()  # Shutdown NVML after usage
    else:
        gpu_processing_utilization_percent = None
        gpu_memory_utilization_percent = None
    return max_values


"""
Experiment
"""


def run_simulator(
    sim: sim_utils.SimulationContext | None,
    scene_entities: dict | InteractiveScene,
    warm_start_length: int = 10,
    experiment_length: int = 100,
    tiled_camera_data_types: list[str] | None = None,
    standard_camera_data_types: list[str] | None = None,
    ray_caster_camera_data_types: list[str] | None = None,
    depth_predicate: Callable = lambda x: "to" in x or x == "depth",
    perspective_depth_predicate: Callable = lambda x: x == "distance_to_camera",
    convert_depth_to_camera_to_image_plane: bool = True,
    max_cameras_per_env: int = 1,
    env: gym.Env | None = None,
) -> dict:
    """Run the simulator with all cameras, and return timing analytics. Visualize if desired."""

    if tiled_camera_data_types is None:
        tiled_camera_data_types = ["rgb"]
    if standard_camera_data_types is None:
        standard_camera_data_types = ["rgb"]
    if ray_caster_camera_data_types is None:
        ray_caster_camera_data_types = ["distance_to_image_plane"]

    # Initialize camera lists
    tiled_cameras = []
    standard_cameras = []
    ray_caster_cameras = []

    # Dynamically extract cameras from the scene entities up to max_cameras_per_env
    for i in range(max_cameras_per_env):
        # Extract tiled cameras
        tiled_camera_key = f"tiled_camera{i}" if i > 0 else "tiled_camera"
        standard_camera_key = f"standard_camera{i}" if i > 0 else "standard_camera"
        ray_caster_camera_key = f"ray_caster_camera{i}" if i > 0 else "ray_caster_camera"

        try:  # if instead you checked ... if key is in scene_entities... # errors out always even if key present
            tiled_cameras.append(scene_entities[tiled_camera_key])
            standard_cameras.append(scene_entities[standard_camera_key])
            ray_caster_cameras.append(scene_entities[ray_caster_camera_key])
        except KeyError:
            break

    # Initialize camera counts
    camera_lists = [tiled_cameras, standard_cameras, ray_caster_cameras]
    camera_data_types = [tiled_camera_data_types, standard_camera_data_types, ray_caster_camera_data_types]
    labels = ["tiled", "standard", "ray_caster"]

    if sim is not None:
        # Set camera world poses
        for camera_list in camera_lists:
            for camera in camera_list:
                num_cameras = camera.data.intrinsic_matrices.size(0)
                positions = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device).repeat(num_cameras, 1)
                targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device).repeat(num_cameras, 1)
                camera.set_world_poses_from_view(positions, targets)

    # Initialize timing variables
    timestep = 0
    total_time = 0.0
    valid_timesteps = 0
    sim_step_time = 0.0

    while simulation_app.is_running() and timestep < experiment_length:
        print(f"On timestep {timestep} of {experiment_length}, with warm start of {warm_start_length}")
        get_utilization_percentages()

        # Measure the total simulation step time
        step_start_time = time.time()

        if sim is not None:
            sim.step()

        if env is not None:
            with torch.inference_mode():
                # compute zero actions
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                # apply actions
                env.step(actions)

        # Update cameras and process vision data within the simulation step
        clouds = {}
        images = {}
        depth_images = {}

        # Loop through all camera lists and their data_types
        for camera_list, data_types, label in zip(camera_lists, camera_data_types, labels):
            for cam_idx, camera in enumerate(camera_list):

                if env is None:  # No env, need to step cams manually
                    # Only update the camera if it hasn't been updated as part of scene_entities.update ...
                    camera.update(dt=sim.get_physics_dt())

                for data_type in data_types:
                    data_label = f"{label}_{cam_idx}_{data_type}"

                    if depth_predicate(data_type):  # is a depth image, want to create cloud
                        depth = camera.data.output[data_type]
                        depth_images[data_label + "_raw"] = depth
                        if perspective_depth_predicate(data_type) and convert_depth_to_camera_to_image_plane:
                            depth = orthogonalize_perspective_depth(
                                camera.data.output[data_type], camera.data.intrinsic_matrices
                            )
                            depth_images[data_label + "_undistorted"] = depth

                        pointcloud = unproject_depth(depth=depth, intrinsics=camera.data.intrinsic_matrices)
                        clouds[data_label] = pointcloud
                    else:  # rgb image, just save it
                        image = camera.data.output[data_type]
                        images[data_label] = image

        # End timing for the step
        step_end_time = time.time()
        sim_step_time += step_end_time - step_start_time

        if timestep > warm_start_length:
            get_utilization_percentages(reset=True)
            total_time += step_end_time - step_start_time
            valid_timesteps += 1

        timestep += 1

    # Calculate average timings
    if valid_timesteps > 0:
        avg_timestep_duration = total_time / valid_timesteps
        avg_sim_step_duration = sim_step_time / experiment_length
    else:
        avg_timestep_duration = 0.0
        avg_sim_step_duration = 0.0

    # Package timing analytics in a dictionary
    timing_analytics = {
        "average_timestep_duration": avg_timestep_duration,
        "average_sim_step_duration": avg_sim_step_duration,
        "total_simulation_time": sim_step_time,
        "total_experiment_duration": sim_step_time,
    }

    system_utilization_analytics = get_utilization_percentages()

    print("--- Benchmark Results ---")
    print(f"Average timestep duration: {avg_timestep_duration:.6f} seconds")
    print(f"Average simulation step duration: {avg_sim_step_duration:.6f} seconds")
    print(f"Total simulation time: {sim_step_time:.6f} seconds")
    print("\nSystem Utilization Statistics:")
    print(
        f"| CPU:{system_utilization_analytics[0]}% | "
        f"RAM:{system_utilization_analytics[1]}% | "
        f"GPU Compute:{system_utilization_analytics[2]}% | "
        f" GPU Memory: {system_utilization_analytics[3]:.2f}% |"
    )

    return {"timing_analytics": timing_analytics, "system_utilization_analytics": system_utilization_analytics}


def main():
    """Main function."""
    # Load simulation context
    if args_cli.num_tiled_cameras + args_cli.num_standard_cameras + args_cli.num_ray_caster_cameras <= 0:
        raise ValueError("You must select at least one camera.")
    if (
        (args_cli.num_tiled_cameras > 0 and args_cli.num_standard_cameras > 0)
        or (args_cli.num_ray_caster_cameras > 0 and args_cli.num_standard_cameras > 0)
        or (args_cli.num_ray_caster_cameras > 0 and args_cli.num_tiled_cameras > 0)
    ):
        print("[WARNING]: You have elected to use more than one camera type.")
        print("[WARNING]: For a benchmark to be meaningful, use ONLY ONE camera type at a time.")
        print(
            "[WARNING]: For example, if num_tiled_cameras=100, for a meaningful benchmark,"
            "num_standard_cameras should be 0, and num_ray_caster_cameras should be 0"
        )
        raise ValueError("Benchmark one camera at a time.")

    print("[INFO]: Designing the scene")
    if args_cli.task is None:
        print("[INFO]: No task environment provided, creating random scene.")
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        scene_entities = design_scene(
            num_tiled_cams=args_cli.num_tiled_cameras,
            num_standard_cams=args_cli.num_standard_cameras,
            num_ray_caster_cams=args_cli.num_ray_caster_cameras,
            tiled_camera_data_types=args_cli.tiled_camera_data_types,
            standard_camera_data_types=args_cli.standard_camera_data_types,
            ray_caster_camera_data_types=args_cli.ray_caster_camera_data_types,
            height=args_cli.height,
            width=args_cli.width,
            num_objects=args_cli.num_objects,
            mesh_prim_paths=args_cli.ray_caster_visible_mesh_prim_paths,
        )
        # Play simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run simulator
        run_simulator(
            sim=sim,
            scene_entities=scene_entities,
            warm_start_length=args_cli.warm_start_length,
            experiment_length=args_cli.experiment_length,
            tiled_camera_data_types=args_cli.tiled_camera_data_types,
            standard_camera_data_types=args_cli.standard_camera_data_types,
            ray_caster_camera_data_types=args_cli.ray_caster_camera_data_types,
            convert_depth_to_camera_to_image_plane=args_cli.convert_depth_to_camera_to_image_plane,
        )
    else:
        print("[INFO]: Using known task environment, injecting cameras.")
        autotune_iter = 0
        max_sys_util_thresh = [0.0, 0.0, 0.0]
        max_num_cams = max(args_cli.num_tiled_cameras, args_cli.num_standard_cameras, args_cli.num_ray_caster_cameras)
        cur_num_cams = max_num_cams
        cur_sys_util = max_sys_util_thresh
        interval = args_cli.autotune_camera_count_interval

        if args_cli.autotune:
            max_sys_util_thresh = args_cli.autotune_max_percentage_util
            max_num_cams = args_cli.autotune_max_camera_count
            print("[INFO]: Auto tuning until any of the following threshold are met")
            print(f"|CPU: {max_sys_util_thresh[0]}% | RAM {max_sys_util_thresh[1]}% | GPU: {max_sys_util_thresh[2]}% |")
            print(f"[INFO]: Maximum number of cameras allowed: {max_num_cams}")
        # Determine which camera is being tested...
        tiled_camera_cfg = create_tiled_camera_cfg("tiled_camera")
        standard_camera_cfg = create_standard_camera_cfg("standard_camera")
        ray_caster_camera_cfg = create_ray_caster_camera_cfg("ray_caster_camera")
        camera_name_prefix = ""
        camera_creation_callable = None
        num_cams = 0
        if tiled_camera_cfg is not None:
            camera_name_prefix = "tiled_camera"
            camera_creation_callable = create_tiled_camera_cfg
            num_cams = args_cli.num_tiled_cameras
        elif standard_camera_cfg is not None:
            camera_name_prefix = "standard_camera"
            camera_creation_callable = create_standard_camera_cfg
            num_cams = args_cli.num_standard_cameras
        elif ray_caster_camera_cfg is not None:
            camera_name_prefix = "ray_caster_camera"
            camera_creation_callable = create_ray_caster_camera_cfg
            num_cams = args_cli.num_ray_caster_cameras

        while (
            all(cur <= max_thresh for cur, max_thresh in zip(cur_sys_util, max_sys_util_thresh))
            and cur_num_cams <= max_num_cams
        ):
            cur_num_cams = num_cams + interval * autotune_iter
            autotune_iter += 1

            env = inject_cameras_into_task(
                task=args_cli.task,
                num_cams=cur_num_cams,
                camera_name_prefix=camera_name_prefix,
                camera_creation_callable=camera_creation_callable,
                num_cameras_per_env=args_cli.task_num_cameras_per_env,
            )
            env.reset()
            print(f"Testing with {cur_num_cams} {camera_name_prefix}")
            analysis = run_simulator(
                sim=None,
                scene_entities=env.unwrapped.scene,
                warm_start_length=args_cli.warm_start_length,
                experiment_length=args_cli.experiment_length,
                tiled_camera_data_types=args_cli.tiled_camera_data_types,
                standard_camera_data_types=args_cli.standard_camera_data_types,
                ray_caster_camera_data_types=args_cli.ray_caster_camera_data_types,
                convert_depth_to_camera_to_image_plane=args_cli.convert_depth_to_camera_to_image_plane,
                max_cameras_per_env=args_cli.task_num_cameras_per_env,
                env=env,
            )

            cur_sys_util = analysis["system_utilization_analytics"]
            print("Triggering reset...")
            env.close()
            create_new_stage()
        print("[INFO]: DONE! Feel free to CTRL + C Me ")
        print(f"[INFO]: If you've made it this far, you can likely simulate {cur_num_cams} {camera_name_prefix}")
        print("Keep in mind, this is without any training running on the GPU.")
        print("Set lower utilization thresholds to account for training.")

        if not args_cli.autotune:
            print("[WARNING]: GPU Util Statistics only correct while autotuning, ignore above.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
