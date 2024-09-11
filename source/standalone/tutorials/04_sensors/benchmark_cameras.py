# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script might help you determine how many cameras your system can realistically run
at different desired settings. This script can also be used to sanity check visually
what the camera images and or point clouds output from the replicator looks like.


.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/benchmark_cameras.py -h

    # Usage with headless
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/benchmark_cameras.py -h --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from omni.isaac.lab.app import AppLauncher

# parse the arguments
args_cli = argparse.Namespace()

parser = argparse.ArgumentParser(description="This script can help you benchmark how many cameras you could run.")


def add_cli_args(parser):
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        required=False,
        help=(
            "Whether to visualize. Only switch to True if you don't care about the benchmarking results"
            " and are instead visually checking replicator output."
        ),
    )

    parser.add_argument(
        "--save_clouds",
        action="store_true",
        default=False,
        required=False,
        help="Whether to save clouds as .npys in addition to figures (visualize should also be on)",
    )

    parser.add_argument(
        "--num_tiled_cameras",
        type=int,
        default=2,
        required=False,
        help="How many tiled cameras to create",
    )

    parser.add_argument(
        "--num_standard_cameras", type=int, default=1, required=False, help="How many normal cameras to create"
    )

    parser.add_argument(
        "--num_ray_caster_cameras", type=int, default=1, required=False, help="How many normal cameras to create"
    )

    parser.add_argument(
        "--tiled_camera_replicators",
        nargs="+",
        type=str,
        default=["rgb", "depth"],
        help="What replicators to use for the tiled camera",
    )

    parser.add_argument(
        "--standard_camera_replicators",
        nargs="+",
        type=str,
        default=["rgb", "distance_to_image_plane"],
        help="What replicators to use for the usd camera",
    )

    parser.add_argument(
        "--ray_caster_camera_replicators",
        nargs="+",
        type=str,
        default=["distance_to_image_plane"],
        help="What replicators to use for the ray caster camera",
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
            "Enable undistorting from perspective view (distance to camera replicator)"
            "to orthogonal view (distance to plane replicator) for depth."
        ),
    )

    parser.add_argument(
        "--keep_raw_depth",
        dest="convert_depth_to_camera_to_image_plane",
        action="store_false",
        help=(
            "Disable undistorting from perspective view (distance to camera)"
            "to orthogonal view (distance to plane replicator) for depth."
        ),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=120,
        required=True,
        help="Height in pixels of cameras",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=140,
        required=True,
        help="Width in pixels of cameras",
    )

    parser.add_argument(
        "--warm_start_length",
        type=int,
        default=3,
        required=False,
        help="How many steps to run the sim before starting benchmark",
    )

    parser.add_argument(
        "--num_objects", type=int, default=10, required=False, help="How many objects to spawn into the scene."
    )

    parser.add_argument(
        "--experiment_length",
        type=int,
        default=30,
        required=False,
        help="How many steps to average over",
    )


add_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.visualize:
    print("[WARNING]: You have selected to visualize. which means your benchmark results will not be meaningful.")
    import matplotlib

    matplotlib.use("Agg")  # Use a non-interactive backend

if len(args_cli.ray_caster_visible_mesh_prim_paths) > 1:
    print("[WARNING]: Ray Casting is only currently supported for a single, static object")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import open3d as o3d
import random
import time
import torch
from matplotlib import pyplot as plt

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sensors.camera import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from omni.isaac.lab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from omni.isaac.lab.utils.math import convert_perspective_depth_image_to_orthogonal_depth_image, unproject_depth


def create_camera_base(
    camera_cls: type[Camera | TiledCamera],
    camera_cfg: type[CameraCfg | TiledCameraCfg],
    num_cams: int,
    data_types: list[str],
    height: int,
    width: int,
) -> Camera | TiledCamera | None:
    """Generalized function to create a camera or tiled camera sensor."""
    # Determine prim prefix based on the camera class
    name = camera_cls.__name__

    # Create the necessary prims
    for idx in range(num_cams):
        prim_utils.create_prim(f"/World/{name}_{idx:02d}", "Xform")

    # If valid camera settings are provided, create the camera
    if num_cams > 0 and len(data_types) > 0 and height > 0 and width > 0:
        cfg = camera_cfg(
            prim_path=f"/World/{name}_.*/{name}",
            update_period=0,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
            ),
        )
        return camera_cls(cfg=cfg)
    else:
        return None


def create_tiled_cameras(
    num_cams: int = 2, data_types: list[str] | None = None, height: int = 100, width: int = 120
) -> TiledCamera | None:
    if data_types is None:
        data_types = ["rgb", "depth"]
    """Defines the tiled camera sensor to add to the scene."""
    return create_camera_base(
        camera_cls=TiledCamera,
        camera_cfg=TiledCameraCfg,
        num_cams=num_cams,
        data_types=data_types,
        height=height,
        width=width,
    )


def create_cameras(
    num_cams: int = 2, data_types: list[str] | None = None, height: int = 100, width: int = 120
) -> Camera | None:
    """Defines the USD/Standard cameras."""
    if data_types is None:
        data_types = ["rgb", "depth"]
    return create_camera_base(
        camera_cls=Camera, camera_cfg=CameraCfg, num_cams=num_cams, data_types=data_types, height=height, width=width
    )


def create_ray_caster_cameras(
    num_cams: int = 2,
    data_types: list[str] = ["distance_to_image_plane"],
    mesh_prim_paths: list[str] = ["/World/ground"],
    height: int = 100,
    width: int = 120,
) -> RayCasterCamera | None:
    """Create the raycaster cameras; different configuration than USD/Tiled camera"""
    for idx in range(num_cams):
        prim_utils.create_prim(f"/World/RayCasterCamera_{idx:02d}/RayCaster", "Xform")

    if num_cams > 0 and len(data_types) > 0 and height > 0 and width > 0:
        cam_cfg = RayCasterCameraCfg(
            prim_path="/World/RayCasterCamera_.*/RayCaster",
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
        return RayCasterCamera(cfg=cam_cfg)

    else:
        return None


def design_scene(
    num_tiled_cams: int = 2,
    num_standard_cams: int = 0,
    num_ray_caster_cams: int = 0,
    tiled_camera_replicators: list[str] | None = None,
    standard_camera_replicators: list[str] | None = None,
    ray_caster_camera_replicators: list[str] | None = None,
    height: int = 100,
    width: int = 200,
    num_objects: int = 20,
    mesh_prim_paths: list[str] = ["/World/ground"],
) -> dict:
    """Design the scene."""
    if tiled_camera_replicators is None:
        tiled_camera_replicators = ["rgb"]
    if standard_camera_replicators is None:
        standard_camera_replicators = ["rgb"]
    if ray_caster_camera_replicators is None:
        ray_caster_camera_replicators = ["rgb"]

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
        num_cams=num_standard_cams, data_types=standard_camera_replicators, height=height, width=width
    )
    tiled_camera = create_tiled_cameras(
        num_cams=num_tiled_cams, data_types=tiled_camera_replicators, height=height, width=width
    )
    ray_caster_camera = create_ray_caster_cameras(
        num_cams=num_ray_caster_cams,
        data_types=ray_caster_camera_replicators,
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


def numpy_to_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert a NumPy Nx3 pointcloud to an Open3d Nx3 pointcloud for easy plotting"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene_entities: dict,
    warm_start_length: int = 10,
    experiment_length: int = 100,
    tiled_camera_replicators: list[str] | None = None,
    standard_camera_replicators: list[str] | None = None,
    ray_caster_camera_replicators: list[str] | None = None,
    depth_predicate: Callable = lambda x: "to" in x or x == "depth",
    perspective_depth_predicate: Callable = lambda x: x == "depth" or x == "depth_to_camera",
    convert_depth_to_camera_to_image_plane: bool = True,
    visualize: bool = False,
) -> dict:
    """Run the simulator with all cameras, and return timing analytics. Visualize if desired."""

    if tiled_camera_replicators is None:
        tiled_camera_replicators = ["rgb"]
    if standard_camera_replicators is None:
        standard_camera_replicators = ["rgb"]
    if ray_caster_camera_replicators is None:
        ray_caster_camera_replicators = ["rgb"]
    # Extract entities for simplified notation
    num_tiled_cameras = 0
    tiled_camera = None
    num_standard_cameras = 0
    standard_camera = None
    num_ray_caster_cameras = 0
    ray_caster_camera = None

    if "tiled_camera" in scene_entities:
        tiled_camera: TiledCamera = scene_entities["tiled_camera"]
        num_tiled_cameras = tiled_camera.data.intrinsic_matrices.size(0)
    if "standard_camera" in scene_entities:
        standard_camera: Camera = scene_entities["standard_camera"]
        num_standard_cameras = standard_camera.data.intrinsic_matrices.size(0)
    if "ray_caster_camera" in scene_entities:
        ray_caster_camera: RayCasterCamera = scene_entities["ray_caster_camera"]
        num_ray_caster_cameras = ray_caster_camera.data.intrinsic_matrices.size(0)

    camera_counts = [num_tiled_cameras, num_standard_cameras, num_ray_caster_cameras]
    cameras = [tiled_camera, standard_camera, ray_caster_camera]
    all_replicators = [tiled_camera_replicators, standard_camera_replicators, ray_caster_camera_replicators]
    labels = ["tiled", "standard", "ray_caster"]

    for num, camera in zip(camera_counts, cameras):
        if num > 0:
            positions = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device).repeat(num, 1)
            targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device).repeat(num, 1)
            camera.set_world_poses_from_view(positions, targets)

    # Initialize timing variables
    timestep = 0
    total_time = 0.0
    valid_timesteps = 0
    sim_step_time = 0.0
    vision_processing_time = 0.0

    while simulation_app.is_running() and timestep < experiment_length:
        print(f"On timestep {timestep} of {experiment_length}, with warm start of {warm_start_length}")

        # Measure simulation step time
        sim_start_time = time.time()

        sim.step()
        sim_end_time = time.time()
        sim_step_time += sim_end_time - sim_start_time

        if timestep > warm_start_length:
            vision_start_time = time.time()

            clouds = {}
            images = {}
            depth_images = {}

            for num_cams, camera, replicators, label in zip(camera_counts, cameras, all_replicators, labels):
                if num_cams > 0:
                    camera.update(dt=sim.get_physics_dt())
                    for replicator in replicators:
                        data_label = label + "_" + str(replicator)

                        if depth_predicate(replicator):  # is a depth image, want to create cloud
                            depth = camera.data.output[replicator]
                            depth_images[data_label + "_raw"] = depth
                            if perspective_depth_predicate(replicator) and convert_depth_to_camera_to_image_plane:
                                depth = convert_perspective_depth_image_to_orthogonal_depth_image(
                                    perspective_depth=camera.data.output[replicator],
                                    intrinsics=camera.data.intrinsic_matrices,
                                )
                                depth_images[data_label + "_undistorted"] = depth

                            pointcloud = unproject_depth(depth=depth, intrinsics=camera.data.intrinsic_matrices)
                            clouds[data_label] = pointcloud
                        else:  # rgb image, just  save it
                            image = camera.data.output[replicator]
                            images[data_label] = image

            vision_end_time = time.time()
            vision_processing_time += vision_end_time - vision_start_time

            if visualize:
                plot_point_clouds(clouds=clouds, save_name=f"saved_clouds_timestep_{timestep}.png")
                if images:
                    plot_images(
                        images,
                        cols=4,
                        cmap="gray",
                        title_prefix="Image",
                        save_name=f"saved_images_timestep_{timestep}.png",
                    )
                if depth_images:
                    plot_images(
                        depth_images,
                        cols=4,
                        cmap="viridis",
                        title_prefix="Depth Image",
                        save_name=f"saved_depth_images_timestep_{timestep}.png",
                    )

            total_time += vision_end_time - vision_start_time
            valid_timesteps += 1

        timestep += 1

    # Calculate average timings
    if valid_timesteps > 0:
        avg_timestep_duration = total_time / valid_timesteps
        avg_sim_step_duration = sim_step_time / experiment_length
        avg_vision_processing_duration = vision_processing_time / valid_timesteps
    else:
        avg_timestep_duration = 0.0
        avg_sim_step_duration = 0.0
        avg_vision_processing_duration = 0.0

    # Package timing analytics in a dictionary
    timing_analytics = {
        "average_timestep_duration": avg_timestep_duration,
        "average_sim_step_duration": avg_sim_step_duration,
        "average_vision_processing_duration": avg_vision_processing_duration,
        "total_simulation_time": sim_step_time,
        "total_vision_processing_time": vision_processing_time,
        "total_experiment_duration": sim_step_time + vision_processing_time,
    }

    print("Benchmark Result:---")
    print(f"Average timestep duration: {avg_timestep_duration:.6f} seconds")
    print(f"Average simulation step duration: {avg_sim_step_duration:.6f} seconds")
    print(f"Average vision processing duration: {avg_vision_processing_duration:.6f} seconds")

    return timing_analytics


def plot_images(
    images: dict, cols: int = 4, cmap: str = "gray", title_prefix: str = "Image", save_name: str | None = None
):
    """Plot the provided images in a grid, labelling each one with the relevant info

    Args:
        images: the images, where their label is key, and the image is value
        cols: how many columns to use in the grid. Defaults to 4.
        cmap: what color map to use. Defaults to "gray".
        title_prefix: what prefix to prepend to image titles. Defaults to "Image".
        save_name: Filename to save the image to. Defaults to None.
    """
    total_images = sum(image_tensor.shape[0] for image_tensor in images.values())

    # Determine the grid size (rows and columns)
    cols = min(total_images, cols)
    rows = (total_images + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier iteration

    img_idx = 0
    for key, image_tensor in images.items():
        for idx, image in enumerate(image_tensor):
            ax = axs[img_idx]
            ax.imshow(image.cpu().numpy(), cmap=cmap)
            ax.set_title(f"{key} - {title_prefix} {idx}")
            ax.axis("off")  # Hide axis for cleaner visualization
            img_idx += 1

    # Hide any remaining empty subplots
    for ax in axs[img_idx:]:
        ax.axis("off")

    if save_name:
        plt.savefig(save_name)
        plt.close()  # Close the figure to free up memory


def plot_point_clouds(
    clouds: dict, viewpoints: list | None = None, cols: int = 4, save_name: str | None = None, timestep: int = 0
):
    """Plot pointclouds together from different views, and save it if a save_name is provided

    Args:
        clouds: the pointclouds to plot, where the key is the label and the value
            is the Open3D PointCloud
        viewpoints: from what Elivs and Azims to view clouds from. Defaults to None.
        cols: how many columns to use. Defaults to 4.
        save_name: _description_. Defaults to None.
        timestep: _description_. Defaults to 0.
    """
    if viewpoints is None:
        viewpoints = [
            {"elev": 30, "azim": 45},
            {"elev": 210, "azim": 315},
        ]

    num_views = len(viewpoints)
    rows = (num_views + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier iteration

    for i, ax in enumerate(axs):
        if i < num_views:
            viewpoint = viewpoints[i]
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

            for key, cloud in clouds.items():
                for idx, single_cloud in enumerate(cloud):
                    points = single_cloud.cpu()

                    # Plot the point cloud with a random color
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        color=np.random.rand(3),
                        label=key + f"_cloud{idx}",
                        s=1,
                    )

            # Set the viewpoint
            ax.view_init(elev=viewpoint["elev"], azim=viewpoint["azim"])
            ax.set_title(f"Cloud {timestep} | View: elev: {viewpoint['elev']}, azim: {viewpoint['azim']}")
            ax.axis("off")  # Hide axes for cleaner visualization
            ax.legend()
        else:
            ax.axis("off")  # Hide any extra subplots that aren't needed

    if save_name:
        plt.savefig(save_name, dpi=300)
        plt.close()  # Close the figure to free up memory


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cpu" if args_cli.cpu else "cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # design the scene
    print("[INFO]: Designing the scene")

    if args_cli.num_tiled_cameras + args_cli.num_standard_cameras + args_cli.num_ray_caster_cameras <= 0:
        raise ValueError("You must select at least one camera.")
    scene_entities = design_scene(
        num_tiled_cams=args_cli.num_tiled_cameras,
        num_standard_cams=args_cli.num_standard_cameras,
        num_ray_caster_cams=args_cli.num_ray_caster_cameras,
        tiled_camera_replicators=args_cli.tiled_camera_replicators,
        standard_camera_replicators=args_cli.standard_camera_replicators,
        ray_caster_camera_replicators=args_cli.ray_caster_camera_replicators,
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
        sim,
        scene_entities,
        warm_start_length=args_cli.warm_start_length,
        experiment_length=args_cli.experiment_length,
        tiled_camera_replicators=args_cli.tiled_camera_replicators,
        standard_camera_replicators=args_cli.standard_camera_replicators,
        ray_caster_camera_replicators=args_cli.ray_caster_camera_replicators,
        convert_depth_to_camera_to_image_plane=args_cli.convert_depth_to_camera_to_image_plane,
        visualize=args_cli.visualize,
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
