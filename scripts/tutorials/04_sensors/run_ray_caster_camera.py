# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the ray-cast camera sensor from the Isaac Lab framework.

The camera sensor is based on using Warp kernels which do ray-casting against static meshes.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/04_sensors/run_ray_caster_camera.py --viz kit

"""

"""Parse the command-line arguments first."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import instantiate

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the ray-cast camera sensor.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to generate.")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")
# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""

import os
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera.utils import save_images_to_file
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import project_points, unproject_depth

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import RayCasterCamera


def define_sensor() -> "RayCasterCamera":
    """Defines the ray-cast camera sensor to add to the scene."""
    # Camera base frames
    # In contras to the USD camera, we associate the sensor to the prims at these locations.
    # This means that parent prim of the sensor is the prim at this location.
    sim_utils.create_prim("/World/Origin_00/CameraSensor", "Xform")
    sim_utils.create_prim("/World/Origin_01/CameraSensor", "Xform")

    # Setup camera sensor
    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/Origin_[^/]+/CameraSensor",
        mesh_prim_paths=["/World/ground"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=True,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=480,
            width=640,
        ),
    )
    # Create camera
    camera = instantiate(camera_cfg)

    return camera


def design_scene():
    # Populate scene
    # -- Rough terrain
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # -- Sensors
    camera = define_sensor()

    # return the scene information
    scene_entities = {"camera": camera}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # extract entities for simplified notation
    camera: RayCasterCamera = scene_entities["camera"]

    # Create the output directory
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera")
    os.makedirs(output_dir, exist_ok=True)

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    eyes = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)
    targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(eyes, targets)
    # -- Option-2: Set pose using ROS
    # position = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    # orientation = torch.tensor([[-0.17591989, 0.33985114, 0.82047325, -0.42470819]], device=sim.device)
    # camera.set_world_poses(position, orientation, indices=[0], convention="ros")

    # Simulate physics
    while sim.is_headless_or_exist_active_visualizer():
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())

        # Print camera info
        print(camera)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        # Extract camera data
        if args_cli.save:
            # Save the depth image of the first camera, normalized to [0, 1], as a PNG file
            camera_index = 0
            depth = camera.data.output["distance_to_image_plane"].torch[camera_index : camera_index + 1]
            depth = torch.nan_to_num(depth, posinf=0.0)
            file_path = os.path.join(output_dir, f"depth_{camera.frame[camera_index]:03d}.png")
            save_images_to_file(depth / depth.max().clamp(min=1e-6), file_path)

            # Pointcloud in world frame
            points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"].torch, camera.data.intrinsic_matrices.torch
            )

            # Check methods are valid
            im_height, im_width = camera.image_shape
            # -- project points to (u, v, d)
            reproj_points = project_points(points_3d_cam, camera.data.intrinsic_matrices.torch)
            reproj_depths = reproj_points[..., -1].view(-1, im_width, im_height).transpose_(1, 2)
            sim_depths = camera.data.output["distance_to_image_plane"].torch.squeeze(-1)
            torch.testing.assert_close(reproj_depths, sim_depths)


def main():
    """Main function."""
    # Configure the simulation
    sim_cfg = sim_utils.SimulationCfg()
    # Launch the simulator runtime that the configuration needs
    with launch_simulation(sim_cfg, args_cli):
        # Initialize the simulation context
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])
        # Design scene
        scene_entities = design_scene()
        # Play simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run simulator
        run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
