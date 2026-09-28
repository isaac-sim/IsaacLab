# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the camera sensor from the Isaac Lab framework.

Instead of using the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.

.. code-block:: bash

    # Usage with GUI
    uv run python scripts/tutorials/04_sensors/run_usd_camera.py --viz kit

    # Usage with no visualizer
    uv run python scripts/tutorials/04_sensors/run_usd_camera.py

"""

"""Parse the command-line arguments first."""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# Camera sensors require the rendering extensions in headless and viewport-free launches.
args_cli.enable_cameras = True

"""Rest everything follows."""

import os
import random

import numpy as np
import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, save_images_to_file
from isaaclab.utils import instantiate, replace

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera


def define_sensor() -> "Camera":
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    sim_utils.create_prim("/World/Origin_00", "Xform")
    sim_utils.create_prim("/World/Origin_01", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_[^/]+/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation",
            "instance_id_segmentation_fast",
        ],
        renderer_cfg=IsaacRtxRendererCfg(
            colorize_semantic_segmentation=True,
            colorize_instance_id_segmentation=True,
            colorize_instance_segmentation=True,
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = instantiate(camera_cfg)

    return camera


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create a dictionary for the scene entities
    scene_entities = {}

    # Xform to hold objects
    sim_utils.create_prim("/World/Objects", "Xform")
    # Random objects
    for i in range(8):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # sample random color
        color = (random.random(), random.random(), random.random())
        # choose random prim type
        prim_type = random.choice(["Cube", "Cone", "Cylinder"])
        common_properties = {
            "rigid_props": sim_utils.UsdPhysicsRigidBodyCfg(),
            "mass_props": sim_utils.MassCfg(mass=5.0),
            "collision_props": sim_utils.UsdPhysicsCollisionCfg(),
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
    camera = define_sensor()

    # return the scene information
    scene_entities["camera"] = camera
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # extract entities for simplified notation
    camera: Camera = scene_entities["camera"]

    # Create the output directory
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    os.makedirs(output_dir, exist_ok=True)

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)
    # These orientations are in ROS-convention, and will position the cameras to view the origin
    camera_orientations = torch.tensor(  # noqa: F841
        [[-0.1759, 0.3399, 0.8205, -0.4247], [-0.4247, 0.8205, -0.3399, 0.1759]], device=sim.device
    )

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    camera.set_world_poses_from_view(camera_positions, camera_targets)
    # -- Option-2: Set pose using ROS
    # camera.set_world_poses(camera_positions, camera_orientations, convention="ros")

    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id

    # Create the markers for the --draw option outside of the simulation loop
    if sim.get_setting("/isaaclab/has_gui") and args_cli.draw:
        cfg = replace(RAY_CASTER_MARKER_CFG, prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    # Simulate physics
    while sim.is_headless_or_exist_active_visualizer():
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())

        # Print camera info
        print(camera)
        if "rgb" in camera.data.output.keys():
            print("Received shape of rgb image        : ", camera.data.output["rgb"].shape)
        if "distance_to_image_plane" in camera.data.output.keys():
            print("Received shape of depth image      : ", camera.data.output["distance_to_image_plane"].shape)
        if "normals" in camera.data.output.keys():
            print("Received shape of normals          : ", camera.data.output["normals"].shape)
        if "semantic_segmentation" in camera.data.output.keys():
            print("Received shape of semantic segm.   : ", camera.data.output["semantic_segmentation"].shape)
        if "instance_segmentation" in camera.data.output.keys():
            print("Received shape of instance segm.   : ", camera.data.output["instance_segmentation"].shape)
        if "instance_id_segmentation_fast" in camera.data.output.keys():
            print("Received shape of instance id segm.: ", camera.data.output["instance_id_segmentation_fast"].shape)
        print("-------------------------------")

        # Extract camera data
        if args_cli.save:
            # Save the 8-bit color images (rgb and colorized segmentations) from camera at camera_index
            for key, data in camera.data.output.items():
                if data.torch.dtype == torch.uint8:
                    file_path = os.path.join(output_dir, f"{key}_{camera.frame[camera_index]}.png")
                    save_images_to_file(data.torch[camera_index : camera_index + 1].float() / 255.0, file_path)

        # Draw pointcloud if there is a GUI and --draw has been passed
        if (
            sim.get_setting("/isaaclab/has_gui")
            and args_cli.draw
            and "distance_to_image_plane" in camera.data.output.keys()
        ):
            # Derive pointcloud from camera at camera_index
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],
                depth=camera.data.output["distance_to_image_plane"][camera_index],
                position=camera.data.pos_w[camera_index],
                orientation=camera.data.quat_w_ros[camera_index],
                device=sim.device,
            )

            # In the first few steps, things are still being instanced and Camera.data
            # can be empty. If we attempt to visualize an empty pointcloud it will crash
            # the sim, so we check that the pointcloud is not empty.
            if pointcloud.size()[0] > 0:
                pc_markers.visualize(translations=pointcloud)


def main():
    """Main function."""
    # Configure the simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # Launch the simulator runtime that the configuration needs
    with launch_simulation(sim_cfg, args_cli):
        # Initialize the simulation context
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        # Design scene
        scene_entities = design_scene()
        # Play simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run simulator
        run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
