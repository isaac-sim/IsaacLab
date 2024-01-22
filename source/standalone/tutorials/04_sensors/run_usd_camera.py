# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the camera sensor from the Orbit framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU device for camera output.")
parser.add_argument("--draw", action="store_true", default=False, help="Draw the obtained pointcloud on viewport.")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import numpy as np
import os
import random
import torch
import traceback

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import omni.replicator.core as rep
from omni.isaac.core.prims import RigidPrim
from pxr import Gf, UsdGeom

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.sensors.camera import Camera, CameraCfg
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.orbit.utils.math import project_points, transform_points, unproject_depth


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contras to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    prim_utils.create_prim("/World/Origin_00", "Xform")
    prim_utils.create_prim("/World/Origin_01", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "normals"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene():
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Xform to hold objects
    prim_utils.create_prim("/World/Objects", "Xform")
    # Random objects
    for i in range(8):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # create prim
        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        _ = prim_utils.create_prim(
            f"/World/Objects/Obj_{i:02d}",
            prim_type,
            translation=position,
            scale=(0.25, 0.25, 0.25),
            semantic_label=prim_type,
        )
        # add rigid properties
        rigid_obj = RigidPrim(f"/World/Objects/Obj_{i:02d}", mass=5.0)
        # cast to geom prim
        geom_prim = getattr(UsdGeom, prim_type)(rigid_obj.prim)
        # set random color
        color = Gf.Vec3f(random.random(), random.random(), random.random())
        geom_prim.CreateDisplayColorAttr()
        geom_prim.GetDisplayColorAttr().Set([color])

    # Sensors
    camera = define_sensor()

    # return the scene information
    scene_entities = {"camera": camera}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # extract entities for simplified notation
    camera: Camera = scene_entities["camera"]

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # Acquire draw interface
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    eyes = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)
    targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(eyes, targets)
    # -- Option-2: Set pose using ROS
    # position = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    # orientation = torch.tensor([[-0.17591989, 0.33985114, 0.82047325, -0.42470819]], device=sim.device)
    # camera.set_world_poses(position, orientation, env_ids=[0], convention="ros")

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(3):
        sim.step()

    # Simulate physics
    while simulation_app.is_running():
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())

        # Print camera info
        print(camera)
        print("Received shape of rgb   image: ", camera.data.output["rgb"].shape)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        # Extract camera data
        if args_cli.save:
            # Save images from camera 1
            camera_index = 1
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            if sim.backend == "torch":
                # tensordict allows easy indexing of tensors in the dictionary
                single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")
            else:
                # for numpy, we need to manually index the data
                single_cam_data = dict()
                for key, value in camera.data.output.items():
                    single_cam_data[key] = value[camera_index]
            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = dict()
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output[key] = {"data": data, "info": info}
                else:
                    rep_output[key] = data
            # Save images
            # Note: We need to provide On-time data for Replicator to save the images.
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

        # Draw pointcloud
        if not args_cli.headless and args_cli.draw:
            # Pointcloud in world frame
            points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
            )
            points_3d_world = transform_points(points_3d_cam, camera.data.pos_w, camera.data.quat_w_ros)

            # Check methods are valid
            im_height, im_width = camera.image_shape
            # -- project points to (u, v, d)
            reproj_points = project_points(points_3d_cam, camera.data.intrinsic_matrices)
            reproj_depths = reproj_points[..., -1].view(-1, im_width, im_height).transpose_(1, 2)
            sim_depths = camera.data.output["distance_to_image_plane"].squeeze(-1)
            torch.testing.assert_allclose(reproj_depths, sim_depths)

            # Convert to numpy for visualization
            if not isinstance(points_3d_world, np.ndarray):
                points_3d_world = points_3d_world.cpu().numpy()
            # Clear any existing points
            draw_interface.clear_points()
            # Obtain drawing settings
            num_batch = points_3d_world.shape[0]
            num_points = points_3d_world.shape[1]
            points_size = [1.25] * num_points
            # Fix random seed
            random.seed(0)
            # Visualize the points
            for index in range(num_batch):
                # generate random color
                color = [random.random() for _ in range(3)]
                color += [1.0]
                # plain color for points
                points_color = [color] * num_points
                draw_interface.draw_points(points_3d_world[index].tolist(), points_color, points_size)


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cpu" if args_cli.cpu else "cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
