# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
This script shows how to use the camera sensor from the Orbit framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU device for camera rendering output.")
parser.add_argument("--draw", action="store_true", default=False, help="Draw the obtained pointcloud on viewport.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import numpy as np
import os
import random

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import omni.replicator.core as rep
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
from omni.isaac.orbit.sensors.camera.utils import create_pointcloud_from_rgbd
from omni.isaac.orbit.utils import convert_dict_to_backend

"""
Helpers
"""


def design_scene():
    """Add prims to the scene."""
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
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


"""
Main
"""


def main():
    """Runs a camera sensor from orbit."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.005, rendering_dt=0.005, backend="torch")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Acquire draw interface
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    # Populate scene
    design_scene()
    # Setup camera sensor
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "normals", "motion_vectors"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg, device="cuda" if args_cli.gpu else "cpu")

    # Spawn camera
    camera.spawn("/World/CameraSensor")

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # Play simulator
    sim.play()
    # Initialize camera
    camera.initialize()

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    # camera.set_world_pose_from_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # -- Option-2: Set pose using ROS
    position = [2.5, 2.5, 2.5]
    orientation = [-0.17591989, 0.33985114, 0.82047325, -0.42470819]
    camera.set_world_pose_ros(position, orientation)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(14):
        sim.render()

    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=0.0)

        # Print camera info
        print(camera)
        print("Received shape of rgb   image: ", camera.data.output["rgb"].shape)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        # Save images
        # note: BasicWriter only supports saving data in numpy format
        rep_writer.write(convert_dict_to_backend(camera.data.output, backend="numpy"))

        # Pointcloud in world frame
        pointcloud_w, pointcloud_rgb = create_pointcloud_from_rgbd(
            camera.data.intrinsic_matrix,
            depth=camera.data.output["distance_to_image_plane"],
            rgb=camera.data.output["rgb"],
            position=camera.data.position,
            orientation=camera.data.orientation,
            normalize_rgb=True,
            num_channels=4,
        )

        # Draw pointcloud
        if not args_cli.headless and args_cli.draw:
            # Convert to numpy for visualization
            if not isinstance(pointcloud_w, np.ndarray):
                pointcloud_w = pointcloud_w.cpu().numpy()
            if not isinstance(pointcloud_rgb, np.ndarray):
                pointcloud_rgb = pointcloud_rgb.cpu().numpy()
            # Visualize the points
            num_points = pointcloud_w.shape[0]
            points_size = [1.25] * num_points
            points_color = pointcloud_rgb
            draw_interface.clear_points()
            draw_interface.draw_points(pointcloud_w.tolist(), points_color, points_size)


if __name__ == "__main__":
    # Runs the main function
    main()
    # Close the simulator
    simulation_app.close()
