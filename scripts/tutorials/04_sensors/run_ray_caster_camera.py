# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the ray-cast camera sensor from the Isaac Lab framework.

The camera sensor is based on using Warp kernels which do ray-casting against static meshes.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster_camera.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the ray-cast camera sensor.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to generate.")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import project_points, unproject_depth


def define_sensor() -> RayCasterCamera:
    """Defines the ray-cast camera sensor to add to the scene."""
    # Camera base frames
    # In contras to the USD camera, we associate the sensor to the prims at these locations.
    # This means that parent prim of the sensor is the prim at this location.
    prim_utils.create_prim("/World/Origin_00/CameraSensor", "Xform")
    prim_utils.create_prim("/World/Origin_01/CameraSensor", "Xform")

    # Setup camera sensor
    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
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
    camera = RayCasterCamera(cfg=camera_cfg)

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

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

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
    while simulation_app.is_running():
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
            # Extract camera data
            camera_index = 0
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # Save images
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

            # Pointcloud in world frame
            points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
            )

            # Check methods are valid
            im_height, im_width = camera.image_shape
            # -- project points to (u, v, d)
            reproj_points = project_points(points_3d_cam, camera.data.intrinsic_matrices)
            reproj_depths = reproj_points[..., -1].view(-1, im_width, im_height).transpose_(1, 2)
            sim_depths = camera.data.output["distance_to_image_plane"].squeeze(-1)
            torch.testing.assert_close(reproj_depths, sim_depths)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
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
    # close sim app
    simulation_app.close()
