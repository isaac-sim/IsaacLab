# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the ray-cast camera sensor from the Orbit framework.

The camera sensor is based on using Warp kernels which do ray-casting against static meshes.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the ray-cast camera sensor.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to generate.")
parser.add_argument("--save", type=int, default=16, help="Number of environments to generate.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import os
import torch
import traceback

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.terrains as terrain_gen
from omni.isaac.orbit.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.orbit.terrains.terrain_importer import TerrainImporter
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.orbit.utils.math import project_points, unproject_depth


def main():
    """Main function."""

    # Load kit helper
    sim = sim_utils.SimulationContext()
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])

    # Populate scene
    # Handler for terrains importing
    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        num_envs=args_cli.num_envs,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(curriculum=True, num_rows=4, num_cols=4),
        debug_vis=True,
    )
    terrain = TerrainImporter(terrain_importer_cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # Camera base frames
    prim_utils.create_prim("/World/envs", "Scope")
    for index in range(args_cli.num_envs):
        prim_utils.create_prim(f"/World/envs/env_{index}", "Xform", translation=terrain.env_origins[index])
        prim_utils.create_prim(f"/World/envs/env_{index}/Camera", "Xform")

    # Setup camera sensor
    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
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
    # create xform because placement of camera directly under world is not supported
    prim_utils.create_prim("/World/Camera", "Xform")
    # Create camera
    camera = RayCasterCamera(cfg=camera_cfg)

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # Play simulator
    sim.reset()

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    eyes = torch.tensor([[-1.0, 0, 2] * args_cli.num_envs], device=sim.device).reshape(-1, 3)
    targets = torch.tensor([[0.0, 0.0, 0.0] * args_cli.num_envs], device=sim.device).reshape(-1, 3)
    camera.set_world_poses_from_view(eyes + terrain.env_origins, targets + terrain.env_origins)
    # -- Option-2: Set pose using ROS
    # position = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    # orientation = torch.tensor([[-0.17591989, 0.33985114, 0.82047325, -0.42470819]], device=sim.device)
    # camera.set_world_pose_ros(position, orientation, indices=[0])

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
            torch.testing.assert_allclose(reproj_depths, sim_depths)


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
