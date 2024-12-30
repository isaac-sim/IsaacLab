# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the different camera sensors that can be attached to a robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/cameras.py --enable_cameras

    # Usage in headless mode
    ./isaaclab.sh -p scripts/demos/cameras.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates the different camera sensor implementations.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(color_scheme="random"),
        visual_material=None,
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=None,  # the camera is already spawned in the scene
        offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    raycast_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=["distance_to_image_plane", "normals"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=480,
            width=640,
        ),
    )


def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    # show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axes = axes.flatten()

    # plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])
    # remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)
    # set title
    if title:
        plt.suptitle(title)

    # adjust layout to fit the title
    plt.tight_layout()
    # save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # close the figure
    plt.close()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print(scene["tiled_camera"])
        print("Received shape of rgb   image: ", scene["tiled_camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["tiled_camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print(scene["raycast_camera"])
        print("Received shape of depth: ", scene["raycast_camera"].data.output["distance_to_image_plane"].shape)
        print("Received shape of normals: ", scene["raycast_camera"].data.output["normals"].shape)

        # save every 10th image (for visualization purposes only)
        # note: saving images will slow down the simulation
        if count % 10 == 0:
            # compare generated RGB images across different cameras
            rgb_images = [scene["camera"].data.output["rgb"][0, ..., :3], scene["tiled_camera"].data.output["rgb"][0]]
            save_images_grid(
                rgb_images,
                subtitles=["Camera", "TiledCamera"],
                title="RGB Image: Cam0",
                filename=os.path.join(output_dir, "rgb", f"{count:04d}.jpg"),
            )

            # compare generated Depth images across different cameras
            depth_images = [
                scene["camera"].data.output["distance_to_image_plane"][0],
                scene["tiled_camera"].data.output["distance_to_image_plane"][0, ..., 0],
                scene["raycast_camera"].data.output["distance_to_image_plane"][0],
            ]
            save_images_grid(
                depth_images,
                cmap="turbo",
                subtitles=["Camera", "TiledCamera", "RaycasterCamera"],
                title="Depth Image: Cam0",
                filename=os.path.join(output_dir, "distance_to_camera", f"{count:04d}.jpg"),
            )

            # save all tiled RGB images
            tiled_images = scene["tiled_camera"].data.output["rgb"]
            save_images_grid(
                tiled_images,
                subtitles=[f"Cam{i}" for i in range(tiled_images.shape[0])],
                title="Tiled RGB Image",
                filename=os.path.join(output_dir, "tiled_rgb", f"{count:04d}.jpg"),
            )

            # save all camera RGB images
            cam_images = scene["camera"].data.output["rgb"][..., :3]
            save_images_grid(
                cam_images,
                subtitles=[f"Cam{i}" for i in range(cam_images.shape[0])],
                title="Camera RGB Image",
                filename=os.path.join(output_dir, "cam_rgb", f"{count:04d}.jpg"),
            )


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
