# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstrate camera and ray-caster camera sensors attached to a robot.

.. code-block:: bash

    # Usage with Newton physics and camera rendering
    uvx isaaclab example camera

    # Usage in headless mode
    uvx isaaclab example camera --viz none --max_steps 20

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import AppLauncher, add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Example on using the different camera sensor implementations.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
parser.add_argument("--log_interval", type=int, default=100, help="Steps between compact sensor summaries.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument("--save", action="store_true", help="Save sampled RGB and depth images.")
parser.add_argument("--save_interval", type=int, default=100, help="Steps between saved image samples.")
parser.add_argument("--output_dir", type=Path, default=Path("output/camera"), help="Directory for saved images.")
parser.add_argument(
    "--physics",
    default="newton_mjwarp",
    choices=["isaacsim_physx", "newton_mjwarp"],
    help="Physics backend.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if args_cli.log_interval < 1:
    parser.error("--log_interval must be at least 1.")
if args_cli.save_interval < 1:
    parser.error("--save_interval must be at least 1.")
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")
if args_cli.physics == "isaacsim_physx":
    args_cli.enable_cameras = True
    app_launcher = AppLauncher(args_cli)
else:
    app_launcher = None

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_newton.renderers import NewtonWarpRendererCfg

if args_cli.physics == "isaacsim_physx":
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    camera_renderer_cfg = IsaacRtxRendererCfg()
else:
    camera_renderer_cfg = NewtonWarpRendererCfg()


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
        renderer_cfg=camera_renderer_cfg,
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
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
    filename: str | Path | None = None,
) -> None:
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrow: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])
    for ax in axes[n_images:]:
        fig.delaxes(ax)
    if title:
        plt.suptitle(title)

    plt.tight_layout()
    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
    plt.close()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    if args_cli.save:
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)

    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or count < args_cli.max_steps):
        # Reset
        if count % 500 == 0:
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_pose = scene["robot"].data.default_root_pose.torch.clone()
            root_pose[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = scene["robot"].data.default_root_vel.torch.clone()
            scene["robot"].write_root_velocity_to_sim_index(root_velocity=root_vel)
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.torch.clone(),
                scene["robot"].data.default_joint_vel.torch.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
            scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos.torch
        # -- apply action to the robot
        scene["robot"].set_joint_position_target_index(target=targets)
        # -- write data to sim
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        if count % args_cli.log_interval == 0:
            camera_output = scene["camera"].data.output
            raycast_output = scene["raycast_camera"].data.output
            print(
                f"[INFO] step={count} rgb={tuple(camera_output['rgb'].shape)} "
                f"depth={tuple(camera_output['distance_to_image_plane'].shape)} "
                f"raycast_depth={tuple(raycast_output['distance_to_image_plane'].shape)}"
            )

        if args_cli.save and count % args_cli.save_interval == 0:
            rgb_images = [scene["camera"].data.output["rgb"][0, ..., :3]]
            save_images_grid(
                rgb_images,
                subtitles=["Camera"],
                title="RGB image",
                filename=str(args_cli.output_dir / "rgb" / f"{count:06d}.jpg"),
            )
            depth_images = [
                scene["camera"].data.output["distance_to_image_plane"][0],
                scene["raycast_camera"].data.output["distance_to_image_plane"][0],
            ]
            save_images_grid(
                depth_images,
                cmap="turbo",
                subtitles=["Camera", "Ray-caster camera"],
                title="Depth comparison",
                filename=str(args_cli.output_dir / "depth" / f"{count:06d}.jpg"),
            )


def main() -> None:
    """Run the camera example."""
    # Initialize the simulation context
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.005, device=args_cli.device, physics=physics_cfg, use_fabric=not args_cli.disable_fabric
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        if app_launcher is not None:
            app_launcher.app.close()
