# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script checks the FrameTransformer sensor by visualizing the frames that it creates.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
import math
from scipy.spatial.transform import Rotation

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script checks the FrameTransformer sensor by visualizing the frames that it creates."
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["feet", "all"],
    default="all",
    help="Dictates the the type of frames to use for FrameTransformer.",
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help=(
        "Whether to enable FrameTransformer's debug_vis (True) or visualize each frame one at a"
        " time and print to console (False)."
    ),
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.sensors import FrameTransformerCfg, OffsetCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.timer import Timer

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.anymal import ANYMAL_C_CFG  # isort:skip


def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (w, x, y, z)."""
    quat = Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat[[3, 0, 1, 2]].tolist())


def euler_rpy_apply(rpy, xyz, degrees=False):
    """Applies rotation from Euler XYZ on position vector."""
    rot = Rotation.from_euler("xyz", rpy, degrees=degrees)
    return tuple(rot.apply(xyz).tolist())


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # terrain - flat terrain plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    # articulation - robot
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    if args_cli.mode == "feet":
        # Example where only feet position frames are created
        frame_transformer = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    name="LF_FOOT",
                    prim_path="{ENV_REGEX_NS}/Robot/LF_SHANK",
                    offset=OffsetCfg(
                        pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(0.08795, 0.01305, -0.33797)),
                        rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    name="RF_FOOT",
                    prim_path="{ENV_REGEX_NS}/Robot/RF_SHANK",
                    offset=OffsetCfg(
                        pos=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(0.08795, -0.01305, -0.33797)),
                        rot=quat_from_euler_rpy(0, 0, math.pi / 2),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    name="LH_FOOT",
                    prim_path="{ENV_REGEX_NS}/Robot/LH_SHANK",
                    offset=OffsetCfg(
                        pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(-0.08795, 0.01305, -0.33797)),
                        rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    name="RH_FOOT",
                    prim_path="{ENV_REGEX_NS}/Robot/RH_SHANK",
                    offset=OffsetCfg(
                        pos=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(-0.08795, -0.01305, -0.33797)),
                        rot=quat_from_euler_rpy(0, 0, math.pi / 2),
                    ),
                ),
            ],
            debug_vis=args_cli.visualize,
        )

    elif args_cli.mode == "all":
        # Example using .* to get full body + LF_FOOT to ensure these work together
        frame_transformer = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*"),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/LF_SHANK",
                    name="LF_FOOT",
                    offset=OffsetCfg(
                        pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(0.08795, 0.01305, -0.33797)),
                        rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                    ),
                ),
            ],
            debug_vis=args_cli.visualize,
        )

    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    # Set main camera
    sim.set_camera_view(eye=[5, 5, 5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    with Timer("Setup scene"):
        scene = InteractiveScene(MySceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0, lazy_sensor_update=False))

    # Play the simulator
    with Timer("Time taken to play the simulator"):
        sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # default joint targets
    robot_actions = scene.articulations["robot"].data.default_joint_pos.clone()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # We only want one visualization at a time. This visualizer will be used
    # to step through each frame so the user can verify that the correct frame
    # is being visualized as the frame names are printing to console
    if not args_cli.visualize:
        cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
        cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        transform_visualizer = VisualizationMarkers(cfg)
    else:
        transform_visualizer = None

    frame_index = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step()
            continue
        # # reset
        if count % 50 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel
            # -- set root state
            # -- robot
            scene.articulations["robot"].write_root_state_to_sim(root_state)
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            root_state[:, 1] += 1.0
            # reset buffers
            scene.reset()
            print(">>>>>>>> Reset!")
        # perform this loop at policy control freq (50 Hz)
        for _ in range(4):
            # set joint targets
            scene.articulations["robot"].set_joint_position_target(robot_actions)
            # write data to sim
            scene.write_data_to_sim()
            # perform step
            sim.step()
            # read data from sim
            scene.update(sim_dt)

        # Change the frame that we are visualizing to ensure that frame names
        # are correctly associated with the frames
        if not args_cli.visualize:
            if count % 50 == 0:
                frame_names = scene["frame_transformer"].data.target_frame_names

                frame_name = frame_names[frame_index]
                print(f"Displaying {frame_index}: {frame_name}")
                frame_index += 1
                frame_index = frame_index % len(frame_names)

            # visualize frame
            pos = scene["frame_transformer"].data.target_pos_w[:, frame_index]
            rot = scene["frame_transformer"].data.target_rot_w[:, frame_index]
            transform_visualizer.visualize(pos, rot)

        # update sim-time
        sim_time += sim_dt * 4
        count += 1


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
