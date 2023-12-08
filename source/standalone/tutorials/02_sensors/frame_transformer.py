# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the FrameTransformer sensor by visualizing the frames that it creates.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/02_sensors/frame_transformer.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script checks the FrameTransformer sensor by visualizing the frames that it creates."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.config.anymal import ANYMAL_C_CFG
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG
from omni.isaac.orbit.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from omni.isaac.orbit.sim import SimulationContext


def design_scene() -> tuple(FrameTransformer, Articulation):
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # -- robot
    robot_cfg = ANYMAL_C_CFG
    robot_cfg.spawn.func("/World/Anymal_c/Robot", robot_cfg.spawn, translation=(1.5, -1.5, 0.65))
    robot = Articulation(robot_cfg.replace(prim_path="/World/Anymal_c/Robot"))

    # add frame transformer
    frame_transformer = add_sensor()

    return frame_transformer, robot


def add_sensor() -> FrameTransformer:
    """Adds FrameTransformer sensor to the scene."""
    # define offset
    rot_offset = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor(-math.pi / 2))
    pos_offset = math_utils.quat_apply(rot_offset, torch.tensor([0.08795, 0.01305, -0.33797]))

    # Example using .* to get full body + LF_FOOT
    frame_transformer_cfg = FrameTransformerCfg(
        prim_path="/World/Anymal_c/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/Anymal_c/Robot/.*"),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/Anymal_c/Robot/LF_SHANK",
                name="LF_FOOT",
                offset=OffsetCfg(pos=tuple(pos_offset.tolist()), rot=tuple(rot_offset[0].tolist())),
            ),
        ],
        debug_vis=False,
    )
    frame_transformer = FrameTransformer(frame_transformer_cfg)

    return frame_transformer


def run_simulator(sim: sim_utils.SimulationContext, robot: Articulation, frame_transformer: FrameTransformer):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # We only want one visualization at a time. This visualizer will be used
    # to step through each frame so the user can verify that the correct frame
    # is being visualized as the frame names are printing to console
    if not args_cli.headless:
        cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
        cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        transform_visualizer = VisualizationMarkers(cfg)

    frame_index = 0
    # Simulate physics
    while simulation_app.is_running():
        # perform this loop at policy control freq (50 Hz)
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # read data from sim
        robot.update(sim_dt)
        frame_transformer.update(dt=sim_dt)

        # Change the frame that we are visualizing to ensure that frame names
        # are correctly associated with the frames
        if not args_cli.headless:
            if count % 50 == 0:
                frame_names = frame_transformer.data.target_frame_names

                frame_name = frame_names[frame_index]
                print(f"Displaying {frame_index}: {frame_name}")
                frame_index += 1
                frame_index = frame_index % len(frame_names)

            # visualize frame
            pos = frame_transformer.data.target_pos_w[:, frame_index]
            rot = frame_transformer.data.target_rot_w[:, frame_index]
            transform_visualizer.visualize(pos, rot)


def main():
    """Main function."""
    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # Design the scene
    frame_transformer, robot = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, robot, frame_transformer)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
