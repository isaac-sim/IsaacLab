# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the FrameTransformer sensor by visualizing the frames that it creates.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_frame_transformer.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

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

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


def define_sensor() -> FrameTransformer:
    """Defines the FrameTransformer sensor to add to the scene."""
    # define offset
    rot_offset = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor(-math.pi / 2))
    pos_offset = math_utils.quat_apply(rot_offset, torch.tensor([0.08795, 0.01305, -0.33797]))

    # Example using .* to get full body + LF_FOOT
    frame_transformer_cfg = FrameTransformerCfg(
        prim_path="/World/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/Robot/.*"),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/Robot/LF_SHANK",
                name="LF_FOOT_USER",
                offset=OffsetCfg(pos=tuple(pos_offset.tolist()), rot=tuple(rot_offset[0].tolist())),
            ),
        ],
        debug_vis=False,
    )
    frame_transformer = FrameTransformer(frame_transformer_cfg)

    return frame_transformer


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # -- Robot
    robot = Articulation(ANYMAL_C_CFG.replace(prim_path="/World/Robot"))
    # -- Sensors
    frame_transformer = define_sensor()

    # return the scene information
    scene_entities = {"robot": robot, "frame_transformer": frame_transformer}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # extract entities for simplified notation
    robot: Articulation = scene_entities["robot"]
    frame_transformer: FrameTransformer = scene_entities["frame_transformer"]

    # We only want one visualization at a time. This visualizer will be used
    # to step through each frame so the user can verify that the correct frame
    # is being visualized as the frame names are printing to console
    if not args_cli.headless:
        cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
        cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        transform_visualizer = VisualizationMarkers(cfg)
        # debug drawing for lines connecting the frame
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    else:
        transform_visualizer = None
        draw_interface = None

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
                # get frame names
                frame_names = frame_transformer.data.target_frame_names
                # increment frame index
                frame_index += 1
                frame_index = frame_index % len(frame_names)
                print(f"Displaying Frame ID {frame_index}: {frame_names[frame_index]}")

            # visualize frame
            source_pos = frame_transformer.data.source_pos_w
            source_quat = frame_transformer.data.source_quat_w
            target_pos = frame_transformer.data.target_pos_w[:, frame_index]
            target_quat = frame_transformer.data.target_quat_w[:, frame_index]
            # draw the frames
            transform_visualizer.visualize(
                torch.cat([source_pos, target_pos], dim=0), torch.cat([source_quat, target_quat], dim=0)
            )
            # draw the line connecting the frames
            draw_interface.clear_lines()
            # plain color for lines
            lines_colors = [[1.0, 1.0, 0.0, 1.0]] * source_pos.shape[0]
            line_thicknesses = [5.0] * source_pos.shape[0]
            draw_interface.draw_lines(source_pos.tolist(), target_pos.tolist(), lines_colors, line_thicknesses)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # Design the scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
