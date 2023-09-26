# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a mobile manipulator.

We currently support the following robots:

* Franka Emika Panda on a Clearpath Ridgeback Omni-drive Base

From the default configuration file for these robots, zero actions imply a default pose.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates how to simulate a mobile manipulator with dummy joints."
)
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import torch

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.config.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from omni.isaac.orbit.sim import SimulationContext

"""
Main
"""


def main():
    """Main function."""

    # Load kit helper
    # note: there is a bug in Isaac Sim 2022.2.1 that prevents the use of GPU pipeline
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False))
    # Set main camera
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights-1
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75), radius=2.5)
    cfg.func("/World/Light/greyLight", cfg, translation=(4.5, 3.5, 10.0))
    # Lights-2
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0), radius=2.5)
    cfg.func("/World/Light/whiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    # Robots
    robot_cfg = RIDGEBACK_FRANKA_PANDA_CFG
    # -- Spawn robot
    robot_cfg.spawn.func("/World/Robot_1", robot_cfg.spawn, translation=(0.0, -1.0, 0.0))
    robot_cfg.spawn.func("/World/Robot_2", robot_cfg.spawn, translation=(0.0, 1.0, 0.0))
    # -- Create interface
    robot = Articulation(cfg=robot_cfg.replace(prim_path="/World/Robot.*"))

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy action
    actions = torch.zeros(robot.root_view.count, robot.num_joints, device=robot.device) + robot.data.default_joint_pos

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if ep_step_count % 1000 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset command
            actions = torch.rand_like(robot.data.default_joint_pos) + robot.data.default_joint_pos
            # -- base
            actions[:, 0::3] = 0.0
            # -- gripper
            actions[:, -2:] = 0.04
            print("[INFO]: Resetting robots state...")
        # change the gripper action
        if ep_step_count % 200 == 0:
            # flip command for the gripper
            actions[:, -2:] = 0.0 if actions[0, -2] > 0.0 else 0.04
        # change the base action
        # -- forward and backward (x-axis)
        if ep_step_count == 200:
            actions[:, :3] = 0.0
            actions[:, 0] = 1.0
        if ep_step_count == 300:
            actions[:, :3] = 0.0
            actions[:, 0] = -1.0
        # -- right and left (y-axis)
        if ep_step_count == 400:
            actions[:, :3] = 0.0
            actions[:, 1] = 1.0
        if ep_step_count == 500:
            actions[:, :3] = 0.0
            actions[:, 1] = -1.0
        # -- turn right and left (z-axis)
        if ep_step_count == 600:
            actions[:, :3] = 0.0
            actions[:, 2] = 1.0
        if ep_step_count == 700:
            actions[:, :3] = 0.0
            actions[:, 2] = -1.0
        if ep_step_count == 900:
            actions[:, :3] = 0.0
            actions[:, 2] = 1.0
        # change the arm action
        if ep_step_count % 100:
            actions[:, 3:10] = (
                torch.rand(robot.root_view.count, 7, device=robot.device) + robot.data.default_joint_pos[:, 3:10]
            )
        # apply action
        robot.set_joint_velocity_target(actions[:, :3], joint_ids=[0, 1, 2])
        robot.set_joint_position_target(actions[:, 3:], joint_ids=[3, 4, 5, 6, 7, 8, 9, 10, 11])
        robot.write_data_to_sim()
        # perform step
        sim.step(render=app_launcher.RENDER)
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
