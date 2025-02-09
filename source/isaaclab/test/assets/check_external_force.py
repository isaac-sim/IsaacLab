# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script checks if the external force is applied correctly on the robot.

.. code-block:: bash

    # Usage to apply force on base
    ./isaaclab.sh -p source/isaaclab/test/assets/check_external_force.py --body base --force 1000
    # Usage to apply force on legs
    ./isaaclab.sh -p source/isaaclab/test/assets/check_external_force.py --body .*_SHANK --force 100
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to external force on a legged robot.")
parser.add_argument("--body", default="base", type=str, help="Name of the body to apply force on.")
parser.add_argument("--force", default=1000.0, type=float, help="Force to apply on the body.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light/greyLight", cfg)

    # Robots
    robot_cfg = ANYMAL_C_CFG
    robot_cfg.spawn.func("/World/Anymal_c/Robot_1", robot_cfg.spawn, translation=(0.0, -0.5, 0.65))
    robot_cfg.spawn.func("/World/Anymal_c/Robot_2", robot_cfg.spawn, translation=(0.0, 0.5, 0.65))
    # create handles for the robots
    robot = Articulation(robot_cfg.replace(prim_path="/World/Anymal_c/Robot.*"))

    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, body_names = robot.find_bodies(args_cli.body)
    # Sample a large force
    external_wrench_b = torch.zeros(robot.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = args_cli.force

    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: Applying force on the robot: ", args_cli.body, " -> ", body_names)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 100 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = robot.data.default_root_state.clone()
            root_state[0, :2] = torch.tensor([0.0, -0.5], device=sim.device)
            root_state[1, :2] = torch.tensor([0.0, 0.5], device=sim.device)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # apply force
            robot.set_external_force_and_torque(
                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
            )
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
