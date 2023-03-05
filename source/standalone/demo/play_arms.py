# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the physics engine to simulate a single-arm manipulator.

We currently support the following robots:

* Franka Emika Panda
* Universal Robot UR10

From the default configuration file for these robots, zero actions imply a default pose.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.config.universal_robots import UR10_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

"""
Main
"""


def main():
    """Spawns a single arm manipulator and applies random joint commands."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
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
    # Table
    table_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    prim_utils.create_prim("/World/Table_1", usd_path=table_usd_path, translation=(0.55, -1.0, 0.0))
    prim_utils.create_prim("/World/Table_2", usd_path=table_usd_path, translation=(0.55, 1.0, 0.0))
    # Robots
    # -- Resolve robot config from command-line arguments
    if args_cli.robot == "franka_panda":
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    elif args_cli.robot == "ur10":
        robot_cfg = UR10_CFG
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # -- Spawn robot
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/Robot_1", translation=(0.0, -1.0, 0.0))
    robot.spawn("/World/Robot_2", translation=(0.0, 1.0, 0.0))

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/Robot.*")
    # Reset states
    robot.reset_buffers()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
    has_gripper = robot.cfg.meta_info.tool_num_dof > 0

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        if ep_step_count % 1000 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset command
            actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
            # reset gripper
            if has_gripper:
                actions[:, -1] = -1
            print("[INFO]: Resetting robots state...")
        # change the gripper action
        if ep_step_count % 200 == 0 and has_gripper:
            # flip command for the gripper
            actions[:, -1] = -actions[:, -1]
            print(f"[INFO]: [Step {ep_step_count:03d}]: Flipping gripper command...")
        # apply action to the robot
        robot.apply_action(actions)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
