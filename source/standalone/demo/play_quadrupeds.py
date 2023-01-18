# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the physics engine to simulate a legged robot.

We currently support the following robots:

* ANYmal-B (from ANYbotics)
* ANYmal-C (from ANYbotics)
* A1 (from Unitree Robotics)

From the default configuration file for these robots, zero actions imply a standing pose.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch
from typing import List

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.markers import PointMarker, StaticMarker
from omni.isaac.orbit.robots.config.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG
from omni.isaac.orbit.robots.config.unitree import UNITREE_A1_CFG
from omni.isaac.orbit.robots.legged_robot import LeggedRobot

"""
Helpers
"""


def design_scene():
    """Add prims to the scene."""
    # Ground-plane
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8,
        improve_patch_friction=True,
    )
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


"""
Main
"""


def main():
    """Imports all legged robots supported in Orbit and applies zero actions."""

    # Load kit helper
    sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.005, backend="torch")
    # Set main camera
    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # -- anymal-b
    robot_b = LeggedRobot(cfg=ANYMAL_B_CFG)
    robot_b.spawn("/World/Anymal_b/Robot_1", translation=(0.0, -1.5, 0.65))
    robot_b.spawn("/World/Anymal_b/Robot_2", translation=(0.0, -0.5, 0.65))
    # -- anymal-c
    robot_c = LeggedRobot(cfg=ANYMAL_C_CFG)
    robot_c.spawn("/World/Anymal_c/Robot_1", translation=(1.5, -1.5, 0.65))
    robot_c.spawn("/World/Anymal_c/Robot_2", translation=(1.5, -0.5, 0.65))
    # -- unitree a1
    robot_a = LeggedRobot(cfg=UNITREE_A1_CFG)
    robot_a.spawn("/World/Unitree_A1/Robot_1", translation=(1.5, 0.5, 0.42))
    robot_a.spawn("/World/Unitree_A1/Robot_2", translation=(1.5, 1.5, 0.42))
    # design props
    design_scene()

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot_b.initialize("/World/Anymal_b/Robot.*")
    robot_c.initialize("/World/Anymal_c/Robot.*")
    robot_a.initialize("/World/Unitree_A1/Robot.*")
    # Reset states
    robot_b.reset_buffers()
    robot_c.reset_buffers()
    robot_a.reset_buffers()

    # Debug visualization markers.
    # -- feet markers
    feet_markers: List[StaticMarker] = list()
    feet_contact_markers: List[PointMarker] = list()
    # iterate over robots
    for robot_name in ["Anymal_b", "Anymal_c", "Unitree_A1"]:
        # foot
        marker = StaticMarker(f"/World/Visuals/{robot_name}/feet", 4 * robot_c.count, scale=(0.1, 0.1, 0.1))
        feet_markers.append(marker)
        # contact
        marker = PointMarker(f"/World/Visuals/{robot_name}/feet_contact", 4 * robot_c.count, radius=0.035)
        feet_contact_markers.append(marker)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy action
    actions = torch.zeros(robot_a.count, robot_a.num_actions, device=robot_a.device)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
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
        if count % 1000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            for robot in [robot_a, robot_b, robot_c]:
                dof_pos, dof_vel = robot.get_default_dof_state()
                robot.set_dof_state(dof_pos, dof_vel)
                robot.reset_buffers()
            # reset command
            actions = torch.zeros(robot_a.count, robot_a.num_actions, device=robot_a.device)
            print(">>>>>>>> Reset!")
        # apply actions
        robot_b.apply_action(actions)
        robot_c.apply_action(actions)
        robot_a.apply_action(actions)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot_b.update_buffers(sim_dt)
            robot_c.update_buffers(sim_dt)
            robot_a.update_buffers(sim_dt)
            # update marker positions
            for foot_marker, contact_marker, robot in zip(
                feet_markers, feet_contact_markers, [robot_b, robot_c, robot_a]
            ):
                # feet
                foot_marker.set_world_poses(
                    robot.data.feet_state_w[..., 0:3].view(-1, 3), robot.data.feet_state_w[..., 3:7].view(-1, 4)
                )
                # contact sensors
                contact_marker.set_world_poses(
                    robot.data.feet_state_w[..., 0:3].view(-1, 3), robot.data.feet_state_w[..., 3:7].view(-1, 4)
                )
                contact_marker.set_status(torch.where(robot.data.feet_air_time.view(-1) > 0.0, 1, 2))


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
