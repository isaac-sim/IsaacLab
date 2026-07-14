# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to view ARL Robot 1.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    ./isaaclab.sh -p scripts/demos/arl_robot_1.py

    # Usage with Newton visualizer and default PhysX physics.
    ./isaaclab.sh -p scripts/demos/arl_robot_1.py --visualizer newton

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="View ARL Robot 1 with Lee Position Controller.",
    conflict_handler="resolve",
)
parser.add_argument("--physics", default="physx", choices=["physx"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils

##
# Pre-defined configs
##
from isaaclab.physics import PhysicsCfg

from isaaclab_contrib.controllers.lee_position_control import LeePosController
from isaaclab_contrib.controllers.lee_position_control_cfg import LeePosControllerCfg

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


def main():
    """Main function to spawn arl_robot_1."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # Create simulation context
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)

        # Create a dome light with light blue color
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.53, 0.81, 0.92))
        light_cfg.func("/World/DomeLight", light_cfg)

        # Spawn ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Spawn robot
        robot_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Robot")
        robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
        robot = robot_cfg.class_type(robot_cfg)

        # Play the simulator
        sim.reset()

        # Create Lee position controller
        controller_cfg = LeePosControllerCfg(
            K_pos_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            max_inclination_angle_rad=1.0471975511965976,
            max_yaw_rate=1.0471975511965976,
        )
        controller = LeePosController(controller_cfg, robot, num_envs=1, device=str(sim.device))

        # Get allocation matrix and compute pseudoinverse
        allocation_matrix = torch.tensor(robot_cfg.allocation_matrix, device=sim.device, dtype=torch.float32)
        # allocation_matrix is (6, num_thrusters), we need pseudoinverse for wrench -> thrust
        alloc_pinv = torch.linalg.pinv(allocation_matrix)  # Shape: (num_thrusters, 6)

        # Position command: hover in place (zero position, zero yaw)
        pos_command = torch.zeros((1, 4), device=sim.device)  # [x, y, z, yaw]
        pos_command[0, 2] = 1.0  # Hover at 1 meter height

        # Simulation loop
        print("[INFO] Starting demo with Lee Position Controller. Press Ctrl+C to stop.")

        # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
        while sim.is_headless_or_exist_active_visualizer():
            # Compute wrench from position controller
            wrench = controller.compute(pos_command)  # Shape: (1, 6)

            # Allocate wrench to thrusters: thrust = pinv(A) @ wrench
            thrust_cmd = torch.matmul(wrench, alloc_pinv.T)  # Shape: (1, num_thrusters)
            thrust_cmd = thrust_cmd.clamp(min=0.0)  # Ensure non-negative thrust

            # Apply thrust
            robot.set_thrust_target(thrust_cmd)

            # Step simulation
            robot.write_data_to_sim()
            sim.step()

            # Update robot
            robot.update(sim_cfg.dt)


if __name__ == "__main__":
    main()
