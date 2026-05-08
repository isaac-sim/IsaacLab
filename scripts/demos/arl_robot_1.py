# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to view ARL Robot 1.

Launch Isaac Sim Simulator first.
"""

# Create argparser
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View ARL Robot 1 with Lee Position Controller.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.usd
from pxr import Gf, UsdLux

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from isaaclab_contrib.assets import Multirotor
from isaaclab_contrib.controllers.lee_position_control import LeePosController
from isaaclab_contrib.controllers.lee_position_control_cfg import LeePosControllerCfg

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


def main():
    """Main function to spawn arl_robot_1."""

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # Create a dome light with light blue color
    stage = omni.usd.get_context().get_stage()
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateColorAttr(Gf.Vec3f(0.53, 0.81, 0.92))  # Light blue
    dome_light.CreateIntensityAttr(1000.0)

    # Spawn ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Spawn robot
    robot_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Robot")
    robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
    robot = Multirotor(robot_cfg)

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

    while simulation_app.is_running():
        # Compute wrench from velocity controller
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

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
