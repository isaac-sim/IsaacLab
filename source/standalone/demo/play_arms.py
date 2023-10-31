# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a single-arm manipulator with Orbit interfaces.

We currently support the following robots:

* Franka Emika Panda
* Universal Robot UR10

From the default configuration file for these robots, zero actions imply a default pose.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates how to use the physics engine to simulate a single-arm manipulator."
)
parser.add_argument(
    "--robot", type=str, default="franka_panda", choices=["franka_panda", "ur10"], help="Name of the robot."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import traceback

import carb

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.assets.config.universal_robots import UR10_CFG
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Main function."""

    # Load kit helper
    # note: there is a bug in Isaac Sim 2022.2.1 that prevents the use of GPU pipeline
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False))
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg, translation=(0.0, 0.0, -1.05))
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Table_1", cfg, translation=(0.55, -1.0, 0.0))
    cfg.func("/World/Table_2", cfg, translation=(0.55, 1.0, 0.0))

    # Robots
    # -- Resolve robot config from command-line arguments
    if args_cli.robot == "franka_panda":
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    elif args_cli.robot == "ur10":
        robot_cfg = UR10_CFG
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported!")
    # -- Spawn robot
    robot_cfg.spawn.func("/World/Robot_1", robot_cfg.spawn, translation=(0.0, -1.0, 0.0))
    robot_cfg.spawn.func("/World/Robot_2", robot_cfg.spawn, translation=(0.0, 1.0, 0.0))
    # -- Create interface
    robot = Articulation(cfg=robot_cfg.replace(prim_path="/World/Robot.*"))

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    actions = torch.rand(robot.root_view.count, robot.num_joints, device=robot.device) + robot.data.default_joint_pos
    has_gripper = args_cli.robot == "franka_panda"

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
            actions = (
                torch.rand(robot.root_view.count, robot.num_joints, device=robot.device) + robot.data.default_joint_pos
            )
            # reset gripper
            if has_gripper:
                actions[:, -2:] = 0.04
            print("[INFO]: Resetting robots state...")
        # change the gripper action
        if ep_step_count % 200 == 0 and has_gripper:
            # flip command for the gripper
            actions[:, -2:] = 0.0 if actions[0, -2] > 0.0 else 0.04
            print(f"[INFO]: [Step {ep_step_count:03d}]: Flipping gripper command...")
        # apply action to the robot
        robot.set_joint_position_target(actions)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update buffers
        robot.update(sim_dt)
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
