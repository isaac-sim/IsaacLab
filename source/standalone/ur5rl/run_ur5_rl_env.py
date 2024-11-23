# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the xxx with ROS 2 integration."""

import argparse

from omni.isaac.lab.app import AppLauncher

# ----------------- ROS -----------------
import rclpy
import sys

# Get the Ur5JointController class from the ur5_basic_control_fpc module
from ros2_humble_ws.src.ur5_parallel_control.ur5_parallel_control.ur5_basic_control_fpc import (
    Ur5JointController,
)
import threading


# Separate thread to run the ROS 2 node in parallel to the simulation
def ros_node_thread(node: Ur5JointController):
    """
    Function to spin the ROS 2 node in a separate thread.
    """
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on running the cartpole RL environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)

parser.add_argument(
    "--pub2ros",
    type=bool,
    default=True,
    help="Publish the action commands via a ros node to a forward position position controller. This will enable real robot parallel control.",
)

# TODO Check that if ros2 is not installed, then only one environment can be spawned
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# from omni.isaac.core.utils.extensions import enable_extension

# Enable the ROS 2 Bridge
# enable_extension("omni.isaac.ros2_bridge")

import torch
from ur5_rl_env import HawUr5EnvCfg, HawUr5Env


def sync_sim_joints_with_real_robot(env: HawUr5Env, ur5_controller: Ur5JointController):
    """Sync the simulated robot joints with the real robot."""
    # Sync sim joints with real robot
    print("[INFO]: Waiting for joint positions from the real robot...")
    while ur5_controller.get_joint_positions() == None:
        pass
    real_joint_positions = ur5_controller.get_joint_positions()
    env.set_joint_angles_absolute(joint_angles=real_joint_positions)


def main():
    """Main function."""
    # Check if the user wants to publish the actions to ROS2
    PUBLISH_2_ROS = args_cli.pub2ros

    # create environment configuration
    env_cfg = HawUr5EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = HawUr5Env(cfg=env_cfg)

    # simulate physics
    count = 0

    if PUBLISH_2_ROS:
        # ROS 2 initialization
        rclpy.init()
        ur5_controller = Ur5JointController()
        # Start the ROS node in a separate thread
        ros_thread = threading.Thread(
            target=ros_node_thread, args=(ur5_controller,), daemon=True
        )
        ros_thread.start()

    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 200 == 0:
                count = 0
                # env.reset()
                print("-" * 80)
                print("[INFO]: Env reset.")
                if PUBLISH_2_ROS:
                    sync_sim_joints_with_real_robot(env, ur5_controller)

            # Sample test action for the gripper
            gripper_action = -1 + count / 100 % 2
            # create a tensor for joint position targets with 7 values (6 for joints, 1 for gripper)
            actions = torch.tensor(
                [
                    [
                        -0.0,
                        -0.0,
                        -0.1,
                        -0.0,
                        -0.0,
                        -0.0,
                        gripper_action,
                    ]
                ]
                * env_cfg.scene.num_envs
            )

            if PUBLISH_2_ROS:
                # Send ros actions to the real robot # TODO implement GRIPPER CONTROL
                ur5_controller.set_joint_delta(actions[0, :7].numpy())
                real_joint_positions = ur5_controller.get_joint_positions()

            print(f"Gripperaction: {gripper_action}")
            # Step the environment
            obs, rew, terminated, truncated, info = env.step(actions)

            # update counter
            count += 1

    # close the environment
    env.close()

    # Shutdown ROS 2 (if initialized)
    # rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
