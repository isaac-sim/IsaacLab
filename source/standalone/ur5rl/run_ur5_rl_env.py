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


PUBLISH_2_ROS = True
# ---------------------------------------

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on running the cartpole RL environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)

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


def main():
    """Main function."""
    if PUBLISH_2_ROS:
        # ROS 2 initialization
        rclpy.init()
        ur5_controller = Ur5JointController()
        # Start the ROS node in a separate thread
        ros_thread = threading.Thread(
            target=ros_node_thread, args=(ur5_controller,), daemon=True
        )
        ros_thread.start()

    # create environment configuration
    env_cfg = HawUr5EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = HawUr5Env(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Env reset.")

            # Sample test action for the gripper
            gripper_action = -1 + count % 2
            # 2Pi/360 helper var
            deg2rad = 0.017453292519943295
            # create a tensor for joint position targets with 7 values (6 for joints, 1 for gripper)
            actions = torch.tensor(
                [
                    [
                        -0.5,
                        -0.5,
                        -0.5,
                        -0.5,
                        -0.5,
                        -0.5,
                        gripper_action,
                    ]
                ]
                * env_cfg.scene.num_envs
            )
            print(f"[INFO]: Shape of actions: {actions.shape}")

            if PUBLISH_2_ROS:
                # Send ros actions to the real robot # TODO (implement GRIPPER CONTROL)
                ur5_controller.set_joint_delta(actions[0, :6].numpy())
            # Step the environment
            obs, rew, terminated, truncated, info = env.step(actions)

            # update counter
            count += 0.01

    # close the environment
    env.close()

    # Shutdown ROS 2 (if initialized)
    # rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
