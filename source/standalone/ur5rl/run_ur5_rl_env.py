# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task with ROS 2 integration."""
# ----------------- CUDA_LAUNCH_BLOCKING -----------------
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# --------------------------------------------------------

import argparse

from omni.isaac.lab.app import AppLauncher

# ----------------- ROS -----------------
import rclpy
import sys

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

from omni.isaac.core.utils.extensions import enable_extension

# Enable the ROS 2 Bridge
enable_extension("omni.isaac.ros2_bridge")

import torch
from ur5_rl_env import HawUr5EnvCfg, HawUr5Env


def main():
    """Main function."""
    # ROS 2 initialization
    rclpy.init()

    # ROS 2 Node example (optional, if needed)
    # node = rclpy.create_node("isaac_sim_ros_bridge")

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
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        gripper_action,
                    ]
                ]
                * env_cfg.scene.num_envs
            )
            print(f"[INFO]: Shape of actions: {actions.shape}")
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
