# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# flake8: noqa
# pylint: skip-file

import argparse

from pink.tasks import FrameTask

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates running an Isaac Lab environment with random actions."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# Add argument for number of steps to run
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run.")
# Add argument for whether to print info
parser.add_argument("--print_info", "-p", action="store_true", help="Whether to print info.")
# Add argument for environment selection
parser.add_argument(
    "--mode",
    type=str,
    default="FixedBaseUpperBodyIKG1Env",
    choices=["LocomanipulationG1Env", "FixedBaseUpperBodyIKG1Env", "FixedBaseUpperBodyIKGR1T2Env"],
    help=(
        "Environment to use: 'LocomanipulationG1Env' for full body control, 'FixedBaseUpperBodyIKG1Env' for upper body only,"
        " or 'FixedBaseUpperBodyIKGR1T2Env' for upper body control only on GR1T2."
    ),
)



# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import os
import time
import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation import (
    fixed_base_upper_body_ik_g1_env_cfg,
    fixed_base_upper_body_ik_gr1t2_env_cfg,
    locomanipulation_g1_env_cfg,
)


def get_environment_config(mode: str):
    """Get environment configuration based on the specified mode.
    
    Args:
        mode: The environment mode/name to use
        
    Returns:
        The environment configuration object
        
    Raises:
        ValueError: If an invalid mode is provided
    """
    if mode == "LocomanipulationG1Env":
        env_cfg = locomanipulation_g1_env_cfg.LocomanipulationG1EnvCfg()
        print(f"[INFO]: Using LocomanipulationG1Env - full body control enabled")
    elif mode == "FixedBaseUpperBodyIKG1Env":
        env_cfg = fixed_base_upper_body_ik_g1_env_cfg.FixedBaseUpperBodyIKG1EnvCfg()
        print(f"[INFO]: Using FixedBaseUpperBodyIKG1Env - upper body control only")
    elif mode == "FixedBaseUpperBodyIKGR1T2Env":
        env_cfg = fixed_base_upper_body_ik_gr1t2_env_cfg.FixedBaseUpperBodyIKGR1T2EnvCfg()
        print(f"[INFO]: Using FixedBaseUpperBodyIKGR1T2Env - upper body control only on GR1T2")
    else:
        raise ValueError(f"Invalid mode '{mode}'. Supported environments are: 'LocomanipulationG1Env', 'FixedBaseUpperBodyIKG1Env', 'FixedBaseUpperBodyIKGR1T2Env'")
    
    return env_cfg


def generate_circular_trajectory(base_pose, radius=0.1, frequency=0.5, time_offset=0.0):
    """Generate a circular trajectory around a base pose.
    
    Args:
        base_pose: Base pose [x, y, z, qw, qx, qy, qz]
        radius: Radius of the circular motion
        frequency: Frequency of the circular motion (Hz)
        time_offset: Time offset for phase shift
        
    Returns:
        Updated pose with circular motion applied
    """
    t = time.time() * frequency + time_offset
    
    # Extract position and orientation
    x, y, z = base_pose[0], base_pose[1], base_pose[2]
    orientation = base_pose[3:]
    
    # Apply circular motion in xz plane
    x += radius * np.cos(t)
    z += radius * np.sin(t)
    
    return [x, y, z] + orientation


def print_info(env):
    # Print all actuated joint names and their corresponding order
    print("-" * 80)
    print("Actuated Joints Information:")
    print("-" * 80)

    # Get joint names from the robot asset
    robot_asset = env.scene["robot"]
    actuated_joint_names = robot_asset.data.joint_names

    print(f"Total actuated joints: {len(actuated_joint_names)}")
    print("Joint order:")
    for i, joint_name in enumerate(actuated_joint_names):
        print(f"  [{i}]: {joint_name}")
    print("-" * 80)

    body_names = robot_asset.data.body_names
    print(f"Total bodies: {len(body_names)}")
    print("Body names:")
    for i, body_name in enumerate(body_names):
        print(f"  [{i}]: {body_name}")
    print("-" * 80)

    # Print joint limits
    print("-" * 80)
    print("Joint Limits:")
    print("-" * 80)
    for i, joint_name in enumerate(actuated_joint_names):
        # Get position limits - tensor has shape [num_envs, 2] where dimensions are:
        # - num_envs: number of environments
        # - 2: [lower limit, upper limit]
        # Since all environments have the same limits, we just take the first one
        joint_pos_limit = robot_asset.data.joint_pos_limits[0, i]  # Take the limits from first environment
        joint_pos_limit_str = f"[{joint_pos_limit[0].item():.3f}, {joint_pos_limit[1].item():.3f}]"
        default_joint_pos = robot_asset.data.default_joint_pos[0, i].item()

        # Get velocity and effort limits
        joint_vel_limit = robot_asset.data.joint_vel_limits[0, i].item()
        default_joint_vel = robot_asset.data.default_joint_vel[0, i].item()
        joint_effort_limit = robot_asset.data.joint_effort_limits[0, i].item()

        print(
            f"  {joint_name}: default_pos={default_joint_pos:.3f}, default_vel={default_joint_vel:.3f},"
            f" pos_limit={joint_pos_limit_str}, vel_limit={joint_vel_limit:.3f},"
            f" effort_limit={joint_effort_limit:.3f}"
        )


def main():
    """Main function."""
    # Get environment configuration based on mode
    env_cfg = get_environment_config(args_cli.mode)
    env_cfg.scene.num_envs = args_cli.num_envs
    # Setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    if args_cli.print_info:
        print_info(env)

    count = 0

    # Reset
    obs_dict = env.unwrapped.reset()[0]
    print("-" * 80)
    print("[INFO]: Resetting environment...")

    # Define base poses for hands
    hand_orientation_quat = [0.7071, 0, 0, 0.7071]
    left_hand_roll_link_pos = [-0.18, 0.22, 1.0]
    left_hand_roll_link_pose = left_hand_roll_link_pos + hand_orientation_quat
    right_hand_roll_link_pos = [0.18, 0.22, 1.0]
    right_hand_roll_link_pose = right_hand_roll_link_pos + hand_orientation_quat

    # Configure circular trajectory parameters
    radius = 0.1  # 10 cm radius
    frequency = 0.5  # 0.5 Hz frequency
    left_hand_offset = 0.0  # No offset for left hand
    right_hand_offset = 0.0  # No offset for right hand

    print(f"[INFO]: Using circular trajectory with radius={radius}m, frequency={frequency}Hz")

    while simulation_app.is_running() and count < args_cli.num_steps:
        with torch.inference_mode():

            action_commands = torch.zeros_like(env.unwrapped.action_manager.action)

            # Generate circular trajectories for both hands
            left_hand_pose = generate_circular_trajectory(left_hand_roll_link_pose, radius, frequency, left_hand_offset)
            right_hand_pose = generate_circular_trajectory(right_hand_roll_link_pose, radius, frequency, right_hand_offset)

            # Combine poses for IK
            setpoint_poses = left_hand_pose + right_hand_pose
            ik_actions = torch.tensor(setpoint_poses, device=env.device, dtype=torch.float32)
            action_commands[:, : ik_actions.shape[0]] = ik_actions

            # Step the environment
            obs_dict, rewards, dones, truncateds, infos = env.unwrapped.step(action_commands)
            # Update counter
            count += 1

    # close the environment
    env.unwrapped.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
