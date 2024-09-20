# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on running the cartpole RL environment."
)
parser.add_argument(
    "--num_envs", type=int, default=16, help="Number of environments to spawn."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from ur5_rl_env import HawUr5EnvCfg, HawUr5Env


def gripper_steer(action: float) -> torch.Tensor:
    """Steer the individual gripper joints.
       This function translates a single action
       between -1 and 1 to the gripper joint position targets.
       value to the gripper joint position targets.

    Args:
        action (float): Action to steer the gripper.

    Returns:
        torch.Tensor: Gripper joint position targets.
    """
    # create joint position targets
    gripper_joint_pos = torch.tensor(
        [
            36 * action,  # "left_outer_knuckle_joint",
            -36 * action,  # "left_inner_finger_joint",
            -36 * action,  # "left_inner_knuckle_joint",
            -36 * action,  # "right_inner_knuckle_joint",
            36 * action,  # "right_outer_knuckle_joint",
            36 * action,  # "right_inner_finger_joint",
        ]
    )
    return gripper_joint_pos


def main():
    """Main function."""
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
                print("[INFO]: Resetting environment...")

            # Sample test action for the gripper
            gripper_action = -1 + count % 2
            # 2Pi/360 helper var
            deg2rad = 0.017453292519943295
            # create a tensor for joint position targets with 12 values (6 for joints, 6 for gripper)
            joint_pos_targets = torch.tensor(
                [
                    [
                        0 * deg2rad,
                        -110 * deg2rad,
                        110 * deg2rad,
                        -180 * deg2rad,
                        -90 * deg2rad,
                        0 * deg2rad,
                        gripper_action,
                    ]
                ]
                * env_cfg.scene.num_envs
                # [[0, -1, -1, 1, 0, 0, gripper_action]]
                # * env_cfg.scene.num_envs
            )

            # Step the environment
            obs, rew, terminated, truncated, info = env.step(joint_pos_targets)

            # update counter
            count += 0.01

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
