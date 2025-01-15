# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
if args_cli.teleop_device.lower() == "handtracking":
    app_launcher_args["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'
# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import omni.log

from isaaclab.devices import Se3Gamepad, Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "handtracking":
        from isaacsim.xr.openxr import OpenXRSpec

        teleop_interface = Se3HandTracking(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, False, True)
        teleop_interface.add_callback("RESET", env.reset)
        viewer = ViewerCfg(eye=(-0.25, -0.3, 0.5), lookat=(0.6, 0, 0), asset_name="viewer")
        ViewportCameraController(env, viewer)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse''handtracking'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.device).repeat(env.num_envs, 1)
            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)
            # apply actions
            env.step(actions)

            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
