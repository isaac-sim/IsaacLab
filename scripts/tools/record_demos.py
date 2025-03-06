# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import time
import torch

import omni.log

from isaaclab.devices import OpenXRDevice, Se3Keyboard, Se3SpaceMouse
from isaaclab.devices.openxr.retargeters.manipulator import GripperRetargeter, Se3AbsRetargeter, Se3RelRetargeter
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    if "handtracking" in args_cli.teleop_device.lower():
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Flags for controlling the demonstration recording process
    should_reset_recording_instance = False
    running_recording_instance = True

    def reset_recording_instance():
        """Reset the current recording instance.

        This function is triggered when the user indicates the current demo attempt
        has failed and should be discarded. When called, it marks the environment
        for reset, which will start a fresh recording instance. This is useful when:
        - The robot gets into an unrecoverable state
        - The user makes a mistake during demonstration
        - The objects in the scene need to be reset to their initial positions
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def start_recording_instance():
        """Start or resume recording the current demonstration.

        This function enables active recording of robot actions. It's used when:
        - Beginning a new demonstration after positioning the robot
        - Resuming recording after temporarily stopping to reposition
        - Continuing demonstration after pausing to adjust approach or strategy

        The user can toggle between stop/start to reposition the robot without
        recording those transitional movements in the final demonstration.
        """
        nonlocal running_recording_instance
        running_recording_instance = True

    def stop_recording_instance():
        """Temporarily stop recording the current demonstration.

        This function pauses the active recording of robot actions, allowing the user to:
        - Reposition the robot or hand tracking device without recording those movements
        - Take a break without terminating the entire demonstration
        - Adjust their approach before continuing with the task

        The environment will continue rendering but won't record actions or advance
        the simulation until recording is resumed with start_recording_instance().
        """
        nonlocal running_recording_instance
        running_recording_instance = False

    def create_teleop_device(device_name: str):
        """Create and configure teleoperation device for robot control.

        Args:
            device_name: Control device to use. Options include:
                - "keyboard": Use keyboard keys for simple discrete movements
                - "spacemouse": Use 3D mouse for precise 6-DOF control
                - "handtracking": Use VR hand tracking for intuitive manipulation
                - "handtracking_abs": Use VR hand tracking for intuitive manipulation with absolute EE pose

        Returns:
            DeviceBase: Configured teleoperation device ready for robot control
        """
        device_name = device_name.lower()
        if device_name == "keyboard":
            return Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5)
        elif device_name == "spacemouse":
            return Se3SpaceMouse(pos_sensitivity=0.2, rot_sensitivity=0.5)
        elif "handtracking" in device_name:
            # Create Franka retargeter with desired configuration
            if "_abs" in device_name:
                retargeter_device = Se3AbsRetargeter(zero_out_xy_rotation=True)
            else:
                retargeter_device = Se3RelRetargeter(zero_out_xy_rotation=True)

            grip_retargeter = GripperRetargeter()

            # Create hand tracking device with retargeter (in a list)
            device = OpenXRDevice(
                env_cfg.xr,
                hand=OpenXRDevice.Hand.RIGHT,
                retargeters=[retargeter_device, grip_retargeter],
            )
            device.add_callback("RESET", reset_recording_instance)
            device.add_callback("START", start_recording_instance)
            device.add_callback("STOP", stop_recording_instance)

            nonlocal running_recording_instance
            running_recording_instance = False
            return device
        else:
            raise ValueError(
                f"Invalid device interface '{device_name}'. Supported: 'keyboard', 'spacemouse', 'handtracking',"
                " 'handtracking_abs'."
            )

    teleop_interface = create_teleop_device(args_cli.teleop_device)
    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    # reset before starting
    env.reset()
    teleop_interface.reset()

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            actions = pre_process_actions(delta_pose, gripper_command)

            # perform action on environment
            if running_recording_instance:
                env.step(actions)
            else:
                env.sim.render()

            if success_term is not None:
                if bool(success_term.func(env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0

            if should_reset_recording_instance:
                env.recorder_manager.reset()
                env.reset()
                should_reset_recording_instance = False
                success_step_count = 0

            # print out the current demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
