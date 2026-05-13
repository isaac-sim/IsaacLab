# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#export PYTHONPATH=$PYTHONPATH:$(pwd)
#./isaaclab.sh -p A10_Single/teleoperation.py --task Isaac-Lift-Cube-A10-IK-Rel --num_envs 1 --teleop_device keyboard

#./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device keyboard

#./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-A10-IK-Rel --num_envs 1 --teleop_device keyboard
"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import logging
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# import logger
logger = logging.getLogger(__name__)

# from .reach.config import a10  # 如果teleoperation.py在包内


def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
   # env_cfg.robot = A10_CFG
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        # External cameras are not supported with XR teleop
        # Check for any camera configs and disable them
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach, we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.04 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.04 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.10 * sensitivity, rot_sensitivity=0.12 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, gamepad, handtracking")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # Infer expected action dimension from action manager (more reliable than action_space for batched envs).
    expected_action_dim = None
    if hasattr(env, "action_manager"):
        expected_action_dim = int(env.action_manager.total_action_dim)
    elif hasattr(env.action_space, "shape") and env.action_space.shape is not None and len(env.action_space.shape) > 0:
        expected_action_dim = int(env.action_space.shape[-1])
    warned_action_dim_once = False
    reset_hold_steps = 8
    reset_hold_counter = reset_hold_steps

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # Handle reset before reading/applying any new action to avoid one stale control step.
                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    reset_hold_counter = reset_hold_steps
                    print("Environment reset complete")
                    continue

                # get device command
                action = teleop_interface.advance()

                # # Only apply teleop commands when active
                # if teleoperation_active:
                #     # process actions
                #     actions = action.repeat(env.num_envs, 1)
                #     print("num actions:", env.action_space.shape)

                #     actions = actions[..., :6]
                #     robot = env.scene["robot"]
                #     print("joint pos:", robot.data.joint_pos[0, :6])

                #     env.step(actions)
                # else:
                #     env.sim.render()

                if teleoperation_active:
                    # Hold zero-actions for a few frames right after reset to avoid stale-input spikes.
                    if reset_hold_counter > 0 and expected_action_dim is not None:
                        zero_actions = torch.zeros((env.num_envs, expected_action_dim), device=env.device, dtype=torch.float32)
                        env.step(zero_actions)
                        reset_hold_counter -= 1
                        continue

                    # Some devices can return tuples/lists (e.g. [delta_pose, gripper]); flatten to a single vector.
                    if isinstance(action, (tuple, list)):
                        tensor_parts = [
                            torch.as_tensor(part, device=env.device, dtype=torch.float32).reshape(-1) for part in action
                        ]
                        actions = torch.cat(tensor_parts, dim=0)
                    else:
                        actions = torch.as_tensor(action, device=env.device, dtype=torch.float32)

                    if actions.ndim == 1:
                        actions = actions.unsqueeze(0)

                    if expected_action_dim is not None and actions.shape[-1] != expected_action_dim:
                        current_dim = int(actions.shape[-1])
                        if current_dim > expected_action_dim:
                            actions = actions[..., :expected_action_dim]
                        else:
                            pad = torch.zeros(
                                actions.shape[0],
                                expected_action_dim - current_dim,
                                device=actions.device,
                                dtype=actions.dtype,
                            )
                            actions = torch.cat([actions, pad], dim=-1)

                        if not warned_action_dim_once:
                            logger.warning(
                                "Teleop action dimension adjusted from %d to %d for task '%s'.",
                                current_dim,
                                expected_action_dim,
                                args_cli.task,
                            )
                            warned_action_dim_once = True

                    if actions.shape[0] == 1 and env.num_envs > 1:
                        actions = actions.repeat(env.num_envs, 1)
                    # A10 keyboard convention fix: keep x as-is, invert y/z/yaw for intuitive teleop direction.
                    if args_cli.teleop_device.lower() == "keyboard" and actions.shape[-1] >= 6:
                        actions[:, 1] *= -1.0  # y-axis (A/D)
                        actions[:, 2] *= -1.0  # z-axis (Q/E)
                        actions[:, 5] *= -1.0  # yaw (C/V)
                    # Filter tiny residual values to prevent drift when no key is pressed.
                    actions = torch.where(actions.abs() < 1e-4, torch.zeros_like(actions), actions)
                    env.step(actions)
                else:
                    env.sim.render()
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
