"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Bimanual Mobile Manipulator(BMM) Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard_bmm", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default="Cabinet-anubis-teleop-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# parse the arguments

app_launcher_args = vars(args_cli)
if args_cli.teleop_device.lower() == "handtracking":
    app_launcher_args["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'
# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import argparse
import numpy as np
import gymnasium as gym
import torch
import omni.log

from isaaclab.app import AppLauncher
from isaaclab.devices import Se3Keyboard_BMM, MobileBaseJoystick
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import ViewerCfg
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401


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

    # Instantiate teleoperation interfaces
    # Keyboard BMM for arms only (disable base) by setting base_sensitivity=0
    arm_ctrl = Se3Keyboard_BMM(
        pos_sensitivity=0.05 * args_cli.sensitivity,
        rot_sensitivity=0.05 * args_cli.sensitivity,
        base_sensitivity=0.0
    )
    # Joystick for mobile base
    base_ctrl = MobileBaseJoystick(
        linear_sensitivity=0.1 * args_cli.sensitivity,
        angular_sensitivity=0.1 * args_cli.sensitivity
    )

    # Teleoperation reset flag and callback
    should_reset = False
    def reset_all():
        nonlocal should_reset
        should_reset = True
        arm_ctrl.reset()
        base_ctrl.reset()

    arm_ctrl.add_callback("R", reset_all)

    # Initial resets
    env.reset()
    arm_ctrl.reset()
    base_ctrl.reset()

    # Main simulation loop
    while sim_app.is_running():
        with torch.inference_mode():
            # Read arm commands from keyboard
            delta_L, grip_L, delta_R, grip_R, _ = arm_ctrl.advance()
            # Read base commands from joystick
            delta_base = base_ctrl.advance()

            # Convert to torch and repeat per environment
            tdelta_L = torch.tensor(delta_L.astype("float32"), device=env.device).repeat(env.num_envs, 1)
            tdelta_R = torch.tensor(delta_R.astype("float32"), device=env.device).repeat(env.num_envs, 1)
            tdelta_base = torch.tensor(delta_base.astype("float32"), device=env.device).repeat(env.num_envs, 1)

            # Build action tensor
            grL = -1.0 if grip_L else 1.0
            grR = -1.0 if grip_R else 1.0
            grL_tensor = torch.full((env.num_envs, 1), grL, device=env.device)
            grR_tensor = torch.full((env.num_envs, 1), grR, device=env.device)
            actions = torch.cat([
                tdelta_L, tdelta_R, grL_tensor, grR_tensor, tdelta_base
            ], dim=1)

            # Step the environment
            env.step(actions)

            # Handle reset
            if should_reset:
                env.reset()
                should_reset = False

    # Cleanup
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()