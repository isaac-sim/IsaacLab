# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omegaconf import OmegaConf

from omni.isaac.lab.app import AppLauncher

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

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import pickle

from force_tool.utils.data_utils import SmartDict, update_config
from force_tool.visualization.plot_utils import get_img_from_fig, save_numpy_as_mp4

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3RobotiqKeyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
import omni.isaac.core.utils.prims as prim_utils
from pxr import  UsdPhysics
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils import parse_env_cfg


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task or "Float" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    elif "Kuka" in args_cli.task:
        return torch.concat([delta_pose, gripper_command], dim=1).float()
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    cfg = OmegaConf.create()
    vv = {
        "scene.screw_type":  "m16_loose", 
        "scene.robot.collision_approximation": "convexHull"
          }
    cfg = update_config(cfg, vv)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric, params=cfg
    )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        if "Kuka" in args_cli.task:
            teleop_interface = Se3RobotiqKeyboard(
                pos_sensitivity=1 * args_cli.sensitivity,
                rot_sensitivity=1 * args_cli.sensitivity,
                gripper_sensitivity=0.02 * args_cli.sensitivity,
            )
        else:
            teleop_interface = Se3Keyboard(
                pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.5 * args_cli.sensitivity
            )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()
    counter = 0
    record_forces = False
    forces, frames = [], []
    for i in range(10):
        frame = env.unwrapped.render()
    # cached_env_state = SmartDict(pickle.load(open("data/kuka_nut_thread_pre_grasp.pkl", "rb")))
    # env.unwrapped.write_state(cached_env_state)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            gripper_command = torch.tensor(gripper_command, device=env.unwrapped.device).repeat(
                env.unwrapped.num_envs, 1
            )
            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)
            # apply actions
            # actions[:, -2:] = torch.rand_like(actions[:, -2:])
            # actions[:, -2:] = gripper_deltas[counter%100]
            counter += 1
            obs, reward, termin, timeout, _ = env.step(actions)
            # print(env.unwrapped.scene["robot"].read_body_pos_w("victor_left_tool0"))
       
            if record_forces:
                frame = env.unwrapped.render()
                frames.append(frame)
                contact_sensor = env.unwrapped.scene["contact_sensor"]
                dt = contact_sensor._sim_physics_dt
                friction_data = contact_sensor.contact_physx_view.get_friction_data(dt)
                contact_data = contact_sensor.contact_physx_view.get_contact_data(dt)
                nforce_mag, npoint, nnormal, ndist, ncount, nstarts = contact_data
                tforce, tpoint, tcount, tstarts = friction_data
                nforce = nnormal * nforce_mag
                nforce = torch.sum(nforce, dim=0)
                tforce = torch.sum(tforce, dim=0)
                total_force = torch.tensor([nforce.norm(), tforce.norm(), torch.norm(nforce + tforce)])
                print(nforce, tforce, total_force)
                print("Total force: ", total_force)
                forces.append(total_force.cpu().numpy())
            if termin:
                print("Episode terminated.")
                env.reset()
                teleop_interface.reset()
                counter = 0
                if record_forces:
                    wrench_frames = []
                    plot_target = np.array(forces)
                    labels = ["Normal Force", "Tangential Force", "Total Force"]
                    max_val = np.max(plot_target)
                    min_val = np.min(plot_target)
                    indices = np.arange(len(plot_target)) + 1
                    num_plots = plot_target.shape[-1]
                    plt.plot(indices, plot_target, label=labels)
                    plt.legend()
                    plt.show()
                    plt.close()
                    for t in tqdm.tqdm(indices):
                        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
                        plt.ylim((min_val, max_val))
                        plt.xlim((0, len(plot_target)))
                        plt.plot(indices[:t], plot_target[:t], label=labels)
                        plt.legend()
                        wrench_frame = get_img_from_fig(fig, width=frame.shape[1] // 2, height=frame.shape[0])
                        wrench_frames.append(wrench_frame)
                        plt.close()
                        # combine frames
                    frames = np.array(frames)
                    wrench_frames = np.array(wrench_frames)
                    combined_frames = np.concatenate([frames, wrench_frames], axis=2)
                    save_numpy_as_mp4(np.array(combined_frames), "nut.mp4")
                    frames = []

            # print("Step: ", counter)
            # print("Reward: ", reward, termin)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
