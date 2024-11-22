# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omegaconf import OmegaConf

from force_tool.visualization.plot import draw_wrench_video
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
import numpy as np
import pickle
import torch

import omni.isaac.core.utils.prims as prim_utils
from force_tool.utils.data_utils import SmartDict, update_config
from force_tool.visualization.plot_utils import save_numpy_as_mp4
from pxr import UsdPhysics

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3RobotiqKeyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils import parse_env_cfg


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    return delta_pose
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
        "scene.screw_type": "m16_loose",
        "events.reset_target": "rigid_grasp_open_align",
        "scene.robot.arm_stiffness": 300.0,
        "scene.robot.arm_damping": 40.0,
        "decimation": 2,
        "sim.dt": 1 / 60,
        "actions.ik_lambda": 1e-3,
    }
    cfg = update_config(cfg, vv)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        params=cfg,
    )
    # modify configuration
    # env_cfg.terminations = {}
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
    record_forces = True
    forces, frames = [], []
    for i in range(10):
        frame = env.unwrapped.render()
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
            nut_root_pose = env.unwrapped.scene["nut"].read_root_state_from_sim()
            gripper_state_w = env.unwrapped.scene["robot"].read_body_state_w("victor_left_tool0")[:, 0]
            relative_pos, relative_quat = math_utils.subtract_frame_transforms(
                nut_root_pose[:, :3], nut_root_pose[:, 3:7], gripper_state_w[:, :3], gripper_state_w[:, 3:7]
            )
            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)
            # apply actions
            # actions[:, -2:] = torch.rand_like(actions[:, -2:])
            # actions[:, -2:] = gripper_deltas[counter%100]
            counter += 1
            obs, reward, termin, timeout, _ = env.step(actions)
            # print(env.unwrapped.scene["robot"].read_body_pos_w("victor_left_tool0"))
            # grasped_state = env.unwrapped.read_state()
            # pickle.dump(grasped_state, open(f"cached/convexHull2/m16_loose/kuka_rigid_grasp_close_align.pkl", "wb"))
            if record_forces:
                frame = env.unwrapped.render()
                frames.append(frame)
                wrench = obs["policy"][0, 13:19]
                # contact_sensor = env.unwrapped.scene["contact_sensor"]
                # dt = contact_sensor._sim_physics_dt
                # friction_data = contact_sensor.contact_physx_view.get_friction_data(dt)
                # contact_data = contact_sensor.contact_physx_view.get_contact_data(dt)
                # nforce_mag, npoint, nnormal, ndist, ncount, nstarts = contact_data
                # tforce, tpoint, tcount, tstarts = friction_data
                # nforce = nnormal * nforce_mag
                # nforce = torch.sum(nforce, dim=0)
                # tforce = torch.sum(tforce, dim=0)
                # total_force = torch.tensor([nforce.norm(), tforce.norm(), torch.norm(nforce + tforce)])
                # print(nforce, tforce, total_force)
                # print("Total force: ", total_force)
                forces.append(wrench.detach().cpu().numpy())
            # if termin or len(forces) > 50:
            if termin:
                print("Episode terminated.")
                env.reset()
                teleop_interface.reset()
                
                counter = 0
                if record_forces:
                    wrench_frames = draw_wrench_video(forces, frame)
                    frames = np.array(frames)
                    wrench_frames = np.array(wrench_frames)
                    combined_frames = np.concatenate([frames, wrench_frames], axis=2)
                    save_numpy_as_mp4(np.array(combined_frames), "nut.mp4")
                    frames = []
                    break

            # print("Step: ", counter)
            # print("Reward: ", reward, termin)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
