# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run teleoperation of a Franka in an IsaacLab environment using PS4, keyboard, or SpaceMouse.
"""
import argparse

from isaaclab.app import AppLauncher


# ---- CLI args ----
parser = argparse.ArgumentParser(
    description="Teleop IK agent for IsaacLab environments."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric interface.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="gamepad",
    choices=["gamepad", "keyboard", "spacemouse"],
    help="Which input device to use for teleoperation",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---- Launch Isaac Sim ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)


# ---- Rest follows ----
def main():
    # 1) Build the Gym environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # 2) Grab the underlying Franka articulation and its indices
    franka = env.unwrapped.robot_left  # name comes from the env's config
    # resolve the end-effector body index:
    ee_body_id = franka.find_bodies([".*hand"])[0][0]
    # for fixed-base robots, jacobian index = body_id - 1
    ee_jacobian_idx = ee_body_id - 1 if franka.is_fixed_base else ee_body_id
    # get joint IDs for the arm (exclude fingers)
    franka_joint_ids = franka.find_joints(["panda_joint.*"])[0]

    # 3) Set up your teleop device

    if args_cli.teleop_device == "gamepad":
        teleop = Se3Gamepad(pos_sensitivity=1.0, rot_sensitivity=1.6, dead_zone=0.01)
    elif args_cli.teleop_device == "keyboard":
        teleop = Se3Keyboard(pos_sensitivity=0.4, rot_sensitivity=0.8)
    else:  # spacemouse
        teleop = Se3SpaceMouse(
            v_x_sensitivity=0.8,
            v_y_sensitivity=0.4,
            omega_z_sensitivity=1.0,
            dead_zone=0.01,
        )
    teleop.reset()  # clear its internals :contentReference[oaicite:2]{index=2}

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",  # full pose control
        use_relative_mode=True,  # commands are deltas
        ik_method="dls",  # damped least squares
    )
    ik_ctrl = DifferentialIKController(
        ik_cfg,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )

    # 5) Main teleop loop
    while simulation_app.is_running():
        # read teleop device: (6-vector delta pose, bool gripper)
        delta_pose, gripper_cmd = (
            teleop.advance()
        )  # :contentReference[oaicite:3]{index=3}

        # get current world-frame quantities
        ee_pose_w = franka.data.body_pose_w[:, ee_body_id, :7]  # (N,7)
        root_pose_w = franka.data.root_pose_w  # (N,7)
        joint_pos = franka.data.joint_pos[:, franka_joint_ids]  # (N, num_joints)
        jacobian = franka.root_physx_view.get_jacobians()[
            :, ee_jacobian_idx, :, franka_joint_ids
        ]  # (N,6,num_joints)

        # transform EE pose into base frame

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )

        # set and solve IK (differential)
        ik_ctrl.reset()  # clear controller state
        ik_ctrl.set_command(
            torch.tensor(delta_pose[None, :], device=args_cli.device),
            ee_pos=ee_pos_b,
            ee_quat=ee_quat_b,
        )  # :contentReference[oaicite:4]{index=4}
        joint_pos_des = ik_ctrl.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        actions[:, :7] = joint_pos_des

        # advance the sim
        env.step(actions)

    # cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
