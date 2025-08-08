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

    # after you build the env
    franka_L = env.unwrapped.robot_left
    franka_R = env.unwrapped.robot_right  # use it to hold the right arm

    left_arm_ids = franka_L.find_joints(["panda_joint[1-7]"])[0]
    left_finger_ids = franka_L.find_joints(["panda_finger_joint.*"])[0]
    right_arm_ids = franka_R.find_joints(["panda_joint[1-7]"])[0]
    right_finger_ids = franka_R.find_joints(["panda_finger_joint.*"])[0]

    # derive safe open/close from soft limits (works across assets)
    lims_low = franka_L.data.soft_joint_pos_limits[0, left_finger_ids, 0]
    lims_high = franka_L.data.soft_joint_pos_limits[0, left_finger_ids, 1]
    F_OPEN = float(torch.min(lims_high).clamp(min=0.03, max=0.05))  # ~0.04 m typical
    F_CLOSED = float(torch.max(lims_low)) + 0.001

    # LATCHED hold targets (don’t overwrite these every frame)
    qR_hold = franka_R.data.joint_pos[:, right_arm_ids].clone()
    qRf_hold = franka_R.data.joint_pos[:, right_finger_ids].clone()

    num_envs = env.unwrapped.num_envs
    act_dim = env.action_space.shape[-1]

    # resolve the end-effector body index:
    ee_body_id = franka_L.find_bodies([".*hand"])[0][0]
    # for fixed-base robots, jacobian index = body_id - 1
    ee_jacobian_idx = ee_body_id - 1 if franka_L.is_fixed_base else ee_body_id
    # get joint IDs for the arm (exclude fingers)
    franka_joint_ids = franka_L.find_joints(["panda_joint.*"])[0]

    # 3) Set up your teleop device

    if args_cli.teleop_device == "gamepad":
        teleop = Se3Gamepad(pos_sensitivity=1.0, rot_sensitivity=1.6, dead_zone=0.07)
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
        # 1) read device
        delta_pose, grip_cmd = teleop.advance()
        print(f"[INFO]: Delta pose: {delta_pose}, Grip command: {grip_cmd}")
        delta = torch.tensor(delta_pose, device=args_cli.device)
        idle = torch.all(delta.abs() < 1e-3).item()
        print(f"[INFO]: Idle: {idle}")

        f_target = F_CLOSED if grip_cmd else F_OPEN
        f_target_vec = torch.full(
            (env.unwrapped.num_envs, 1), f_target, device=env.unwrapped.device
        )

        # 2) current states
        qL = franka_L.data.joint_pos[:, left_arm_ids]
        qR = franka_R.data.joint_pos[:, right_arm_ids]

        # 3) ee in base frame
        ee_pose_w = franka_L.data.body_pose_w[:, ee_body_id, :7]
        root_pose_w = franka_L.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )
        jacobian = franka_L.root_physx_view.get_jacobians()[
            :, ee_jacobian_idx, :, franka_joint_ids
        ]  # (N,6,num_joints)

        # 4) target for left arm
        if not idle:
            ik_ctrl.set_command(delta.unsqueeze(0), ee_pos=ee_pos_b, ee_quat=ee_quat_b)
            qL_des = ik_ctrl.compute(ee_pos_b, ee_quat_b, jacobian, qL)
        else:
            qL_des = qL  # HOLD

        # 5) build actions: (N, 18)
        actions = torch.zeros((num_envs, act_dim), device=env.unwrapped.device)

        # left arm (7 joints)
        actions[:, 0:7] = qL_des
        # left fingers (keep open at current, or set a value)
        actions[:, 7:9] = f_target_vec.expand(
            -1, 2
        )  # command both left finger joints equally

        # right: HOLD (latched)
        actions[:, 9:16] = qR_hold
        actions[:, 16:18] = qRf_hold

        env.step(actions)

    # cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
