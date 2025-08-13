# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse

from isaaclab.app import AppLauncher

# ---- CLI args ----
parser = argparse.ArgumentParser(description="Teleop IK agent for IsaacLab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric interface.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---- Launch Isaac Sim ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se3Gamepad
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_tasks.utils import parse_env_cfg


def main():
    # 1) Build env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # Robots
    franka = env.unwrapped.robot

    # IDs
    arm_ids = franka.find_joints(["panda_joint[1-7]"])[0]
    finger_ids = franka.find_joints(["panda_finger_joint.*"])[0]

    # Gripper (binary)
    lims_low = franka.data.soft_joint_pos_limits[0, finger_ids, 0]
    lims_high = franka.data.soft_joint_pos_limits[0, finger_ids, 1]
    F_OPEN = float(torch.min(lims_high).clamp(min=0.03, max=0.05))
    F_CLOSED = float(torch.max(lims_low)) + 0.001

    num_envs = env.unwrapped.num_envs
    act_dim = env.action_space.shape[-1]
    device = env.unwrapped.device

    # Hand link / Jacobian index
    hand_body = franka.find_bodies([".*hand"])[0][0]
    hand_jac = hand_body - 1 if franka.is_fixed_base else hand_body

    # Gamepad
    teleop = Se3Gamepad(pos_sensitivity=0.1, rot_sensitivity=0.16, dead_zone=0.07)
    teleop.reset()

    # Controller: relative, position-only
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",  # position only
        use_relative_mode=True,  # deltas in EE frame
        ik_method="dls",
        ik_params={"lambda_val": 0.15},
    )
    ik_ctrl = DifferentialIKController(ik_cfg, num_envs=args_cli.num_envs, device=args_cli.device)

    # ---- Main loop ----
    while simulation_app.is_running():
        delta_pose, grip_cmd = teleop.advance()
        delta_pose[0] = delta_pose[0] * -1.0  # flip y-axis for consistent control form viewpoint

        base_pos_w = franka.data.root_pose_w[:, :3]
        base_quat_w = franka.data.root_pose_w[:, 3:7]
        hand_pos_w = franka.data.body_pos_w[:, hand_body]
        hand_quat_w = franka.data.body_quat_w[:, hand_body]

        # EE (hand) in base
        ee_pos_b, ee_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, hand_pos_w, hand_quat_w)

        # Device delta already in EE frame → position-only relative step
        delta = torch.tensor(delta_pose, device=ee_pos_b.device, dtype=ee_pos_b.dtype)
        dpos_e = delta[:3].unsqueeze(0).expand_as(ee_pos_b)  # (N,3)

        # Jacobian and IK
        J = franka.root_physx_view.get_jacobians()[:, hand_jac, :, arm_ids]
        ik_ctrl.set_command(dpos_e, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        qL_des = ik_ctrl.compute(ee_pos_b, ee_quat_b, J, franka.data.joint_pos[:, arm_ids])

        # Build actions
        actions = torch.zeros((num_envs, act_dim), device=device)
        actions[:, 0:7] = qL_des
        # left gripper from grip_cmd
        f_target = F_CLOSED if grip_cmd else F_OPEN
        f_vec = torch.full((num_envs, 1), f_target, device=device)
        actions[:, 7:9] = f_vec.expand(-1, 2)

        env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
