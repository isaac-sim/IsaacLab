# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

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

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---- Launch Isaac Sim ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.devices import Se3Gamepad
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


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
    franka_L = env.unwrapped.robot_left
    franka_R = env.unwrapped.robot_right

    left_arm_ids = franka_L.find_joints(["panda_joint[1-7]"])[0]
    left_finger_ids = franka_L.find_joints(["panda_finger_joint.*"])[0]
    right_arm_ids = franka_R.find_joints(["panda_joint[1-7]"])[0]
    right_finger_ids = franka_R.find_joints(["panda_finger_joint.*"])[0]

    # Gripper (binary)
    lims_low = franka_L.data.soft_joint_pos_limits[0, left_finger_ids, 0]
    lims_high = franka_L.data.soft_joint_pos_limits[0, left_finger_ids, 1]
    F_OPEN = float(torch.min(lims_high).clamp(min=0.03, max=0.05))
    F_CLOSED = float(torch.max(lims_low)) + 0.001

    # Right arm hold
    qR_hold = franka_R.data.joint_pos[:, right_arm_ids].clone()
    qRf_hold = franka_R.data.joint_pos[:, right_finger_ids].clone()

    num_envs = env.unwrapped.num_envs
    act_dim = env.action_space.shape[-1]

    # Hand link / Jacobian index
    hand_body_id = franka_L.find_bodies([".*hand"])[0][0]
    hand_jac_idx = hand_body_id - 1 if franka_L.is_fixed_base else hand_body_id
    left_arm_ids = franka_L.find_joints(["panda_joint[1-7]"])[0]

    teleop = Se3Gamepad(pos_sensitivity=0.1, rot_sensitivity=0.16, dead_zone=0.07)

    teleop.reset()

    # Relative, position-only DIK
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",  # position only
        use_relative_mode=True,  # deltas in EE frame
        ik_method="dls",
        ik_params={"lambda_val": 0.15},
    )
    ik_ctrl = DifferentialIKController(
        ik_cfg, num_envs=args_cli.num_envs, device=args_cli.device
    )

    # Start by holding current joints
    qL_des = franka_L.data.joint_pos[:, left_arm_ids].clone()

    # ---- Main loop ----
    while simulation_app.is_running():
        delta_pose, grip_cmd = teleop.advance()

        # current base & hand pose
        base_pos_w = franka_L.data.root_pose_w[:, :3]
        base_quat_w = franka_L.data.root_pose_w[:, 3:7]
        hand_pos_w = franka_L.data.body_pos_w[:, hand_body_id]
        hand_quat_w = franka_L.data.body_quat_w[:, hand_body_id]

        # EE (hand) in BASE (state for IK)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pos_w, base_quat_w, hand_pos_w, hand_quat_w
        )

        # --- joystick is ALREADY in EE frame ---
        delta = torch.tensor(delta_pose, device=ee_pos_b.device, dtype=ee_pos_b.dtype)
        dpos_e = (
            delta[:3].unsqueeze(0).expand_as(ee_pos_b)
        )  # (N,3) relative EE-frame step

        # Jacobian at hand
        J = franka_L.root_physx_view.get_jacobians()[:, hand_jac_idx, :, left_arm_ids]

        # relative, position-only IK
        ik_ctrl.set_command(dpos_e, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        qL_des = ik_ctrl.compute(
            ee_pos_b, ee_quat_b, J, franka_L.data.joint_pos[:, left_arm_ids]
        )

        # Actions
        actions = torch.zeros((num_envs, act_dim), device=env.unwrapped.device)
        actions[:, 0:7] = qL_des
        f_target = F_CLOSED if grip_cmd else F_OPEN
        f_target_vec = torch.full((num_envs, 1), f_target, device=env.unwrapped.device)
        actions[:, 7:9] = f_target_vec.expand(-1, 2)
        actions[:, 9:16] = qR_hold
        actions[:, 16:18] = qRf_hold

        env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
