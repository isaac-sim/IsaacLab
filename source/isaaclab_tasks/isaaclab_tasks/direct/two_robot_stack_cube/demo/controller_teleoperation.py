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

from carb.input import GamepadInput  # <-- for callbacks

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

    # IDs
    L_arm_ids = franka_L.find_joints(["panda_joint[1-7]"])[0]
    L_fing_ids = franka_L.find_joints(["panda_finger_joint.*"])[0]
    R_arm_ids = franka_R.find_joints(["panda_joint[1-7]"])[0]
    R_fing_ids = franka_R.find_joints(["panda_finger_joint.*"])[0]

    # Gripper (binary)
    lims_low = franka_L.data.soft_joint_pos_limits[0, L_fing_ids, 0]
    lims_high = franka_L.data.soft_joint_pos_limits[0, L_fing_ids, 1]
    F_OPEN = float(torch.min(lims_high).clamp(min=0.03, max=0.05))
    F_CLOSED = float(torch.max(lims_low)) + 0.001

    num_envs = env.unwrapped.num_envs
    act_dim = env.action_space.shape[-1]
    device = env.unwrapped.device

    # Hand link / Jacobian index
    L_hand_body = franka_L.find_bodies([".*hand"])[0][0]
    R_hand_body = franka_R.find_bodies([".*hand"])[0][0]
    L_hand_jac = L_hand_body - 1 if franka_L.is_fixed_base else L_hand_body
    R_hand_jac = R_hand_body - 1 if franka_R.is_fixed_base else R_hand_body

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
    ik_ctrl = DifferentialIKController(
        ik_cfg, num_envs=args_cli.num_envs, device=args_cli.device
    )

    # State
    state = {"active": "left"}  # "left" or "right"

    # Latches to hold the inactive arm
    holds = {
        "L_q": franka_L.data.joint_pos[:, L_arm_ids].clone(),
        "L_qf": franka_L.data.joint_pos[:, L_fing_ids].clone(),
        "R_q": franka_R.data.joint_pos[:, R_arm_ids].clone(),
        "R_qf": franka_R.data.joint_pos[:, R_fing_ids].clone(),
    }

    # ---- Callback: press RIGHT SHOULDER to swap active arm ----
    def on_swap_arm():
        if state["active"] == "left":
            # before switching away, latch current left to hold
            holds["L_q"] = franka_L.data.joint_pos[:, L_arm_ids].clone()
            holds["L_qf"] = franka_L.data.joint_pos[:, L_fing_ids].clone()
            state["active"] = "right"
            print("[Teleop] Switched control → RIGHT arm")
        else:
            holds["R_q"] = franka_R.data.joint_pos[:, R_arm_ids].clone()
            holds["R_qf"] = franka_R.data.joint_pos[:, R_fing_ids].clone()
            state["active"] = "left"
            print("[Teleop] Switched control → LEFT arm")

    # Bind callback (change to any GamepadInput.* you like, e.g., GamepadInput.Y)
    teleop.add_callback(GamepadInput.RIGHT_SHOULDER, on_swap_arm)

    # Start by holding current joints for both, but we'll compute only active side each frame
    qL_des = franka_L.data.joint_pos[:, L_arm_ids].clone()
    qR_des = franka_R.data.joint_pos[:, R_arm_ids].clone()

    # ---- Main loop ----
    while simulation_app.is_running():
        delta_pose, grip_cmd = teleop.advance()

        # Active arm selection
        if state["active"] == "left":
            # Bases & hand pose
            base_pos_w = franka_L.data.root_pose_w[:, :3]
            base_quat_w = franka_L.data.root_pose_w[:, 3:7]
            hand_pos_w = franka_L.data.body_pos_w[:, L_hand_body]
            hand_quat_w = franka_L.data.body_quat_w[:, L_hand_body]

            # EE (hand) in base
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w, hand_pos_w, hand_quat_w
            )

            # Device delta already in EE frame → position-only relative step
            delta = torch.tensor(
                delta_pose, device=ee_pos_b.device, dtype=ee_pos_b.dtype
            )
            dpos_e = delta[:3].unsqueeze(0).expand_as(ee_pos_b)  # (N,3)

            # Jacobian and IK
            J = franka_L.root_physx_view.get_jacobians()[:, L_hand_jac, :, L_arm_ids]
            ik_ctrl.set_command(dpos_e, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
            qL_des = ik_ctrl.compute(
                ee_pos_b, ee_quat_b, J, franka_L.data.joint_pos[:, L_arm_ids]
            )

            # Right arm stalled at latched
            qR_des = holds["R_q"]

            # Build actions
            actions = torch.zeros((num_envs, act_dim), device=device)
            actions[:, 0:7] = qL_des
            # left gripper from grip_cmd
            f_target = F_CLOSED if grip_cmd else F_OPEN
            f_vec = torch.full((num_envs, 1), f_target, device=device)
            actions[:, 7:9] = f_vec.expand(-1, 2)
            # right hold
            actions[:, 9:16] = qR_des
            actions[:, 16:18] = holds["R_qf"]

        else:  # active == "right"
            base_pos_w = franka_R.data.root_pose_w[:, :3]
            base_quat_w = franka_R.data.root_pose_w[:, 3:7]
            hand_pos_w = franka_R.data.body_pos_w[:, R_hand_body]
            hand_quat_w = franka_R.data.body_quat_w[:, R_hand_body]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w, hand_pos_w, hand_quat_w
            )

            delta = torch.tensor(
                delta_pose, device=ee_pos_b.device, dtype=ee_pos_b.dtype
            )
            dpos_e = delta[:3].unsqueeze(0).expand_as(ee_pos_b)

            J = franka_R.root_physx_view.get_jacobians()[:, R_hand_jac, :, R_arm_ids]
            ik_ctrl.set_command(dpos_e, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
            qR_des = ik_ctrl.compute(
                ee_pos_b, ee_quat_b, J, franka_R.data.joint_pos[:, R_arm_ids]
            )

            # Left arm stalled
            qL_des = holds["L_q"]

            actions = torch.zeros((num_envs, act_dim), device=device)
            # left hold
            actions[:, 0:7] = qL_des
            actions[:, 7:9] = holds["L_qf"]
            # right controlled
            actions[:, 9:16] = qR_des
            # right gripper from grip_cmd
            f_target = F_CLOSED if grip_cmd else F_OPEN
            f_vec = torch.full((num_envs, 1), f_target, device=device)
            actions[:, 16:18] = f_vec.expand(-1, 2)

        env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
