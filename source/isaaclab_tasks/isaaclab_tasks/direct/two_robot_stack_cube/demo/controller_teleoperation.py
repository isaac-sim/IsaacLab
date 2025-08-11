# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import time
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

    # Gripper (binary) – use left's limits; both Frankas share limits
    lims_low = franka_L.data.soft_joint_pos_limits[0, left_finger_ids, 0]
    lims_high = franka_L.data.soft_joint_pos_limits[0, left_finger_ids, 1]
    F_OPEN = float(torch.min(lims_high).clamp(min=0.03, max=0.05))
    F_CLOSED = float(torch.max(lims_low)) + 0.001

    num_envs = env.unwrapped.num_envs
    act_dim = env.action_space.shape[-1]

    # Hand links / Jacobian indices
    left_hand_body_id = franka_L.find_bodies([".*hand"])[0][0]
    left_hand_jac_idx = (
        left_hand_body_id - 1 if franka_L.is_fixed_base else left_hand_body_id
    )
    right_hand_body_id = franka_R.find_bodies([".*hand"])[0][0]
    right_hand_jac_idx = (
        right_hand_body_id - 1 if franka_R.is_fixed_base else right_hand_body_id
    )

    # Input device (EE-frame deltas)
    teleop = Se3Gamepad(pos_sensitivity=0.1, rot_sensitivity=0.16, dead_zone=0.07)
    teleop.reset()

    # Relative, position-only IK (shared controller)
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",  # 3D position only
        use_relative_mode=True,  # deltas in EE frame
        ik_method="dls",
        ik_params={"lambda_val": 0.15},
    )
    ik_ctrl = DifferentialIKController(
        ik_cfg, num_envs=args_cli.num_envs, device=args_cli.device
    )

    # Start by controlling LEFT
    active_left = True
    print("[TELEOP] Active robot: LEFT  (double-tap GRIP to switch)")

    # Hold targets for the inactive arm/fingers
    qL_hold = franka_L.data.joint_pos[:, left_arm_ids].clone()
    qLf_hold = franka_L.data.joint_pos[:, left_finger_ids].clone()
    qR_hold = franka_R.data.joint_pos[:, right_arm_ids].clone()
    qRf_hold = franka_R.data.joint_pos[:, right_finger_ids].clone()

    # Double-tap detection on gripper button
    prev_grip = False
    last_rise_time = 0.0
    DOUBLE_TAP_WINDOW = 0.35  # seconds

    # ---- Main loop ----
    while simulation_app.is_running():
        delta_pose, grip_cmd = teleop.advance()

        # Detect double-tap on the gripper button to toggle active robot
        now = time.time()
        switch_now = False
        if grip_cmd and not prev_grip:  # rising edge
            if now - last_rise_time <= DOUBLE_TAP_WINDOW:
                switch_now = True
                last_rise_time = 0.0
            else:
                last_rise_time = now
        prev_grip = grip_cmd

        if switch_now:
            active_left = not active_left
            side = "LEFT" if active_left else "RIGHT"
            print(f"[TELEOP] Switched active robot -> {side}")

        # Read current states for both robots
        # LEFT
        L_base_pos_w = franka_L.data.root_pose_w[:, :3]
        L_base_quat_w = franka_L.data.root_pose_w[:, 3:7]
        L_hand_pos_w = franka_L.data.body_pos_w[:, left_hand_body_id]
        L_hand_quat_w = franka_L.data.body_quat_w[:, left_hand_body_id]
        L_ee_pos_b, L_ee_quat_b = subtract_frame_transforms(
            L_base_pos_w, L_base_quat_w, L_hand_pos_w, L_hand_quat_w
        )
        # RIGHT
        R_base_pos_w = franka_R.data.root_pose_w[:, :3]
        R_base_quat_w = franka_R.data.root_pose_w[:, 3:7]
        R_hand_pos_w = franka_R.data.body_pos_w[:, right_hand_body_id]
        R_hand_quat_w = franka_R.data.body_quat_w[:, right_hand_body_id]
        R_ee_pos_b, R_ee_quat_b = subtract_frame_transforms(
            R_base_pos_w, R_base_quat_w, R_hand_pos_w, R_hand_quat_w
        )

        # Build command for the ACTIVE robot only (EE-frame deltas already)
        delta = torch.tensor(
            delta_pose, device=env.unwrapped.device, dtype=L_ee_pos_b.dtype
        )
        dpos_e = delta[:3].unsqueeze(0).expand(num_envs, -1)  # (N,3)

        # Compute IK for the active side
        if active_left:
            J = franka_L.root_physx_view.get_jacobians()[
                :, left_hand_jac_idx, :, left_arm_ids
            ]
            ik_ctrl.set_command(dpos_e, ee_pos=L_ee_pos_b, ee_quat=L_ee_quat_b)
            qL_des = ik_ctrl.compute(
                L_ee_pos_b, L_ee_quat_b, J, franka_L.data.joint_pos[:, left_arm_ids]
            )

            # Inactive RIGHT: refresh hold at current joint pos
            qR_hold = franka_R.data.joint_pos[:, right_arm_ids].clone()
            qRf_hold = franka_R.data.joint_pos[:, right_finger_ids].clone()
        else:
            J = franka_R.root_physx_view.get_jacobians()[
                :, right_hand_jac_idx, :, right_arm_ids
            ]
            ik_ctrl.set_command(dpos_e, ee_pos=R_ee_pos_b, ee_quat=R_ee_quat_b)
            qR_des = ik_ctrl.compute(
                R_ee_pos_b, R_ee_quat_b, J, franka_R.data.joint_pos[:, right_arm_ids]
            )

            # Inactive LEFT: refresh hold at current joint pos
            qL_hold = franka_L.data.joint_pos[:, left_arm_ids].clone()
            qLf_hold = franka_L.data.joint_pos[:, left_finger_ids].clone()

        # Actions (N,18): [L arm(0:7), L grip(7:9), R arm(9:16), R grip(16:18)]
        actions = torch.zeros((num_envs, act_dim), device=env.unwrapped.device)

        # Active side gets IK result; inactive side gets hold
        if active_left:
            actions[:, 0:7] = qL_des
            actions[:, 9:16] = qR_hold
        else:
            actions[:, 0:7] = qL_hold
            actions[:, 9:16] = qR_des

        # Gripper command goes to the ACTIVE robot; inactive keeps its hold
        if active_left:
            f_active = F_CLOSED if grip_cmd else F_OPEN
            f_vec = torch.full((num_envs, 1), f_active, device=env.unwrapped.device)
            actions[:, 7:9] = f_vec.expand(-1, 2)  # left fingers
            actions[:, 16:18] = qRf_hold  # right fingers hold
        else:
            f_active = F_CLOSED if grip_cmd else F_OPEN
            f_vec = torch.full((num_envs, 1), f_active, device=env.unwrapped.device)
            actions[:, 7:9] = qLf_hold  # left fingers hold
            actions[:, 16:18] = f_vec.expand(-1, 2)  # right fingers

        env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
