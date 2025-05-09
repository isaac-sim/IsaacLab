# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
import sys

# Import pinocchio in the main script to force the use of the dependencies installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controller
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import torch
import unittest

from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix, quat_inv

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class TestPinkIKController(unittest.TestCase):
    """Test fixture for the Pink IK controller with the GR1T2 humanoid robot.

    This test validates that the Pink IK controller can accurately track commanded
    end-effector poses for a humanoid robot. It specifically:

    1. Creates a GR1T2 humanoid robot with the Pink IK controller
    2. Sends target pose commands to the left and right hand roll links
    3. Checks that the observed poses of the links match the target poses within tolerance
    4. Tests adaptability by moving the hands up and down multiple times

    The test succeeds when the controller can accurately converge to each new target
    position, demonstrating both accuracy and adaptability to changing targets.
    """

    def setUp(self):

        # End effector position mean square error tolerance in meters
        self.pos_tolerance = 0.02  # 2 cm
        # End effector orientation mean square error tolerance in radians
        self.rot_tolerance = 0.17  # 10 degrees

        # Number of environments
        self.num_envs = 1

        # Number of joints in the 2 robot hands
        self.num_joints_in_robot_hands = 22

        # Number of steps to wait for controller convergence
        self.num_steps_controller_convergence = 25

        self.num_times_to_move_hands_up = 3
        self.num_times_to_move_hands_down = 3

        # Create starting setpoints with respect to the env origin frame
        # These are the setpoints for the forward kinematics result of the
        # InitialStateCfg specified in `PickPlaceGR1T2EnvCfg`
        y_axis_z_axis_90_rot_quaternion = [0.5, 0.5, -0.5, 0.5]
        left_hand_roll_link_pos = [-0.23, 0.28, 1.1]
        self.left_hand_roll_link_pose = left_hand_roll_link_pos + y_axis_z_axis_90_rot_quaternion
        right_hand_roll_link_pos = [0.23, 0.28, 1.1]
        self.right_hand_roll_link_pose = right_hand_roll_link_pos + y_axis_z_axis_90_rot_quaternion

    """
    Test fixtures.
    """

    def test_gr1t2_ik_pose_abs(self):
        """Test IK controller for GR1T2 humanoid."""

        env_name = "Isaac-PickPlace-GR1T2-Abs-v0"
        device = "cuda:0"
        env_cfg = parse_env_cfg(env_name, device=device, num_envs=self.num_envs)

        # create environment from loaded config
        env = gym.make(env_name, cfg=env_cfg).unwrapped

        # reset before starting
        obs, _ = env.reset()

        num_runs = 0  # Counter for the number of runs

        move_hands_up = True
        test_counter = 0

        # simulate environment -- run everything in inference mode
        with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():

                num_runs += 1
                setpoint_poses = self.left_hand_roll_link_pose + self.right_hand_roll_link_pose
                actions = setpoint_poses + [0.0] * self.num_joints_in_robot_hands
                actions = torch.tensor(actions, device=device)
                actions = torch.stack([actions for _ in range(env.num_envs)])

                obs, _, _, _, _ = env.step(actions)

                left_hand_roll_link_pose_obs = obs["policy"]["robot_links_state"][
                    :, env.scene["robot"].data.body_names.index("left_hand_roll_link"), :7
                ]
                right_hand_roll_link_pose_obs = obs["policy"]["robot_links_state"][
                    :, env.scene["robot"].data.body_names.index("right_hand_roll_link"), :7
                ]

                # The setpoints are wrt the env origin frame
                # The observations are also wrt the env origin frame
                left_hand_roll_link_feedback = left_hand_roll_link_pose_obs
                left_hand_roll_link_setpoint = (
                    torch.tensor(self.left_hand_roll_link_pose, device=device).unsqueeze(0).repeat(env.num_envs, 1)
                )
                left_hand_roll_link_pos_error = (
                    left_hand_roll_link_setpoint[:, :3] - left_hand_roll_link_feedback[:, :3]
                )
                left_hand_roll_link_rot_error = axis_angle_from_quat(
                    quat_from_matrix(
                        matrix_from_quat(left_hand_roll_link_setpoint[:, 3:])
                        * matrix_from_quat(quat_inv(left_hand_roll_link_feedback[:, 3:]))
                    )
                )

                right_hand_roll_link_feedback = right_hand_roll_link_pose_obs
                right_hand_roll_link_setpoint = (
                    torch.tensor(self.right_hand_roll_link_pose, device=device).unsqueeze(0).repeat(env.num_envs, 1)
                )
                right_hand_roll_link_pos_error = (
                    right_hand_roll_link_setpoint[:, :3] - right_hand_roll_link_feedback[:, :3]
                )
                right_hand_roll_link_rot_error = axis_angle_from_quat(
                    quat_from_matrix(
                        matrix_from_quat(right_hand_roll_link_setpoint[:, 3:])
                        * matrix_from_quat(quat_inv(right_hand_roll_link_feedback[:, 3:]))
                    )
                )

                if num_runs % self.num_steps_controller_convergence == 0:
                    # Check if the left hand roll link is at the target position
                    torch.testing.assert_close(
                        torch.mean(torch.abs(left_hand_roll_link_pos_error), dim=1),
                        torch.zeros(env.num_envs, device="cuda:0"),
                        rtol=0.0,
                        atol=self.pos_tolerance,
                    )

                    # Check if the right hand roll link is at the target position
                    torch.testing.assert_close(
                        torch.mean(torch.abs(right_hand_roll_link_pos_error), dim=1),
                        torch.zeros(env.num_envs, device="cuda:0"),
                        rtol=0.0,
                        atol=self.pos_tolerance,
                    )

                    # Check if the left hand roll link is at the target orientation
                    torch.testing.assert_close(
                        torch.mean(torch.abs(left_hand_roll_link_rot_error), dim=1),
                        torch.zeros(env.num_envs, device="cuda:0"),
                        rtol=0.0,
                        atol=self.rot_tolerance,
                    )

                    # Check if the right hand roll link is at the target orientation
                    torch.testing.assert_close(
                        torch.mean(torch.abs(right_hand_roll_link_rot_error), dim=1),
                        torch.zeros(env.num_envs, device="cuda:0"),
                        rtol=0.0,
                        atol=self.rot_tolerance,
                    )

                    # Change the setpoints to move the hands up and down as per the counter
                    test_counter += 1
                    if move_hands_up and test_counter > self.num_times_to_move_hands_up:
                        move_hands_up = False
                    elif not move_hands_up and test_counter > (
                        self.num_times_to_move_hands_down + self.num_times_to_move_hands_up
                    ):
                        # Test is done after moving the hands up and down
                        break
                    if move_hands_up:
                        self.left_hand_roll_link_pose[1] += 0.05
                        self.left_hand_roll_link_pose[2] += 0.05
                        self.right_hand_roll_link_pose[1] += 0.05
                        self.right_hand_roll_link_pose[2] += 0.05
                    else:
                        self.left_hand_roll_link_pose[1] -= 0.05
                        self.left_hand_roll_link_pose[2] -= 0.05
                        self.right_hand_roll_link_pose[1] -= 0.05
                        self.right_hand_roll_link_pose[2] -= 0.05

        env.close()
