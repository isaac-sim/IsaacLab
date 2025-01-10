# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from isaaclab_tasks.manager_based.manipulation.stack.mdp import cubes_stacked


class FrankaCubeStackIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Franka Cube Stack IK Rel env.
    """

    def get_robot_eef_pose(self, env_ind=0):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (torch.Tensor): 4x4 eef pose matrix
        """

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ind]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ind]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(self, target_eef_pose, relative=True, env_ind=0):
        """
        Takes a target pose for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.

        Args:
            target_eef_pose (torch.Tensor): 4x4 target eef pose
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            action (torch.Tensor): action compatible with env.step (minus gripper actuation)
        """

        # target position and rotation
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(env_ind=env_ind)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        if relative:
            # normalized delta position action
            delta_position = target_pos - curr_pos
            # delta_position = np.clip(delta_position / max_dpos, -1., 1.)

            # normalized delta rotation action
            delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
            delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
            delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

            # delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
            return torch.cat([delta_position, delta_rotation], dim=0)
        else:
            raise NotImplementedError("Absolute pose actions are not implemented.")
            return

    def action_to_target_eef_pos(self, action, relative=True, env_ind=0):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (torch.Tensor): environment action
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            target_eef_pose (torch.Tensor): 4x4 target eef pose that @action corresponds to
        """

        target_poses = []

        for env_ind in range(self.scene.num_envs):

            delta_position = action[env_ind][:3]
            delta_rotation = action[env_ind][3:6]

            # current position and rotation
            curr_pose = self.get_robot_eef_pose(env_ind=env_ind)
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            # get pose target
            target_pos = curr_pos + delta_position

            # Convert delta_rotation to axis angle form
            delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
            # make sure that axis is a unit vector
            # Check for invalid division
            if torch.isclose(delta_rotation_angle, torch.tensor([0.0], device=delta_rotation_angle.device)):
                # Quaternion format is wxyz
                delta_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=delta_rotation_angle.device)
            else:
                delta_rotation_axis = delta_rotation / delta_rotation_angle
                delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle, delta_rotation_axis).squeeze(0)
            delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)

            target_rot = torch.matmul(delta_rot_mat, curr_rot)

            target_pose = PoseUtils.make_pose(target_pos, target_rot).clone()

            target_poses.append(target_pose)
        return target_poses

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (torch.Tensor): environment action of shape N x action_dim. Where N is number of steps in a demo

        Returns:
            gripper_action (torch.Tensor): subset of environment action for gripper actuation of shape N x gripper_action_dim
        """

        # last dimension is gripper action
        return action[:, -1:]

    def get_subtask_term_signals(self, env_ind=0):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. Isaac Lab Mimic only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.
        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_1"] = subtask_terms["grasp_1"][env_ind]
        signals["grasp_2"] = subtask_terms["grasp_2"][env_ind]
        signals["stack_1"] = subtask_terms["stack_1"][env_ind]
        # final subtask is placing cubeC on cubeA (motion relative to cubeA) - but final subtask signal is not needed
        return signals

    def is_success(self):
        return cubes_stacked(self, atol=0.001, rtol=0.001)
