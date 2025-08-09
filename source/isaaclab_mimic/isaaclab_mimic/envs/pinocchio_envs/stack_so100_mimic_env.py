# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class StackSO100MimicEnv(ManagerBasedRLMimicEnv):

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector (for SO100, this is typically "gripper").
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # For SO100, the end effector pose is stored in policy observations
        eef_pos_name = "eef_pos"
        eef_quat_name = "eef_quat"

        target_eef_position = self.obs_buf["policy"][eef_pos_name][env_ids]
        target_rot_mat = PoseUtils.matrix_from_quat(self.obs_buf["policy"][eef_quat_name][env_ids])

        return PoseUtils.make_pose(target_eef_position, target_rot_mat)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary containing target eef pose. For SO100, expects key "gripper".
            gripper_action_dict: Dictionary containing gripper action. For SO100, expects key "gripper".
            action_noise_dict: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """

        # target position and rotation for single arm
        target_eef_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["gripper"])
        target_eef_rot_quat = PoseUtils.quat_from_matrix(target_rot)

        # gripper action
        gripper_action = gripper_action_dict["gripper"]

        if action_noise_dict is not None:
            pos_noise = action_noise_dict["gripper"] * torch.randn_like(target_eef_pos)
            quat_noise = action_noise_dict["gripper"] * torch.randn_like(target_eef_rot_quat)

            target_eef_pos += pos_noise
            target_eef_rot_quat += quat_noise

        return torch.cat(
            (
                target_eef_pos,
                target_eef_rot_quat,
                gripper_action,
            ),
            dim=0,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim).

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to.
        """
        target_poses = {}

        # For SO100: action dimensions are [pos(3), quat(4), gripper(1)]
        target_eef_position = action[:, 0:3]
        target_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_pose = PoseUtils.make_pose(target_eef_position, target_rot_mat)
        target_poses["gripper"] = target_pose

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key is "gripper".
        """
        print("actions", actions)
        print("actions", actions.shape)
        return {"gripper": actions[:, -1:]}  # Gripper is at index 7 (single dimension)

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
        return signals 