# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class FrankaCubeStackIKAbsMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Franka Cube Stack IK Abs env.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose."""
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self, target_eef_pose_dict: dict, gripper_action_dict: dict, noise: float | None = None, env_id: int = 0
    ) -> torch.Tensor:
        """Convert target pose to action.

        This method transforms a dictionary of target end-effector poses and gripper actions
        into a single action tensor that can be used by the environment.

        The function:
        1. Extracts target position and rotation from the pose dictionary
        2. Extracts gripper action for the end effector
        3. Concatenates position and quaternion rotation into a pose action
        4. Optionally adds noise to the pose action for exploration
        5. Combines pose action with gripper action into a final action tensor

        Args:
            target_eef_pose_dict: Dictionary containing target end-effector pose(s),
                with keys as eef names and values as pose tensors.
            gripper_action_dict: Dictionary containing gripper action(s),
                with keys as eef names and values as action tensors.
            noise: Optional noise magnitude to apply to the pose action for exploration.
                If provided, random noise is generated and added to the pose action.
            env_id: Environment ID for multi-environment setups, defaults to 0.

        Returns:
            torch.Tensor: A single action tensor combining pose and gripper commands.
        """
        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0)
        if noise is not None:
            noise = noise * torch.randn_like(pose_action)
            pose_action += noise

        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert action to target pose."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        target_pos = action[:, :3]
        target_quat = action[:, 3:7]
        target_rot = PoseUtils.matrix_from_quat(target_quat)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions."""
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals."""
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
        signals["grasp_2"] = subtask_terms["grasp_2"][env_ids]
        signals["stack_1"] = subtask_terms["stack_1"][env_ids]
        return signals
