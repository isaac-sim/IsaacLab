# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic environment wrapper for the OpenArm pick-up (IK relative) task.

The OpenArm "IK-Abs" task uses use_relative_mode=True in the differential IK controller,
so the action format is a RELATIVE delta pose (same pattern as Franka IK-Rel):

  action (14D):
    [0:3]   left arm delta position  (dx, dy, dz)
    [3:6]   left arm delta rotation  (axis-angle, rad)
    [6]     left gripper command     (+1.0 = open, -1.0 = close)
    [7:13]  right arm delta pose     (zeroed — right arm is idle during pick-up)
    [13]    right gripper command    (+1.0 = open, held constant)

Mimic only controls the LEFT arm; the right arm slots are zeroed so the right arm
stays at its reset pose throughout generation.
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

# Left arm occupies indices [0:7], right arm [7:14]
_LEFT_DELTA = slice(0, 6)   # left IK delta (pos + rot)
_LEFT_GRP = 6               # left gripper scalar index
_ACTION_DIM = 14            # full flat action dimension


class OpenArmPickUpIKAbsMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic wrapper for Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0.

    Controls only the LEFT arm; right arm action is zeroed throughout.
    Reads EEF pose from the 'policy' observation group (eef_pos, eef_quat)
    which are populated by the FrameTransformer tracking openarm_left_ee_tcp.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return left-arm EEF pose as a (N, 4, 4) homogeneous matrix.

        Reads eef_pos / eef_quat from the policy observation buffer.
        Quaternion convention: w, x, y, z.
        """
        if env_ids is None:
            env_ids = slice(None)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert a target left-arm EEF pose → 14D environment action.

        Computes the delta (relative) pose from the current EEF to the target,
        encodes it as [delta_pos(3), axis_angle(3), gripper(1)] in the left-arm
        slots; the right-arm slots are zeroed (right arm stays still).

        Returns shape (7,) — the 7D left-arm+gripper part is embedded in a 14D
        tensor but only the first 7 elements are filled; the caller does not need
        to split.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos

        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action = pose_action + noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        # Build full 14D action: left arm [0:7], right arm [7:14] = zeros
        full = torch.zeros(_ACTION_DIM, dtype=pose_action.dtype, device=pose_action.device)
        full[_LEFT_DELTA] = pose_action
        full[_LEFT_GRP] = gripper_action.squeeze()
        # Keep right gripper open so it doesn't accidentally interfere
        full[13] = 1.0

        return full

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert 14D env actions → left-arm target EEF poses (N, 4, 4).

        For a relative controller, target = current_pose ⊕ delta.
        Only the left-arm delta (action[:, 0:6]) is consumed.
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]    # (N, 3)
        delta_rotation = action[:, 3:6]   # (N, 3)  axis-angle

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero_angle = torch.isclose(
            delta_rotation_angle, torch.zeros_like(delta_rotation_angle)
        ).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(
            delta_rotation_angle.squeeze(1), delta_rotation_axis
        ).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract left gripper command from a batch of actions.

        Left gripper is at index 6 in the 14D action vector.
        """
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, _LEFT_GRP:_LEFT_GRP + 1]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return subtask termination signals for auto-annotation.

        Only "grasp" is returned — the final subtask (lift) has no explicit
        term signal; it ends when the episode success condition is met.
        """
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {"grasp": subtask_terms["grasp"][env_ids]}
