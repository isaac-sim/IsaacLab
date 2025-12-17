# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from dataclasses import dataclass

import isaaclab.utils.math as PoseUtils

from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class FiiRetargeter(RetargeterBase):

    def __init__(self, cfg: "FiiRetargeterCfg"):
        """Initialize the retargeter."""
        self.cfg = cfg
        self._sim_device = cfg.sim_device

    def retarget(self, data: dict) -> torch.Tensor:

        base_vel = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self._sim_device)
        base_height = torch.tensor([0.7], dtype=torch.float32, device=self._sim_device)

        left_eef_pose = torch.tensor(
            [-0.3, 0.3, 0.72648, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self._sim_device
        )
        right_eef_pose = torch.tensor(
            [-0.3, 0.3, 0.72648, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self._sim_device
        )

        left_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_RIGHT]
        left_wrist = left_hand_poses.get("wrist")
        right_wrist = right_hand_poses.get("wrist")

        left_eef_pose_np = self._retarget_abs(left_wrist, is_left=True)
        right_eef_pose_np = self._retarget_abs(right_wrist, is_left=False)

        if left_wrist is not None:
            left_eef_pose = torch.from_numpy(left_eef_pose_np).to(dtype=torch.float32, device=self._sim_device)
        if right_wrist is not None:
            right_eef_pose = torch.from_numpy(right_eef_pose_np).to(dtype=torch.float32, device=self._sim_device)

        gripper_value_left = self._hand_data_to_gripper_values(data[OpenXRDevice.TrackingTarget.HAND_LEFT])
        gripper_value_right = self._hand_data_to_gripper_values(data[OpenXRDevice.TrackingTarget.HAND_RIGHT])

        return torch.cat(
            [left_eef_pose, right_eef_pose, gripper_value_left, gripper_value_right, base_vel, base_height]
        )

    def _hand_data_to_gripper_values(self, hand_data):
        thumb_tip = hand_data["thumb_tip"]
        index_tip = hand_data["index_tip"]

        distance = np.linalg.norm(thumb_tip[:3] - index_tip[:3])

        finger_dist_closed = 0.00
        finger_dist_open = 0.06

        gripper_value_closed = 0.06
        gripper_value_open = 0.00

        t = np.clip((distance - finger_dist_closed) / (finger_dist_open - finger_dist_closed), 0, 1)
        # t = 1 -> open
        # t = 0 -> closed
        gripper_joint_value = (1.0 - t) * gripper_value_closed + t * gripper_value_open

        return torch.tensor([gripper_joint_value, gripper_joint_value], dtype=torch.float32, device=self._sim_device)

    def _retarget_abs(self, wrist: np.ndarray, is_left: bool) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            wrist: Wrist pose data from OpenXR.
            is_left: True for the left hand, False for the right hand.

        Returns:
            Retargeted wrist pose in USD control frame.
        """
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))

        if is_left:
            # Corresponds to a rotation of (0, 90, 90) in euler angles (x,y,z)
            combined_quat = torch.tensor([0, 0.7071, 0, 0.7071], dtype=torch.float32)
        else:
            # Corresponds to a rotation of (0, -90, -90) in euler angles (x,y,z)
            combined_quat = torch.tensor([0, -0.7071, 0, 0.7071], dtype=torch.float32)

        offset = torch.tensor([0.0, 0.25, -0.15])
        transform_pose = PoseUtils.make_pose(offset, PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])


@dataclass
class FiiRetargeterCfg(RetargeterCfg):
    retargeter_type: type[RetargeterBase] = FiiRetargeter
