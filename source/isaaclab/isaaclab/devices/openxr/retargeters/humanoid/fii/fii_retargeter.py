# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from dataclasses import dataclass

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

        if left_wrist is not None:
            left_eef_pose = torch.tensor(torch.from_numpy(left_wrist), dtype=torch.float32, device=self._sim_device)
            left_eef_pose[2] = left_eef_pose[2]
        if right_wrist is not None:
            right_eef_pose = torch.tensor(torch.from_numpy(right_wrist), dtype=torch.float32, device=self._sim_device)
            right_eef_pose[2] = right_eef_pose[2]

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


@dataclass
class FiiRetargeterCfg(RetargeterCfg):
    retargeter_type: type[RetargeterBase] = FiiRetargeter
