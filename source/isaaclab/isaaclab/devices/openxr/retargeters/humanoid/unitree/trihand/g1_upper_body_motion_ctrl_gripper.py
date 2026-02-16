# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class G1TriHandUpperBodyMotionControllerGripperRetargeter(RetargeterBase):
    """Retargeter for G1 gripper that outputs a boolean state based on controller trigger input,
    concatenated with the retargeted wrist pose.

    Gripper:
    - Uses hysteresis to prevent flickering when the trigger is near the threshold.
    - Output is 0.0 for open, 1.0 for close.

    Wrist:
    - Retargets absolute pose from controller to robot frame.
    - Applies a fixed offset rotation for comfort/alignment.
    """

    def __init__(self, cfg: G1TriHandUpperBodyMotionControllerGripperRetargeterCfg):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter.
        """
        super().__init__(cfg)
        self._cfg = cfg
        # Track previous state for hysteresis (left, right)
        self._prev_left_state: float = 0.0
        self._prev_right_state: float = 0.0

    def retarget(self, data: dict) -> torch.Tensor:
        """Retarget controller inputs to gripper boolean state and wrist pose.

        Args:
            data: Dictionary with MotionControllerTrackingTarget.LEFT/RIGHT keys
                 Each value is a 2D array: [pose(7), inputs(7)]

        Returns:
            Tensor: [left_gripper_state(1), right_gripper_state(1), left_wrist(7), right_wrist(7)]
            Wrist format: [x, y, z, qw, qx, qy, qz]
        """
        # Get controller data
        left_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))

        # --- Gripper Logic ---
        # Extract hand state from controller data with hysteresis
        left_hand_state: float = self._extract_hand_state(left_controller_data, self._prev_left_state)
        right_hand_state: float = self._extract_hand_state(right_controller_data, self._prev_right_state)

        # Update previous states
        self._prev_left_state = left_hand_state
        self._prev_right_state = right_hand_state

        gripper_tensor = torch.tensor([left_hand_state, right_hand_state], dtype=torch.float32, device=self._sim_device)

        # --- Wrist Logic ---
        # Default wrist poses (position + quaternion [x, y, z, w])
        # Format: [x, y, z, qx, qy, qz, qw]
        default_wrist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        # Extract poses from controller data
        left_wrist = self._extract_wrist_pose(left_controller_data, default_wrist)
        right_wrist = self._extract_wrist_pose(right_controller_data, default_wrist)

        # Convert to tensors
        left_wrist_tensor = torch.tensor(self._retarget_abs(left_wrist), dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(self._retarget_abs(right_wrist), dtype=torch.float32, device=self._sim_device)

        # Concatenate: [gripper(2), left_wrist(7), right_wrist(7)]
        return torch.cat([gripper_tensor, left_wrist_tensor, right_wrist_tensor])

    def _extract_hand_state(self, controller_data: np.ndarray, prev_state: float) -> float:
        """Extract hand state from controller data with hysteresis.

        Args:
            controller_data: 2D array [pose(7), inputs(7)]
            prev_state: Previous hand state (0.0 or 1.0)

        Returns:
            Hand state as float (0.0 for open, 1.0 for close)
        """
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return 0.0

        # Extract inputs from second row
        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        if len(inputs) < len(DeviceBase.MotionControllerInputIndex):
            return 0.0

        # Extract specific inputs using enum
        trigger = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]  # 0.0 to 1.0 (analog)

        # Apply hysteresis
        if prev_state < 0.5:  # Currently open
            return 1.0 if trigger > self._cfg.threshold_high else 0.0
        else:  # Currently closed
            return 0.0 if trigger < self._cfg.threshold_low else 1.0

    def _extract_wrist_pose(self, controller_data: np.ndarray, default_pose: np.ndarray) -> np.ndarray:
        """Extract wrist pose from controller data.

        Args:
            controller_data: 2D array [pose(7), inputs(7)]
            default_pose: Default pose to use if no data

        Returns:
            Wrist pose array [x, y, z, w, x, y, z]
        """
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        return default_pose

    def _retarget_abs(self, wrist: np.ndarray) -> np.ndarray:
        """Handle absolute pose retargeting for controller wrists."""
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        # Combined -75° (rather than -90° for wrist comfort) Y rotation + 90° Z rotation
        # This is equivalent to (0, -75, 90) in euler angles
        combined_quat = torch.tensor([-0.4619, 0.5358, 0.4619, 0.5358], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class G1TriHandUpperBodyMotionControllerGripperRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 boolean gripper and wrist retargeter."""

    threshold_high: float = 0.6  # Threshold to close hand
    threshold_low: float = 0.4  # Threshold to open hand
    retargeter_type: type[RetargeterBase] = G1TriHandUpperBodyMotionControllerGripperRetargeter
