# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from dataclasses import MISSING, dataclass

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class DexMotionController(RetargeterBase):
    """Retargeter that maps motion controller inputs to robot hand joint angles.

    This class implements simple logic to map button presses and trigger/joystick inputs
    to finger joint angles. It is specifically designed for the G1/Inspire hands mapping logic.
    """

    def __init__(self, cfg: DexMotionControllerCfg):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter.
        """
        super().__init__(cfg)
        self._sim_device = cfg.sim_device
        self._hand_joint_names = cfg.hand_joint_names
        self._target = cfg.target
        self._is_left = self._target == DeviceBase.TrackingTarget.CONTROLLER_LEFT

        # Visualization not implemented for this simple mapper yet, but cfg supports it
        self._enable_visualization = cfg.enable_visualization

    def retarget(self, data: dict) -> torch.Tensor:
        """Process input data and return retargeted joint angles.

        Args:
            data: Dictionary mapping tracking targets to controller data.

        Returns:
            torch.Tensor: Tensor of hand joint angles.
        """
        controller_data = data.get(self._target, np.array([]))

        # Map controller inputs to hand joints
        # Returns 7 joints: [ThumbRot, Thumb1, Thumb2, Index1, Index2, Middle1, Middle2]
        joints = self._map_to_hand_joints(controller_data, is_left=self._is_left)

        if self._is_left:
            # Negate left hand joints for proper mirroring
            joints = -joints

        # The internal map returns 7 DOFs. We assume the hand_joint_names provided
        # correspond to these 7 DOFs in order, or we return them as is if names match count.
        # Internal Order:
        # 0: Thumb Rotation
        # 1: Thumb Middle/Proximal
        # 2: Thumb Distal
        # 3: Index Proximal
        # 4: Index Distal
        # 5: Middle Proximal
        # 6: Middle Distal

        # If the user provided specific joint names, we might need to map, but
        # for this simple controller we return the 7-dof vector.
        # Note: The previous implementation hardcoded a specific interleaving for two hands.
        # Here we return the single hand 7-dof vector.

        return torch.tensor(joints, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _map_to_hand_joints(self, controller_data: np.ndarray, is_left: bool) -> np.ndarray:
        """Map controller inputs to hand joint angles."""
        hand_joints = np.zeros(7)

        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return hand_joints

        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        if len(inputs) < len(DeviceBase.MotionControllerInputIndex):
            return hand_joints

        trigger = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
        squeeze = inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value]

        thumb_button = max(trigger, squeeze)
        thumb_angle = -thumb_button

        thumb_rotation = 0.5 * trigger - 0.5 * squeeze
        if not is_left:
            thumb_rotation = -thumb_rotation

        hand_joints[0] = thumb_rotation
        hand_joints[1] = thumb_angle * 0.4
        hand_joints[2] = thumb_angle * 0.7

        index_angle = trigger * 1.0
        hand_joints[3] = index_angle
        hand_joints[4] = index_angle

        middle_angle = squeeze * 1.0
        hand_joints[5] = middle_angle
        hand_joints[6] = middle_angle

        return hand_joints


@dataclass(kw_only=True)
class DexMotionControllerCfg(RetargeterCfg):
    """Configuration for the dexterous motion controller retargeter."""

    retargeter_type: type[RetargeterBase] = DexMotionController

    # Target
    target: DeviceBase.TrackingTarget = MISSING

    # Configuration
    hand_joint_names: list[str] = MISSING
    hand_urdf: str = MISSING  # Added as requested parameter

    enable_visualization: bool = False
