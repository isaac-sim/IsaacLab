# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch
from dataclasses import dataclass
from typing import Final

from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass
class GripperRetargeterCfg(RetargeterCfg):
    """Configuration for gripper retargeter."""

    bound_hand: OpenXRDevice.TrackingTarget = OpenXRDevice.TrackingTarget.HAND_RIGHT


class GripperRetargeter(RetargeterBase):
    """Retargeter specifically for gripper control based on hand tracking data.

    This retargeter analyzes the distance between thumb and index finger tips to determine
    whether the gripper should be open or closed. It includes hysteresis to prevent rapid
    toggling between states when the finger distance is near the threshold.

    Features:
    - Tracks thumb and index finger distance
    - Implements hysteresis for stable gripper control
    - Outputs boolean command (True = close gripper, False = open gripper)
    """

    GRIPPER_CLOSE_METERS: Final[float] = 0.03
    GRIPPER_OPEN_METERS: Final[float] = 0.05

    def __init__(
        self,
        cfg: GripperRetargeterCfg,
    ):
        super().__init__(cfg)
        """Initialize the gripper retargeter."""
        # Store the hand to track
        if cfg.bound_hand not in [OpenXRDevice.TrackingTarget.HAND_LEFT, OpenXRDevice.TrackingTarget.HAND_RIGHT]:
            raise ValueError(
                "bound_hand must be either OpenXRDevice.TrackingTarget.HAND_LEFT or"
                " OpenXRDevice.TrackingTarget.HAND_RIGHT"
            )
        self.bound_hand = cfg.bound_hand
        # Initialize gripper state
        self._previous_gripper_command = False

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to gripper command.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.
                The joint names are defined in isaaclab.devices.openxr.common.HAND_JOINT_NAMES

        Returns:
            torch.Tensor: Tensor containing a single bool value where True = close gripper, False = open gripper
        """
        # Extract key joint poses
        hand_data = data[self.bound_hand]
        thumb_tip = hand_data["thumb_tip"]
        index_tip = hand_data["index_tip"]

        # Calculate gripper command with hysteresis
        gripper_command_bool = self._calculate_gripper_command(thumb_tip[:3], index_tip[:3])
        gripper_value = -1.0 if gripper_command_bool else 1.0

        return torch.tensor([gripper_value], dtype=torch.float32, device=self._sim_device)

    def _calculate_gripper_command(self, thumb_pos: np.ndarray, index_pos: np.ndarray) -> bool:
        """Calculate gripper command from finger positions with hysteresis.

        Args:
            thumb_pos: 3D position of thumb tip
            index_pos: 3D position of index tip

        Returns:
            bool: Gripper command (True = close, False = open)
        """
        distance = np.linalg.norm(thumb_pos - index_pos)

        # Apply hysteresis to prevent rapid switching
        if distance > self.GRIPPER_OPEN_METERS:
            self._previous_gripper_command = False
        elif distance < self.GRIPPER_CLOSE_METERS:
            self._previous_gripper_command = True

        return self._previous_gripper_command
