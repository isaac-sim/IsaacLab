# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse controller for SE(3) control."""

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from isaaclab.utils.array import convert_to_torch

from ..spacemouse.base_spacemouse import SpaceMouseBase, SpaceMouseBaseCfg


@dataclass
class Se3SpaceMouseCfg(SpaceMouseBaseCfg):
    """Configuration for SE3 space mouse devices."""

    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    retargeters: None = None


class Se3SpaceMouse(SpaceMouseBase):
    """A SpaceMouse controller for sending SE(3) commands as delta poses.

    This class is useful for controlling a robot in SE(3) space, for instance, a gripper attached to a robotic arm.
    It outputs the (x, y, z, roll, pitch, yaw) command, where roll, pitch, and yaw are the rotations around the
    x, y, and z axes, respectively. The gripper can be opened and closed using the left button of the 3D mouse.
    """

    def __init__(self, cfg: Se3SpaceMouseCfg):
        """Initialize the space-mouse layer.

        Args:
            cfg: Configuration object for space-mouse settings.
        """
        super().__init__(cfg=cfg)

        # store inputs
        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity

        # read rotations
        self._read_rotation = False

        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    """
    Public Methods
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind spacemouse.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from spacemouse event state.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        delta_pose = np.concatenate([self._delta_pos, rot_vec])
        gripper_value = -1.0 if self._close_gripper else 1.0
        command = np.append(delta_pose, gripper_value)
        print("SIMDEVICE:", self._sim_device)
        return convert_to_torch(command, device=self._sim_device)

    """
    Internal helpers.
    """

    def _listen_for_updates(self):
        """
        This method implements the abstract method in the base class.
        It reads the current mouse input state and runs operations for the current user input.
        """

        # Restart the timer to call this function again after the specified interval
        self._start_timer()

        # read the device state
        self._read_mouse_state()

        # operations for the current user input

        # close gripper when left button is pressed
        if self._state.buttons[0] and not self._state.buttons[1]:
            self._close_gripper = not self._close_gripper

            # run additional callbacks
            if "L" in self._additional_callbacks:
                self._additional_callbacks["L"]

        # reset commands when right button is pressed
        elif self._state.buttons[1] and not self._state.buttons[0]:
            self.reset()

            # run additional callbacks
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]

        # toggle read rotation state when both buttons are pressed
        elif self._state.buttons[0] and self._state.buttons[1]:
            self._read_rotation = not self._read_rotation

        # transform the SpaceMouse state into base command
        twist = self._transform_state_to_twist(self._state)

        # call the callback for the twist command
        self._process_twist_command(twist)

    def _process_twist_command(self, twist) -> None:
        """Process Se3 twist into delta position and rotation commands.
        Args:
            twist: The Se3 twist linear and angular velocity in order: [vx, vy, vz, wx, wy, wz].
        """
        self._delta_pos = self._pos_sensitivity * twist[:3]
        self._delta_rot = self._rot_sensitivity * twist[3:]
