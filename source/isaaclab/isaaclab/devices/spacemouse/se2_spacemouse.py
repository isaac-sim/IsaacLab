# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse controller for SE(2) control."""

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass

from isaaclab.utils.array import convert_to_torch

from ..spacemouse.base_spacemouse import SpaceMouseBase, SpaceMouseBaseCfg


@dataclass
class Se2SpaceMouseCfg(SpaceMouseBaseCfg):
    """Configuration for SE2 space mouse devices."""

    v_x_sensitivity: float = 0.8
    v_y_sensitivity: float = 0.4
    omega_z_sensitivity: float = 1.0


class Se2SpaceMouse(SpaceMouseBase):
    r"""A SpaceMouse controller for sending SE(2) commands as delta poses.

    This class is useful for controlling a robot in SE(2) space, for instance, a differential drive robot.
    It provides the output as (x, y, yaw), where yaw is the rotation around the z-axis.
    """

    cfg: Se2SpaceMouseCfg

    def __init__(self, cfg: Se2SpaceMouseCfg):
        """Initialize the spacemouse layer.

        Args:
            cfg: Configuration for the spacemouse device.
        """
        super().__init__(cfg=cfg)
        # store inputs
        self._v_x_sensitivity = cfg.v_x_sensitivity
        self._v_y_sensitivity = cfg.v_y_sensitivity
        self._omega_z_sensitivity = cfg.omega_z_sensitivity
        # command buffers
        self._base_command = np.zeros(3)

    """
    Public Methods
    """

    def reset(self):
        # default flags
        self._base_command.fill(0.0)

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
            A 3D tensor containing the linear (x,y) and angular velocity (z).
        """
        return convert_to_torch(self._base_command, device=self._sim_device)

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

        # callback when left button is pressed
        if self._state.buttons[0] and not self._state.buttons[1]:
            # run additional callbacks
            if "L" in self._additional_callbacks:
                self._additional_callbacks["L"]

        # callback when right button is pressed
        elif self._state.buttons[1] and not self._state.buttons[0]:
            self.reset()  # reset the base command

            # run additional callbacks
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]

        # transform the SpaceMouse state into base command
        twist = self._transform_state_to_twist(self._state)

        # call the callback for the twist command
        self._process_twist_command(twist)

    def _process_twist_command(self, twist) -> None:
        """Project the Se3 twist into Se2 twist command.
        Args:
            twist: The Se3 twist linear and angular velocity in order: [vx, vy, vz, wx, wy, wz].
        """
        self._base_command[0] = self._v_x_sensitivity * twist[0]  # x
        self._base_command[1] = self._v_y_sensitivity * twist[1]  # y
        self._base_command[2] = self._omega_z_sensitivity * twist[5]  # yaw
