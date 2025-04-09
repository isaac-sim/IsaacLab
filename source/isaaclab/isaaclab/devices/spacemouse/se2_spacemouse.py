# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# Copyright (c) 2023, Boston Dynamics AI Institute, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SpaceMouse controller for SE(2) control."""

import numpy as np

from .base_spacemouse import SpaceMouseBase


class Se2SpaceMouse(SpaceMouseBase):
    """A SpaceMouse controller for sending SE(2) commands as delta poses.
    The use-case of this class is to control the robot in SE(2) space (x, y, yaw),
    where theta is the rotation about the z-axis.
    This is useful for controlling a differential drive robot.

    """

    def __init__(
        self,
        v_x_sensitivity: float = 0.8,
        v_y_sensitivity: float = 0.4,
        omega_z_sensitivity: float = 1.0,
        interval_ms: float = 10.0,
    ):
        """Initialize the SpaceMouse layer.

        Args:
            v_x_sensitivity (float): Magnitude of linear velocity along x-direction scaling. Defaults to 0.8.
            v_y_sensitivity (float): Magnitude of linear velocity along y-direction scaling. Defaults to 0.4.
            omega_z_sensitivity (float): Magnitude of angular velocity along z-direction scaling. Defaults to 1.0.
            interval_ms (float): Update interval for the SpaceMouse in milliseconds. Defaults to 10ms (100Hz).
        """
        # call the base class constructor
        super().__init__(interval_ms)

        # store inputs
        self._v_x_sensitivity = v_x_sensitivity
        self._v_y_sensitivity = v_y_sensitivity
        self._omega_z_sensitivity = omega_z_sensitivity

        # set the base command to zero
        self._base_command = np.zeros(3)

    """
    Public methods
    """

    def advance(self) -> np.ndarray:
        """Provides the result from SpaceMouse event state.

        Returns:
            np.ndarray -- A 3D array containing the linear (x,y) and angular velocity (z).
        """
        return self._base_command

    def reset(self):
        # reset the base command
        self._base_command.fill(0.0)

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
        """Transform the raw SpaceMouse state into base command.

        Args:
            state: The raw SpaceMouse state.

        Returns:
            np.ndarray -- A 3D array containing the linear (x,y) and angular velocity (z).
        """
        self._base_command = np.zeros(3)
        self._base_command[0] = self._v_x_sensitivity * twist[0]  # x
        self._base_command[1] = self._v_y_sensitivity * twist[1]  # y
        self._base_command[2] = self._omega_z_sensitivity * twist[5]  # yaw
