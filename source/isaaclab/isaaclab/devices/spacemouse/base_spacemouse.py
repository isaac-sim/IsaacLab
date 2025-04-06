# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# Copyright (c) 2023, Boston Dynamics AI Institute, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for 3D SpaceMouse controller for SE(2) and SE(3) control."""

import numpy as np
import threading
from collections.abc import Callable

import pyspacemouse


class SpaceMouseBase:
    """A base class for SpaceMouse controllers.

    This class provides a common interface for SpaceMouse controllers.
    It abstracts away the details of connecting to and reading from the SpaceMouse device.
    It uses 'pyspacemouse'_ which is a Python library for 3Dconnexion SpaceMouse devices.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Note:
        The interface finds and uses the first supported device connected to the computer.

    Currently tested for following devices:

    - SpaceMouse Wireless <https://3dconnexion.com/uk/product/spacemouse-wireless/>

    .. _pyspacemouse: https://pypi.org/project/pyspacemouse/

    Note that pyspacemouse library needs to be installed within IsaacSim python
    environment using the following command: `./python.sh -m pip install pyspacemouse`.
    Also, the instructions for setting up the device on Linux from the website above
    needs to be followed.

    """

    def __init__(self, interval_ms):
        """Initialize the SpaceMouse base class.

        This method opens a connection to the SpaceMouse device and starts a timer
        to listen for updates at the given interval.

        Args:
            interval_ms (float): Update interval for the SpaceMouse in milliseconds.

        """
        # open a connection to the SpaceMouse device
        self._device = pyspacemouse.open()
        if not self._device:
            raise OSError("No device found by SpaceMouse. Is the device connected?")

        # start a timer to listen for updates with the given interval
        self._start_timer(interval_ms / 1000.0)

        self._additional_callbacks = dict()
        self._state = None

    def __del__(self):
        """Destructor for the SpaceMouse base class.

        This method terminates the timer and closes the connection to the SpaceMouse device.

        """
        self._timer.cancel()
        pyspacemouse.close()

    def __str__(self) -> str:
        """Returns: A string containing the information of SpaceMouse device."""
        msg = "----------------------------------------------\n"
        msg += "The SpaceMouse Controller connected:\n"
        msg += f"\tManufacturer: {self._device.vendor_name}\n"
        msg += f"\tProduct: {self._device.product_name}\n"
        msg += "----------------------------------------------\n"
        return msg

    """
    Public Methods
    """

    def add_callback(self, key: str, func: Callable):
        """Add a custom callback for a button press."""

        # check keys supported by callback
        if key not in ["L", "R"]:
            raise ValueError(f"Only left (L) and right (R) buttons supported. Provided: {key}.")
        # TODO: Improve this to allow multiple buttons on same key.
        self._additional_callbacks[key] = func

    """
    All the following methods are to be implemented by the derived class.
    """

    def _listen_for_updates(self):
        """Callback for listening for updates from the SpaceMouse device.

        This method is called periodically by a timer to listen for updates from the SpaceMouse device.
        This method is implemented by the derived class.

        """
        pass

    def advance(self):
        """Provides the result from SpaceMouse event state.

        This method is implemented by the derived class.

        """
        pass

    """
    Helper Methods
    """

    def _read_mouse_state(self):
        """This method reads the current state of the SpaceMouse device."""

        self._state = pyspacemouse.read()

    def _start_timer(self, interval_sec: float = 0.01):
        """This method starts a timer with the given interval to run the callback function for updates."""

        self._timer = threading.Timer(interval_sec, self._listen_for_updates)
        self._timer.start()

    def _transform_state_to_twist(self, state):
        """Transform the raw SpaceMouse state into twist command.

        Args:
            state: The raw SpaceMouse state.

        Returns:
            np.ndarray: The transformed twist 6D command (linear followed by angular)
        """

        twist = np.zeros(6)
        twist[0] = -state.y
        twist[1] = state.x
        twist[2] = state.z
        twist[3] = -state.roll
        twist[4] = -state.pitch
        twist[5] = -state.yaw

        return twist
