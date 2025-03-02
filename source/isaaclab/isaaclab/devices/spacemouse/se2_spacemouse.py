# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse controller for SE(2) control."""

import hid
import numpy as np
import threading
import time
from collections.abc import Callable

from ..device_base import DeviceBase
from .utils import convert_buffer


class Se2SpaceMouse(DeviceBase):
    r"""A space-mouse controller for sending SE(2) commands as delta poses.

    This class implements a space-mouse controller to provide commands to mobile base.
    It uses the `HID-API`_ which interfaces with USD and Bluetooth HID-class devices across multiple platforms.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Note:
        The interface finds and uses the first supported device connected to the computer.

    Currently tested for following devices:

    - SpaceMouse Compact: https://3dconnexion.com/de/product/spacemouse-compact/

    .. _HID-API: https://github.com/libusb/hidapi

    """

    def __init__(self, v_x_sensitivity: float = 0.8, v_y_sensitivity: float = 0.4, omega_z_sensitivity: float = 1.0):
        """Initialize the spacemouse layer.

        Args:
            v_x_sensitivity: Magnitude of linear velocity along x-direction scaling. Defaults to 0.8.
            v_y_sensitivity: Magnitude of linear velocity along y-direction scaling. Defaults to 0.4.
            omega_z_sensitivity: Magnitude of angular velocity along z-direction scaling. Defaults to 1.0.
        """
        # store inputs
        self.v_x_sensitivity = v_x_sensitivity
        self.v_y_sensitivity = v_y_sensitivity
        self.omega_z_sensitivity = omega_z_sensitivity
        # acquire device interface
        self._device = hid.device()
        self._find_device()
        # command buffers
        self._base_command = np.zeros(3)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        # run a thread for listening to device updates
        self._thread = threading.Thread(target=self._run_device)
        self._thread.daemon = True
        self._thread.start()

    def __del__(self):
        """Destructor for the class."""
        self._thread.join()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Spacemouse Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tManufacturer: {self._device.get_manufacturer_string()}\n"
        msg += f"\tProduct: {self._device.get_product_string()}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tMove mouse laterally: move base horizontally in x-y plane\n"
        msg += "\tTwist mouse about z-axis: yaw base about a corresponding axis"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable):
        # check keys supported by callback
        if key not in ["L", "R"]:
            raise ValueError(f"Only left (L) and right (R) buttons supported. Provided: {key}.")
        # TODO: Improve this to allow multiple buttons on same key.
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Provides the result from spacemouse event state.

        Returns:
            A 3D array containing the linear (x,y) and angular velocity (z).
        """
        return self._base_command

    """
    Internal helpers.
    """

    def _find_device(self):
        """Find the device connected to computer."""
        found = False
        # implement a timeout for device search
        for _ in range(5):
            for device in hid.enumerate():
                if device["product_string"] == "SpaceMouse Compact":
                    # set found flag
                    found = True
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    # connect to the device
                    self._device.open(vendor_id, product_id)
            # check if device found
            if not found:
                time.sleep(1.0)
            else:
                break
        # no device found: return false
        if not found:
            raise OSError("No device found by SpaceMouse. Is the device connected?")

    def _run_device(self):
        """Listener thread that keeps pulling new messages."""
        # keep running
        while True:
            # read the device data
            data = self._device.read(13)
            if data is not None:
                # readings from 6-DoF sensor
                if data[0] == 1:
                    # along y-axis
                    self._base_command[1] = self.v_y_sensitivity * convert_buffer(data[1], data[2])
                    # along x-axis
                    self._base_command[0] = self.v_x_sensitivity * convert_buffer(data[3], data[4])
                elif data[0] == 2:
                    # along z-axis
                    self._base_command[2] = self.omega_z_sensitivity * convert_buffer(data[3], data[4])
                # readings from the side buttons
                elif data[0] == 3:
                    # press left button
                    if data[1] == 1:
                        # additional callbacks
                        if "L" in self._additional_callbacks:
                            self._additional_callbacks["L"]
                    # right button is for reset
                    if data[1] == 2:
                        # reset layer
                        self.reset()
                        # additional callbacks
                        if "R" in self._additional_callbacks:
                            self._additional_callbacks["R"]
