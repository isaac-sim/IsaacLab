# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse controller for SE(3) control."""

import hid
import numpy as np
import threading
import time
import torch
from collections.abc import Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from ..device_base import DeviceBase, DeviceCfg
from .utils import convert_buffer


@dataclass
class Se3SpaceMouseCfg(DeviceCfg):
    """Configuration for SE3 space mouse devices."""

    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    retargeters: None = None


class Se3SpaceMouse(DeviceBase):
    """A space-mouse controller for sending SE(3) commands as delta poses.

    This class implements a space-mouse controller to provide commands to a robotic arm with a gripper.
    It uses the `HID-API`_ which interfaces with USD and Bluetooth HID-class devices across multiple platforms [1].

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Note:
        The interface finds and uses the first supported device connected to the computer.

    Currently tested for following devices:

    - SpaceMouse Compact: https://3dconnexion.com/de/product/spacemouse-compact/

    .. _HID-API: https://github.com/libusb/hidapi

    """

    def __init__(self, cfg: Se3SpaceMouseCfg):
        """Initialize the space-mouse layer.

        Args:
            cfg: Configuration object for space-mouse settings.
        """
        # store inputs
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self._sim_device = cfg.sim_device
        # acquire device interface
        self._device = hid.device()
        self._find_device()
        # read rotations
        self._read_rotation = False

        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
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
        msg = f"Spacemouse Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tManufacturer: {self._device.get_manufacturer_string()}\n"
        msg += f"\tProduct: {self._device.get_product_string()}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tLeft button: toggle gripper command (open/close)\n"
        msg += "\tMove mouse laterally: move arm horizontally in x-y plane\n"
        msg += "\tMove mouse vertically: move arm vertically\n"
        msg += "\tTwist mouse about an axis: rotate arm about a corresponding axis"
        return msg

    """
    Operations
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
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    """
    Internal helpers.
    """

    def _find_device(self):
        """Find the device connected to computer."""
        found = False
        # implement a timeout for device search
        for _ in range(5):
            for device in hid.enumerate():
                if (
                    device["product_string"] == "SpaceMouse Compact"
                    or device["product_string"] == "SpaceMouse Wireless"
                ):
                    # set found flag
                    found = True
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    # connect to the device
                    self._device.close()
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
            data = self._device.read(7)
            if data is not None:
                # readings from 6-DoF sensor
                if data[0] == 1:
                    self._delta_pos[1] = self.pos_sensitivity * convert_buffer(data[1], data[2])
                    self._delta_pos[0] = self.pos_sensitivity * convert_buffer(data[3], data[4])
                    self._delta_pos[2] = self.pos_sensitivity * convert_buffer(data[5], data[6]) * -1.0
                elif data[0] == 2 and not self._read_rotation:
                    self._delta_rot[1] = self.rot_sensitivity * convert_buffer(data[1], data[2])
                    self._delta_rot[0] = self.rot_sensitivity * convert_buffer(data[3], data[4])
                    self._delta_rot[2] = self.rot_sensitivity * convert_buffer(data[5], data[6]) * -1.0
                # readings from the side buttons
                elif data[0] == 3:
                    # press left button
                    if data[1] == 1:
                        # close gripper
                        self._close_gripper = not self._close_gripper
                        # additional callbacks
                        if "L" in self._additional_callbacks:
                            self._additional_callbacks["L"]()
                    # right button is for reset
                    if data[1] == 2:
                        # reset layer
                        self.reset()
                        # additional callbacks
                        if "R" in self._additional_callbacks:
                            self._additional_callbacks["R"]()
                    if data[1] == 3:
                        self._read_rotation = not self._read_rotation
