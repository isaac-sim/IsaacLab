# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad controller for SE(3) control."""


import numpy as np
from scipy.spatial.transform.rotation import Rotation
from typing import Callable, Tuple

import carb
import omni

from ..device_base import DeviceBase


class Se3Gamepad(DeviceBase):
    """A gamepad controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a gamepad controller for a robotic arm with a gripper.
    It uses the gamepad interface to listen to gamepad events and map them to the robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Stick and Button bindings:
        ============================ ========================= =========================
        Description                  Stick/Button (+ve axis)   Stick/Button (-ve axis)
        ============================ ========================= =========================
        Toggle gripper(open/close)   X Button                  X Button
        Move along x-axis            Left Stick Up             Left Stick Down
        Move along y-axis            Left Stick Left           Left Stick Right
        Move along z-axis            Right Stick Up            Right Stick Down
        Rotate along x-axis          D-Pad Left                D-Pad Right
        Rotate along y-axis          D-Pad Down                D-Pad Up
        Rotate along z-axis          Right Stick Left          Right Stick Right
        ============================ ========================= =========================

    .. seealso::

        The official documentation for the gamepad interface: `Carb Gamepad Interface <https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html#carb.input.Gamepad>`__.

    """

    def __init__(self, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.6, deadzone: float = 0.01):
        """Initialize the gamepad layer.

        Args:
            pos_sensitivity (float): Magnitude of input position command scaling. Defaults to 1.0.
            rot_sensitivity (float): Magnitude of scale input rotation commands scaling. Defaults to 1.6.
            deadzone (float): Magnitude of deadzone for gamepad input. Defaults to 0.01.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.deadzone = deadzone
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(self._gamepad, self._on_gamepad_event)
        # bindings for gamepad to command
        self._create_key_bindings()
        # command buffers
        self._close_gripper = False
        self._delta_pose = np.zeros([2, 6])  # (pos, neg) (x, y, z, roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Unsubscribe from gamepad events."""
        self._input.unsubscribe_from_gamepad_events(self._gamepad, self._gamepad_sub)
        self._gamepad_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Gamepad Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tDevice name: {self._input.get_gamepad_name(self._gamepad)}\n"
        msg += "----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): X\n"
        msg += "\tMove arm along x-axis: Left Stick Up/Down\n"
        msg += "\tMove arm along y-axis: Left Stick Left/Right\n"
        msg += "\tMove arm along z-axis: Right Stick Up/Down\n"
        msg += "\tRotate arm along x-axis: D-Pad Right/Left\n"
        msg += "\tRotate arm along y-axis: D-Pad Down/Up\n"
        msg += "\tRotate arm along z-axis: Right Stick Left/Right\n"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pose.fill(0.0)  # (pos, neg) (x, y, z, roll, pitch, yaw) all values >= 0

    def add_callback(self, key: carb.input.GamepadInput, func: Callable):
        """Add additional functions to bind gamepad.

        A list of available gamepad keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=gamepadeventtype#carb.input.GamepadInput>`__.

        The callback function should not take any arguments.

        Args:
            key (carb.input.GamepadInput): The gamepad button to check against.
            func (Callable): The function to call when key is pressed.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> Tuple[np.ndarray, bool]:
        """Provides the result from gamepad event state.

        Returns:
            Tuple[np.ndarray, bool]: A tuple containing the delta pose command and gripper commands.
        """
        # In self._delta_rot and self._delta_rot,
        #   the [0,:] represents value in the positive direction, [1,:] represents the negative direction
        #   One of the two values is always 0, the other is the magnitude of the command
        # -- resolve rotation command
        delta_rot = self._delta_pose[:, 3:]
        # compare the pos and neg value decide the sign of the value
        delta_rot_sgn = delta_rot[0, :] > delta_rot[1, :]
        # extract the command value
        delta_rot = delta_rot.max(axis=0)
        # apply the sign
        delta_rot[~delta_rot_sgn] *= -1

        # -- resolve position command
        delta_pos = self._delta_pose[:, :3]
        # compare the pos and neg value decide the sign of the value
        delta_pos_sgn = delta_pos[0, :] > delta_pos[1, :]
        # extract the command value
        delta_pos = delta_pos.max(axis=0)
        # apply the sign
        delta_pos[~delta_pos_sgn] *= -1

        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        # return the command and gripper state
        return np.concatenate([delta_pos, rot_vec]), self._close_gripper

    """
    Internal helpers.
    """

    def _on_gamepad_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=gamepadeventtype#carb.input.Gamepad
        """
        # check if the event is a button press
        cur_val = event.value
        if abs(cur_val) < self.deadzone:
            cur_val = 0
        # toggle gripper based on the button pressed
        if event.input == carb.input.GamepadInput.X:
            if cur_val > 0.5:
                self._close_gripper = not self._close_gripper
        # update the delta pose based on the stick/dpad pressed
        if event.input in self._INPUT_STICK_VALUE_MAPPING:
            i, j, v = self._INPUT_STICK_VALUE_MAPPING[event.input]
            self._delta_pose[i, j] = v * cur_val
        elif event.input in self._INPUT_DPAD_VALUE_MAPPING:
            i, j, v = self._INPUT_DPAD_VALUE_MAPPING[event.input]
            if cur_val > 0.5:
                self._delta_pose[i, j] = v
                self._delta_pose[1 - i, j] = 0
            else:
                self._delta_pose[:, j] = 0
        # additional callbacks
        if event.input in self._additional_callbacks:
            self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        # map gamepad input to the element in self._delta_pose
        #   the first index is the direction (0: positive, 1: negative)
        #   the second index is the axis (0: x, 1: y, 2: z, 3: roll, 4: pitch, 5: yaw)
        self._INPUT_STICK_VALUE_MAPPING = {
            # forward command
            carb.input.GamepadInput.LEFT_STICK_UP: (0, 0, self.pos_sensitivity),
            # backward command
            carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 0, self.pos_sensitivity),
            # right command
            carb.input.GamepadInput.LEFT_STICK_RIGHT: (0, 1, self.pos_sensitivity),
            # left command
            carb.input.GamepadInput.LEFT_STICK_LEFT: (1, 1, self.pos_sensitivity),
            # upward command
            carb.input.GamepadInput.RIGHT_STICK_UP: (0, 2, self.pos_sensitivity),
            # downward command
            carb.input.GamepadInput.RIGHT_STICK_DOWN: (1, 2, self.pos_sensitivity),
            # yaw command (positive)
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (0, 5, self.rot_sensitivity),
            # yaw command (negative)
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (1, 5, self.rot_sensitivity),
        }

        self._INPUT_DPAD_VALUE_MAPPING = {
            # pitch command (postive)
            carb.input.GamepadInput.DPAD_UP: (1, 4, self.rot_sensitivity * 0.8),
            # pitch command (negative)
            carb.input.GamepadInput.DPAD_DOWN: (0, 4, self.rot_sensitivity * 0.8),
            # roll command (positive)
            carb.input.GamepadInput.DPAD_RIGHT: (1, 3, self.rot_sensitivity * 0.8),
            # roll command (negative)
            carb.input.GamepadInput.DPAD_LEFT: (0, 3, self.rot_sensitivity * 0.8),
        }
