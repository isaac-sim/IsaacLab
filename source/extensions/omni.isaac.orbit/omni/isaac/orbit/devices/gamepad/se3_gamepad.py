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
    ================================= ================================ ================================
    Description                        Stick/Button (+ve axis)         Stick/Button (-ve axis)
    ================================= ================================ ================================
    Toggle gripper                     B Button (open)                 A Button (close)
    Move along x-axis                  Left Stick Up                   Left Stick Down
    Move along y-axis                  Left Stick Left                 Left Stick Right
    Move along z-axis                  Right Stick Up                  Right Stick Down
    Rotate along x-axis                D-Pad Right                     D-Pad Left
    Rotate along y-axis                D-Pad Up                        D-Pad Down
    Rotate along z-axis                Right Stick Left                Right Stick Right
    ================================= ================================ ================================


    Reference:
        https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html#carb.input.Gamepad
    """

    def __init__(self, pos_sensitivity: float = 1, rot_sensitivity: float = 2, deadzone: float = 0.01):
        """Initialize the gamepad layer.

        Args:
            pos_sensitivity (float): Magnitude of input position command scaling. Defaults to 1.
            rot_sensitivity (float): Magnitude of scale input rotation commands scaling. Defaults to 2.
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
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Gamepad Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\tToggle gripper (open/close): B/A\n"
        msg += "\tMove arm along x-axis: Left Stick Up/Down\n"
        msg += "\tMove arm along y-axis: Left Stick Left/Right\n"
        msg += "\tMove arm along z-axis: Right Stick Up/Down\n"
        msg += "\tRotate arm along x-axis: D-Pad Left/Right\n"
        msg += "\tRotate arm along y-axis: D-Pad Up/Down\n"
        msg += "\tRotate arm along z-axis: Right Stick Left/Right\n"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros([2, 3])  # (pos, neg) (x, y, z)
        self._delta_rot = np.zeros([2, 3])  # (pos, neg) (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind gamepad.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=gamepadeventtype#carb.input.GamepadInput>`_.

        The callback function should not take any arguments.

        Args:
            key (str): The gamepad button to check against.
            func (Callable): The function to call when key is pressed.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> Tuple[np.ndarray, bool]:
        """Provides the result from gamepad event state.

        Returns:
            Tuple[np.ndarray, bool] -- A tuple containing the delta pose command and gripper commands.
        """
        delta_rot_sgn = self._delta_rot[0, :] > self._delta_rot[1, :]
        delta_rot = self._delta_rot.max(axis=0)
        delta_rot[~delta_rot_sgn] *= -1
        delta_pos_sgn = self._delta_pos[0, :] > self._delta_pos[1, :]
        delta_pos = self._delta_pos.max(axis=0)
        delta_pos[~delta_pos_sgn] *= -1

        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        # if new command received, reset event flag to False until gamepad updated.
        return np.concatenate([delta_pos, rot_vec]), self._close_gripper

    """
    Internal helpers.
    """

    def _on_gamepad_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=gamepadeventtype#carb.input.Gamepad
        """

        cur_val = event.value
        absval = abs(event.value)

        # Ignore 0 since it signifies the movement  of the stick has stopped,
        # but doesn't mean it's at center...could be being held steady

        if absval < self.deadzone:
            cur_val = 0

        if event.input == carb.input.GamepadInput.A:
            if cur_val > 0.5:
                self._close_gripper = True
        elif event.input == carb.input.GamepadInput.B:
            if cur_val > 0.5:
                self._close_gripper = False
        elif event.input in self._INPUT_KEY_POS_VALUE_MAPPING:
            i, j, v = self._INPUT_KEY_POS_VALUE_MAPPING[event.input]
            self._delta_pos[i, j] = v * cur_val
        elif event.input in self._INPUT_KEY_ROT_VALUE_MAPPING:
            i, j, v = self._INPUT_KEY_ROT_VALUE_MAPPING[event.input]
            self._delta_rot[i, j] = v * cur_val
        elif event.input in self._INPUT_KEY_ROT_BOOL_MAPPING:
            i, j, v = self._INPUT_KEY_ROT_BOOL_MAPPING[event.input]
            if cur_val > 0.5:
                self._delta_rot[i, j] = v
            else:
                self._delta_rot[:, j] = 0
        elif event.input.name in self._additional_callbacks:
            self._additional_callbacks[event.input.name]()
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_POS_VALUE_MAPPING = {
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
        }

        self._INPUT_KEY_ROT_BOOL_MAPPING = {
            # forward command
            carb.input.GamepadInput.DPAD_UP: (0, 1, self.rot_sensitivity * 0.8),
            # backward command
            carb.input.GamepadInput.DPAD_DOWN: (1, 1, self.rot_sensitivity * 0.8),
            # right command
            carb.input.GamepadInput.DPAD_RIGHT: (0, 0, self.rot_sensitivity * 0.8),
            # left command
            carb.input.GamepadInput.DPAD_LEFT: (1, 0, self.rot_sensitivity * 0.8),
        }

        self._INPUT_KEY_ROT_VALUE_MAPPING = {
            # yaw command (positive)
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (0, 2, self.rot_sensitivity),
            # yaw command (negative)
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (1, 2, self.rot_sensitivity),
        }
