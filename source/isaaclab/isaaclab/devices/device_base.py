# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for teleoperation interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class DeviceBase(ABC):
    """An interface class for teleoperation devices."""

    def __init__(self):
        """Initialize the teleoperation interface."""
        pass

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        return f"{self.__class__.__name__}"

    """
    Operations
    """

    @abstractmethod
    def reset(self):
        """Reset the internals."""
        raise NotImplementedError

    @abstractmethod
    def add_callback(self, key: Any, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self) -> Any:
        """Provides the joystick event state.

        Returns:
            The processed output form the joystick.
        """
        raise NotImplementedError
