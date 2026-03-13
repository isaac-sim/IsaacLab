# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for teleoperation interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass
class DeviceCfg:
    """Configuration for teleoperation devices."""

    # Whether teleoperation should start active by default
    teleoperation_active_default: bool = True
    # Torch device string to place output tensors on
    sim_device: str = "cpu"
    # Retargeters that transform device data into robot commands
    retargeters: list[RetargeterCfg] = field(default_factory=list)
    # Concrete device class to construct for this config. Set by each device module.
    class_type: type["DeviceBase"] | None = None


@dataclass
class DevicesCfg:
    """Configuration for all supported teleoperation devices."""

    devices: dict[str, DeviceCfg] = field(default_factory=dict)


class DeviceBase(ABC):
    """An interface class for teleoperation devices.

    Derived classes have two implementation options:

    1. Override _get_raw_data() and use the base advance() implementation:
       This approach is suitable for devices that want to leverage the built-in
       retargeting logic but only need to customize the raw data acquisition.

    2. Override advance() completely:
       This approach gives full control over the command generation process,
       and _get_raw_data() can be ignored entirely.
    """

    def __init__(self, retargeters: list[RetargeterBase] | None = None):
        """Initialize the teleoperation interface.

        Args:
            retargeters: List of components that transform device data into robot commands.
                        If None or empty list, the device will output its native data format.
        """
        # Initialize empty list if None is provided
        self._retargeters = retargeters or []
        # Aggregate required features across all retargeters
        self._required_features = set()
        for retargeter in self._retargeters:
            self._required_features.update(retargeter.get_requirements())

    def __str__(self) -> str:
        """Returns: A string identifier for the device."""
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

    def _get_raw_data(self) -> Any:
        """Internal method to get the raw data from the device.

        This method is intended for internal use by the advance() implementation.
        Derived classes can override this method to customize raw data acquisition
        while still using the base class's advance() implementation.

        Returns:
            Raw device data in a device-specific format

        Note:
            This is an internal implementation detail. Clients should call advance()
            instead of this method.
        """
        raise NotImplementedError("Derived class must implement _get_raw_data() or override advance()")

    def advance(self) -> torch.Tensor:
        """Process current device state and return control commands.

        This method retrieves raw data from the device and optionally applies
        retargeting to convert it to robot commands.

        Derived classes can either:
        1. Override _get_raw_data() and use this base implementation, or
        2. Override this method completely for custom command processing

        Returns:
            When no retargeters are configured, returns raw device data in its native format.
            When retargeters are configured, returns a torch.Tensor containing the concatenated
            outputs from all retargeters.
        """
        raw_data = self._get_raw_data()

        # If no retargeters, return raw data directly (not as a tuple)
        if not self._retargeters:
            return raw_data

        # With multiple retargeters, return a tuple of outputs
        # Concatenate retargeted outputs into a single tensor
        return torch.cat([retargeter.retarget(raw_data) for retargeter in self._retargeters], dim=-1)

    # -----------------------------
    # Shared data layout helpers (for retargeters across devices)
    # -----------------------------
    class TrackingTarget(Enum):
        """Standard tracking targets shared across devices."""

        HAND_LEFT = 0
        HAND_RIGHT = 1
        HEAD = 2
        CONTROLLER_LEFT = 3
        CONTROLLER_RIGHT = 4

    class MotionControllerDataRowIndex(Enum):
        """Rows in the motion-controller 2x7 array."""

        POSE = 0
        INPUTS = 1

    class MotionControllerInputIndex(Enum):
        """Indices in the motion-controller input row."""

        THUMBSTICK_X = 0
        THUMBSTICK_Y = 1
        TRIGGER = 2
        SQUEEZE = 3
        BUTTON_0 = 4
        BUTTON_1 = 5
        PADDING = 6
