# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Base classes for legacy retargeting.

.. deprecated::
    :class:`RetargeterBase` and :class:`RetargeterCfg` are deprecated.
    Please use the IsaacTeleop retargeting engine via :mod:`isaaclab_teleop`
    instead. See :class:`isaaclab_teleop.IsaacTeleopCfg` for pipeline-based
    retargeting configuration.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass
class RetargeterCfg:
    """Base configuration for hand tracking retargeters.

    .. deprecated::
        Use the IsaacTeleop retargeting engine via :mod:`isaaclab_teleop` instead.
    """

    sim_device: str = "cpu"
    # Concrete retargeter class to construct for this config. Set by each retargeter module.
    retargeter_type: type[RetargeterBase] | None = None


class RetargeterBase(ABC):
    """Base interface for input data retargeting.

    .. deprecated::
        Use the IsaacTeleop retargeting engine via :mod:`isaaclab_teleop` instead.

    This abstract class defines the interface for components that transform
    raw device data into robot control commands. Implementations can handle
    various types of transformations including:
    - Hand joint data to end-effector poses
    - Input device commands to robot movements
    - Sensor data to control signals
    """

    def __init__(self, cfg: RetargeterCfg):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        warnings.warn(
            "RetargeterBase is deprecated. Please use the IsaacTeleop retargeting engine via isaaclab_teleop instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._sim_device = cfg.sim_device

    class Requirement(Enum):
        """Features a retargeter may require from a device's raw data feed."""

        HAND_TRACKING = "hand_tracking"
        HEAD_TRACKING = "head_tracking"
        MOTION_CONTROLLER = "motion_controller"

    @abstractmethod
    def retarget(self, data: Any) -> Any:
        """Retarget input data to desired output format.

        Args:
            data: Raw input data to be transformed

        Returns:
            Retargeted data in implementation-specific format
        """
        pass

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Return the list of required data features for this retargeter.

        Defaults to requesting all available features for backward compatibility.
        Implementations should override to narrow to only what they need.
        """
        return [
            RetargeterBase.Requirement.HAND_TRACKING,
            RetargeterBase.Requirement.HEAD_TRACKING,
            RetargeterBase.Requirement.MOTION_CONTROLLER,
        ]
