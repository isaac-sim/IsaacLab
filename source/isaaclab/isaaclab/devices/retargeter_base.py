# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RetargeterCfg:
    """Base configuration for hand tracking retargeters."""

    sim_device: str = "cpu"


class RetargeterBase(ABC):
    """Base interface for input data retargeting.

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
        self._sim_device = cfg.sim_device

    @abstractmethod
    def retarget(self, data: Any) -> Any:
        """Retarget input data to desired output format.

        Args:
            data: Raw input data to be transformed

        Returns:
            Retargeted data in implementation-specific format
        """
        pass
