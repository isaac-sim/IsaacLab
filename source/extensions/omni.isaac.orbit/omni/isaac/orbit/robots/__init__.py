# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Submodule containing all robot abstractions.
"""

from .legged_robot import LeggedRobot, LeggedRobotCfg, LeggedRobotData
from .mobile_manipulator import (
    LeggedMobileManipulator,
    LeggedMobileManipulatorCfg,
    LeggedMobileManipulatorData,
    MobileManipulator,
    MobileManipulatorCfg,
    MobileManipulatorData,
)
from .single_arm import SingleArmManipulator, SingleArmManipulatorCfg, SingleArmManipulatorData

__all__ = [
    # single-arm manipulators
    "SingleArmManipulatorCfg",
    "SingleArmManipulatorData",
    "SingleArmManipulator",
    # legged robots
    "LeggedRobotCfg",
    "LeggedRobotData",
    "LeggedRobot",
    # mobile manipulators
    "MobileManipulatorCfg",
    "MobileManipulatorData",
    "MobileManipulator",
    # legged mobile manipulators
    "LeggedMobileManipulator",
    "LeggedMobileManipulatorCfg",
    "LeggedMobileManipulatorData",
]
