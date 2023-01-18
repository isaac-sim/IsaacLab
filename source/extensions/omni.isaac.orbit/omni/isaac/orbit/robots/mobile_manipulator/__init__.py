# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for handling mobile manipulators."""

from .mobile_manipulator import LeggedMobileManipulator, MobileManipulator
from .mobile_manipulator_cfg import LeggedMobileManipulatorCfg, MobileManipulatorCfg
from .mobile_manipulator_data import LeggedMobileManipulatorData, MobileManipulatorData

__all__ = [
    # general mobile manipulator
    "MobileManipulator",
    "MobileManipulatorCfg",
    "MobileManipulatorData",
    # mobile manipulator with a legged base
    "LeggedMobileManipulator",
    "LeggedMobileManipulatorCfg",
    "LeggedMobileManipulatorData",
]
