# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass

from ..legged_robot import LeggedRobotCfg
from ..single_arm import SingleArmManipulatorCfg

__all__ = ["MobileManipulatorCfg", "LeggedMobileManipulatorCfg"]


@configclass
class MobileManipulatorCfg(SingleArmManipulatorCfg):
    """Properties for a mobile manipulator."""

    @configclass
    class MetaInfoCfg(SingleArmManipulatorCfg.MetaInfoCfg):
        """Meta-information about the mobile manipulator."""

        base_num_dof: int = MISSING
        """Number of degrees of freedom of base"""

    ##
    # Initialize configurations.
    ##

    meta_info: MetaInfoCfg = MetaInfoCfg()
    """Meta-information about the mobile manipulator."""


@configclass
class LeggedMobileManipulatorCfg(MobileManipulatorCfg, LeggedRobotCfg):
    """Properties for a mobile manipulator."""

    pass
