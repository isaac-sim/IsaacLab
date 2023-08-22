# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

from omni.isaac.orbit.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .rigid_object import RigidObject


@configclass
class RigidObjectCfg(AssetBaseCfg):
    """Configuration parameters for a rigid object."""

    cls_name = RigidObject

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the rigid body."""

        lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    ##
    # Initialize configurations.
    ##

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose with zero velocity."""
