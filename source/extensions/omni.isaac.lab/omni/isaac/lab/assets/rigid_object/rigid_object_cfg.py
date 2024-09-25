# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .rigid_object import RigidObject


@configclass
class RigidObjectCfg(AssetBaseCfg):
    """Configuration parameters for a rigid object."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the rigid body."""

        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    ##
    # Initialize configurations.
    ##

    class_type: type = RigidObject

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose with zero velocity."""

    use_first_matching_path: bool = True
    """Determines whether to parse only the first matching path to the regex expression and assume that
       all other matching assets follow the same topology as the first matching path.

       If set to False, each path matching to the regex will be parsed to find the required predicate.
       Only set to False when the regex encapsulates assets that have predicates specified at different
       locations in the topology tree. This will impact scene creation performance.
    """
