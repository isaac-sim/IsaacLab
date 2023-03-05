# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Dict, Optional, Tuple

from omni.isaac.orbit.utils import configclass

from ..robot_base_cfg import RobotBaseCfg


@configclass
class LeggedRobotCfg(RobotBaseCfg):
    """Properties for a legged robot (floating-base)."""

    @configclass
    class FootFrameCfg:
        """Information about the end-effector/foot frame location."""

        body_name: str = MISSING
        """Name of the body to track."""
        pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Additional position offset from the body frame. (default: (0, 0, 0)."""
        rot_offset: Tuple[float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Additional rotation offset (w, x, y, z) from the body frame. (default: (1, 0, 0, 0)."""

    @configclass
    class PhysicsMaterialCfg:
        """Physics material applied to the feet of the robot."""

        prim_path = "/World/Materials/footMaterial"
        """Path to the physics material prim. Defaults to /World/Materials/footMaterial.

        Note:
            If the prim path is not absolute, it will be resolved relative to the path specified when spawning
            the object.
        """
        static_friction: float = 1.0
        """Static friction coefficient. Defaults to 1.0."""
        dynamic_friction: float = 1.0
        """Dynamic friction coefficient. Defaults to 1.0."""
        restitution: float = 0.0
        """Restitution coefficient. Defaults to 0.0."""

    ##
    # Initialize configurations.
    ##

    feet_info: Dict[str, FootFrameCfg] = MISSING
    """Information about the feet to track (added to :obj:`data`).

    The returned tensor for feet state is in the same order as that of the provided list.
    """
    physics_material: Optional[PhysicsMaterialCfg] = PhysicsMaterialCfg()
    """Settings for the physics material to apply to feet.

    If set to None, no physics material will be created and applied.
    """
