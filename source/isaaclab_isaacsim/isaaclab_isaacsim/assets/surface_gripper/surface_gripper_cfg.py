# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility configuration for the native PhysX surface gripper."""

from typing import TYPE_CHECKING

from isaaclab.assets.surface_gripper import SurfaceGripperCfg as BaseSurfaceGripperCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .surface_gripper import SurfaceGripper


@configclass
class SurfaceGripperCfg(BaseSurfaceGripperCfg):
    """Configure an authored Isaac Sim surface-gripper prim using native PhysX."""

    class_type: type["SurfaceGripper"] | str = "{DIR}.surface_gripper:SurfaceGripper"
    cloning_contexts: tuple[type | str, ...] | None = None
