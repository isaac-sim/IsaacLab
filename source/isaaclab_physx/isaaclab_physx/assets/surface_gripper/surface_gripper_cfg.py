# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.utils import REQUIRED

if TYPE_CHECKING:
    from .surface_gripper import SurfaceGripper


@dataclass
class SurfaceGripperCfg(AssetBaseCfg):
    """Configuration parameters for a surface gripper actuator."""

    prim_path: str = REQUIRED
    """The expression to find the grippers in the stage."""

    max_grip_distance: float | None = None
    """The maximum grip distance of the gripper."""

    coaxial_force_limit: float | None = None
    """The coaxial force limit of the gripper."""

    shear_force_limit: float | None = None
    """The shear force limit of the gripper."""

    retry_interval: float | None = None
    """The amount of time the gripper will spend trying to grasp an object."""

    class_type: type["SurfaceGripper"] | str = "isaaclab_physx.assets.surface_gripper.surface_gripper:SurfaceGripper"
