# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for one free-root ProHand articulation."""

from typing import Literal

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

from .prohand_actions import ProHandAction


@configclass
class ProHandActionCfg(ActionTermCfg):
    """Configure an AVP palm pose and 20 ProHand finger targets."""

    class_type: type[ProHandAction] = ProHandAction

    side: Literal["left", "right"] = "left"
    """Hand chirality; controls the ProHand joint-name prefix."""

    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-space offset applied to the tracked palm position, in meters."""

    root_to_palm_position: tuple[float, float, float] = (0.0, 0.2512, 0.02045)
    """Neutral ProHand root-to-palm translation from the published URDF."""

    root_to_palm_quaternion: tuple[float, float, float, float] = (0.70710678, 0.0, 0.70710678, 0.0)
    """Neutral ProHand root-to-palm quaternion in xyzw order."""
