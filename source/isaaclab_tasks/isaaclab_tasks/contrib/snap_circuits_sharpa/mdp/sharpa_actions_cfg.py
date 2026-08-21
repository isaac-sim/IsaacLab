# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for bimanual Sharpa Wave teleoperation."""

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

from .sharpa_actions import SharpaWaveBimanualAction


@configclass
class SharpaWaveBimanualActionCfg(ActionTermCfg):
    """Configure AVP wrist poses and finger targets for the dual Sharpa Wave asset."""

    class_type: type[SharpaWaveBimanualAction] = SharpaWaveBimanualAction

    position_scale: float = 1.0
    """Scale applied to the tracked wrist positions."""

    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-space offset applied to both tracked wrist positions, in meters."""
