# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .actions import JointZeroOffsetPositionAction


@configclass
class JointZeroOffsetPositionActionCfg(JointPositionActionCfg):
    """Configuration for the joint position action term with joint-zero calibration offsets.

    See :class:`JointZeroOffsetPositionAction` for more details.
    """

    class_type: type[JointZeroOffsetPositionAction] | str = "{DIR}.actions:JointZeroOffsetPositionAction"

    zero_offset_range: tuple[float, float] = (0.0, 0.0)
    """Range [rad] of the joint-zero offset sampled once for each environment and joint."""
