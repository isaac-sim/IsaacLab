# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action-term configuration for UR10 particle pushing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import ClampedRelativeJointPositionAction


@configclass
class ClampedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for one bounded joint-delta target per policy step."""

    joint_limit_margin: float = 0.0
    """Margin retained inside the articulation's soft joint-position limits [rad]."""

    class_type: type[ClampedRelativeJointPositionAction] | str = "{DIR}.actions:ClampedRelativeJointPositionAction"
