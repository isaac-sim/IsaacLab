# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import thrust_actions

##
# Drone actions.
##


@configclass
class ThrustActionCfg(ActionTermCfg):
    """Configuration for the thrust action term.

    See :class:`ThrustAction` for more details.
    """

    class_type: type[ActionTerm] = thrust_actions.ThrustAction

    asset_name: str = MISSING
    """Name or regex expression of the asset that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""

    preserve_order: bool = False
    """Whether to preserve the order of the asset names in the action output. Defaults to False."""

    use_default_offset: bool = True
    """Whether to use default thrust (e.g. hover thrust) configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default thrust values
    from the articulation asset.
    """
