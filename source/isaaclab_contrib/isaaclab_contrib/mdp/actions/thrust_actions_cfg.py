# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab_contrib.controllers import LeeAccControllerCfg, LeePosControllerCfg, LeeVelControllerCfg

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


@configclass
class NavigationActionCfg(ActionTermCfg):
    """Configuration for the navigation action term.

    See :class:`NavigationAction` for more details.
    """

    class_type: type[ActionTerm] = thrust_actions.NavigationAction

    asset_name: str = MISSING
    """Name or regex expression of the asset that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""

    preserve_order: bool = False
    """Whether to preserve the order of the asset names in the action output. Defaults to False."""

    use_default_offset: bool = False
    """Whether to use default thrust (e.g. hover thrust) configured in the articulation asset as offset.
    Defaults to False.

    If True, this flag results in overwriting the values of :attr:`offset` to the default thrust values
    from the articulation asset.
    """

    command_type: str = "vel"
    """Type of command to apply: "vel" for velocity commands, "pos" for position commands.
    "acc" for acceleration commands. Defaults to "vel".
    """

    action_dim: dict[str, int] = {"vel": 3, "pos": 4, "acc": 4}
    """Dimension of the action space for each command type."""

    controller_cfg: LeeVelControllerCfg | LeePosControllerCfg | LeeAccControllerCfg = MISSING
    """The configuration for the Lee velocity controller."""
