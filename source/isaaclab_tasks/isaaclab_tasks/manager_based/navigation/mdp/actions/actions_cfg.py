# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers.lee_velocity_control_cfg import LeeVelControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions import binary_joint_actions, joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from . import thrust_actions

##
# Joint actions.
##
    
@configclass
class ThrustActionCfg(JointActionCfg):
    """Configuration for the joint thrust action term.

    See :class:`ThrustAction` for more details.
    """

    class_type: type[ActionTerm] = thrust_actions.ThrustAction

    use_default_offset: bool = True
    """Whether to use default thrust (e.g. hover thrust) configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default thrust values
    from the articulation asset.
    """
    
@configclass
class NavigationActionCfg(JointActionCfg):
    """Configuration for the joint navigation action term.

    See :class:`NavigationAction` for more details.
    """

    class_type: type[ActionTerm] = thrust_actions.NavigationAction

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
    
    controller_cfg: LeeVelControllerCfg = MISSING
    """The configuration for the Lee velocity controller."""
    
    
