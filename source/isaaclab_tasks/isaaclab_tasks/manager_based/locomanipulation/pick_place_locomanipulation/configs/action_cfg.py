# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from isaaclab.envs.mdp.actions import actions_cfg as mdp
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import JointPositionPolicyAction, LowerBodyAction

@configclass
class LowerBodyActionCfg(ActionTermCfg):
    """Configuration for the lower body action term."""

    class_type: type[ActionTerm] = LowerBodyAction
    """The class type for the lower body action term."""

    policy_path: str = MISSING
    """The path to the policy model."""

    joint_names: list[str] = MISSING
    """The names of the joints to control."""

    scale: float = 1.0
    """The scale of the action."""
    
    offset: float = 0.0
    """The offset of the action."""


@configclass
class JointPositionPolicyActionCfg(mdp.JointPositionActionCfg):
    """Configuration for the locomotion policy action term."""

    class_type: type[ActionTerm] = JointPositionPolicyAction
    """The class type for the joint position policy action term."""

    policy_path: str = MISSING
