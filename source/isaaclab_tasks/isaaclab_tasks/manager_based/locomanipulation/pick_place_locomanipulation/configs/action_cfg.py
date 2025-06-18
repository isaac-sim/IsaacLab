# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from isaaclab.envs.mdp.actions import actions_cfg as mdp
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from ..mdp.actions import JointPositionPolicyAction


@configclass
class JointPositionPolicyActionCfg(mdp.JointPositionActionCfg):
    """Configuration for the locomotion policy action term."""

    class_type: type[ActionTerm] = JointPositionPolicyAction
    """The class type for the joint position policy action term."""

    policy_path: str = MISSING
