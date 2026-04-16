# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term configuration (experimental, minimal).

This module mirrors the stable :mod:`isaaclab.envs.mdp.actions.actions_cfg` but only keeps what
the experimental Cartpole task needs.
"""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_experimental.managers.action_manager import ActionTerm, ActionTermCfg

from . import joint_actions


@configclass
class JointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term."""

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@configclass
class JointEffortActionCfg(JointActionCfg):
    """Configuration for the joint effort action term."""

    class_type: type[ActionTerm] = joint_actions.JointEffortAction
