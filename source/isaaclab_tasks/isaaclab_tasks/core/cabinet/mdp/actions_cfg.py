# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for cabinet position commands with bounded target motion."""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Limit commanded joint motion while preserving the absolute position action convention."""

    class_type: str = "{DIR}.actions:RateLimitedJointPositionAction"
    max_velocity: float = 0.3
    """Maximum position-target speed [rad/s], independent of the physics backend."""
