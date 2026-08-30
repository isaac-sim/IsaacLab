# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass

_ACTIONS_MODULE = "isaaclab_tasks.contrib.deploy.mdp.delayed_joint_actions"


@configclass
class DelayedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for delayed relative joint-position actions."""

    class_type: type | str = f"{_ACTIONS_MODULE}:DelayedRelativeJointPositionAction"

    latency_s: float = 0.02
    """Command latency in seconds. Defaults to one 50 Hz collection sample, 20 ms."""

    latency_steps: int | None = None
    """Optional explicit delay in action-application steps. Overrides latency_s when set."""


@configclass
class ShapedDelayedRelativeJointPositionActionCfg(DelayedRelativeJointPositionActionCfg):
    """Configuration for delayed relative joint-position actions with command shaping."""

    class_type: type | str = f"{_ACTIONS_MODULE}:ShapedDelayedRelativeJointPositionAction"

    command_velocity_limit: float = 0.2
    """Maximum command-target slew velocity in rad/s. Set to zero to disable."""

    command_acceleration_limit: float = 0.5
    """Maximum command-target slew acceleration in rad/s^2. Set to zero to disable."""
