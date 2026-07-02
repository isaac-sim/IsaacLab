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


@configclass
class FlexivDynamicsAwareRelativeJointPositionActionCfg(DelayedRelativeJointPositionActionCfg):
    """Configuration for Flexiv-style dynamics-aware relative joint-position control."""

    class_type: type | str = f"{_ACTIONS_MODULE}:FlexivDynamicsAwareRelativeJointPositionAction"

    stiffness: float | dict[str, float] = {
        "joint1": 6000.0,
        "joint2": 6000.0,
        "joint3": 4200.0,
        "joint4": 4200.0,
        "joint5": 1500.0,
        "joint6": 1500.0,
        "joint7": 1500.0,
    }
    """Joint stiffness in Nm/rad, matching Flexiv's SetJointImpedance stiffness argument."""

    damping_ratio: float | dict[str, float] = 0.7
    """Joint damping ratio, matching Flexiv's SetJointImpedance damping-ratio argument."""

    damping_scale: float = 1.0
    """Multiplier applied after converting stiffness and damping ratio to joint damping."""

    min_effective_inertia: float = 1.0e-6
    """Lower bound for mass-matrix diagonal entries used in the damping calculation."""

    max_damping: float | None = None
    """Optional upper clamp for the computed damping in Nms/rad."""

    mass_matrix_mode: str = "diagonal"
    """Mass-matrix reduction mode. Only ``diagonal`` is currently supported."""

    rewrite_stiffness_each_step: bool = False
    """Whether to write stiffness on every action application instead of once."""

    rewrite_stiffness_on_reset: bool = True
    """Whether to rewrite stiffness after reset in case reset events changed drive parameters."""
