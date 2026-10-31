# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Suction action configuration without importing robot runtime views before Kit."""

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

TOOL_OFFSET = (0.0, 0.0, 0.105)
DOWN_QUATERNION = (0.9238795325, 0.3826834324, 0.0, 0.0)


@configclass
class CartesianSuctionArmCfg(DifferentialInverseKinematicsActionCfg):
    """Three Cartesian residuals, with orientation constrained by the tool."""

    class_type: type | str = "{DIR}.actions:CartesianSuctionArm"
    max_target_error: float = 0.04
    """Maximum Cartesian target lead from the measured tool position [m]."""


@configclass
class VacuumActionCfg(ActionTermCfg):
    """One binary action for a Newton suction tool."""

    class_type: type | str = "{DIR}.actions:VacuumAction"
    asset_name: str = "robot"
