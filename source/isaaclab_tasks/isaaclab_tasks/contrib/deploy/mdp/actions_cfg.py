# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action configuration classes."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class DeployRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for deploy relative joint actions with explicit LEAPP current-joint input."""

    class_type: type | str = "isaaclab_tasks.contrib.deploy.mdp.actions:DeployRelativeJointPositionAction"


@configclass
class DeployOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    """OSC action that exports scaled pose deltas for LEAPP instead of joint efforts."""

    class_type: type | str = "isaaclab_tasks.contrib.deploy.mdp.actions:DeployOperationalSpaceControllerAction"


@configclass
class DeployDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """DiffIK action that exports scaled pose deltas for LEAPP instead of joint targets."""

    class_type: type | str = "isaaclab_tasks.contrib.deploy.mdp.actions:DeployDifferentialInverseKinematicsAction"
