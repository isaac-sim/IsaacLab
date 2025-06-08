# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import JointActionCfg
from . import action


@configclass
class LimitsScaledJointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`LimitsScaledJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = action.LimitsScaledJointPositionAction

    ema_lambda: float = 0.9

