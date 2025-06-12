# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import JointActionCfg
from . import action
from .fabric_action import FabricAction


@configclass
class LimitsScaledJointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`LimitsScaledJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = action.LimitsScaledJointPositionAction

    ema_lambda: float = 0.9


@configclass
class FabricActionCfg(ActionTermCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = FabricAction
    
    action_dim = 11

    max_pose_angle = -1.

    fabric_damping_gain = 10

    fabrics_dt = 1/60.

    fabric_decimation = 2
    
    observation_annealing_coefficient = 0.0

    pd_vel_factor = 1.0
    
    robot_joint_pos_bias_width = 0.0

    robot_joint_vel_bias_width = 0.0
    
    robot_joint_pos_noise = 0.0
    
    robot_joint_vel_noise = 0.0

    actuated_joint_names = [
        "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3", "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
        "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
        "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
        "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
        "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ]
    
    hand_body_names = ["palm_link", "index_biotac_tip", "middle_biotac_tip", "ring_biotac_tip", "thumb_biotac_tip"]
