# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
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
class PCAHandActionCfg(ActionTermCfg):

    class_type: type[ActionTerm] = action.PCAAction

    pca_joints_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=[
            "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
            "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
            "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
            "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
        ],
        preserve_order=True
    )

    arm_joints_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="iiwa7_.*")
    hand_pca_mins = [0.2475, -0.3286, -0.7238, -0.0192, -0.5532]
    hand_pca_maxs = [3.8336, 3.0025, 0.8977, 1.0243, 0.0629]
    pca_matrix = [
            [
                -3.8872e-02,  3.7917e-01,  4.4703e-01,  7.1016e-03,
                2.1159e-03, 3.2014e-01,  4.4660e-01,  5.2108e-02,  
                5.6869e-05,  2.9845e-01, 3.8575e-01,  7.5774e-03,
                -1.4790e-02,  9.8163e-02,  4.3551e-02, 3.1699e-01
            ],
            [
                -5.1148e-02, -1.3007e-01,  5.7727e-02,  5.7914e-01,
                1.0156e-02, -1.8469e-01,  5.3809e-02,  5.4888e-01,
                1.3351e-04, -1.7747e-01, 2.7809e-02,  4.8187e-01,
                2.9753e-02,  2.6149e-02,  6.6994e-02, 1.8117e-01
            ],
            [
                -5.7137e-02, -3.4707e-01,  3.3365e-01, -1.8029e-01,
                -4.3560e-02, -4.7666e-01,  3.2517e-01, -1.5208e-01,
                -5.9691e-05, -4.5790e-01, 3.6536e-01, -1.3916e-01,
                2.3925e-03,  3.7238e-02, -1.0124e-01, -1.7442e-02
            ],
            [
                2.2795e-02, -3.4090e-02,  3.4366e-02, -2.6531e-02,
                2.3471e-02, 4.6123e-02,  9.8059e-02, -1.2619e-03,
                -1.6452e-04, -1.3741e-02, 1.3813e-01,  2.8677e-02,
                2.2661e-01, -5.9911e-01,  7.0257e-01, -2.4525e-01
            ],
            [
                -4.4911e-02, -4.7156e-01,  9.3124e-02,  2.3135e-01,
                -2.4607e-03, 9.5564e-02,  1.2470e-01,  3.6613e-02, 
                1.3821e-04,  4.6072e-01, 9.9315e-02, -8.1080e-02,
                -4.7617e-01, -2.7734e-01, -2.3989e-01, -3.1222e-01
            ]
        ]


@configclass
class FabricActionCfg(ActionTermCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = FabricAction

    palm_rot_range = 0.7854

    fabric_damping_gain = 10

    fabrics_dt = 1/60.

    fabric_decimation = 2

    pd_vel_factor = 1.0

    pca_feat_min = [0.2475, -0.3286, -0.7238, -0.0192, -0.5532]

    pca_feat_max = [3.8336, 3.0025, 0.8977, 1.0243, 0.0629]
    
    fabric_robot_scene_cfg = SceneEntityCfg(
        "robot",
        body_names=["palm_link", "index_biotac_tip", "middle_biotac_tip", "ring_biotac_tip", "thumb_biotac_tip"],
        joint_names=[
            "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3", "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
            "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
            "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
            "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
            "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
        ],
        preserve_order=True,
    )