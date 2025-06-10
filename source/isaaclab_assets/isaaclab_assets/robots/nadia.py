# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for IHMC robots.

The following configurations are available:

* :obj:`NADIA_V17_SIMPLEKNEES`: nadia with cycloid arms and nubs and simplized knee joints

"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


NADIA_V17_SIMPLEKNEES_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="../nadia-orbit-data/robot/nadia/usds/nadiaV17.fullRobot.simpleKnees.cycloidArms/instances/nadiaV17.fullRobot.simpleKnees.cycloidArms.partialMeshCollisions_Instance.usd",
        usd_path="../nadia-orbit-data/robot/nadia/usds/nadiaV17.fullRobot.simpleKnees.cycloidArms/instances/nadiaV17.fullRobot.simpleKnees.cycloidArms.partialMeshCollisions.noMaterials_Instance.usd",
        # usd_path="../nadia-orbit-data/robot/nadia/usds/nadiaV17.fullRobot.simpleKnees.cycloidArms/instances/nadiaV17.fullRobot.simpleKnees.cycloidArms.partialMeshCollisions.noVisuals_Instance.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.2),
        joint_pos={
            "LEFT_HIP_Z": 0.0,
            "RIGHT_HIP_Z": 0.0,
            "LEFT_HIP_X": 0.0,
            "RIGHT_HIP_X": 0.0,
            "LEFT_HIP_Y": -0.0,
            "RIGHT_HIP_Y": -0.0,
            "LEFT_KNEE": 0.0,
            "RIGHT_KNEE": 0.0,
            "LEFT_ANKLE_Y": 0.0,
            "RIGHT_ANKLE_Y": 0.0,
            "LEFT_ANKLE_X": 0.0,
            "RIGHT_ANKLE_X": 0.0,
            "SPINE_Z": 0.0,
            "SPINE_X": 0.0,
            "SPINE_Y": 0.0,
            "LEFT_SHOULDER_Y": 0.0,
            "RIGHT_SHOULDER_Y": 0.0,
            "LEFT_SHOULDER_X": 0.0,
            "RIGHT_SHOULDER_X": 0.0,
            "LEFT_SHOULDER_Z": 0.0,
            "RIGHT_SHOULDER_Z": 0.0,
            "LEFT_ELBOW_Y": 0.0,
            "RIGHT_ELBOW_Y": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP.*", ".*KNEE.*", ".*ANKLE.*"],
            stiffness={
                ".*HIP_Z": 120.0,
                ".*HIP_X": 120.0,
                ".*HIP_Y": 120.0,
                ".*KNEE":  80.0,
                ".*ANKLE_Y": 60.0,
                ".*ANKLE_X": 60.0,
            },
            damping={
                ".*HIP_Z": 2.0,
                ".*HIP_X": 2.0,
                ".*HIP_Y": 2.0,
                ".*KNEE":  1.33,
                ".*ANKLE_Y": 1.0,
                ".*ANKLE_X": 1.0,
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["SPINE.*"],
            stiffness={
                "SPINE_Z": 100.0,
                "SPINE_X": 200.0,
                "SPINE_Y": 200.0,
            },
            damping={
                "SPINE_Z": 5,
                "SPINE_X": 10,
                "SPINE_Y": 10,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*SHOULDER.*",".*ELBOW.*"],
            stiffness={
                ".*SHOULDER_Y": 10.0,
                ".*SHOULDER_X": 10.0,
                ".*SHOULDER_Z": 10.0,
                ".*ELBOW_Y":    10.0,
            },
            damping={
                ".*SHOULDER_Y": 0.3,
                ".*SHOULDER_X": 0.3,
                ".*SHOULDER_Z": 0.3,
                ".*ELBOW_Y":    0.3,
            },
        ),
    },
    # actuators={
    #     "legs": ImplicitActuatorCfg(
    #         joint_names_expr=[".*HIP.*", ".*KNEE.*", ".*ANKLE.*"],
    #         stiffness={
    #             ".*HIP_Z": 0.0,
    #             ".*HIP_X": 0.0,
    #             ".*HIP_Y": 0.0,
    #             ".*KNEE":  0.0,
    #             ".*ANKLE_Y": 0.0,
    #             ".*ANKLE_X": 0.0,
    #         },
    #         damping={
    #             ".*HIP_Z": 0.0,
    #             ".*HIP_X": 0.0,
    #             ".*HIP_Y": 0.0,
    #             ".*KNEE":  0.0,
    #             ".*ANKLE_Y": 0.0,
    #             ".*ANKLE_X": 0.0,
    #         },
    #     ),
    #     "torso": ImplicitActuatorCfg(
    #         joint_names_expr=["SPINE.*"],
    #         stiffness={
    #             "SPINE_Z": 0.0,
    #             "SPINE_X": 0.0,
    #             "SPINE_Y": 0.0,
    #         },
    #         damping={
    #             "SPINE_Z": 0,
    #             "SPINE_X": 0,
    #             "SPINE_Y": 0,
    #         },
    #     ),
    #     "arms": ImplicitActuatorCfg(
    #         joint_names_expr=[".*SHOULDER.*",".*ELBOW.*"],
    #         stiffness={
    #             ".*SHOULDER_Y": 0.0,
    #             ".*SHOULDER_X": 0.0,
    #             ".*SHOULDER_Z": 0.0,
    #             ".*ELBOW_Y":    0.0,
    #         },
    #         damping={
    #             ".*SHOULDER_Y": 0.0,
    #             ".*SHOULDER_X": 0.0,
    #             ".*SHOULDER_Z": 0.0,
    #             ".*ELBOW_Y":    0.0,
    #         },
    #     ),
    # },
)
