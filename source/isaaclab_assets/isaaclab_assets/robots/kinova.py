# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kinova Robotics arms.

The following configuration parameters are available:

* :obj:`KINOVA_JACO2_N7S300_CFG`: The Kinova JACO2 (7-Dof) arm with a 3-finger gripper.
* :obj:`KINOVA_JACO2_N6S300_CFG`: The Kinova JACO2 (6-Dof) arm with a 3-finger gripper.
* :obj:`KINOVA_GEN3_N7_CFG`: The Kinova Gen3 (7-Dof) arm with no gripper.

Reference: https://github.com/Kinovarobotics/kinova-ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

KINOVA_JACO2_N7S300_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Kinova/Jaco2/J2N7S300/j2n7s300_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "j2n7s300_joint_1": 0.0,
            "j2n7s300_joint_2": 2.76,
            "j2n7s300_joint_3": 0.0,
            "j2n7s300_joint_4": 2.0,
            "j2n7s300_joint_5": 2.0,
            "j2n7s300_joint_6": 0.0,
            "j2n7s300_joint_7": 0.0,
            "j2n7s300_joint_finger_[1-3]": 0.2,  # close: 1.2, open: 0.2
            "j2n7s300_joint_finger_tip_[1-3]": 0.2,  # close: 1.2, open: 0.2
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint_[1-7]"],
            effort_limit_sim={
                ".*_joint_[1-2]": 80.0,
                ".*_joint_[3-4]": 40.0,
                ".*_joint_[5-7]": 20.0,
            },
            stiffness={
                ".*_joint_[1-4]": 40.0,
                ".*_joint_[5-7]": 15.0,
            },
            damping={
                ".*_joint_[1-4]": 1.0,
                ".*_joint_[5-7]": 0.5,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger_[1-3]", ".*_finger_tip_[1-3]"],
            effort_limit_sim=2.0,
            stiffness=1.2,
            damping=0.01,
        ),
    },
)
"""Configuration of Kinova JACO2 (7-Dof) arm with 3-finger gripper."""


KINOVA_JACO2_N6S300_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "j2n6s300_joint_1": 0.0,
            "j2n6s300_joint_2": 2.76,
            "j2n6s300_joint_3": 2.76,
            "j2n6s300_joint_4": 2.5,
            "j2n6s300_joint_5": 2.0,
            "j2n6s300_joint_6": 0.0,
            "j2n6s300_joint_finger_[1-3]": 0.2,  # close: 1.2, open: 0.2
            "j2n6s300_joint_finger_tip_[1-3]": 0.2,  # close: 1.2, open: 0.2
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint_[1-6]"],
            effort_limit_sim={
                ".*_joint_[1-2]": 80.0,
                ".*_joint_3": 40.0,
                ".*_joint_[4-6]": 20.0,
            },
            stiffness={
                ".*_joint_[1-3]": 40.0,
                ".*_joint_[4-6]": 15.0,
            },
            damping={
                ".*_joint_[1-3]": 1.0,
                ".*_joint_[4-6]": 0.5,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger_[1-3]", ".*_finger_tip_[1-3]"],
            effort_limit_sim=2.0,
            stiffness=1.2,
            damping=0.01,
        ),
    },
)
"""Configuration of Kinova JACO2 (6-Dof) arm with 3-finger gripper."""


KINOVA_GEN3_N7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Kinova/Gen3/gen3n7_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.65,
            "joint_3": 0.0,
            "joint_4": 1.89,
            "joint_5": 0.0,
            "joint_6": 0.6,
            "joint_7": -1.57,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            effort_limit={
                "joint_[1-4]": 39.0,
                "joint_[5-7]": 9.0,
            },
            stiffness={
                "joint_[1-4]": 40.0,
                "joint_[5-7]": 15.0,
            },
            damping={
                "joint_[1-4]": 1.0,
                "joint_[5-7]": 0.5,
            },
        ),
    },
)
"""Configuration of Kinova Gen3 (7-Dof) arm with no gripper."""
