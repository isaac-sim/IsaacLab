# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Fiibot humanoid robot.

The following configuration parameters are available:

* :obj:`FIIBOT_CFG`: The Fiibot humanoid robot with wheeled base.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# For local debugging, use the local data directory
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

# Toggle between local (for debugging) and Nucleus (for production)
USE_LOCAL_ASSETS = True

if USE_LOCAL_ASSETS:
    FIIBOT_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Fiibot/Fiibot/fiibot.usd"
else:
    FIIBOT_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Fiibot/Fiibot/fiibot.usd"


FIIBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FIIBOT_USD_PATH,
        activate_contact_sensors=True,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.7071068, 0.0, 0.0, 0.7071068),
        joint_pos={
            "jack_joint": 0.7,
            "left_1_joint": 0.0,
            "left_2_joint": 0.785398,
            "left_3_joint": 0.0,
            "left_4_joint": 1.570796,
            "left_5_joint": 0.0,
            "left_6_joint": -0.785398,
            "left_7_joint": 0.0,
            "right_1_joint": 0.0,
            "right_2_joint": 0.785398,
            "right_3_joint": 0.0,
            "right_4_joint": 1.570796,
            "right_5_joint": 0.0,
            "right_6_joint": -0.785398,
            "right_7_joint": 0.0,
        },
    ),
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_[1-7]_joint"],
            stiffness=None,
            damping=None,
        ),
        "jack": ImplicitActuatorCfg(
            joint_names_expr=["jack_joint"],
            stiffness=500000.0,
            damping=5000.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_.*"],
            stiffness=None,
            damping=None,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_.*", "walk_.*"],
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration for the Fiibot humanoid robot with wheeled base.

The Fiibot is a wheeled humanoid robot featuring:
- Dual 7-DOF arms with grippers for manipulation
- Three-wheeled swerve drive base for omnidirectional movement
- Height-adjustable jack mechanism for variable working height
- Suitable for locomanipulation and pick-and-place tasks
"""

