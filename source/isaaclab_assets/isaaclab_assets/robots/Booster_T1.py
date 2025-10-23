# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Booster T1 Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os


##
# Configuration
##

BOOSTER_T1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path = os.path.expanduser("~/IsaacLab-nomadz/source/isaaclab_assets/data/Robots/T1/t1.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
        
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.34),
        joint_pos={
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arms
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Legs
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 30.0,
                ".*_Hip_Yaw": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={".*_Ankle_Pitch": 24, ".*_Ankle_Roll": 15},
            velocity_limit_sim={".*_Ankle_Pitch": 18.8, ".*_Ankle_Roll": 12.4},
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim=18.0,
            velocity_limit_sim=18.8,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot."""
