# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators_warp import ImplicitActuatorWarpCfg
from isaaclab.assets import ArticulationCfg, ArticulationWarpCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

HUMANOID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Humanoid/humanoid_instanceable.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.34),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            control_mode="position",
            stiffness={
                ".*_waist.*": 20.0,
                ".*_upper_arm.*": 10.0,
                "pelvis": 10.0,
                ".*_lower_arm": 2.0,
                ".*_thigh:0": 10.0,
                ".*_thigh:1": 20.0,
                ".*_thigh:2": 10.0,
                ".*_shin": 5.0,
                ".*_foot.*": 2.0,
            },
            damping={
                ".*_waist.*": 5.0,
                ".*_upper_arm.*": 5.0,
                "pelvis": 5.0,
                ".*_lower_arm": 1.0,
                ".*_thigh:0": 5.0,
                ".*_thigh:1": 5.0,
                ".*_thigh:2": 5.0,
                ".*_shin": 0.1,
                ".*_foot.*": 1.0,
            },
            velocity_limit_sim={".*": 100.0},
            armature={".*": 0.01},
            friction=0.00001,
            effort_limit_sim=150.0,
        ),
    },
)
"""Configuration for the Mujoco Humanoid robot."""

HUMANOID_CFG_WARP = ArticulationWarpCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Humanoid/humanoid_instanceable.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationWarpCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.34),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorWarpCfg(
            joint_names_expr=[".*"],
            control_mode="position",
            stiffness={
                ".*_waist.*": 20.0,
                ".*_upper_arm.*": 10.0,
                "pelvis": 10.0,
                ".*_lower_arm": 2.0,
                ".*_thigh:0": 10.0,
                ".*_thigh:1": 20.0,
                ".*_thigh:2": 10.0,
                ".*_shin": 5.0,
                ".*_foot.*": 2.0,
            },
            damping={
                ".*_waist.*": 5.0,
                ".*_upper_arm.*": 5.0,
                "pelvis": 5.0,
                ".*_lower_arm": 1.0,
                ".*_thigh:0": 5.0,
                ".*_thigh:1": 5.0,
                ".*_thigh:2": 5.0,
                ".*_shin": 0.1,
                ".*_foot.*": 1.0,
            },
            velocity_limit_sim={".*": 100.0},
            armature={".*": 0.01},
            friction=0.00001,
            effort_limit_sim=150.0,
        ),
    },
)
"""Configuration for the Mujoco Humanoid robot."""
