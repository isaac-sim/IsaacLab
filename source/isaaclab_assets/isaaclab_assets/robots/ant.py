# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Ant robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

ANT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Ant/ant_instanceable.usd",
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*_leg": 0.0,
            "front_left_foot": 0.785398,  # 45 degrees
            "front_right_foot": -0.785398,
            "left_back_foot": -0.785398,
            "right_back_foot": 0.785398,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            control_mode="none",
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=30.0,
            friction=0.0001,
        ),
    },
)
"""Configuration for the Mujoco Ant robot."""
