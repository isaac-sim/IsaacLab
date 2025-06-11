# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple pick and place robot with a suction cup."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

PICK_AND_PLACE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Tests/PickAndPlace/pick_and_place_robot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "x_axis": 0.0,
            "y_axis": 0.0,
            "z_axis": 0.0,
        },
    ),
    actuators={
        "x_gantry": ImplicitActuatorCfg(
            joint_names_expr=["x_axis"],
            effort_limit=400.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "y_gantry": ImplicitActuatorCfg(
            joint_names_expr=["y_axis"],
            effort_limit=400.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "z_gantry": ImplicitActuatorCfg(
            joint_names_expr=["z_axis"],
            effort_limit=400.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)
"""Configuration for a simple pick and place robot with a suction cup."""
