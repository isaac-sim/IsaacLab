# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base Rizon4s and Sharpa hand with a measured rear-housing pregrasp."""

from isaaclab_newton.sim.schemas import MujocoJointCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .assets import (
    RIZON_SHARPA_ARM_JOINT_NAMES,
    RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES,
    RIZON_SHARPA_USD_PATH,
)

ROBOT_BASE_POSITION_E = (0.40706384, -0.44981234, 1.78624670)
ROBOT_BASE_ROTATION_XYZW = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
PREGRASP_ARM_RAD = (-0.28426774, -0.68145812, 0.31863682, 2.04760345, 0.40848495, 2.17470681, -2.85753436)
# Thumb on the housing roof, index and middle underneath; no clip compression.
CLOSE_HAND_RAD = (
    0.84772129,
    -0.10509452,
    0.67820725,
    -0.10205317,
    0.00001000,
    0.83700016,
    -0.08471779,
    0.86494442,
    0.00001000,
    1.24722315,
    -0.19387218,
    0.40609412,
    0.49765344,
    1.1,
    0.0,
    1.4,
    1.0,
    0.0,
    1.1,
    0.0,
    1.4,
    1.0,
)


def rizon_sharpa_articulation_cfg() -> ArticulationCfg:
    """Return the fixed-base arm and 22-DoF hand in the collision-checked pregrasp."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RIZON_SHARPA_USD_PATH,
            joint_drive_props={"/Physics/.*": [MujocoJointCfg(actuatorgravcomp=True)]},
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POSITION_E,
            rot=ROBOT_BASE_ROTATION_XYZW,
            joint_pos={
                **dict(zip(RIZON_SHARPA_ARM_JOINT_NAMES, PREGRASP_ARM_RAD, strict=True)),
                **dict(zip(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, CLOSE_HAND_RAD, strict=True)),
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                joint_effort_limit=123.0,
                joint_velocity_limit=2.094,
                stiffness=6000.0,
                damping=108.5,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-4]"],
                joint_effort_limit=64.0,
                joint_velocity_limit=2.443,
                stiffness=4200.0,
                damping=90.7,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-7]"],
                joint_effort_limit=39.0,
                joint_velocity_limit=4.887,
                stiffness=1500.0,
                damping=54.2,
            ),
            "right_thumb": ImplicitActuatorCfg(
                joint_names_expr=["right_thumb_.*"],
                joint_effort_limit=3.3,
                joint_velocity_limit=6.0,
                stiffness=24.0,
                damping=1.2,
            ),
            "right_fingers": ImplicitActuatorCfg(
                joint_names_expr=["right_(?:index|middle|ring|pinky)_.*"],
                joint_effort_limit=3.3,
                joint_velocity_limit=6.0,
                stiffness=24.0,
                damping=1.2,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )
