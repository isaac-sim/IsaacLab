# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base Rizon4s and Sharpa hand with a measured rear-housing pregrasp."""

from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.rizon_sharpa import (
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    ROBOT_USD_PATH,
    arm_actuators,
)

from .sharpa_contacts import spawn_sharpa_contact_meshes

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
            usd_path=ROBOT_USD_PATH,
            func=spawn_sharpa_contact_meshes,
            collision_props={
                "/.*": [NewtonCollisionCfg(contact_gap=0.001)],
                # Tucked digits are visual-only for this three-finger pinch.
                "/right_(?:ring|pinky)_.*/.*": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)],
            },
            joint_drive_props={"/Physics/.*": [MujocoJointCfg(actuatorgravcomp=True)]},
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POSITION_E,
            rot=ROBOT_BASE_ROTATION_XYZW,
            joint_pos={
                **dict(zip(ARM_JOINT_NAMES, PREGRASP_ARM_RAD, strict=True)),
                **dict(zip(HAND_JOINT_NAMES, CLOSE_HAND_RAD, strict=True)),
            },
        ),
        actuators={
            **arm_actuators(),
            "right_thumb": ImplicitActuatorCfg(
                joint_names_expr=["right_thumb_.*"],
                joint_velocity_limit=6.0,
                stiffness=96.0,
                damping=2.4,
                armature=2.0e-4,
            ),
            "right_fingers": ImplicitActuatorCfg(
                joint_names_expr=["right_(?:index|middle|ring|pinky)_.*"],
                joint_velocity_limit=6.0,
                stiffness=96.0,
                damping=2.4,
                armature=2.0e-4,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )
