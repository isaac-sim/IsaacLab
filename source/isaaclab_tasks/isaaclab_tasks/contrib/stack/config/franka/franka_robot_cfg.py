# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific Franka configuration for cube stacking."""

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG

FRANKA_PANDA_DEXSUITE_CFG = FRANKA_PANDA_MENAGERIE_CFG.copy()
FRANKA_PANDA_DEXSUITE_CFG.spawn.rigid_props.disable_gravity = False
FRANKA_PANDA_DEXSUITE_CFG.actuators = {
    "panda_arm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        effort_limit_sim={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
        # Record the Panda's rated joint-speed envelope. MJWarp exposes but
        # does not enforce these fields; action scaling, gains, effort limits,
        # and armature determine the live Newton response.
        velocity_limit_sim={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
        stiffness={
            "panda_joint[1-4]": 600.0,
            "panda_joint5": 250.0,
            "panda_joint6": 150.0,
            "panda_joint7": 50.0,
        },
        damping={
            "panda_joint[1-4]": 50.0,
            "panda_joint5": 30.0,
            "panda_joint6": 25.0,
            "panda_joint7": 15.0,
        },
        armature={
            "panda_joint[1-2]": 0.6057,
            "panda_joint[3-4]": 0.4625,
            "panda_joint[5-7]": 0.2055,
        },
    ),
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint[1-2]"],
        effort_limit_sim=70.0,
        velocity_limit_sim=2.0,
        stiffness=350.0,
        damping=175.0,
        armature=0.1,
    ),
}
"""DexSuite-calibrated Franka with physical gravity enabled.

The arm controller combines these joint-impedance gains with configuration-
dependent gravity feedforward in the stack task's action term.
"""
