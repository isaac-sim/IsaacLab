# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-calibrated Franka configuration for conveyor manipulation."""

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG

FRANKA_PANDA_CONVEYOR_CFG = FRANKA_PANDA_MENAGERIE_CFG.copy()
FRANKA_PANDA_CONVEYOR_CFG.spawn.rigid_props.disable_gravity = False
FRANKA_PANDA_CONVEYOR_CFG.actuators = {
    "panda_arm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        effort_limit_sim={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
        velocity_limit_sim={"panda_joint[1-4]": 20.0, "panda_joint[5-7]": 25.0},
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
        # Keep the 0.1 kg m^2 armature close to critical damping so the
        # fingers establish contact within a few 50 Hz policy steps.
        damping=10.0,
        armature=0.1,
    ),
}
"""Menagerie Franka with explicit manipulation gains and gravity compensation-ready dynamics."""
