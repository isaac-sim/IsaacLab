# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific Franka configuration for cube stacking."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import clone

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

FRANKA_PANDA_DEXSUITE_CFG = clone(FRANKA_PANDA_CFG)
FRANKA_PANDA_DEXSUITE_CFG.spawn.variants["Physics"] = "mujoco"
FRANKA_PANDA_DEXSUITE_CFG.init_state = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "panda_joint1": 0.0444,
        "panda_joint2": -0.1894,
        "panda_joint3": -0.1107,
        "panda_joint4": -2.5148,
        "panda_joint5": 0.0044,
        "panda_joint6": 2.3775,
        "panda_joint7": 0.6952,
        "panda_finger_joint.*": 0.04,
    },
)
# Keep the shared asset's passive mimic follower; tune only the active drives.
FRANKA_PANDA_DEXSUITE_CFG.actuators.update(
    {
        "panda_arm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            joint_effort_limit={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
            # Record the Panda's rated joint-speed envelope. MJWarp exposes but
            # does not enforce these fields; action scaling, gains, effort limits,
            # and armature determine the live Newton response.
            joint_velocity_limit={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
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
            viscous_friction=0.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint1"],
            joint_effort_limit=70.0,
            joint_velocity_limit=2.0,
            stiffness=350.0,
            damping=175.0,
            armature=0.1,
            viscous_friction=0.0,
        ),
    }
)
"""DexSuite-calibrated Franka with physical gravity enabled.

The arm controller combines these joint-impedance gains with configuration-
dependent gravity feedforward in the stack task's action term.
"""
