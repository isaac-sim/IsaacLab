# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

LEG_JOINT_NAMES = [
    ".*_hip_roll",
    ".*_hip_yaw",
    ".*_hip_pitch",
    ".*_knee",
    ".*_toe_a",
    ".*_toe_b",
]

ARM_JOINT_NAMES = [".*_arm_.*"]


DIGIT_V4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=".*",
            stiffness=None,
            damping=None,
        ),
    },
)
