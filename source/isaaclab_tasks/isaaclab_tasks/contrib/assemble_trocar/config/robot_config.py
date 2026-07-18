# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot configuration for the `install_trocar` task.

This file is intentionally **minimal**:
- Supported robot: **Unitree G1 (29 DOF body)**
- Supported hands: **Dex3**

The only public entry point expected by the task is
`G1RobotPresets.g1_29dof_dex3_base_fix(...)`.
"""

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX

# Joint indices in the full robot joint vector for observation extraction.
# Body joints: 29 DOF (legs, waist, arms, wrists)
G1_29DOF_BODY_JOINT_INDICES: list[int] = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

# Dex3 hand joints: 14 DOF (left + right)
G1_DEX3_JOINT_INDICES: list[int] = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]

# Default joint positions for the supported setup (G1 29DOF + Dex3).
DEFAULT_JOINT_POS: dict[str, float] = {
    # legs
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": -0.754599,
    "left_shoulder_roll_joint": 0.550010,
    "left_shoulder_yaw_joint": -0.399298,
    "left_elbow_joint": 0.278886,
    "left_wrist_roll_joint": 0.320559,
    "left_wrist_pitch_joint": -0.203525,
    "left_wrist_yaw_joint": -0.387435,
    "right_shoulder_pitch_joint": -0.340858,
    "right_shoulder_roll_joint": -0.186152,
    "right_shoulder_yaw_joint": 0.015023,
    "right_elbow_joint": -0.777159,
    "right_wrist_roll_joint": 0.019805,
    "right_wrist_pitch_joint": 1.182285,
    "right_wrist_yaw_joint": -0.022848,
    # dex3 hands (left)
    "left_hand_index_0_joint": -60.0 * math.pi / 180.0,
    "left_hand_middle_0_joint": -60.0 * math.pi / 180.0,
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_index_1_joint": -40.0 * math.pi / 180.0,
    "left_hand_middle_1_joint": -40.0 * math.pi / 180.0,
    "left_hand_thumb_1_joint": 0.0,
    "left_hand_thumb_2_joint": 0.0,
    # dexterous hand joint - right hand
    "right_hand_index_0_joint": 60.0 * math.pi / 180.0,
    "right_hand_middle_0_joint": 60.0 * math.pi / 180.0,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_index_1_joint": 40.0 * math.pi / 180.0,
    "right_hand_middle_1_joint": 40.0 * math.pi / 180.0,
    "right_hand_thumb_1_joint": 0.0,
    "right_hand_thumb_2_joint": 0.0,
}


def make_g1_29dof_dex3_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    init_pos: tuple[float, float, float] = (-0.15, 0.0, 0.744),
    init_rot: tuple[float, float, float, float] = (0, 0, 0.7071, 0.7071),
    custom_joint_pos: dict[str, float] | None = None,
    base_config: ArticulationCfg = G129_CFG_WITH_DEX3_BASE_FIX,
) -> ArticulationCfg:
    """Create the only supported robot articulation cfg for this task."""
    joint_pos = DEFAULT_JOINT_POS.copy()
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return base_config.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class G1RobotPresets:
    """G1 robot preset configuration collection"""

    @classmethod
    def g1_29dof_dex3_base_fix(
        cls,
        init_pos: tuple[float, float, float] = (-0.15, 0.0, 0.76),
        init_rot: tuple[float, float, float, float] = (0, 0, 0.7071, 0.7071),
    ) -> ArticulationCfg:
        """pick-place task configuration - dex3 hand"""
        return make_g1_29dof_dex3_cfg(init_pos=init_pos, init_rot=init_rot)
