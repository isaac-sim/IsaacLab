# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "AGIBOT_A2D_CFG",
    "LEG_JOINT_NAMES",
    "ARM_JOINT_NAMES",
    "DIGIT_V4_CFG",
    "ALLEGRO_HAND_CFG",
    "ANT_CFG",
    "ANYDRIVE_3_SIMPLE_ACTUATOR_CFG",
    "ANYDRIVE_3_LSTM_ACTUATOR_CFG",
    "ANYMAL_B_CFG",
    "ANYMAL_C_CFG",
    "ANYMAL_D_CFG",
    "ANYMAL_LIDAR_CFG",
    "CART_DOUBLE_PENDULUM_CFG",
    "CARTPOLE_CFG",
    "CASSIE_CFG",
    "GR1T2_CFG",
    "GR1T2_HIGH_PD_CFG",
    "FRANKA_PANDA_CFG",
    "FRANKA_PANDA_HIGH_PD_CFG",
    "FRANKA_ROBOTIQ_GRIPPER_CFG",
    "GALBOT_ONE_CHARLIE_CFG",
    "HUMANOID_CFG",
    "HUMANOID_28_CFG",
    "KINOVA_JACO2_N7S300_CFG",
    "KINOVA_JACO2_N6S300_CFG",
    "KINOVA_GEN3_N7_CFG",
    "KUKA_ALLEGRO_CFG",
    "PICK_AND_PLACE_CFG",
    "CRAZYFLIE_CFG",
    "RIDGEBACK_FRANKA_PANDA_CFG",
    "SAWYER_CFG",
    "SHADOW_HAND_CFG",
    "joint_parameter_lookup",
    "SPOT_CFG",
    "GO1_ACTUATOR_CFG",
    "UNITREE_A1_CFG",
    "UNITREE_GO1_CFG",
    "UNITREE_GO2_CFG",
    "H1_CFG",
    "H1_MINIMAL_CFG",
    "G1_CFG",
    "G1_MINIMAL_CFG",
    "G1_29DOF_CFG",
    "G1_INSPIRE_FTP_CFG",
    "UR10_CFG",
    "UR10e_CFG",
    "UR10_LONG_SUCTION_CFG",
    "UR10_SHORT_SUCTION_CFG",
    "UR10e_ROBOTIQ_GRIPPER_CFG",
    "UR10e_ROBOTIQ_2F_85_CFG",
]

from .agibot import AGIBOT_A2D_CFG
from .agility import LEG_JOINT_NAMES, ARM_JOINT_NAMES, DIGIT_V4_CFG
from .allegro import ALLEGRO_HAND_CFG
from .ant import ANT_CFG
from .anymal import (
    ANYDRIVE_3_SIMPLE_ACTUATOR_CFG,
    ANYDRIVE_3_LSTM_ACTUATOR_CFG,
    ANYMAL_B_CFG,
    ANYMAL_C_CFG,
    ANYMAL_D_CFG,
    ANYMAL_LIDAR_CFG,
)
from .cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG
from .cartpole import CARTPOLE_CFG
from .cassie import CASSIE_CFG
from .fourier import GR1T2_CFG, GR1T2_HIGH_PD_CFG
from .franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG, FRANKA_ROBOTIQ_GRIPPER_CFG
from .galbot import GALBOT_ONE_CHARLIE_CFG
from .humanoid import HUMANOID_CFG
from .humanoid_28 import HUMANOID_28_CFG
from .kinova import KINOVA_JACO2_N7S300_CFG, KINOVA_JACO2_N6S300_CFG, KINOVA_GEN3_N7_CFG
from .kuka_allegro import KUKA_ALLEGRO_CFG
from .pick_and_place import PICK_AND_PLACE_CFG
from .quadcopter import CRAZYFLIE_CFG
from .ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from .sawyer import SAWYER_CFG
from .shadow_hand import SHADOW_HAND_CFG
from .spot import joint_parameter_lookup, SPOT_CFG
from .unitree import (
    GO1_ACTUATOR_CFG,
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
    H1_CFG,
    H1_MINIMAL_CFG,
    G1_CFG,
    G1_MINIMAL_CFG,
    G1_29DOF_CFG,
    G1_INSPIRE_FTP_CFG,
)
from .universal_robots import (
    UR10_CFG,
    UR10e_CFG,
    UR10_LONG_SUCTION_CFG,
    UR10_SHORT_SUCTION_CFG,
    UR10e_ROBOTIQ_GRIPPER_CFG,
    UR10e_ROBOTIQ_2F_85_CFG,
)
