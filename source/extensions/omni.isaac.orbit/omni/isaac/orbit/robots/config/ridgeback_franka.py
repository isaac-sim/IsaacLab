# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Ridgeback-Manipulation robots.

The following configurations are available:

* :obj:`RIDGEBACK_FRANKA_PANDA_CFG`: Clearpath Ridgeback base with Franka Emika arm

Reference: https://github.com/ridgeback/ridgeback_manipulation
"""


from omni.isaac.orbit.actuators.config.franka import PANDA_HAND_MIMIC_GROUP_CFG
from omni.isaac.orbit.actuators.group import ActuatorControlCfg, ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from ..mobile_manipulator import MobileManipulatorCfg

_RIDGEBACK_FRANKA_PANDA_ARM_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"


RIDGEBACK_FRANKA_PANDA_CFG = MobileManipulatorCfg(
    meta_info=MobileManipulatorCfg.MetaInfoCfg(
        usd_path=_RIDGEBACK_FRANKA_PANDA_ARM_USD,
        base_num_dof=3,
        arm_num_dof=7,
        tool_num_dof=2,
        tool_sites_names=["panda_leftfinger", "panda_rightfinger"],
    ),
    init_state=MobileManipulatorCfg.InitialStateCfg(
        dof_pos={
            # base
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            # franka arm
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            # tool
            "panda_finger_joint*": 0.035,
        },
        dof_vel={".*": 0.0},
    ),
    ee_info=MobileManipulatorCfg.EndEffectorFrameCfg(
        body_name="panda_hand", pos_offset=(0.0, 0.0, 0.1034), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),
    actuator_groups={
        "base": ActuatorGroupCfg(
            dof_names=["dummy_base_.*"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=1000.0),
            control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 0.0}, damping={".*": 1e5}),
        ),
        "panda_shoulder": ActuatorGroupCfg(
            dof_names=["panda_joint[1-4]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
                dof_pos_offset={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                },
            ),
        ),
        "panda_forearm": ActuatorGroupCfg(
            dof_names=["panda_joint[5-7]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=12.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
                dof_pos_offset={"panda_joint5": 0.0, "panda_joint6": 3.037, "panda_joint7": 0.741},
            ),
        ),
        "panda_hand": PANDA_HAND_MIMIC_GROUP_CFG,
    },
)
"""Configuration of Franka arm with Franka Hand on a Clearpath Ridgeback base using implicit actuator models.

The following control configuration is used:

* Base: velocity control with damping
* Arm: position control with damping (contains default position offsets)
* Hand: mimic control

"""
