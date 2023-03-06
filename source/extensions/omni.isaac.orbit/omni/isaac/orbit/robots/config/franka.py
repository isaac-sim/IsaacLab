# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG`: Franka Emika Panda robot with Panda hand

Reference: https://github.com/frankaemika/franka_ros
"""


from omni.isaac.orbit.actuators.config.franka import PANDA_HAND_MIMIC_GROUP_CFG
from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

from ..single_arm import SingleArmManipulatorCfg

_FRANKA_PANDA_ARM_INSTANCEABLE_USD = f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"


FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG = SingleArmManipulatorCfg(
    meta_info=SingleArmManipulatorCfg.MetaInfoCfg(
        usd_path=_FRANKA_PANDA_ARM_INSTANCEABLE_USD,
        arm_num_dof=7,
        tool_num_dof=2,
        tool_sites_names=["panda_leftfinger", "panda_rightfinger"],
    ),
    init_state=SingleArmManipulatorCfg.InitialStateCfg(
        dof_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint*": 0.04,
        },
        dof_vel={".*": 0.0},
    ),
    ee_info=SingleArmManipulatorCfg.EndEffectorFrameCfg(
        body_name="panda_hand", pos_offset=(0.0, 0.0, 0.1034), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),
    rigid_props=SingleArmManipulatorCfg.RigidBodyPropertiesCfg(
        max_depenetration_velocity=5.0,
    ),
    collision_props=SingleArmManipulatorCfg.CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0,
    ),
    articulation_props=SingleArmManipulatorCfg.ArticulationRootPropertiesCfg(
        enable_self_collisions=True,
    ),
    actuator_groups={
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
"""Configuration of Franka arm with Franka Hand using implicit actuator models."""
