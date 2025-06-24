# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`SO_100_CFG`: SO-100 robot

Reference: https://github.com/TheRobotStudio/SO-ARM100
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

SO_100_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Users/ashwinvk@nvidia.com/so-100/so-100_pure_usd.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "a_1": 0.0,
            "a_2": 0.0,
            "a_3": 0.0,
            "a_4": 0.0,
            "a_5": 0.0,
            "a_6": 0.0,
        },
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=["a_[1-5]"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=0.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["a_6"],
            effort_limit_sim=0.1,
            velocity_limit_sim=2.175,
            stiffness=8.0,
            damping=0.4,
            armature=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


SO_100_HIGH_PD_CFG = SO_100_CFG.copy()
SO_100_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# SO_100_HIGH_PD_CFG.actuators["all_joints"].stiffness = 400.0
# SO_100_HIGH_PD_CFG.actuators["all_joints"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
