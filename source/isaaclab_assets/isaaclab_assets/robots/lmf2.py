# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the NTNU robots.

The following configuration parameters are available:

* :obj:`LMF2_CFG`: The LMF2 robot with (TODO add motor propeller combination)
"""

from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ThrusterCfg
from isaaclab.assets.articulation import ArticulationWithThrustersCfg
# from isaaclab.sensors import RayCasterCfg

from isaaclab import ISAACLAB_EXT_DIR

##
# Configuration - Actuators.
##

LMF2_THRUSTER = ThrusterCfg()

##
# Configuration - Articulation.
##

LMF2_CFG = ArticulationWithThrustersCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_EXT_DIR}/../isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/LMF2/LMF2_model/lmf2/lmf2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationWithThrustersCfg.InitialThrusterStateCfg(
        pos=(0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0,0.0),
        rps={
            "base_link_to_back_left_prop": 200.0,  
            "base_link_to_back_right_prop": 200.0, 
            "base_link_to_front_left_prop": 200.0, 
            "base_link_to_front_right_prop": 200.0,  
        },
    ),
    actuators={"thrusters": LMF2_THRUSTER},
    soft_joint_pos_limit_factor=0.95,
    rotor_directions=[1, -1, 1, -1],
    allocation_matrix=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.07, 0.07, -0.07, 0.07],
        ]
)

##
# Configuration - Sensors.
##