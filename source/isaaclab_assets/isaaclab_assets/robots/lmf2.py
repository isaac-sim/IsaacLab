# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the NTNU robots.

The following configuration parameters are available:

* :obj:`LMF2_CFG`: The LMF2 robot with (TODO add motor propeller combination)
"""

from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import ThrusterLMF2Cfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.sensors import RayCasterCfg

from isaaclab import ISAACLAB_EXT_DIR

##
# Configuration - Actuators.
##

LMF2_THRUSTER = ThrusterLMF2Cfg()

##
# Configuration - Articulation.
##

LMF2_CFG = ArticulationCfg(
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
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "base_link": 0.0,  # all HAA
            "back_left_prop": 0.0,  # both front HFE
            "back_right_prop": 0.0,  # both hind HFE
            "front_left_prop": 0.0,  # both front KFE
            "front_right_prop": 0.0,  # both hind KFE
        },
    ),
    actuators={"thrusters": LMF2_THRUSTER},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-C robot using actuator-net."""

##
# Configuration - Sensors.
##

#TODO add depth camera

# ANYMAL_LIDAR_CFG = VELODYNE_VLP_16_RAYCASTER_CFG.replace(
#     offset=RayCasterCfg.OffsetCfg(pos=(-0.310, 0.000, 0.159), rot=(0.0, 0.0, 0.0, 1.0))
# )
# """Configuration for the Velodyne VLP-16 sensor mounted on the ANYmal robot's base."""
