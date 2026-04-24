# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a tendon-actuated capsule robot.

The following configurations are available:

* :obj:`CAPSULES_CFG`: Capsule robot with fixed tendon actuators.

"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

##
# Configuration
##

CAPSULES_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/rgresia/Repositories/tendon-assets/tendon-capsules.usda/capsules.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
    ),
    actuators={},
)
"""Configuration of Spatial Tendon Finger robot."""