# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots.

The following configuration parameters are available:

* :obj:`ANYMAL_B_CFG`: The ANYmal-B robot with ANYdrives 3.0
* :obj:`ANYMAL_C_CFG`: The ANYmal-C robot with ANYdrives 3.0
* :obj:`ANYMAL_D_CFG`: The ANYmal-D robot with ANYdrives 3.0

Reference:

* https://github.com/ANYbotics/anymal_b_simple_description
* https://github.com/ANYbotics/anymal_c_simple_description
* https://github.com/ANYbotics/anymal_d_simple_description

"""


import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##

ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    control_mode="position",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)
"""Configuration for ANYdrive 3.x with DC actuator model."""

##
# Configuration - Articulation.
##

ANYMAL_D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-D robot using simple DC motor model.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""
