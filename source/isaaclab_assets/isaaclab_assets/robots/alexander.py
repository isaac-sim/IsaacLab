"""Configuration for IHMC robots.

The following configurations are available:

* :obj:`ALEXANDER_V1`: 

"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


ALEXANDER_V1 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="../nadia-orbit-data/robot/alexander/usds/v1/test_flat2.usd",
        activate_contact_sensors=False,
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
        pos=(0.0, 0.0, 0.96),
        joint_pos={
            "NECK.*": 0.0,
            "SPINE.*": 0.0,
            ".*HIP_X": 0.0,
            ".*HIP_Y": 0.0,
            ".*HIP_Z": 0.0,
            ".*KNEE_Y": 0.0,
            ".*ANKLE_Y": 0.0,
            ".*ANKLE_X": 0.0,
            ".*SHOULDER.*": 0.0,
            ".*ELBOW.*": 0.0,
            ".*WRIST.*": 0.0,
            ".*GRIPPER.*": 0.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP.*", ".*KNEE.*", ".*ANKLE.*"],
            stiffness={
                ".*HIP_X": 100.0,
                ".*HIP_Y": 100.0,
                ".*HIP_Z": 100.0,
                ".*KNEE_Y": 100.0,
                ".*ANKLE_Y": 20.0,
                ".*ANKLE_X": 20.0,
            },
            damping={".*": 2.0},
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["SPINE.*"],
            stiffness={".*": 100.0},
            damping={".*": 2.0},
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=[".*SHOULDER.*", ".*ELBOW.*", ".*WRIST.*", ".*GRIPPER.*"],
            stiffness={".*": 20.0},
            damping={".*": 2.0},
        ),
    },
    
)
