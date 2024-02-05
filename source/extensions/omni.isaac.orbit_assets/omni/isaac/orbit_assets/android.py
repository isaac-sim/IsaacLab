"""Configuration for Fourier humanoid robots.

The following configurations are available:

"""

from __future__ import annotations

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
# from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR
from omni.isaac.orbit_assets import ORBIT_ASSETS_DATA_DIR

##
# Configuration
##

ANDROID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ORBIT_ASSETS_DATA_DIR}/Robots/Fourier/gr1t1_22dof.usd",
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
            articulation_enabled=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.935),
        rot=(1.0, 0.0, 0.0, 0.0),   # Quaternion rotation (w, x, y, z) of the root in simulation world frame.
        joint_pos={
            # left leg
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.,
            'l_hip_pitch': -0.5236,
            'l_knee_pitch': 1.0472,
            'l_ankle_pitch': -0.5236,
            'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': -0.,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.5236,
            'r_knee_pitch': 1.0472,
            'r_ankle_pitch': -0.5236,
            'r_ankle_roll': 0.0,

            # waist
            'waist_yaw': 0.0,
            'waist_pitch': 0.1,
            'waist_roll': 0.0,

            # head
            'head_yaw': 0.0,
            'head_pitch': 0.0,
            'head_roll': 0.0,

            # left arm
            'l_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.3,
            'l_shoulder_yaw': 0.3,
            'l_elbow_pitch': -0.1,
            'l_wrist_yaw': 0.0,
            'l_wrist_roll': 0.0,
            'l_wrist_pitch': 0.0,

            # right arm
            'r_shoulder_pitch': 0.0,
            'r_shoulder_roll': -0.3,
            'r_shoulder_yaw': 0.3,
            'r_elbow_pitch': -0.1,
            'r_wrist_yaw': 0.0,
            'r_wrist_roll': 0.0,
            'r_wrist_pitch': 0.0    
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,        
    actuators={ # TODO: check effort_limit and velocity_limit
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness={
                ".*hip_roll": 251.625,
                ".*hip_yaw": 453.15,
                ".*hip_pitch": 285.8131,
                ".*knee_pitch": 285.8131,
                ".*ankle_pitch": 21.961,
                ".*ankle_roll": 2.0761,
            },
            damping={
                ".*hip_roll": 14.72,
                ".*hip_yaw": 50.4164,
                ".*hip_pitch": 16.5792,
                ".*knee_pitch": 16.5792,
                ".*ankle_pitch": 1.195,
                ".*ankle_roll": 0.1233,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*"],
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness={
                "waist_.*": 453.15,
            },
            damping={
                "waist_.*": 50.4164,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_.*"],
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness={
                "head_.*": 100.0,
            },
            damping={
                "head_.*": 1.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness={
                ".*shoulder_pitch": 92.85,
                ".*shoulder_roll": 92.85,
                ".*shoulder_yaw": 112.06,
                ".*elbow_pitch": 112.06,
                "wrist_.*": 10.0,
            },
            damping={
                ".*shoulder_pitch": 2.575,
                ".*shoulder_roll": 2.575,
                ".*shoulder_yaw": 3.1,
                ".*elbow_pitch": 3.1,
                "wrist_.*": 1.0,
            },
        ),
    },
)