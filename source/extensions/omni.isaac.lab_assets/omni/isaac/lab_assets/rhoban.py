"""Configuration for Rhoban robots.

The following configurations are available:

* :obj:`SIGMBAN_CFG`: Sigmaban humanoid robot

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg


SIGMABAN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/extensions/omni.isaac.lab_assets/data/Robots/Rhoban/sigmaban_mjcf.usda",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.35),
        rot=(0.997, 0.0, 0.074, 0.0),
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0, 
            ".*_hip_pitch": -0.907571, #-52deg
            ".*_knee": 1.37881, #79deg
            ".*_ankle_pitch": -0.6283185, #-36.5deg
            ".*_ankle_roll": 0.0,
            ".*_shoulder_pitch": 0.3316126, #19.5deg
            ".*_shoulder_roll": 0.0,
            ".*_elbow": -0.8552113, #-49.5deg
        },
        # joint_pos={
        #     "head_yaw": 0.0,
        #     "head_pitch": 0.0,
        #     ".*_hip_yaw": 0.0,
        #     ".*_hip_roll": 0.0, 
        #     ".*_hip_pitch": 0.0,
        #     ".*_knee": 0.0,
        #     ".*_ankle_pitch": 0.0,
        #     ".*_ankle_roll": 0.0,
        #     ".*_shoulder_pitch": 0.8,
        #     "left_shoulder_roll": -0.0872665,
        #     "right_shoulder_roll": 0.0872665,
        #     ".*_elbow": 0.0,
        # },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0, # 100% of the joint limits
    actuators={
        "MX64": IdealPDActuatorCfg(
            joint_names_expr=["head_yaw", "head_pitch", ".*_hip_yaw", ".*_shoulder_pitch", ".*_shoulder_roll", ".*_elbow"],
            # saturation_effort=7,
            effort_limit=5, #5Nm
            # velocity_limit=6.2831, #2*pi rad/s
            armature=0.012,
            stiffness=12.5, # considered as kp
            friction=0.09,
            damping= 0.66,
        ),
        "MX106": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_roll", ".*_hip_pitch", ".*_knee", ".*_ankle_pitch", ".*_ankle_roll"],
            # saturation_effort=10, #8Nm
            effort_limit=8, #8Nm
            # velocity_limit=6.2831, #2*pi rad/s
            armature=0.025,
            stiffness=21, # considered as kp
            friction=0.10,
            damping= 1.7,
        ),
    },
)
"""Configuration for the Rhoban Sigmaban Humanoid robot."""