"""Configuration for the Eigenbot modular robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Path to the USDZ file (co-located in this assets directory)
EIGENBOT_USD_PATH = os.path.join(os.path.dirname(__file__), "eigenbot_new.usdz")

EIGENBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=EIGENBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "bendy_joints": ImplicitActuatorCfg(
            joint_names_expr=["bendy_joint_.*"],
            effort_limit_sim=100.0,
            stiffness=40.0,
            damping=5.0,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["wheel_joint_.*"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint_.*"],
            effort_limit_sim=100.0,
            stiffness=40.0,
            damping=5.0,
        ),
    },
)
