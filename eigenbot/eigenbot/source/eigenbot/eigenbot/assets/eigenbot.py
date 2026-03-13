"""Configuration for the Eigenbot hexapod modular robot."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg

# Path to the URDF file
EIGENBOT_URDF_PATH = os.path.join(os.path.dirname(__file__), "eigenbot", "urdf", "eigenbot_hexapod.urdf")

EIGENBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=EIGENBOT_URDF_PATH,
        force_usd_conversion=True,
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=20.0,
                damping=0.5,
            ),
        ),
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
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            "bendy_joint_M1_S1": -math.pi / 4,
            "bendy_joint_M2_S2": 0.0,
            "bendy_joint_M3_S3": math.pi / 4,
            "bendy_joint_M4_S4": -math.pi / 4,
            "bendy_joint_M5_S5": 0.0,
            "bendy_joint_M6_S6": math.pi / 4,
            "bendy_joint_M7_S7": math.pi / 4,
            "bendy_joint_M8_S8": math.pi / 4,
            "bendy_joint_M9_S9": math.pi / 4,
            "bendy_joint_M10_S10": math.pi / 4,
            "bendy_joint_M11_S11": math.pi / 4,
            "bendy_joint_M12_S12": math.pi / 4,
            "bendy_joint_M13_S13": math.pi / 4,
            "bendy_joint_M14_S14": math.pi / 4,
            "bendy_joint_M15_S15": math.pi / 4,
            "bendy_joint_M16_S16": math.pi / 4,
            "bendy_joint_M17_S17": math.pi / 4,
            "bendy_joint_M18_S18": math.pi / 4,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "bendy_joints": ImplicitActuatorCfg(
            joint_names_expr=["bendy_joint_.*"],
            effort_limit_sim=100.0,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)
