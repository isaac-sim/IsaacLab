"""Canonical configuration for the K-Bot v2.0 (headless)."""

import os
from pathlib import Path
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# Use absolute path relative to this file's location
KBOT_USD = os.path.join(os.path.dirname(__file__), "temp_kbot_usd", "robot.usd")

_INIT_JOINT_POS = {
    "dof_right_shoulder_pitch_03": 0.0,
    "dof_right_shoulder_roll_03": math.radians(-10.0),
    "dof_right_shoulder_yaw_02": 0.0,
    "dof_right_elbow_02": math.radians(90.0),
    "dof_right_wrist_00": 0.0,
    "dof_left_shoulder_pitch_03": 0.0,
    "dof_left_shoulder_roll_03": math.radians(10.0),
    "dof_left_shoulder_yaw_02": 0.0,
    "dof_left_elbow_02": math.radians(-90.0),
    "dof_left_wrist_00": 0.0,
    "dof_right_hip_pitch_04": math.radians(-20.0),
    "dof_right_hip_roll_03": math.radians(0.0),
    "dof_right_hip_yaw_03": 0.0,
    "dof_right_knee_04": math.radians(-50.0),
    "dof_right_ankle_02": math.radians(30.0),
    "dof_left_hip_pitch_04": math.radians(20.0),
    "dof_left_hip_roll_03": math.radians(0.0),
    "dof_left_hip_yaw_03": 0.0,
    "dof_left_knee_04": math.radians(50.0),
    "dof_left_ankle_02": math.radians(-30.0),
}


_JOINT_META = {
    "dof_right_shoulder_pitch_03": {
        "kp": 100.0,
        "kd": 8.284,
        "torque": 42.0,
        "vmax": 18.849,
        "arm": 0.02,
    },
    "dof_right_shoulder_roll_03": {
        "kp": 100.0,
        "kd": 8.257,
        "torque": 42.0,
        "vmax": 18.849,
        "arm": 0.02,
    },
    "dof_right_shoulder_yaw_02": {
        "kp": 40.0,
        "kd": 0.945,
        "torque": 11.9,
        "vmax": 37.699,
        "arm": 0.0042,
    },
    "dof_right_elbow_02": {
        "kp": 40.0,
        "kd": 1.266,
        "torque": 11.9,
        "vmax": 37.699,
        "arm": 0.0042,
    },
    "dof_right_wrist_00": {
        "kp": 20.0,
        "kd": 0.295,
        "torque": 9.8,
        "vmax": 27.227,
        "arm": 0.001,
    },
    "dof_left_shoulder_pitch_03": {
        "kp": 100.0,
        "kd": 8.284,
        "torque": 42.0,
        "vmax": 18.849,
        "arm": 0.02,
    },
    "dof_left_shoulder_roll_03": {
        "kp": 100.0,
        "kd": 8.257,
        "torque": 42.0,
        "vmax": 18.849,
        "arm": 0.02,
    },
    "dof_left_shoulder_yaw_02": {
        "kp": 40.0,
        "kd": 0.945,
        "torque": 11.9,
        "vmax": 37.699,
        "arm": 0.0042,
    },
    "dof_left_elbow_02": {
        "kp": 40.0,
        "kd": 1.266,
        "torque": 11.9,
        "vmax": 37.699,
        "arm": 0.0042,
    },
    "dof_left_wrist_00": {
        "kp": 20.0,
        "kd": 0.295,
        "torque": 9.8,
        "vmax": 27.227,
        "arm": 0.001,
    },
    "dof_right_hip_pitch_04": {
        "kp": 150.0,
        "kd": 24.722,
        "torque": 84.0,
        "vmax": 17.488,
        "arm": 0.04,
    },
    "dof_right_hip_roll_03": {
        "kp": 200.0,
        "kd": 26.387,
        "torque": 42.0,
        "vmax": 18.849,
        "arm": 0.02,
    },
    "dof_right_hip_yaw_03": {
        "kp": 100.0,
        "kd": 3.419,
        "torque": 42.0,
        "vmax": 18.849,
        "arm": 0.02,
    },
    "dof_right_knee_04": {
        "kp": 150.0,
        "kd": 8.654,
        "torque": 84.0,
        "vmax": 17.488,
        "arm": 0.04,
    },
    "dof_right_ankle_02": {
        "kp": 40.0,
        "kd": 0.99,
        "torque": 11.9,
        "vmax": 37.699,
        "arm": 0.0042,
    },
    "dof_left_hip_pitch_04": {
        "kp": 150.0,
        "kd": 24.722,
        "torque": 84.0,
        "vmax": 17.488,
        "arm": 0.04,
    },
    "dof_left_hip_roll_03": {
        "kp": 200.0,
        "kd": 26.387,
        "torque": 42.0,
        "vmax": 18.849,
        "arm": 0.02,
    },
    "dof_left_hip_yaw_03": {
        "kp": 100.0,
        "kd": 3.419,
        "torque": 42.0,
        "vmax": 18.849,
        "arm": 0.02,
    },
    "dof_left_knee_04": {
        "kp": 150.0,
        "kd": 8.654,
        "torque": 84.0,
        "vmax": 17.488,
        "arm": 0.04,
    },
    "dof_left_ankle_02": {
        "kp": 40.0,
        "kd": 0.99,
        "torque": 11.9,
        "vmax": 37.699,
        "arm": 0.0042,
    },
}


# Build one ImplicitActuatorCfg per joint
_ACTUATORS = {
    jn: ImplicitActuatorCfg(
        joint_names_expr=[jn],
        effort_limit=meta["torque"],
        velocity_limit=meta["vmax"],
        stiffness={jn: meta["kp"]},
        damping={jn: meta["kd"]},
        armature=meta["arm"],
    )
    for jn, meta in _JOINT_META.items()
}


KBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1_000.0,
            max_angular_velocity=1_000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos=_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=_ACTUATORS,
)
