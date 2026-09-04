# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for the Asimov-1 humanoid robot.

Asimov-1 is a 23-DoF full-body humanoid:

* 12 leg joints (6 per leg): hip pitch/roll/yaw, knee, ankle pitch/roll
* 1 waist joint: waist yaw
* 10 arm joints (5 per arm): shoulder pitch/roll/yaw, elbow, wrist yaw
* head links are welded (not actuated)

One articulation profile is provided:

* :obj:`ASIMOV_1_DELAYED_CFG`: explicit delayed PD actuators
  (:class:`~isaaclab.actuators.DelayedPDActuator`) with a randomized 0-5
  physics-step command delay that models the actuation round-trip latency.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Actuator command delay in physics substeps (0-25 ms at 200 Hz physics).
DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 5

# Policy action scale: joint target = default pose + scale * action.
ASIMOV_1_ACTION_SCALE = 0.25

##
# Joint ordering
##

ASIMOV_1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
]

##
# Configuration - Actuators
##

ASIMOV_1_ACTUATORS = {
    "hip_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_pitch_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.0698,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "hip_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_roll_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.1400,
        friction=0.20,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "hip_yaw": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_yaw_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=28.0,
        armature=0.0687,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "knee": DelayedPDActuatorCfg(
        joint_names_expr=[".*_knee_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.0330,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "ankle_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint"],
        stiffness=110.0,
        damping=5.0,
        effort_limit=40.0,
        armature=0.0484,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "ankle_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_ankle_roll_joint"],
        stiffness=110.0,
        damping=5.0,
        effort_limit=17.0,
        armature=0.0484,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "waist": DelayedPDActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        stiffness=65.0,
        damping=5.0,
        effort_limit=40.0,
        armature=0.0698,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_pitch_joint"],
        stiffness=57.0,
        damping=5.0,
        effort_limit=30.0,
        armature=0.1400,
        friction=0.20,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_roll_joint"],
        stiffness=86.0,
        damping=5.0,
        effort_limit=25.0,
        armature=0.0330,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_yaw": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_yaw_joint"],
        stiffness=96.0,
        damping=5.0,
        effort_limit=20.0,
        armature=0.0687,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "elbow_wrist": DelayedPDActuatorCfg(
        joint_names_expr=[".*_elbow_joint", ".*_wrist_yaw_joint"],
        stiffness=40.0,
        damping=2.0,
        effort_limit=12.0,
        armature=0.0242,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
}


##
# Initial state
##

ASIMOV_1_STANDING_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.639),
    joint_pos={
        "left_hip_pitch_joint": -0.15,
        "right_hip_pitch_joint": 0.15,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.45,
        "right_knee_joint": -0.45,
        "left_ankle_pitch_joint": -0.30,
        "right_ankle_pitch_joint": 0.30,
        ".*_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": -0.25,
        "right_shoulder_pitch_joint": 0.25,
        "left_shoulder_roll_joint": -0.05,
        "right_shoulder_roll_joint": 0.05,
        ".*_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.40,
        "right_elbow_joint": -0.40,
        ".*_wrist_yaw_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Articulation configs
##


ASIMOV_1_DELAYED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Asimov/asimov_1/asimov_1.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0
            ),
        ),
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ASIMOV_1_STANDING_INIT_STATE,
    soft_joint_pos_limit_factor=0.9,
    actuators=ASIMOV_1_ACTUATORS,
)
"""Asimov-1 with explicit delayed PD actuators (0-5 physics-step command delay).
"""
