# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dexterous hand from Shadow Robot.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Shadow Hand with implicit actuator model.

Reference:

* https://www.shadowrobot.com/dexterous-hand-series/

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import MUJOCO_MENAGERIE_DIR

##
# Configuration
##

# MuJoCo Menagerie ``right_hand`` uses the ``rh_`` joint prefix. Legacy Isaac Shadow assets used ``robot0_``.
MENAGERIE_SHADOW_FINGERS_ACTUATOR_BASE = ImplicitActuatorCfg(
    joint_names_expr=["rh_.*"],
    effort_limit_sim={
        "rh_WRJ1": 4.785,
        "rh_WRJ2": 2.175,
        "rh_(FF|MF|RF)J4": 0.9,
        "rh_(FF|MF|RF)J(3|2)": 0.9,
        "rh_(FF|MF|RF)J1": 0.7245,
        "rh_LFJ5": 0.9,
        "rh_LFJ(4|3|2)": 0.9,
        "rh_LFJ1": 0.7245,
        "rh_THJ5": 0.81,
        "rh_THJ4": 2.3722,
        "rh_THJ3": 1.45,
        "rh_THJ(2|1)": 0.99,
    },
    stiffness={
        "rh_WRJ.*": 5.0,
        "rh_(FF|MF|RF|LF|TH)J.*": 1.0,
    },
    damping={
        "rh_WRJ.*": 0.5,
        "rh_(FF|MF|RF|LF|TH)J.*": 0.1,
    },
)

SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MUJOCO_MENAGERIE_DIR}/shadow_hand/right_hand/right_hand.usda",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, -0.7071, 0.7071, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={"fingers": MENAGERIE_SHADOW_FINGERS_ACTUATOR_BASE},
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot."""


SHADOW_FINGERTIP_BODY_NAMES: list[str] = [
    "robot0_ffdistal",
    "robot0_mfdistal",
    "robot0_rfdistal",
    "robot0_lfdistal",
    "robot0_thdistal",
]
"""Shadow Hand fingertip body names (identical on every backend asset)."""

SHADOW_ACTUATED_JOINT_NAMES: list[str] = [
    "robot0_WRJ1",
    "robot0_WRJ0",
    "robot0_FFJ3",
    "robot0_FFJ2",
    "robot0_FFJ1",
    "robot0_MFJ3",
    "robot0_MFJ2",
    "robot0_MFJ1",
    "robot0_RFJ3",
    "robot0_RFJ2",
    "robot0_RFJ1",
    "robot0_LFJ4",
    "robot0_LFJ3",
    "robot0_LFJ2",
    "robot0_LFJ1",
    "robot0_THJ4",
    "robot0_THJ3",
    "robot0_THJ2",
    "robot0_THJ1",
    "robot0_THJ0",
]
"""Shadow Hand actuated joint names, in the Direct task's actuation order."""
