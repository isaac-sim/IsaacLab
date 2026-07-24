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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHand/shadow_hand_instanceable.usd",
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
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["robot0_WR.*", "robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
            effort_limit_sim={
                "robot0_WRJ1": 4.785,
                "robot0_WRJ0": 2.175,
                "robot0_(FF|MF|RF|LF)J1": 0.7245,
                "robot0_FFJ(3|2)": 0.9,
                "robot0_MFJ(3|2)": 0.9,
                "robot0_RFJ(3|2)": 0.9,
                "robot0_LFJ(4|3|2)": 0.9,
                "robot0_THJ4": 2.3722,
                "robot0_THJ3": 1.45,
                "robot0_THJ(2|1)": 0.99,
                "robot0_THJ0": 0.81,
            },
            stiffness={
                "robot0_WRJ.*": 5.0,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 1.0,
                "robot0_(LF|TH)J4": 1.0,
                "robot0_THJ0": 1.0,
            },
            damping={
                "robot0_WRJ.*": 0.5,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.1,
                "robot0_(LF|TH)J4": 0.1,
                "robot0_THJ0": 0.1,
            },
        ),
    },
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
