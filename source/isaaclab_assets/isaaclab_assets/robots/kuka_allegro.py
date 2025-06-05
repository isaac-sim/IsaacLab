# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
Defines the Kuka-Allegro robot configuration for simulation with Isaac Sim.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
ISAACLAB_NUCLEUS_DIR = "source/isaaclab_assets/data"

##
# Configuration
##


KUKA_ALLEGRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/KUKA/kuka_allegro.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "iiwa7_joint_(1|2|3|4|5|6|7)": 0.,
            "index_joint_(0|1|2|3)": 0.,
            "middle_joint_(0|1|2|3)": 0.,
            "ring_joint_(0|1|2|3)": 0.,
            "thumb_joint_0": 0.5,
            "thumb_joint_(1|2|3)": 0.
        },
    ),
    actuators={
        "kuka_allegro_actuators": ImplicitActuatorCfg(
            joint_names_expr=["iiwa7_joint_(1|2|3|4|5|6|7)",
                              "index_joint_(0|1|2|3)",
                              "middle_joint_(0|1|2|3)",
                              "ring_joint_(0|1|2|3)",
                              "thumb_joint_(0|1|2|3)"],
            effort_limit_sim={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 300.,
                "index_joint_(0|1|2|3)": 0.5,
                "middle_joint_(0|1|2|3)": 0.5,
                "ring_joint_(0|1|2|3)": 0.5,
                "thumb_joint_(0|1|2|3)": 0.5,
            },
            stiffness={
                "iiwa7_joint_(1|2|3|4)": 300.,
                "iiwa7_joint_5": 100.,
                "iiwa7_joint_6": 50.,
                "iiwa7_joint_7": 25.,
                "index_joint_(0|1|2|3)": 3.0,
                "middle_joint_(0|1|2|3)": 3.0,
                "ring_joint_(0|1|2|3)": 3.0,
                "thumb_joint_(0|1|2|3)": 3.0,
            },
            damping={
                "iiwa7_joint_(1|2|3|4)": 45.,
                "iiwa7_joint_5": 20.,
                "iiwa7_joint_6": 15.,
                "iiwa7_joint_7": 15.,
                "index_joint_(0|1|2|3)": 0.1,
                "middle_joint_(0|1|2|3)": 0.1,
                "ring_joint_(0|1|2|3)": 0.1,
                "thumb_joint_(0|1|2|3)": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Kuka Allegro robot."""
