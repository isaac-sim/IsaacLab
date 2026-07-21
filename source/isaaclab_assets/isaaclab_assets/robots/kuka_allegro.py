# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka-lbr-iiwa arm robots and Allegro Hand.

The following configurations are available:

* :obj:`KUKA_ALLEGRO_CFG`: Kuka Allegro with implicit actuator model.

Reference:

* https://www.kuka.com/en-us/products/robotics-systems/industrial-robots/lbr-iiwa
* https://www.wonikrobotics.com/robot-hand

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

KUKA_ALLEGRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "iiwa7_joint_(1|2|7)": 0.0,
            "iiwa7_joint_3": 0.7854,
            "iiwa7_joint_4": 1.5708,
            "iiwa7_joint_(5|6)": -1.5708,
            "(index|middle|ring)_joint_0": 0.0,
            "(index|middle|ring)_joint_1": 0.3,
            "(index|middle|ring)_joint_2": 0.3,
            "(index|middle|ring)_joint_3": 0.3,
            "thumb_joint_0": 1.5,
            "thumb_joint_1": 0.60147215,
            "thumb_joint_2": 0.33795027,
            "thumb_joint_3": 0.60845138,
        },
    ),
    actuators={
        "kuka_allegro_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "iiwa7_joint_(1|2|3|4|5|6|7)",
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            # iiwa7, wonic references
            # https://github.com/RobotLocomotion/models/blob/master/iiwa_description/sdf/iiwa7_no_collision.sdf
            # https://www.wonikrobotics.com/robot-hand
            # https://github.com/RobotLocomotion/models/blob/master/allegro_hand_description/sdf/allegro_hand_description_right.sdf
            effort_limit_sim={
                "iiwa7_joint_(1|2)": 176.0,
                "iiwa7_joint_(3|4|5)": 110.0,
                "iiwa7_joint_(6|7)": 40.0,
                "(index|middle|ring|thumb)_joint_(0|1|2|3)": 0.7,
            },
            # motor velocity limits for mdp checking — deliberately NOT velocity_limit_sim.
            velocity_limit={
                "iiwa7_joint_(1|2)": 1.7104,
                "iiwa7_joint_3": 1.7453,
                "iiwa7_joint_4": 2.2689,
                "iiwa7_joint_5": 2.4435,
                "iiwa7_joint_(6|7)": 3.1416,
                "(index|middle|ring|thumb)_joint_(0|1|2|3)": 9.52,  # (https://www.wonikrobotics.com/robot-hand
            },
            stiffness={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 200.0,
                "(index|middle|ring|thumb)_joint_(0|1|2|3)": 3.0,
            },
            damping={
                "iiwa7_joint_1": 42.2,
                "iiwa7_joint_2": 54.4,
                "iiwa7_joint_3": 45.6,
                "iiwa7_joint_4": 37.8,
                "iiwa7_joint_5": 25.3,
                "iiwa7_joint_6": 25.5,
                "iiwa7_joint_7": 23.5,
                "(index|middle|ring|thumb)_joint_(0|1|2|3)": 0.1,
            },
            friction={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 1.0,
                "index_joint_(0|1|2|3)": 0.01,
                "middle_joint_(0|1|2|3)": 0.01,
                "ring_joint_(0|1|2|3)": 0.01,
                "thumb_joint_(0|1|2|3)": 0.01,
            },
            # references:
            # https://github.com/RobotLocomotion/models/blob/master/iiwa_description/sdf/iiwa7_no_collision.sdf
            # https://github.com/RobotLocomotion/drake/pull/20420
            # https://github.com/RobotLocomotion/drake/blob/master/examples/allegro_hand/allegro_single_object_simulation.cc)
            armature={
                "iiwa7_joint_1": 1.836,
                "iiwa7_joint_2": 4.447,
                "iiwa7_joint_3": 3.242,
                "iiwa7_joint_4": 1.817,
                "iiwa7_joint_5": 1.392,
                "iiwa7_joint_6": 1.402,
                "iiwa7_joint_7": 1.392,
                "(index|middle|ring|thumb)_joint_(0|1|2|3)": 0.136,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
