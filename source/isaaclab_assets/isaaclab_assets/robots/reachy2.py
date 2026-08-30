# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pollen Robotics Reachy 2 bimanual humanoid robot.

The following configurations are available:

* :obj:`REACHY2_CFG`: Reachy 2 with both arms, neck, and grippers using implicit actuators.
* :obj:`REACHY2_HIGH_PD_CFG`: Reachy 2 with stiffer PD gains for task-space (IK) tracking.

Reachy 2 has:
* 7 DOF per arm: shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch,
  wrist_roll, wrist_pitch, wrist_yaw (Orbita 2D + 3D parallel mechanisms)
* 3 DOF neck: neck_roll, neck_pitch, neck_yaw (Orbita 3D)
* 5 gripper joints per hand: hand_finger, hand_finger_proximal,
  hand_finger_proximal_mimic, hand_finger_distal, hand_finger_distal_mimic

Reference:
    https://github.com/pollen-robotics/reachy2_core
    https://www.pollen-robotics.com/reachy-2/

Asset generation:
    The USD asset is generated locally from the bundled ``reachy2_fixed.urdf``
    (pending hosting on Nucleus). The URDF references meshes via ROS
    ``package://`` URLs, resolved from a clone of ``reachy2_core``:

    .. code-block:: bash

        git clone https://github.com/pollen-robotics/reachy2_core /path/to/reachy2_core

    .. code-block:: python

        # run via: ./isaaclab.sh -p <this-script>
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        base = "source/isaaclab_assets/isaaclab_assets/robots/reachy2"
        cfg = UrdfConverterCfg(
            asset_path=f"{base}/reachy2_fixed.urdf",
            usd_dir=f"{base}/reachy2.usd",
            fix_base=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="position"),
            ros_package_paths=[
                {"name": "reachy_description", "path": "/path/to/reachy2_core/reachy_description"},
                {"name": "dynamixel_description",
                 "path": "/path/to/reachy2_core/reachy_controllers/dynamixel_control/dynamixel_description"},
            ],
        )
        UrdfConverter(cfg)

    Note: the URDF has been pre-processed from the original in
    ``reachy2_symbolic_ik``: Gazebo-specific blocks and non-existent meshes
    removed, mobile-base and torso-bar visuals stripped (fixed-base
    simulation), and gripper ``<mimic>`` tags removed — the importer drops the
    mimic offset, which displaces the finger linkage; the finger joints are
    instead position-driven directly with kinematically consistent commands.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# USD path — local to this repo until the asset is hosted on Nucleus.
# Generated from reachy2_fixed.urdf via scripts/tools/convert_urdf.py --fix-base:
#   source/isaaclab_assets/isaaclab_assets/robots/reachy2/reachy2.usd/reachy2_fixed/reachy2_fixed.usda
##
_REACHY2_USD = os.path.join(
    os.path.dirname(__file__),
    "reachy2",
    "reachy2.usd",
    "reachy2_fixed",
    "reachy2_fixed.usda",
)

##
# Configuration
##

REACHY2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_REACHY2_USD,
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Neck — neutral
            "neck_roll": 0.0,
            "neck_pitch": 0.0,
            "neck_yaw": 0.0,
            # Right arm — resting alongside torso
            "r_shoulder_pitch": 0.0,
            "r_shoulder_roll": -0.2,
            "r_elbow_yaw": 0.0,
            "r_elbow_pitch": -0.5,
            "r_wrist_roll": 0.0,
            "r_wrist_pitch": 0.0,
            "r_wrist_yaw": 0.0,
            # Left arm — resting alongside torso
            "l_shoulder_pitch": 0.0,
            "l_shoulder_roll": 0.2,
            "l_elbow_yaw": 0.0,
            "l_elbow_pitch": -0.5,
            "l_wrist_roll": 0.0,
            "l_wrist_pitch": 0.0,
            "l_wrist_yaw": 0.0,
            # Grippers — open
            "r_hand_finger": 0.0,
            "r_hand_finger_proximal": 0.0,
            "r_hand_finger_proximal_mimic": 0.0,
            "r_hand_finger_distal": 0.0,
            "r_hand_finger_distal_mimic": 0.0,
            "l_hand_finger": 0.0,
            "l_hand_finger_proximal": 0.0,
            "l_hand_finger_proximal_mimic": 0.0,
            "l_hand_finger_distal": 0.0,
            "l_hand_finger_distal_mimic": 0.0,
        }
    ),
    actuators={
        # Note: non-zero armature is required for stability. The URDF's Orbita
        # dummy links and gripper finger links have near-zero inertias (down to
        # 1e-6 kg m^2); stiff position drives on such bodies are numerically
        # unstable at the 100 Hz physics rate without added rotor inertia.
        # Right arm — 7 DOF Orbita actuators
        "r_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "r_shoulder_pitch",
                "r_shoulder_roll",
                "r_elbow_yaw",
                "r_elbow_pitch",
                "r_wrist_roll",
                "r_wrist_pitch",
                "r_wrist_yaw",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=100.0,
            damping=5.0,
            armature=0.01,
        ),
        # Left arm — 7 DOF Orbita actuators
        "l_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "l_shoulder_pitch",
                "l_shoulder_roll",
                "l_elbow_yaw",
                "l_elbow_pitch",
                "l_wrist_roll",
                "l_wrist_pitch",
                "l_wrist_yaw",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=100.0,
            damping=5.0,
            armature=0.01,
        ),
        # Neck — 3 DOF Orbita 3D
        "neck": ImplicitActuatorCfg(
            joint_names_expr=["neck_roll", "neck_pitch", "neck_yaw"],
            effort_limit_sim=50.0,
            velocity_limit_sim=5.0,
            stiffness=80.0,
            damping=4.0,
            armature=0.01,
        ),
        # Grippers — Dynamixel fingers
        "r_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "r_hand_finger",
                "r_hand_finger_proximal",
                "r_hand_finger_proximal_mimic",
                "r_hand_finger_distal",
                "r_hand_finger_distal_mimic",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.005,
        ),
        "l_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "l_hand_finger",
                "l_hand_finger_proximal",
                "l_hand_finger_proximal_mimic",
                "l_hand_finger_distal",
                "l_hand_finger_distal_mimic",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.005,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Reachy 2 with implicit actuators."""


REACHY2_HIGH_PD_CFG = REACHY2_CFG.copy()
REACHY2_HIGH_PD_CFG.actuators["r_arm"].stiffness = 400.0
REACHY2_HIGH_PD_CFG.actuators["r_arm"].damping = 40.0
REACHY2_HIGH_PD_CFG.actuators["l_arm"].stiffness = 400.0
REACHY2_HIGH_PD_CFG.actuators["l_arm"].damping = 40.0
"""Configuration of Reachy 2 with stiffer PD gains for task-space (IK) control."""
