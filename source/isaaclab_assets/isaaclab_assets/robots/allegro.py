# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Allegro Hand robots from Wonik Robotics.

The following configurations are available:

* :obj:`ALLEGRO_HAND_CFG`: Allegro Hand with implicit actuator model.
* :obj:`ALLEGRO_HAND_MENAGERIE_CFG`: Mujoco Menagerie conversion, MuJoCo physics variant (Newton/MJWarp).

Reference:

* https://www.wonikrobotics.com/robot-hand

"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .menagerie import MENAGERIE_ASSET_ROOT, MenageriePatchedUsdFileCfg

##
# Configuration
##

ALLEGRO_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.283045, 0.683330, -0.621782, 0.257551),
        joint_pos={"^(?!thumb_joint_0).*": 0.0, "thumb_joint_0": 0.28},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Allegro Hand robot."""


ALLEGRO_HAND_MENAGERIE_CFG = ALLEGRO_HAND_CFG.replace(
    spawn=MenageriePatchedUsdFileCfg(
        usd_path=f"{MENAGERIE_ASSET_ROOT}/wonik_allegro/right_hand/right_hand.usda",
        # The Menagerie exports default to the "physx" variant; Newton/MJWarp needs "mujoco".
        # For PhysX runs, replace with variants={"Physics": "physx"}.
        variants={"Physics": "mujoco"},
        activate_contact_sensors=False,
        # Same solver-facing properties as the legacy hand. Self-collisions stay enabled:
        # the impossible tip<->medial pairs manufactured by the conversion's weld split are
        # filtered in the patched asset (see ``menagerie._ASSET_PATCH_RECIPES``).
        rigid_props=ALLEGRO_HAND_CFG.spawn.rigid_props.replace(),
        articulation_props=ALLEGRO_HAND_CFG.spawn.articulation_props.replace(),
        # The mujoco variant authors no UsdPhysics drives (actuation is declared as MjcActuator
        # prims); IsaacLab's implicit actuators require the drive APIs to exist.
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force", ensure_drives_exist=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Offset so the cube's measured rest point coincides with the task's in-hand target
        # (palm-to-palm alignment leaves the cube resting on the palm heel, 8.5 cm behind
        # the fingers' grasp zone).
        pos=(0.0, -0.167, 0.461),
        # Maps the Menagerie base frame (fingers +X, spread +Y) onto the legacy asset's
        # spawned palm-up catch pose (fingers -Y, spread +X, palm facing up), derived from
        # the measured legacy fingertip layout.
        rot=(0.063, 0.0592, -0.6823, 0.7259),
        joint_pos={"^(?!thj0).*": 0.0, "thj0": 0.28},
    ),
)
"""Configuration of the Mujoco Menagerie Allegro Hand (right), MuJoCo physics variant.

Derived from :obj:`ALLEGRO_HAND_CFG`: same solver properties and actuator gains; only the
asset source (patched Menagerie conversion), the spawn-frame compensation, and the joint
naming differ. Physical parameters the asset can express (armature, joint velocity limits,
friction, collision filters) come from the patched asset itself rather than the config.
Unlike the legacy BioTac-equipped hand, this is the stock Allegro Hand v4 with
density-derived link masses (~0.81 kg total vs ~2.14 kg) and per-joint calibrated limits;
training rewards are expected to differ accordingly.
"""
