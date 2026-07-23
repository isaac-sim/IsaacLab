# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dexterous hand from Shadow Robot.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Shadow Hand with implicit actuator model.
* :obj:`SHADOW_HAND_MENAGERIE_CFG`: Mujoco Menagerie conversion, MuJoCo physics variant (Newton/MJWarp).
* :obj:`SHADOW_HAND_MENAGERIE_PHYSX_CFG`: Mujoco Menagerie conversion, PhysX physics variant.

Reference:

* https://www.shadowrobot.com/dexterous-hand-series/

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .menagerie import MENAGERIE_ASSET_ROOT, MenageriePatchedUsdFileCfg

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


_SHADOW_MENAGERIE_USD_PATH = f"{MENAGERIE_ASSET_ROOT}/shadow_hand/right_hand/right_hand.usda"
"""Entry layer of the Mujoco Menagerie Shadow Hand conversion (both physics variants)."""


SHADOW_HAND_MENAGERIE_CFG = SHADOW_HAND_CFG.replace(
    spawn=MenageriePatchedUsdFileCfg(
        usd_path=_SHADOW_MENAGERIE_USD_PATH,
        # The Menagerie exports default to the "physx" variant; Newton/MJWarp needs the
        # "mujoco" variant, which composes the MjcTendon distal couplings.
        variants={"Physics": "mujoco"},
        activate_contact_sensors=False,
        # Same solver-facing properties as the legacy hand.
        rigid_props=SHADOW_HAND_CFG.spawn.rigid_props.replace(),
        articulation_props=SHADOW_HAND_CFG.spawn.articulation_props.replace(),
        # The mujoco variant authors no UsdPhysics drives (actuation is declared as
        # MjcActuator prims); IsaacLab's implicit actuators require the drive APIs to exist.
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force", ensure_drives_exist=True),
        # MjcTendon couplings take damping only (no PhysX limit-stiffness semantics).
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # Same compensation as the ShadowHandNewton reference: Newton's import bakes the root
        # body's USD orientation into the root fixed joint, so it must be re-applied at spawn.
        rot=(0.0, 0.0, -0.70710678118, 0.70710678118),
        joint_pos={".*": 0.0},
    ),
    actuators={
        # Derived from the ShadowHandNewton reference actuator table via the naming map:
        # fingers keep their index (robot0_FFJn -> rh_FFJn), thumb/wrist shift by +1
        # (robot0_THJn -> rh_THJ{n+1}, robot0_WRJn -> rh_WRJ{n+1}).
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF)J(3|2|1)", "rh_LFJ4", "rh_THJ.*"],
            effort_limit_sim={
                "rh_WRJ2": 4.785,
                "rh_WRJ1": 2.175,
                "rh_(FF|MF|RF|LF)J1": 0.7245,
                "rh_FFJ(3|2)": 0.9,
                "rh_MFJ(3|2)": 0.9,
                "rh_RFJ(3|2)": 0.9,
                "rh_LFJ(4|3|2)": 0.9,
                "rh_THJ5": 2.3722,
                "rh_THJ4": 1.45,
                "rh_THJ(3|2)": 0.99,
                "rh_THJ1": 0.81,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF)J(3|2|1)": 1.0,
                "rh_LFJ4": 1.0,
                "rh_THJ.*": 1.0,
            },
            damping={
                "rh_WRJ.*": 0.5,
                "rh_(FF|MF|RF|LF)J(3|2|1)": 0.1,
                "rh_LFJ4": 0.1,
                "rh_THJ.*": 0.1,
            },
            friction=1e-2,
            armature=2e-3,
        ),
    },
)
"""Configuration of the Mujoco Menagerie Shadow Hand (right), MuJoCo physics variant.

Intended for the Newton/MJWarp backend. Physics content matches the
``ShadowHandNewton`` reference asset exactly (joints, limits, masses, tendons);
only the joint/body naming differs.
"""


SHADOW_HAND_MENAGERIE_PHYSX_CFG = SHADOW_HAND_CFG.replace(
    spawn=MenageriePatchedUsdFileCfg(
        usd_path=_SHADOW_MENAGERIE_USD_PATH,
        # The patched asset authors the distal-finger tendon couplings and the legacy
        # 5729.6 deg/s joint velocity limit into the ``physx`` layer (the converter
        # translates neither); see ``menagerie._ASSET_PATCH_RECIPES``.
        variants={"Physics": "physx"},
        activate_contact_sensors=False,
        # Same solver-facing properties and tendon tuning as the legacy hand.
        rigid_props=SHADOW_HAND_CFG.spawn.rigid_props.replace(),
        articulation_props=SHADOW_HAND_CFG.spawn.articulation_props.replace(),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=SHADOW_HAND_CFG.spawn.fixed_tendons_props.replace(),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # Maps the Menagerie native frame onto the legacy asset's spawned pose under the
        # task's PhysX preset (fingers -Y beneath the falling cube), derived from the
        # measured legacy fingertip layout.
        rot=(-0.0061, -0.0054, -0.6743, 0.7384),
        joint_pos={".*": 0.0},
    ),
    actuators={
        # Derived from SHADOW_HAND_CFG (legacy 0-indexed naming) via the naming map:
        # all joints shift by +1 (robot0_FFJn -> rh_FFJ{n+1}, ..., robot0_WRJn -> rh_WRJ{n+1}).
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF)J(4|3|2)", "rh_LFJ5", "rh_THJ.*"],
            effort_limit_sim={
                "rh_WRJ2": 4.785,
                "rh_WRJ1": 2.175,
                "rh_(FF|MF|RF|LF)J2": 0.7245,
                "rh_FFJ(4|3)": 0.9,
                "rh_MFJ(4|3)": 0.9,
                "rh_RFJ(4|3)": 0.9,
                "rh_LFJ(5|4|3)": 0.9,
                "rh_THJ5": 2.3722,
                "rh_THJ4": 1.45,
                "rh_THJ(3|2)": 0.99,
                "rh_THJ1": 0.81,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF)J(4|3|2)": 1.0,
                "rh_LFJ5": 1.0,
                "rh_THJ.*": 1.0,
            },
            damping={
                "rh_WRJ.*": 0.5,
                "rh_(FF|MF|RF|LF)J(4|3|2)": 0.1,
                "rh_LFJ5": 0.1,
                "rh_THJ.*": 0.1,
            },
        ),
    },
)
"""Configuration of the Mujoco Menagerie Shadow Hand (right), PhysX physics variant.

.. note::
    The Menagerie PhysX variant carries no fixed-tendon authoring (converter gap), so the
    Menagerie patcher authors the four distal-middle couplings from the legacy asset's
    template into a patched copy of the asset's ``physx`` layer on disk. See
    :func:`~isaaclab_assets.robots.menagerie.patch_menagerie_asset` and
    :class:`~isaaclab_assets.robots.menagerie.MenageriePatchedUsdFileCfg`.
"""
