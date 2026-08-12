# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dexterous hand from Shadow Robot.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Shadow Hand on the PhysX asset with implicit actuator model.
* :obj:`SHADOW_HAND_NEWTON_CFG`: Shadow Hand on the Newton (MJWarp) asset.

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
"""Configuration of the Shadow Hand robot on the PhysX asset."""


SHADOW_HAND_NEWTON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Newton/MuJoCo use a separate USD schema; this asset renumbers the finger
        # joints (+1) relative to the PhysX asset above (e.g. FFJ4 vs FFJ3, LFJ5 vs LFJ4).
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHandNewton/shadow_hand_instanceable.usda",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force", ensure_drives_exist=True),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # WARNING(Octi): Newton's import_usd.py bakes the USD body xformOp rotation into
        # joint_X_p for the root fixed joint, which cancels with the matching localPose1
        # rotation in joint_X_c during FK (joint_X_p * inv(joint_X_c) ≈ identity). This
        # discards the root body's native USD orientation, so we must re-apply it here as a
        # spawn rotation. PhysX or USD does not have this issue. Remove once Newton fixes root joint
        # transform handling in import_usd.py.
        rot=(0.0, 0.0, -0.70710678118, 0.70710678118),
        joint_pos={".*": 0.0},
    ),
    actuators={
        # Drives the joints named by :obj:`SHADOW_ACTUATED_JOINT_NAMES`, which resolve on this
        # asset despite its +1 finger renumbering.
        #
        # The per-finger ``J1``/``J2`` pair is coupled by a fixed tendon (``coef=[1, 1]``) that
        # the MJWarp solver currently skips, so the configuration is what holds the pair
        # together: both ends are driven with identical gains and effort limits, which
        # reproduces the 1:1 coupling. Keep them symmetric and in the same actuator group --
        # driving one end while clamping the other makes the two fight, and an uncapped effort
        # limit on either end diverges to NaN within a few hundred steps.
        #
        # Known limitation: the ``J4`` knuckle abduction joints (and ``LFJ5``) are left
        # undriven, so the fingers cannot spread laterally. Correcting that requires resolving
        # the skipped tendon first and is deferred to a follow-up.
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "robot0_WR.*",
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)",
                "robot0_(LF|TH)J4",
                "robot0_THJ0",
            ],
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
            # Default gains match the PhysX cfg (wrists 5.0/0.5, fingers 1.0/0.1). Tasks that
            # need more joint authority override these -- e.g. the handover catch on MJWarp
            # raises them to 20.0/2.0, since MJWarp's implicit-PD path lacks PhysX's
            # fixed-tendon limit stiffness + solver-iteration torque amplification.
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
            friction=1e-2,
            armature=2e-3,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the Shadow Hand robot on the Newton (MJWarp) asset.

The Newton USD renumbers the finger joints (+1) relative to :obj:`SHADOW_HAND_CFG`, but the
names in :obj:`SHADOW_ACTUATED_JOINT_NAMES` resolve on both assets, so the two backends share
one actuated-joint list. Gains default to the PhysX values; tasks override them as needed.
"""


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
"""Shadow Hand actuated joint names, in the Direct task's actuation order.

These names resolve on both the PhysX and Newton assets, so every backend shares this list.
"""
