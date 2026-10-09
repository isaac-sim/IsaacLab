# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the AgiBot G2 mobile humanoid (``G2_t2_crs`` variant).

The following configurations are available:

* :obj:`AGIBOT_G2_T2_CRS_CFG`: AgiBot G2 with a fixed root, for arm teleoperation.

The USD is not on Nucleus; convert it locally from the URDF shipped with
``ik_7d``'s ``genie_robot_description``::

    ./isaaclab.sh -p scripts/tools/convert_urdf.py \
        ${IK7D_ROOT}/genie_robot_description/urdf/G2_t2_crs/G2_t2_crs.urdf \
        ${AGIBOT_G2_USD_DIR} --joint-target-type position --fix-base --headless

The second argument is the output *directory*, producing
``${AGIBOT_G2_USD_DIR}/G2_t2_crs/G2_t2_crs.usda``.

Do not pass ``--merge-joints``: ``arm_l_end_link`` / ``arm_r_end_link`` must survive
as rigid bodies for the FK cross-check. ``--fix-base`` is required -- see
``articulation_props`` for why the base cannot be pinned from the cfg side.

``ik_7d``'s ``arm_plane_angle`` is code-generated per arm variant, so the asset and
the solver must both target ``G2_t2_crs``.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Asset paths
##

AGIBOT_G2_USD_DIR = os.environ.get("AGIBOT_G2_USD_DIR", "/opt/agibot_g2_ik7d_assets/usd")
"""Directory holding the locally converted AgiBot G2 USD assets."""

AGIBOT_G2_T2_CRS_USD = os.path.join(AGIBOT_G2_USD_DIR, "G2_t2_crs", "G2_t2_crs.usda")
"""Path to the converted ``G2_t2_crs`` stage (``.usda`` -- see the module docstring)."""

AGIBOT_G2_URDF_DIR = os.environ.get("AGIBOT_G2_URDF_DIR", "/opt/agibot_g2_ik7d_assets/urdf")
"""Directory holding the AgiBot URDFs shipped with ``ik_7d``."""

AGIBOT_G2_T2_CRS_URDF = os.path.join(AGIBOT_G2_URDF_DIR, "G2_t2_crs", "G2_t2_crs.urdf")
"""Path to the ``G2_t2_crs`` URDF.

The same file the USD above was converted from, and the one ``Ik7dController``
loads. Keeping both paths here is the only way the asset and the solver are
guaranteed to describe the same robot.
"""

##
# Joint-name regexes
##
# The URDF prefixes every joint with its hardware index, so the group patterns
# have to disambiguate on the index as well as the name: ``idx11_head_joint1``
# and ``idx111_chassis_lwheel_front_joint1`` both start ``idx11``.

G2_WAIST_JOINTS = "idx0[1-5]_body_joint[1-5]"
G2_HEAD_JOINTS = "idx1[1-3]_head_joint[1-3]"
G2_LEFT_ARM_JOINTS = "idx2[1-7]_arm_l_joint[1-7]"
G2_RIGHT_ARM_JOINTS = "idx6[1-7]_arm_r_joint[1-7]"
G2_WHEEL_JOINTS = "idx1[1-4][1-2]_chassis_.*"

##
# Home pose
##
# ``ik_7d.IK7D.getModelInfo().home_joints``, projected into ``[lb_ik, ub_ik]``.
# The shipped home is not feasible against the solver's own limits, and seeding
# the articulation at the raw values makes the first control tick snap to the
# limit -- a phantom discontinuity in every trajectory starting from home. Keep
# these in sync with ``getModelInfo()`` clipped, not with ``home_joints`` raw.

G2_T2_CRS_HOME_JOINT_POS = {
    # Waist / torso -- a forward-leaning working posture, not a zero pose.
    "idx01_body_joint1": -1.0000,
    "idx02_body_joint2": 2.3000,
    "idx03_body_joint3": -1.3000,
    "idx04_body_joint4": 0.0000,
    "idx05_body_joint5": 0.0000,
    # Head
    "idx11_head_joint1": 0.0000,
    "idx12_head_joint2": 0.0000,
    "idx13_head_joint3": 0.0000,
    # Left arm
    "idx21_arm_l_joint1": 0.0000,
    "idx22_arm_l_joint2": -0.6600,
    "idx23_arm_l_joint3": -0.1700,  # projected from 0.0 into [-2.960, -0.170]
    "idx24_arm_l_joint4": -1.6000,
    "idx25_arm_l_joint5": 0.0000,
    "idx26_arm_l_joint6": -0.8000,
    "idx27_arm_l_joint7": 0.0000,
    # Right arm (mirrored)
    "idx61_arm_r_joint1": 0.0000,
    "idx62_arm_r_joint2": -0.6600,
    "idx63_arm_r_joint3": 0.1700,  # projected from 0.0 into [+0.170, +2.960]
    "idx64_arm_r_joint4": -1.6000,
    "idx65_arm_r_joint5": 0.0000,
    "idx66_arm_r_joint6": -0.8000,
    "idx67_arm_r_joint7": 0.0000,
    # Chassis wheels -- not modelled by ik_7d; the mobile base is a separate
    # control path and is parked here.
    "idx1[1-4][1-2]_chassis_.*": 0.0,
}

##
# Configuration
##

AGIBOT_G2_T2_CRS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=AGIBOT_G2_T2_CRS_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # Deliberately *not* fix_root_link=True. The base is wheeled and
            # unactuated; pinning it at conversion time with ``--fix-base`` keeps
            # the chassis still so arm tracking error is attributable to IK rather
            # than chassis drift under reaction torques. Asking Lab to do it
            # instead fails: the importer authors ``root_joint`` with both bodies
            # named, ``find_global_fixed_joint_prim`` expects a single target, and
            # Lab tries to create a joint on the ``Geometry`` scope, which has no
            # RigidBodyAPI, raising NotImplementedError.
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=G2_T2_CRS_HOME_JOINT_POS,
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        # Waist / torso. Stiff because this chain carries the upper body and any
        # sag shows up as arm tracking error. ``effort_limit_sim`` is deliberately
        # far above the URDF's 50 Nm: at the folded home posture all three waist
        # joints saturate there and the torso collapses, dragging the hands over a
        # metre. The URDF figure reads as a continuous-duty rating; the real arm
        # holds this pose through gearing the rigid-body model does not represent.
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[G2_WAIST_JOINTS],
            effort_limit_sim=2000.0,
            velocity_limit_sim=1.21,
            stiffness=20000.0,
            damping=1000.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[G2_HEAD_JOINTS],
            effort_limit_sim=50.0,
            velocity_limit_sim=2.50,
            stiffness=80.0,
            damping=4.0,
        ),
        # Arms. The URDF tapers effort down the chain (108 / 35 / 18 Nm), so the
        # gains taper with it rather than using one value for all seven joints.
        # Effort limits are the URDF's and are honoured; only the position gains
        # are raised, because a soft arm sags at home under gravity and that sag
        # would show up later as IK tracking error the solver cannot see.
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=[G2_LEFT_ARM_JOINTS],
            effort_limit_sim={
                "idx2[1-2]_arm_l_joint[1-2]": 108.0,
                "idx2[3-5]_arm_l_joint[3-5]": 35.0,
                "idx2[6-7]_arm_l_joint[6-7]": 18.0,
            },
            velocity_limit_sim={
                "idx2[1-2]_arm_l_joint[1-2]": 3.90,
                "idx2[3-4]_arm_l_joint[3-4]": 3.60,
                "idx2[5-7]_arm_l_joint[5-7]": 4.70,
            },
            stiffness={
                "idx2[1-2]_arm_l_joint[1-2]": 4000.0,
                "idx2[3-5]_arm_l_joint[3-5]": 2000.0,
                "idx2[6-7]_arm_l_joint[6-7]": 1000.0,
            },
            damping={
                "idx2[1-2]_arm_l_joint[1-2]": 200.0,
                "idx2[3-5]_arm_l_joint[3-5]": 100.0,
                "idx2[6-7]_arm_l_joint[6-7]": 50.0,
            },
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[G2_RIGHT_ARM_JOINTS],
            effort_limit_sim={
                "idx6[1-2]_arm_r_joint[1-2]": 108.0,
                "idx6[3-5]_arm_r_joint[3-5]": 35.0,
                "idx6[6-7]_arm_r_joint[6-7]": 18.0,
            },
            velocity_limit_sim={
                "idx6[1-2]_arm_r_joint[1-2]": 3.90,
                "idx6[3-4]_arm_r_joint[3-4]": 3.60,
                "idx6[5-7]_arm_r_joint[5-7]": 4.70,
            },
            stiffness={
                "idx6[1-2]_arm_r_joint[1-2]": 4000.0,
                "idx6[3-5]_arm_r_joint[3-5]": 2000.0,
                "idx6[6-7]_arm_r_joint[6-7]": 1000.0,
            },
            damping={
                "idx6[1-2]_arm_r_joint[1-2]": 200.0,
                "idx6[3-5]_arm_r_joint[3-5]": 100.0,
                "idx6[6-7]_arm_r_joint[6-7]": 50.0,
            },
        ),
        # Wheels are held passively. They still need an actuator entry: Isaac Lab
        # requires every articulation joint to be claimed by exactly one group.
        "chassis_passive": ImplicitActuatorCfg(
            joint_names_expr=[G2_WHEEL_JOINTS],
            effort_limit_sim=15.0,
            velocity_limit_sim=20.9,
            stiffness=0.0,
            damping=1.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the AgiBot G2 (``G2_t2_crs``) with a fixed root link."""
