# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Shadow Robot Dexterous Hand.

The hand has 24 physical joints and 20 motor coordinates. The middle and distal joints of each
non-thumb finger (``J2`` and ``J1``) share one tendon motor, ``J0 = J1 + J2``; the remaining 16
motors drive one joint each.

Reference:

* https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand
* https://www.shadowrobot.com/dexterous-hand-series/

"""

import os

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

JOINT_NAMES = [
    "rh_WRJ2",
    "rh_WRJ1",
    "rh_FFJ4",
    "rh_FFJ3",
    "rh_MFJ4",
    "rh_MFJ3",
    "rh_RFJ4",
    "rh_RFJ3",
    "rh_LFJ5",
    "rh_LFJ4",
    "rh_LFJ3",
    "rh_THJ5",
    "rh_THJ4",
    "rh_THJ3",
    "rh_THJ2",
    "rh_THJ1",
]
"""The 16 motors that drive a joint of their own [rad]."""

TENDON_NAMES = ["rh_FFJ0", "rh_MFJ0", "rh_RFJ0", "rh_LFJ0"]
"""The 4 motors that drive a tendon, one per non-thumb finger [rad].

A tendon spans that finger's middle and distal joints, which move together as ``J0 = J1 + J2``
and take no command of their own. Tendons have their own index space, so a joint action term
cannot reach them.
"""

TENDON_POSITION_LIMITS = (0.0, 3.1415)
"""Commandable range of each tendon motor [rad].

The asset's value: each tendon actuator authors ``mjc:ctrlRange:max = 3.1415``, which is pi to
four decimals. It is pi by construction rather than by choice -- the coordinate is
``J0 = J1 + J2`` and each of those joints travels ``[0, pi/2]``, so their sum spans ``[0, pi]``.
That matches the hardware: the hand's middle and distal joints each move through 90 degrees on one
tendon.

Restated here because no *runtime* accessor reports it on both backends. The tendons carry no
position limit of their own -- MEASURED, ``mujoco.tendon_limited`` is ``2`` (MuJoCo's "auto") and
``mujoco.tendon_range`` is all zeros -- so ``fixed_tendon_pos_limits`` reads zeros. What bounds
the command is the actuator's control range, which the MuJoCo backend exposes and PhysX does not.
"""

FINGERTIP_NAMES = [
    "rh_ffdistal",
    "rh_mfdistal",
    "rh_rfdistal",
    "rh_lfdistal",
    "rh_thdistal",
]
"""Fingertip bodies, in the order the observation terms expect."""

SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # The published asset must carry the four PhysX tendon schemas the mujoco variant
        # expresses as MjcTendon prims, and a world weld; MEASURED, an asset without them
        # gives fixed_base=False tendons=0 where this expects fixed_base=True tendons=4.
        usd_path=os.environ.get(
            "SHADOW_HAND_USD",
            f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHandMultiPhysics/shadow_hand_instanceable.usda",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Shared with Allegro, the other hand in these tasks, and with most robot configs
            # (23/30 set disable_gravity, 27/30 set max_depenetration_velocity).
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
            # PhysX only. MEASURED: dropping it costs mean reward 258 -> 207 over one seed,
            # inside that run's own 153-254 spread. Kept until multiple seeds settle it.
            retain_accelerations=True,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                # MEASURED: without these the hand reaches non-finite observations at iteration
                # 49 (reorient) and 3 (handover). The scene-level clamp does not substitute.
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # Newton's importer cancels the root xform during FK, so the orientation is re-applied.
        # MEASURED: of the 24 axis-aligned orientations this is 33 mm from the reference pose and
        # every other is 545-615 mm. One rotation serves both engines; the palm lands at the same
        # env-local (0.0, -0.247, 0.51) on each.
        rot=(0.0, 0.0, -0.70710678118, 0.70710678118),
        joint_pos={".*": 0.0},
    ),
    # Each backend enumerates differently -- MEASURED, index 3 is rh_FFJ3 on Newton and rh_MFJ4
    # on PhysX, and a policy moved across scores 0.0005 against a 0.6165 native baseline.
    # Newton's order is the public one, so PhysX permutes to match.
    joint_ordering="mjwarp",
    body_ordering="mjwarp",
    actuators={
        "direct_motors": ImplicitActuatorCfg(
            joint_names_expr=JOINT_NAMES,
            joint_effort_limit={
                "rh_WRJ2": 10.0,
                "rh_WRJ1": 5.0,
                "rh_(FF|MF|RF)J(4|3)": 1.0,
                "rh_LFJ(5|4|3)": 1.0,
                "rh_THJ5": 3.0,
                "rh_THJ4": 2.0,
                "rh_THJ(3|2|1)": 1.0,
            },
            stiffness={
                "rh_WRJ2": 10.0,
                "rh_WRJ1": 8.0,
                "rh_(FF|MF|RF)J(4|3)": 1.0,
                "rh_LFJ(5|4|3)": 1.0,
                "rh_THJ5": 0.4,
                "rh_THJ4": 1.0,
                "rh_THJ3": 0.5,
                "rh_THJ2": 1.5,
                "rh_THJ1": 1.0,
            },
            # MEASURED: these joints do not inherit the model's damping (only the tendon-coupled
            # J1/J2 do), and stiffness without damping rings rather than settles.
            damping={"rh_WRJ.*": 0.5, "rh_(FF|MF|RF|LF|TH)J.*": 0.1},
            # The asset's own values, restated because they do not survive import -- MEASURED,
            # leaving them None lands every joint at 0.0 where stiffness and damping do survive.
            armature=2.0e-4,
            friction=1.0e-2,
        ),
        "coupled_joints": ImplicitActuatorCfg(
            joint_names_expr=["rh_(FF|MF|RF|LF)J(2|1)"],
            # A PhysX fixed tendon has no force range, so the joint's effort limit is where the
            # model's cap lives. MEASURED: unset, these eight land at 3.4e38 while every directly
            # driven joint gets 1-10 N-m.
            joint_effort_limit={
                "rh_(FF|MF|RF|LF)J2": 0.9,
                "rh_(FF|MF|RF|LF)J1": 0.7245,
            },
            # The tendon drives these, so stiffness and damping stay as authored (None keeps the
            # USD value); armature and friction are restated because they do not survive import.
            stiffness=None,
            damping=None,
            armature=2.0e-4,
            friction=1.0e-2,
        ),
    },
)
"""Shadow Hand, with the asset's own ``Physics`` variant left selected.

One asset serves both engines; per-engine values are authored in the asset rather than restated
here. Prefer :data:`SHADOW_HAND_PHYSX_CFG` or :data:`SHADOW_HAND_NEWTON_CFG`, which name the
engine at the call site instead of relying on the asset's default.
"""

SHADOW_HAND_PHYSX_CFG = SHADOW_HAND_CFG.copy()
SHADOW_HAND_PHYSX_CFG.spawn.variants = {"Physics": "physx"}
"""Shadow Hand on the asset's PhysX variant."""

SHADOW_HAND_NEWTON_CFG = SHADOW_HAND_CFG.copy()
SHADOW_HAND_NEWTON_CFG.spawn.variants = {"Physics": "mujoco"}
"""Shadow Hand on the asset's MuJoCo variant, for the Newton (MJWarp) solver."""
