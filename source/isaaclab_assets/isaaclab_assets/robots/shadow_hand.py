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

SHADOW_HAND_USD_PATH = os.environ.get(
    "SHADOW_HAND_USD",
    f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHandMultiPhysics/shadow_hand_instanceable.usda",
)
"""The asset. Its ``Physics`` USD variant selects the engine.

The published asset must carry the four PhysX tendon schemas the ``mujoco`` variant expresses as
``MjcTendon`` prims, and a world weld PhysX honours -- MEASURED, an asset without them gives
``fixed_base=False tendons=0`` where this configuration expects ``fixed_base=True joints=24
tendons=4``. It publishes beside the existing ``ShadowHand`` rather than replacing it, so assets still
referencing the single-engine hand keep working. ``SHADOW_HAND_USD`` points at a local copy
until it lands.
"""

SHADOW_HAND_JOINT_ORDERING = "mjwarp"
"""Public joint order, so both engines expose the hand's joints identically.

Left unset each backend uses its own enumeration order, and MEASURED they differ: Newton walks
depth-first per finger, PhysX breadth-first by joint level, so index 3 is ``rh_FFJ3`` on one and
``rh_MFJ4`` on the other. Every observation past index 2 then names a different joint, and a policy
trained on one engine scores 0.0005 on the other against a 0.6165 native baseline.

``"mjwarp"`` makes Newton's order the public one, so PhysX permutes to match rather than the other
way round. Actions were never affected -- the action terms resolve by name with
``preserve_order`` -- so this only realigns the observations.
"""

SHADOW_HAND_BODY_ORDERING = "mjwarp"
"""Public body order, chosen to match :attr:`joint_ordering` for the same reason."""

SHADOW_HAND_JOINT_NAMES = [
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

SHADOW_HAND_TENDON_NAMES = ["rh_FFJ0", "rh_MFJ0", "rh_RFJ0", "rh_LFJ0"]
"""The 4 motors that drive a tendon, one per non-thumb finger [rad].

A tendon spans that finger's middle and distal joints, which move together as ``J0 = J1 + J2``
and take no command of their own. Tendons have their own index space, so a joint action term
cannot reach them.
"""

SHADOW_HAND_TENDON_POSITION_LIMITS = (0.0, 3.1415)
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

SHADOW_HAND_FINGERTIP_NAMES = [
    "rh_ffdistal",
    "rh_mfdistal",
    "rh_rfdistal",
    "rh_lfdistal",
    "rh_thdistal",
]
"""Fingertip bodies, in the order the observation terms expect."""

SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SHADOW_HAND_USD_PATH,
        variants={"Physics": "physx"},
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Shared with Allegro, the other hand in these tasks, and with most robot configs
            # (23/30 set disable_gravity, 27/30 set max_depenetration_velocity).
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
            # PhysX only, and inherited from the pre-unification Shadow Hand config.
            # MEASURED 2026-08-20: dropping it does NOT reintroduce the divergence -- 160
            # iterations of handover on PhysX, no NaN, where the earlier solver regression
            # failed at iteration 3. But mean reward over iters 145-155 was 207 against the
            # baseline's 258, and one seed cannot separate that from noise (that run's own
            # spread was 153-254). Kept until a multi-seed comparison settles it.
            retain_accelerations=True,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                # MEASURED 2026-08-19: without these the hand diverges to non-finite
                # observations at iteration 49 (reorient) and 3 (handover), reproducibly,
                # while Allegro on the same backend trains clean. The scene-level
                # `min_position_iteration_count` clamp does NOT substitute -- a run with it
                # produced a byte-identical log.
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
        # Newton's importer bakes the root body's xformOp rotation into the root joint and
        # cancels it during FK, so the orientation is re-applied here. Same value and same
        # reason as the previous Newton asset, which trains this task to 0.926 success.
        #
        # MEASURED: of all 24 axis-aligned orientations scored by fingertip distance from
        # that asset's imported pose, this one is 33 mm away and every other is 545-615 mm.
        #
        # Both engines take this one rotation. The asset pair this replaces needed two,
        # because the PhysX asset baked a root orientation that the Newton one did not;
        # this asset bakes none, so re-applying it here is right for either engine.
        # MEASURED: the palm lands at the same env-local (0.0, -0.247, 0.51) on both.
        rot=(0.0, 0.0, -0.70710678118, 0.70710678118),
        joint_pos={".*": 0.0},
    ),
    joint_ordering=SHADOW_HAND_JOINT_ORDERING,
    body_ordering=SHADOW_HAND_BODY_ORDERING,
    actuators={
        "direct_motors": ImplicitActuatorCfg(
            joint_names_expr=SHADOW_HAND_JOINT_NAMES,
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
            # Zeroing the damping on the grounds that the MuJoCo model supplies it is wrong for these
            # joints: MEASURED, they end at damping 0.0 while only the tendon-coupled J1/J2 inherit the
            # model's. A driven joint with stiffness and no damping is an undamped spring -- it rings
            # rather than settles, and an object resting on it is flung.
            damping={"rh_WRJ.*": 0.5, "rh_(FF|MF|RF|LF|TH)J.*": 0.1},
            # The asset's own drivetrain values, restated because they do not survive import: the asset
            # authors ``armature = 0.0002`` and ``frictionloss``/``jointFriction = 0.01`` in both its
            # physx and mujoco payloads, but MEASURED, leaving these at None lands every joint at 0.0 --
            # stiffness and damping do survive, armature and friction do not.
            #
            # An earlier revision forced ``armature = 2e-3``, the value the PREVIOUS asset's
            # configuration used. That is ten times what this asset describes and reads as a heavier,
            # slower hand.
            armature=2.0e-4,
            friction=1.0e-2,
        ),
        "coupled_joints": ImplicitActuatorCfg(
            joint_names_expr=["rh_(FF|MF|RF|LF)J(2|1)"],
            # Bound what the tendon can apply. MEASURED, leaving this unset lands these eight joints at
            # 3.4e38 -- float max -- while every directly driven joint gets 1 to 10 N-m. They are exactly
            # the joints the tendon pulls, so an unbounded torque there is a route to a diverging
            # velocity under contact. A PhysX fixed tendon has no force-range field of its own, so the
            # joint's effort limit is the only place the model's cap can live: the MuJoCo payload caps
            # each tendon actuator at ``mjc:forceRange`` +/-1, and the previous PhysX Shadow Hand gave
            # these same physical joints 0.9 and 0.7245.
            joint_effort_limit={
                "rh_(FF|MF|RF|LF)J2": 0.9,
                "rh_(FF|MF|RF|LF)J1": 0.7245,
            },
            # These take no position command -- the tendon drives them -- so their stiffness and damping
            # stay as the model authored them, which does survive import (None keeps the USD value).
            # Armature and friction do not survive, so they are restated from the asset as above.
            stiffness=None,
            damping=None,
            armature=2.0e-4,
            friction=1.0e-2,
        ),
    },
)
"""Shadow Hand on the asset's PhysX variant.

One USD asset serves both engines and its ``Physics`` variant selects which; nothing else differs
between them. Per-engine values that cannot be shared, such as the tendon gains each engine
expresses in its own units, are authored in the asset's variants rather than restated here.
"""

SHADOW_HAND_NEWTON_CFG = SHADOW_HAND_CFG.copy()
SHADOW_HAND_NEWTON_CFG.spawn.variants = {"Physics": "mujoco"}
"""Shadow Hand on the asset's MuJoCo variant, for the Newton (MJWarp) solver.

Identical to :data:`SHADOW_HAND_CFG` apart from the selected variant -- see it for why the spawn
pose and both orderings are shared.
"""
