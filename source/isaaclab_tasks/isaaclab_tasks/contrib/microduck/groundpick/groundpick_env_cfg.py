# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground-pick environment for the Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks3 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the shared mjlab base template live in the companion
``historical original source extraction`` and are cited as "reference section N".

Bring the mouth to the floor, as close as possible **without touching**, correctly oriented; hold it
there; stand back up cleanly; rest. Then do it again. The cycle is a 4 s clock the policy is told the
phase of, so this is a scheduled gesture rather than a goal-seeking one -- there is no target to
reach and no success flag, only a segmented profile the reward stack pays for following:

===========  =====================  ==========  ===================================================
segment      phase                  duration    what the stack pays for
===========  =====================  ==========  ===================================================
descent      ``[0, 0.375)``         1.5 s       mouth low and pointing down, neck moving slowly
low dwell    ``[0.375, 0.425)``     0.2 s       the same, held
rise         ``[0.425, 0.80)``      1.5 s       legs and neck back to the stand pose, trunk vertical
standing     ``[0.80, 1.0)``        0.8 s       the same, held
===========  =====================  ==========  ===================================================

Three things follow from the task rather than from the recipe:

* The robot is :data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_ALLCOLLISIONS_CFG`. The whole task is
bringing the
  head assembly to within millimetres of the floor, and on the walking model the head carries no
  collider at all -- the "close but not touching" equilibrium would have nothing to push back with
  and the mouth would sink through the plane (addendum section 12.3).
* **The "without touching" is an equilibrium, not a constraint.** ``mouth_ground_proximity`` pulls
  the mouth down and ``head_impact_penalty`` charges the contact force it would arrive with; the
  hover is where the two balance. Delete either one and the task becomes a different task.
* The **payload** is physics, not reward. A 10-40 g point mass is drawn per episode and hung off the
  mouth tip from the moment the mouth closes, so the return is a lift rather than an unweighted
  extension. Upstream smuggles it in as a weight-zero reward term; see :class:`EventsCfg`.

Isaac Lab has no MJCF **site** concept, and this task is the family's most site-dependent: upstream
measures both mouth terms on the ``mouth_tip`` site. The port carries the site as a fixed offset in
its parent body's frame -- see :data:`MICRODUCK_MOUTH_TIP_OFFSET` -- which is the same adaptation the
foot terms make, with the numbers measured off the pinned MJCF rather than assumed.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ALLCOLLISIONS_CFG

# The body carrying the three head collision shells and the mouth. Imported from the forward-roll
# task, which named it first for the same reason -- it is the surface that task pivots on and the one
# this task hovers over -- rather than restated here.
from isaaclab_tasks.contrib.microduck.roulade.roulade_env_cfg import MICRODUCK_HEAD_BODY_NAME
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR,
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

##
# The cycle (addendum section 5.1)
##

MICRODUCK_GROUND_PICK_PERIOD = 4.0
"""Length [s] of one bend-and-return cycle, upstream's ``GP_PERIOD``.

At the 50 Hz control rate this is 200 steps per cycle, and at the inherited 20 s episode it is five
complete cycles per episode.
"""

MICRODUCK_DESCENT_END = 0.375
MICRODUCK_HOLD_END = 0.425
MICRODUCK_RISE_END = 0.80
"""Phase boundaries of the four cycle segments; see the module docstring for their durations.

Upstream's file carries a second, contradictory description of the same profile inline -- a 6 s
period with a 0.6 s dwell and a 2.4 s rest -- which is the text of an earlier era and is not what the
constants say (addendum section 13.17 c). These are the constants.
"""

MICRODUCK_MOUTH_TIP_OFFSET = (-0.00809334, 0.0, -0.0777383)
"""Position [m] of the mouth tip in the frame of the body that carries it.

Upstream measures both mouth rewards on the MJCF ``mouth_tip`` **site**, which Isaac Lab has no
equivalent of. The site is rigidly attached to :data:`MICRODUCK_HEAD_BODY_NAME` (``jaw_soft``), so
this is that attachment, read straight off the pinned ``robot_allcollisions.xml``. At the model's
home pose it puts the mouth tip 0.1075 m above the trunk frame and 0.0776 m ahead of it, which is
what makes the descent a genuine forward fold rather than a squat.
"""

MICRODUCK_MOUTH_TIP_AXIS = (-0.00872562, 0.0, -0.99996193)
"""The mouth's pointing direction [-] in the same body frame.

The ``mouth_tip`` site's own ``x`` axis, which is what upstream's perpendicularity reward projects
onto world down. Reading it off the site's fixed quaternion rather than assuming a body axis matters:
it is half a degree off ``-z`` of ``jaw_soft``, and the term it feeds is a signed cosine.
"""

MICRODUCK_MOUTH_PAYLOAD_RANGE = (0.01, 0.04)
"""Mass range [kg] of the object the robot is asked to lift, drawn once per episode."""

MICRODUCK_MOUTH_PAYLOAD_RAMP = 0.05
"""Fraction of a cycle the payload fades in over, starting at :data:`MICRODUCK_HOLD_END`.

0.05 of a 4 s cycle is 0.2 s. The gate never falls back to zero before the cycle wraps, so the
payload hangs from phase 0.475 to the wrap -- about 52 % of every cycle -- and drops abruptly there
(addendum section 13.20).
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_MOUTH_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_HEAD_BODY_NAME])
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
# no ``preserve_order``: the material randomization resolves body IDs into backend shape ranges and
# documents that callers must not pre-swizzle them
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")
_FEET_GROUND_SENSOR_CFG = SceneEntityCfg("feet_ground_contact")
_HEAD_IMPACT_SENSOR_CFG = SceneEntityCfg("head_impact_contact")

_TERRAIN_PRIM_PATH = "/World/ground"

_FOOT_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?(left_foot_collision|right_foot_collision)/[^/]*"
)
"""Shape-level expression selecting the two sole colliders.

Upstream's ``feet_ground_contact`` sensor matches the same two geoms by name against the terrain
body. The reward that reads it counts contacts rather than indexing them, so the order the two
resolve in is not a contract here.
"""

_HEAD_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?(top_head_shell_1|jaw_1|bottom_head_shell_1)/[^/]*"
)
"""Shape-level expression selecting the three head shells, which is upstream's ``neck`` subtree.

Upstream senses the whole subtree rooted at the ``neck`` body against the terrain. Measured on the
pinned ``robot_allcollisions.xml``, that subtree is ``neck``, ``neck_pitch``, ``yaw_roll_motion`` and
``jaw_soft`` -- and only ``jaw_soft`` carries colliders, the three head shells named here. The
subtree and the three shells are therefore the same sensor, not an approximation of it.
"""

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle, upstream's velocity-matched value."""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency (reference section 8)."""

_IMU_MAX_LAG = 3
"""Worst-case IMU latency [control steps] the actor is trained against.

Upstream's value **for this task alone**. Every other MicroDuck task uses 1, the value the velocity
recipe's 2026-07 audit settled on after measuring the real Dynamixel IMU path at a +/-20 ms envelope,
and this task's inline comment claims it matches velocity -- which it stopped doing when that audit
landed (addendum section 13.16). It is transcribed rather than corrected, because it is a training
decision and not a port bug: 3 steps is 60 ms, so a policy trained here tolerates a *wider* latency
envelope than the hardware has.
"""


def _iterations(count: int) -> int:
    """Convert an upstream PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Physics preset
##


@configclass
class MicroDuckGroundPickPhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck ground-pick environment.

    MicroDuck is trained on MuJoCo upstream, so MJWarp is the reference backend and the only one
    offered; it is also the only backend that can run this task as configured, because the
    environment sets ``sim.use_newton_actuators = True`` and the solver-hosted BAM model is rejected
    on the PhysX family's host adapter. See
    :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.MicroDuckPhysicsCfg`.

    Upstream leaves ``cfg.sim`` untouched here and inherits the mjlab template's ``nconmax = 35``
    with 10 solver and 20 line-search iterations (addendum section 12.2). The iteration counts are
    transcribed; the buffers are measured, as on every sibling.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Measured, not inherited. Profiling under random actions -- the regime where the robots
            # collapse onto the floor and grind every collider into it, with the tilt termination
            # dropped and the pushes forced to full magnitude -- peaks at **27 contacts and 86
            # constraints** per environment. Logs:
            # ``artifacts/microduck/profile_microduck_contacts_groundpick_{256,2048,4096}envs.log``,
            # from ``artifacts/microduck/profile_microduck_contacts.py``.
            #
            # Profiled at three sizes, because the smallest one undercuts the constraint tail:
            #
            # ============  =========  =========  ===========  ========
            # environments  peak ncon  peak nefc  median ncon  overflow
            # ============  =========  =========  ===========  ========
            # 256                  26         74           21         0
            # 2048                 26         86           23         0
            # 4096                 27         86           24         0
            # ============  =========  =========  ===========  ========
            #
            # 4096 is this task's own training default and therefore the size that matters. The
            # constraint peak **saturates** from 2048 upward and the contact peak moves by one, which
            # is what says the tail has been sampled rather than that it keeps growing; the medians
            # agree across all three sizes. The figures are the sit-stand task's to within a contact,
            # which is the expected answer -- the same robot sprawled on the same floor -- and this
            # task never visits the deliberate seated multi-contact state that made sit-stand's
            # solver diverge, so it keeps the template's iteration counts rather than its buffers.
            #
            # ``njmax`` is a hard per-environment cap and carries the margin: 128 against 86 is 42
            # constraints, about ten further contacts' worth. ``nconmax`` is a per-environment
            # *share* of one shared pool rather than a cap, so an environment spiking past 32 borrows
            # from the pool; at 4096 environments the shipped share provides 131 072 contacts against
            # a measured worst-case total of 47 121, and the measured mean is 11.5 per environment.
            njmax=128,
            nconmax=32,
            # the mjlab template's, which upstream inherits unchanged on this task
            iterations=10,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        # upstream steps MuJoCo once per 0.005 s physics tick (reference section 1)
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MicroDuckGroundPickSceneCfg(InteractiveSceneCfg):
    """Scene with the all-collisions MicroDuck robot on a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path=_TERRAIN_PRIM_PATH,
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(
            # neutral, so the friction the robot pushes off is the 1.0 the MJCF authors on the foot
            # soles and the foot-friction randomization scales
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # The all-collisions model, and this is the task with the strongest reason for it in the family:
    # the three ``jaw_soft`` head shells are the only thing standing between "mouth close to the
    # floor" and "mouth through the floor". On the walking model the head cannot touch the ground at
    # all, so ``head_impact_penalty`` would read zero forever and the task's defining equilibrium
    # would not exist.
    robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Feet and trunk, as every task in the family does: upstream tracks ground contact on the two
    # soles and drops the base template's terrain and foot-height scanners wholesale. This one feeds
    # the three critic foot observations.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    # The "keep both feet planted" signal. Upstream matches the two sole *geoms* against the terrain
    # *body*; this is that sensor on the Newton backend's shape-level expressions.
    #
    # Filtering against the terrain rather than reading a net force is load-bearing on this task in a
    # way it is not on the walking ones: a robot folded far enough to put its mouth on the floor has
    # its own shins and head shells within reach of its soles, and an unfiltered sole would report
    # "grounded" for touching them.
    #
    # The terrain expression is absolute rather than ``{ENV_REGEX_NS}``-relative because the ground
    # plane is a single shape shared by every environment.
    feet_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_FOOT_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
    )

    # The other half of the task's central equilibrium: what the mouth is about to hit. Upstream
    # senses the ``neck`` subtree against the terrain and reduces to a net force; this is the same
    # set of colliders (see :data:`_HEAD_COLLIDER_SHAPE_EXPR`) filtered against the ground plane.
    #
    # Terrain-filtered rather than unfiltered, unlike the forward-roll task's head sensor: there the
    # question is "is the head touching anything", here it is "how hard is the head hitting the
    # *floor*", and a knee brushing a head shell must not be charged as a face-plant.
    head_impact_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_HEAD_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
    )

    # Self-collision sensor, identical to the rest of the family's: the model's ten enabled colliders
    # against each other, many-to-many, which is upstream's trunk-subtree-against-itself sensor. The
    # reward saturates it back to upstream's 0-or-1 scale.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP (addendum section 5.3).

    One live command and two shape placeholders. The twist slot carries the cycle phase instead of a
    velocity, which is the task; the head and body slots are zero padding, filled in
    :class:`ObservationsCfg` rather than by a command term, because this task drives the head itself
    and has nothing to steer the trunk with.
    """

    # The cycle clock, riding in the twist slot so the deployed 61-wide vector keeps its shape. The
    # inherited velocity ranges are never sampled -- the term writes the encoding directly -- and
    # neither is the resampling time, so neither is configured beyond the shape the parent requires.
    base_velocity = mdp.GroundPickPhaseCommandCfg(
        asset_name="robot",
        resampling_time_range=(MICRODUCK_GROUND_PICK_PERIOD, MICRODUCK_GROUND_PICK_PERIOD),
        heading_command=False,
        debug_vis=False,
        period=MICRODUCK_GROUND_PICK_PERIOD,
        # decorrelates the environments, so they do not all bend at once. Upstream's roller trick
        # tasks pass False instead, to match a runtime whose cycle starts from a standing button
        # press; this one wants the spread.
        randomize_phase=True,
        ranges=mdp.GroundPickPhaseCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Identical to the velocity task's: the same 14 servos in the same deploy order, at the same unit
    scale, closing the same encoder-bias loop. Upstream notes that the head servos are deliberately
    **not** wrapped in its neck-offset action term here, because the head is part of the motion the
    task is asking for rather than a separately steerable payload.
    """

    joint_pos = mdp.ArticulationEncoderBiasActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (addendum section 11).

    The actor group is byte-for-byte the velocity task's 61-wide deploy contract, which is the whole
    point of the family: one runtime on the robot feeds every MicroDuck policy from the same buffer.
    Here **ten of those columns are constant zero** -- the head and body command slots -- because the
    task steers neither. They are shape placeholders for a runtime hot-swap, not inputs to wake up.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract.

        See :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.ObservationsCfg` for
        what each corruption models. The one number that is not the velocity task's is the IMU
        latency bound; see :data:`_IMU_MAX_LAG`.
        """

        base_ang_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.base_ang_vel_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": _IMU_MAX_LAG,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        projected_gravity = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.projected_gravity_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": _IMU_MAX_LAG,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_articulation_biased,
            params={"asset_cfg": _SERVO_JOINT_CFG, "biased": True},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.joint_vel_rel,
                "term_params": {"asset_cfg": _SERVO_JOINT_CFG},
                "min_lag": 1,
                "max_lag": 1,
                "update_period": 0,
            },
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function (addendum section 11.2).

        The actor's terms with every corruption removed, plus the base linear velocity and the three
        foot terms the robot has no sensor for. It is narrower than the velocity task's critic by the
        two ``foot_height`` columns, which upstream deletes here because this scene carries no height
        sensor and no foot-height reward to justify one.

        The two sensor-derived foot terms are the NaN-guarded variants, as on every sibling port and
        unlike upstream here; see :class:`TerminationsCfg` for why.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_articulation_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False}
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_air_time = ObsTerm(func=mdp.foot_group_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_group_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_group_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events (addendum section 5.7).

    The domain-randomization suite is the velocity task's, term for term, with **two deliberate
    narrowings** upstream makes here and says why:

    * The pushes are **half** the family's magnitude, +/-0.15 m/s. Upstream's note is measured: the
      gesture is quasi-static, and at +/-0.3 the robot fell over even standing straight.
    * The neck-offset randomization is **off**, because the head is the task's working end rather
      than a payload to be jittered.

    There is no ground-state reset: every episode spawns upright from :attr:`reset_base`, so the
    declaration-order coupling that governs the sit-stand and ball-kick resets does not arise here.

    :attr:`mouth_payload_force` is this task's one structural deviation from upstream, and it is
    physics rather than a reward. Upstream registers the same function as a **weight-zero reward
    term**, so that the reward manager calls it every step and it can write an external wrench
    (addendum section 13.20). This port registers it where Isaac Lab writes state, as an interval
    event on a zero-width interval -- which fires every control step, for every environment -- so
    that a later pass that prunes zero-weight rewards cannot silently delete the payload.
    """

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": _FOOT_MATERIAL_CFG,
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    encoder_bias = EventTerm(
        func=mdp.randomize_articulation_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.015, 0.015)},
    )

    mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    ##
    # Reset: redrawn every episode, in this order.
    ##

    # Upstream samples the absolute base height in (0.12, 0.13) m; Isaac Lab samples an offset from
    # the configured default, which is the 0.125 m midpoint of that band.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

    # Zero-width on purpose: upstream resets every joint exactly to the stand pose, which is also the
    # pose the return terms score against.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # The object in the mouth, drawn fresh every episode. The robot is never told what it weighs.
    sample_mouth_payload = EventTerm(
        func=mdp.sample_mouth_payload,
        mode="reset",
        params={"min_kg": MICRODUCK_MOUTH_PAYLOAD_RANGE[0], "max_kg": MICRODUCK_MOUTH_PAYLOAD_RANGE[1]},
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _HEAD_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    ##
    # Interval.
    ##

    # Every control step, for every environment: a zero-width interval is how Isaac Lab spells "each
    # step". See the class docstring for why the payload lives here rather than in the reward stack.
    mouth_payload_force = EventTerm(
        func=mdp.apply_mouth_payload_force,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": _MOUTH_BODY_CFG,
            "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            "command_name": "base_velocity",
            "hold_end": MICRODUCK_HOLD_END,
            "ramp": MICRODUCK_MOUTH_PAYLOAD_RAMP,
        },
    )

    # Half the family's magnitude, and live at full strength from step 0 -- there is no push
    # curriculum on this task. Upstream measured +/-0.3 m/s toppling the robot even standing
    # straight, which a quasi-static reaching gesture cannot absorb.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (addendum sections 5.4 and 5.5).

    Four groups, and the phase gate is what keeps them from fighting:

    * **The bend** -- ``mouth_ground_proximity`` and ``mouth_perpendicular_to_ground``, both gated on
      the descent and the low dwell.
    * **The return** -- the two ``ground_pick_return_pose`` terms and ``return_upright``, gated on
      the rise and the standing rest. Their combined weight, 16.0, is the largest block in the stack:
      upstream's experience is that the bend is easy and the clean return is not.
    * **The posture floor** -- ``feet_grounded``, ``feet_flat``, ``upright``, ``self_collisions``.
      Ungated, because a foot must not leave the ground at any phase.
    * **Regularization**, deliberately heavier than the walking task's. Upstream's note: the gesture
      is slow and precise, so strong smoothness helps transfer, unlike the dynamic stand-up recovery
      where heavy regularization blocked the motion outright.

    Two of the shapes deserve stating because they are easy to "fix" into something else:

    * ``upright`` is deliberately **weak** at 0.2 against the velocity task's 2.0. The approach
      *requires* a deep forward lean, so a strong always-on uprightness reward would price the task
      out; ``return_upright`` supplies the verticality where it is actually wanted.
    * ``mouth_perpendicular_to_ground`` can go **negative**: its kernel is a signed cosine, so a
      mouth pointing up during the descent gate is charged rather than merely unpaid.

    Two upstream slots are not carried over:

    * ``soft_landing``. The term is **live upstream** at ``-1e-5``; what is a no-op is the
      *assignment*, because the mjlab base template already ships that weight. This port's shared
      base carries no such term, so there is nothing to restate, and dropping it is a real deviation
      of order ``1e-5`` against a stack summing to order 10. It is worth naming for what it
      demonstrates rather than what it weighs: it is gated on the *magnitude* of the twist command,
      and this task's twist is a unit vector on the circle, so an inherited "only while moving" gate
      is permanently open on any phase-command environment (addendum section 13.10).
    * ``mouth_payload_force``, a weight-zero reward that is really a physics hook; see
      :class:`EventsCfg`.
    """

    ##
    # The bend: bring the mouth to the floor, pointing down.
    ##

    mouth_ground_proximity = RewTerm(
        func=mdp.mouth_ground_proximity_phased,
        weight=3.0,
        params={
            "asset_cfg": _MOUTH_BODY_CFG,
            "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
            "std": 0.10,
            "target_height": 0.0,
            "command_name": "base_velocity",
            "descent_end": MICRODUCK_DESCENT_END,
            "hold_end": MICRODUCK_HOLD_END,
            "rise_end": MICRODUCK_RISE_END,
        },
    )
    mouth_perpendicular_to_ground = RewTerm(
        func=mdp.mouth_perpendicular_phased,
        weight=2.0,
        params={
            "asset_cfg": _MOUTH_BODY_CFG,
            "mouth_axis_b": MICRODUCK_MOUTH_TIP_AXIS,
            "command_name": "base_velocity",
            "descent_end": MICRODUCK_DESCENT_END,
            "hold_end": MICRODUCK_HOLD_END,
            "rise_end": MICRODUCK_RISE_END,
        },
    )

    ##
    # The return: stand back up, cleanly.
    ##

    # Loose width on the legs, whose extension *is* the return.
    ground_pick_return_pose_legs = RewTerm(
        func=mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "asset_cfg": _LEG_JOINT_CFG,
            "std": 0.3,
            "command_name": "base_velocity",
            "hold_end": MICRODUCK_HOLD_END,
            "rise_end": MICRODUCK_RISE_END,
        },
    )
    # Half the width on the neck, and upstream says why: the head shells sit close to the trunk at
    # the stand pose, so a neck that overshoots on the way back drives them into it -- and the
    # self-collision sensor cannot see that on the model upstream measured it on.
    ground_pick_return_pose_neck = RewTerm(
        func=mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "asset_cfg": _HEAD_JOINT_CFG,
            "std": 0.15,
            "command_name": "base_velocity",
            "hold_end": MICRODUCK_HOLD_END,
            "rise_end": MICRODUCK_RISE_END,
        },
    )
    return_upright = RewTerm(
        func=mdp.ground_pick_return_upright_phased,
        weight=4.0,
        params={
            "std": 0.4,
            "command_name": "base_velocity",
            "hold_end": MICRODUCK_HOLD_END,
            "rise_end": MICRODUCK_RISE_END,
        },
    )

    ##
    # The posture floor: whatever the phase, stay on your feet.
    ##

    feet_grounded = RewTerm(
        func=mdp.feet_grounded_reward,
        weight=3.0,
        params={"sensor_cfg": _FEET_GROUND_SENSOR_CFG},
    )
    # Ungated, unlike the roller tasks' use of the same kernel: this gesture has no swing phase, so
    # both soles are asked to lie flat at every phase. It is what stops a foot pivoting onto its edge
    # while keeping the single contact point ``feet_grounded`` is satisfied by.
    feet_flat = RewTerm(
        func=mdp.feet_flat_penalty,
        weight=-2.0,
        params={"asset_cfg": _FOOT_BODY_CFG, "normal_axis": (0.0, 1.0, 0.0)},
    )
    upright = RewTerm(
        func=mdp.upright_selected_body,
        weight=0.2,
        params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG},
    )
    # The other side of the task's defining balance. ``saturate`` is not used here, unlike the
    # self-collision term: this is a force, and its magnitude is the signal.
    head_impact_penalty = RewTerm(
        func=mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_cfg": _HEAD_IMPACT_SENSOR_CFG, "threshold": 1.0},
    )
    # ``saturate`` keeps the many-to-many sensor on upstream's 0-or-1 scale, so the weight is the
    # penalty for touching yourself at all rather than a per-collider tariff.
    self_collisions = RewTerm(
        func=mdp.self_collision_cost_saturable,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG, "saturate": True},
    )

    ##
    # Regularization, heavier than the walking task's.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2_raw, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2_raw, weight=-0.02)
    # Inherited silently by upstream -- it is simply not in this task's deletion list (addendum
    # section 13.18) -- so it is stated here rather than left to be discovered.
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # The stage-0 weight of the ``action_rate_weight`` curriculum, which ramps it to -2.0 by
    # iteration 500. Upstream declares -2.0 here *and* -0.8 at stage 0; the curriculum manager runs
    # before the first reward evaluation, so only -0.8 was ever live and the declared literal was
    # dead (addendum section 13.12). The pair is collapsed to the number that applies.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.8)
    # On top of the whole-body action rate, not instead of it: the head is the working end here and
    # its jitter is what a real servo pays for.
    neck_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2,
        weight=-1.0,
        params={"action_name": "joint_pos", "asset_cfg": _HEAD_JOINT_CFG},
    )
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5e-3, params={"asset_cfg": _SERVO_JOINT_CFG})
    # The anti-dive term, gated on the descent and the low dwell with a hard step rather than a ramp:
    # the neck is free the instant the rise begins.
    neck_vel_descent = RewTerm(
        func=mdp.neck_vel_descent_penalty,
        weight=-0.1,
        params={
            "asset_cfg": _HEAD_JOINT_CFG,
            "command_name": "base_velocity",
            "hold_end": MICRODUCK_HOLD_END,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (addendum section 5.7).

    The tilt termination is **kept** here, unlike on the sit-stand and stand-up tasks: this robot is
    supposed to stay on its feet at every phase, so a fall is a failed episode rather than something
    to be paid for and recovered from.

    Upstream also inherits a terrain-bounds termination, which returns all-false on a ground plane
    and is therefore dead on this flat-only task (addendum section 7.24); it is not carried over.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

    # Catches a *broken* robot rather than a fallen one.
    #
    # Deviation from upstream, deliberately and in line with every sibling port: upstream leaves this
    # term's sensor list empty here, which the extraction reads as drift rather than design and
    # recommends closing everywhere in the port (addendum section 14). This task carries the longest
    # list in the family, because it has the most force paths that feed something: the critic's three
    # foot observations, the grounded reward and -- uniquely -- a *reward* read straight off a
    # contact force, where a single non-finite value poisons the episode sum.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan_optional_sensors,
        time_out=False,
        params={"sensor_names": ("contact_forces", "feet_ground_contact", "head_impact_contact")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (addendum section 5.8).

    Nothing schedules the task: the whole cycle is rewarded from step 0, and the pushes are live at
    full strength from step 0 too, because they are already half magnitude. What is scheduled is the
    cost of moving and the width of the hardware spread.

    The centre-of-mass ramp is the **widest in the family**, reaching +/-20 mm against the sit-stand
    and ball-kick tasks' +/-15 mm and the roller family's +/-10 mm. That is the task talking: the
    gesture is a slow quasi-static fold, so the robot has time to compensate for a badly placed
    centre of mass, and being able to is exactly what makes the trained gesture survive a real
    battery pack sitting a centimetre off where the model puts it.

    Upstream's schedules are written in PPO iterations; :func:`_iterations` converts them to the
    global environment-step count these terms compare against.
    """

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -0.8},
                {"step": _iterations(250), "weight": -1.5},
                {"step": _iterations(500), "weight": -2.0},
            ],
        },
    )

    com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
                {"step": _iterations(1500), "range": 0.015},
                {"step": _iterations(2000), "range": 0.02},
            ],
        },
    )

    head_com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_head_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckGroundPickFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck ground-pick environment on flat ground."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckGroundPickPhysicsCfg())
    # ``env_spacing`` is upstream's scene extent (reference section 1).
    scene: MicroDuckGroundPickSceneCfg = MicroDuckGroundPickSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 0.005 s physics steps decimated by 4 give the 50 Hz control rate the deployed policy runs
        # at. Episodes are the family's inherited 20 s -- 1000 control steps -- which is exactly five
        # complete cycles at :data:`MICRODUCK_GROUND_PICK_PERIOD`.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        # Run the BAM servos through the backend-native path, as the rest of the family does and as
        # upstream does. The decimation above is even, which is what lets the stateful servo delay
        # line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.feet_ground_contact is not None:
            self.scene.feet_ground_contact.update_period = self.sim.dt
        if self.scene.head_impact_contact is not None:
            self.scene.head_impact_contact.update_period = self.sim.dt
        if self.scene.self_collision is not None:
            self.scene.self_collision.update_period = self.sim.dt
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
