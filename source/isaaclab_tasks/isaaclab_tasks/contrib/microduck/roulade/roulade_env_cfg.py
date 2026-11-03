# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-roll ("roulade") environment for the Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks2 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the shared mjlab base template live in the companion
``historical original source extraction`` and are cited as "reference section N".

The task is a trick, not a skill: the robot starts standing, tips forward over the flat top of its
head, rolls through 360 degrees and lands back on its feet. It is triggered at deployment the way
sit and stand-up are -- the policy switch *is* the trigger, so there is no phase clock, no reference
motion and no command to follow.

Three things make the recipe different from every other MicroDuck task, and the rest follows from
them:

* **The reward is driven by a rotation accumulator, not by the instantaneous state.** Nothing in a
  single frame says how far around the robot has come, so the forward pitch rate is integrated by
  hand into :class:`~isaaclab_tasks.contrib.microduck.pollen_mdp.events.RouladeRollState`, and the landing
  rewards are gated on that integral. The integral only advances while the robot is *supported* and
  while the roll is *sagittal*, which is what stops a ballistic whip or a shoulder roll from
  counting -- both of which upstream's earlier runs discovered and preferred.
* **Nothing may oppose the flip.** There is no fall termination (falling over is the task), no
  always-on upright reward, and the two motion-blocking regularizers the walking recipe shares are
  kept 25 times lighter here or introduced late by curriculum. Upstream established twice that a
  motion tax active during discovery stops the manoeuvre being found at all.
* **Half of every episode starts part-way through the roll.** The mid-roll spawn is a reverse
  curriculum: the second half of a roulade is the face-up recovery problem, which the stand-up task
  already shows is learnable, so those spawns give dense data on the landing while the flip itself
  is still being discovered.

The robot is :data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_ALLCOLLISIONS_CFG`, as the stand-up task's is,
and
here the three head shells it adds are what the whole task rests on: they are the surface the robot
pivots on and the only thing the head-ground sensor can report (addendum sections 2.2 and 2.4).

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

import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ALLCOLLISIONS_CFG
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR,
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

##
# Task constants (addendum section 4.1)
##

MICRODUCK_STAND_HEIGHT = 0.115
"""Trunk height [m] the roll has to land back on, which every completion-gated reward targets.

Upstream's ``STAND_Z``, restated here rather than imported from the stand-up task because upstream
restates it too and the two tasks are independent recipes. It is the height a velocity policy holds
the robot still at, 2 mm below the 0.11718 m the model reaches geometrically at the stand pose
(addendum section 2.2).
"""

MICRODUCK_HEAD_BODY_NAME = "jaw_soft"
"""The body carrying the three head collision shells (``top_head_shell``, ``jaw``,
``bottom_head_shell``) on the all-collisions model, which is the surface the robot rolls over."""

MICRODUCK_TUCK_JOINT_POS = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "neck_pitch": -1.0,
    "head_pitch": 1.0,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}
"""The tuck anchor [rad] a mid-roll spawn is lerped toward, as absolute joint positions.

Legs folded into the crouch the stand-up work measured, plus a **chin tuck**: ``neck_pitch`` at -1
and ``head_pitch`` at +1 put the flat top of the head squarely on the floor. That pair is not
cosmetic -- upstream measured the head-top axis at -0.99 with the chin tucked against +0.6 for a
passive face-plant, and the over-the-head latch only fires below -0.3 (addendum section 4.3). A
mid-roll spawn must therefore demonstrate the tucked configuration, or it spawns in a pose the task
does not reward.

Upstream keys this by servo index; the port keys it by name, because the converted asset resolves
joints in Newton's order rather than the MJCF's.
"""

MICRODUCK_MIDROLL_PITCH_RANGE = (math.radians(50.0), math.radians(340.0))
"""How far into the roll a mid-roll episode spawns [rad].

90 degrees is balanced on the head, 180 is flat on the back, 270 is supine and about 340 is seated
and leaning back. Upstream widened the upper bound from 185 to 340 degrees after a run in which the
second half of the roll -- supine, to seated, to standing -- was never spawned and never learned;
spawns past about 300 degrees open the landing gate at birth, which is what puts dense on-policy
data on the last mile.
"""

MICRODUCK_MIDROLL_OMEGA_RANGE = (0.0, 3.0)
"""Forward roll rate [rad/s] a mid-roll spawn is born with."""

MICRODUCK_FORWARD_VEL_RANGE = (0.0, 0.0)
"""Forward base velocity [m/s] a standing spawn is born with -- upstream's "élan" hook, disabled.

Widening it to e.g. ``(0.0, 0.3)`` would train rolls entered out of a walk, approximating a hand-off
from the walking policy without simulating the walk. Shipped at zero, so the branch never runs; it
is carried across because it is a documented extension point, not dead configuration.
"""

MICRODUCK_LANDING_GATE = (math.radians(260.0), math.radians(330.0))
"""Rotation band [rad] over which the landing rewards fade in.

Below 260 degrees the robot has not finished the roll and the whole landing annuity is zero, which
is what stops a standing spawn farming it by standing still.
"""

MICRODUCK_RISE_GATE = (math.radians(180.0), math.radians(260.0))
"""Rotation band [rad] over which the exit-rise reward fades in, one quadrant earlier.

It opens on the back, because from there the rest of the roll is the face-up recovery problem and
the end-state rewards have no gradient at zero motion.
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
# the head *shell* body, which the roll's two accumulator terms measure the tuck on -- not the four
# head bodies the centre-of-mass randomization perturbs
_HEAD_SHELL_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_HEAD_BODY_NAME])
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
# no ``preserve_order``: the material randomization resolves body IDs into backend shape ranges and
# documents that callers must not pre-swizzle them
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")
_HEAD_GROUND_SENSOR_CFG = SceneEntityCfg("head_ground_contact")
_ROBOT_GROUND_SENSOR_CFG = SceneEntityCfg("robot_ground_contact")

_TERRAIN_PRIM_PATH = "/World/ground"
"""Where the terrain is imported, shared by every environment.

Named once because the support sensor filters against it: a terrain that moved without the filter
following would silently leave that sensor with no contact partner and open its gate permanently.
"""

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle, upstream's velocity-matched value."""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency (reference section 8)."""


def _iterations(count: int) -> int:
    """Convert an upstream PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Physics preset
##


@configclass
class MicroDuckRouladePhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck forward-roll environment.

    MicroDuck is trained on MuJoCo upstream, so MJWarp is the reference backend and the only one
    offered. Upstream inherits the mjlab base template's solver limits unchanged and never revisits
    them for this task (addendum section 7.4); this port measures them instead, because this is the
    task in the family that deliberately slams a 0.28 kg head assembly into the floor.

    MJWarp is also the only backend that can run this task as configured: the environment sets
    ``sim.use_newton_actuators = True`` and the solver-hosted BAM model is rejected on the PhysX
    family's host adapter. See
    :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.MicroDuckPhysicsCfg`.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Measured, not inherited. Profiled under random actions with every episode forced into
            # the mid-roll bucket at the top of its angular-momentum range, which is the regime that
            # slams the head and both hips into the floor at once: **86 constraints and 26 contacts**
            # per environment at 2048 environments, 74 and 26 at 256. The contact peak agrees to
            # zero between the two scales, which is what says the tail has been sampled; the
            # constraint peak does not -- it grew from 74 to 86 -- so ``njmax`` carries the wider
            # margin of the two. Logs:
            # ``artifacts/microduck/profile_microduck_contacts_roulade_{256,2048}envs.log``, from
            # ``artifacts/microduck/profile_microduck_contacts.py``.
            #
            # ``njmax`` is a hard per-environment cap; ``nconmax`` is a per-environment share of one
            # shared buffer and cannot overflow at the measured peak, so it sits just above it, at
            # the same 1.2x the stand-up task's measured budget uses.
            njmax=160,
            nconmax=32,
            # upstream's flat solver profile, which this task inherits unchanged
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
class MicroDuckRouladeSceneCfg(InteractiveSceneCfg):
    """Scene with the all-collisions MicroDuck robot on a ground plane, and four contact sensors."""

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

    # The all-collisions model, for a stronger reason than the stand-up task has: its three
    # ``jaw_soft`` head shells are the surface this whole task pivots on. On the walking model the
    # head cannot touch the ground at all, so the head sensor below would never fire, the latch
    # could never be earned, and every completion-gated reward would be permanently zero on the
    # standing bucket (addendum section 2.4).
    robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Feet and trunk, as the stand-up and velocity tasks do: upstream tracks ground contact on the
    # two soles and drops the base template's terrain and foot-height scanners wholesale.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    # Self-collision sensor, identical to the stand-up task's: the model's ten enabled colliders
    # against each other, many-to-many, which is upstream's trunk-subtree-against-itself sensor. The
    # reward saturates it back to upstream's 0-or-1 scale.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
    )

    # The roll's pivot signal. Upstream matches the ``jaw_soft`` body against the terrain and reads
    # whether its single contact slot found anything; this reads the same body's net contact force
    # and asks whether it is non-zero, which is the same 0-or-1 signal for everything the head can
    # actually reach. The narrowing is that a *self*-contact -- a knee driven into the head shell of
    # a deeply tucked robot -- also reads as contact here. The latch it feeds is guarded on top of
    # this by the head-top-down test and the first-quadrant rotation window, so a spurious latch
    # needs the robot to be mid-roll with its head pointing at the floor anyway.
    head_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/jaw_soft",
    )

    # The support gate. Upstream matches the whole trunk subtree against the terrain, and so does
    # this: the model's ten world colliders filtered against the ground plane, which answers "is the
    # robot touching the *floor*" rather than "is the robot touching anything".
    #
    # That distinction is the anti-breakdance gate's whole job. Read unfiltered, a net contact force
    # cannot tell the floor from the robot's own shin, so a tucked airborne robot registered as
    # supported and earned the rotation the gate exists to deny -- the gate's failure direction, not
    # its safe one. The terrain filter closes it: the accumulator reads this sensor's force matrix
    # rather than its net force (see ``_update_roulade_state``).
    #
    # The terrain expression is absolute rather than ``{ENV_REGEX_NS}``-relative because the ground
    # plane is a single shape shared by every environment.
    robot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
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
    """Command specifications for the MDP (addendum section 4.9).

    The roll takes no command at all: it is triggered by the policy switch, and the head is part of
    the manoeuvre rather than something to steer. The twist survives only so the deployed 61-wide
    observation keeps its three-wide slot, and the head and body slots are zero padding rather than
    commands (see :class:`ObservationsCfg`).
    """

    # No reward reads this. The ranges are a hundredth of the velocity task's and it resamples at
    # most once per episode. ``rel_forward_envs = 0.2`` is inherited from upstream's base template
    # and never overridden there (addendum section 7.22): a fifth of the resamples get a commanded
    # surge of 0.3 m/s that nothing acts on, which is a quirk of the observation distribution rather
    # than of the behaviour, and is reproduced rather than quietly cleaned up.
    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.01, 0.01),
            lin_vel_y=(-0.01, 0.01),
            ang_vel_z=(-0.05, 0.05),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Identical to the velocity and stand-up tasks': the same 14 servos in the same deploy order, at
    the same unit scale, closing the same encoder-bias loop.
    """

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (addendum section 6).

    The actor group is byte-for-byte the family's 61-wide deploy contract, which is the whole point:
    one runtime on the robot feeds every MicroDuck policy from the same buffer, so a trick policy
    and a walking policy have to read the same vector. This task has no head-pose and no body-pose
    command, so both slots are zero padding -- the deployed runtime sends zeros for them too.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract.

        See :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.ObservationsCfg` for
        what each corruption models; the terms, their order, their noise and their delays are
        upstream's velocity values, which this task copies deliberately for sim-to-real parity.
        """

        base_ang_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.base_ang_vel_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": 1,
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
                "max_lag": 1,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_biased,
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
        """Privileged observations for the value function (addendum section 6.2).

        The actor's terms with every corruption removed, plus the base linear velocity and the three
        foot terms the robot has no sensor for. It is the stand-up task's critic with the two
        command blocks replaced by the same padding the actor carries.

        The two sensor-derived terms are the NaN-guarded variants. Upstream applies them on the
        stand-up task only and gives its reason there -- a robot that lands and flips constantly
        produces degenerate contacts far more often than a walking one -- and the extraction reads
        their absence here as drift rather than design (addendum section 7.9). This task flips
        harder than the stand-up task does, so the guard is applied here too; see
        :class:`TerminationsCfg` for the matching half of that deviation.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_air_time = ObsTerm(func=mdp.foot_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events (addendum section 4.7).

    The domain-randomization suite is the stand-up task's, term for term and range for range, minus
    the push: upstream deletes ``push_robot`` here with the one-line reason that *a push mid-roll is
    incoherent*. What this task adds is :attr:`set_roulade_state`, which replaces the root pose with
    a standing start or a mid-roll one and seeds the rotation bookkeeping to match.

    **Declaration order is behaviour.** Isaac Lab fires reset events in the order they are declared.
    :attr:`set_roulade_state` overwrites the root pose and velocity :attr:`reset_base` wrote, and its
    mid-roll tuck lerps *from* the pose :attr:`reset_robot_joints` wrote, so it has to run after both
    (addendum sections 4.5 and 7.11).
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
        func=mdp.randomize_encoder_bias,
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

    # Upstream also randomizes the base height and yaw here, and ``set_roulade_state`` then
    # overwrites both (addendum section 7.11). Only the horizontal spread is live, so only the
    # horizontal spread is configured.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )

    # Zero-width on purpose: upstream resets every joint exactly to the stand pose, and the mid-roll
    # bucket of ``set_roulade_state`` is what tucks them afterwards.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # The episode distribution, and the rotation bookkeeping that goes with it. The two
    # probabilities here are the ``roulade_spawn_mix`` curriculum's first stage, which walks them
    # toward the standing start as the flip gets discovered; the height bands, the tuck and the
    # angular momentum are left alone by that curriculum.
    set_roulade_state = EventTerm(
        func=mdp.reset_roulade_state,
        mode="reset",
        params={
            "standing_prob": 0.5,
            "midroll_prob": 0.5,
            "standing_z_range": (0.11, 0.12),
            "standing_tilt_max": math.radians(5.0),
            "forward_vel_range": MICRODUCK_FORWARD_VEL_RANGE,
            "midroll_pitch_range": MICRODUCK_MIDROLL_PITCH_RANGE,
            "midroll_z_range": (0.05, 0.10),
            "midroll_omega_range": MICRODUCK_MIDROLL_OMEGA_RANGE,
            "tuck_joint_pos": MICRODUCK_TUCK_JOINT_POS,
            "tuck_factor_range": (0.3, 1.0),
            # about 4.6 degrees per joint, so a mid-roll spawn is a *family* of tucked poses rather
            # than one keyframe the policy can overfit
            "joint_noise_std": 0.08,
            "asset_cfg": _SERVO_JOINT_CFG,
        },
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP (addendum section 4.4).

    One dense signal drives the roll and one gated annuity drives the landing:

    * **Progress.** :attr:`roulade_progress` pays increments of the rotation frontier, capped in
      rate. It is potential-based, so a full roll pays a fixed total however it is spread out, and
      the cap makes a violent whip collect *less* than a controlled roll rather than the same amount
      sooner.
    * **The landing annuity.** Six terms are multiplied by a completion gate that only opens past
      260 degrees of supported, sagittal rotation *and* requires the over-the-head latch. They are a
      broad composite, two bootstrap layers under it, a sharp layer above it, a shortfall tax that
      makes crumpling in a heap net negative, and an exit-rise term gated one quadrant earlier.
    * **Straightness.** Three ungated penalties -- out-of-plane rotation, lateral velocity and
      trunk tilt out of the sagittal plane -- give the per-step gradient back toward a clean roll.
      The accumulator's own sagittal gate already makes a shoulder roll unprofitable; these are what
      steer out of one.

    Sign convention: :attr:`roulade_stand_tax` and :attr:`gentle_landing` negate themselves and
    therefore take **positive** weights.

    :attr:`roulade_progress` is declared **first** on purpose: it is the term that advances the
    rotation accumulator, and Isaac Lab evaluates reward terms in declaration order, so every gated
    term below reads the frontier it has just computed.
    """

    ##
    # The roll itself.
    ##

    roulade_progress = RewTerm(
        func=mdp.roulade_progress,
        weight=8.0,
        params={
            "target_angle": 2.0 * math.pi,
            # Upstream raised this from 3 rad/s after measuring the over-the-top transit of a
            # trained policy at 3.5 to 5.5 rad/s: a 10 cm robot has a fast natural tumble timescale,
            # and the lower cap was forfeiting most of the physically necessary rotation. Style
            # pressure lives in the impact and action-rate penalties, not in fighting gravity.
            "max_paid_rate": 5.0,
            "support_sensor_cfg": _ROBOT_GROUND_SENSOR_CFG,
            "head_sensor_cfg": _HEAD_GROUND_SENSOR_CFG,
            "asset_cfg": _HEAD_SHELL_CFG,
        },
    )
    # Above the measured p90 transit speed of about 5.5 rad/s, so it taxes genuine whips and not the
    # natural tumble.
    roulade_overspeed = RewTerm(func=mdp.roulade_overspeed_penalty, weight=-0.1, params={"omega_max": 7.0})
    roulade_head_pivot = RewTerm(
        func=mdp.roulade_head_pivot,
        weight=0.5,
        params={
            "sensor_cfg": _HEAD_GROUND_SENSOR_CFG,
            "angle_lo": math.radians(30.0),
            "angle_hi": math.radians(240.0),
            "rate_norm": 2.0,
            "asset_cfg": _HEAD_SHELL_CFG,
        },
    )

    ##
    # The landing annuity, all gated on completing the roll over the head.
    ##

    # Broad widths on purpose: the stand-up composite's lesson is that a partial landing has to
    # score visibly, or the product of three Gaussians is numerically zero everywhere.
    roulade_landing_composite = RewTerm(
        func=mdp.roulade_landing_composite,
        weight=4.0,
        params={
            "target_height": MICRODUCK_STAND_HEIGHT,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "gate_lo": MICRODUCK_LANDING_GATE[0],
            "gate_hi": MICRODUCK_LANDING_GATE[1],
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )
    roulade_upright_after_roll = RewTerm(
        func=mdp.roulade_upright_after_roll,
        weight=1.5,
        params={"gate_lo": MICRODUCK_LANDING_GATE[0], "gate_hi": MICRODUCK_LANDING_GATE[1]},
    )
    roulade_height_after_roll = RewTerm(
        func=mdp.roulade_height_after_roll,
        weight=1.0,
        params={
            "target_height": MICRODUCK_STAND_HEIGHT,
            "std": 0.04,
            "gate_lo": MICRODUCK_LANDING_GATE[0],
            "gate_hi": MICRODUCK_LANDING_GATE[1],
        },
    )
    # The last mile. Upstream's run-3 policies all parked a centimetre low and 27 degrees off
    # vertical, where the broad composite already scores about 0.5; this layer scores about 0.1
    # there and about 1.0 upright.
    roulade_landing_sharp = RewTerm(
        func=mdp.roulade_landing_sharp,
        weight=2.0,
        params={
            "target_height": MICRODUCK_STAND_HEIGHT,
            "height_std": 0.015,
            "upright_std": 0.3,
            "gate_lo": MICRODUCK_LANDING_GATE[0],
            "gate_hi": MICRODUCK_LANDING_GATE[1],
        },
    )
    # Self-negating, so the weight is positive. It is what makes lying in a heap after the roll cost
    # something rather than merely earn nothing.
    roulade_stand_tax = RewTerm(
        func=mdp.roulade_stand_tax,
        weight=5.0,
        params={
            "target_height": MICRODUCK_STAND_HEIGHT,
            "gate_lo": MICRODUCK_LANDING_GATE[0],
            "gate_hi": MICRODUCK_LANDING_GATE[1],
        },
    )
    roulade_rise_velocity = RewTerm(
        func=mdp.roulade_rise_velocity,
        weight=0.75,
        params={
            "max_height": MICRODUCK_STAND_HEIGHT + 0.01,
            "gate_lo": MICRODUCK_RISE_GATE[0],
            "gate_hi": MICRODUCK_RISE_GATE[1],
        },
    )

    ##
    # Straightness: keep the roll in the sagittal plane.
    ##

    roulade_sagittal = RewTerm(func=mdp.roulade_sagittal_penalty, weight=-0.1)
    roulade_lateral_vel = RewTerm(func=mdp.roulade_lateral_velocity_penalty, weight=-0.5)
    roulade_flatness = RewTerm(func=mdp.roulade_flatness_penalty, weight=-0.5)

    ##
    # Sim-to-real regularizers, matched to the velocity task except where the roll forbids it.
    ##

    # Ramped to -0.4 by iteration 3000. Upstream softened the ceiling from -0.6 after its landing
    # metrics peaked and then declined in step with the tightening.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # Weight 0.0, ramped to -1e-3 from iteration 3500.
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=0.0, params={"asset_cfg": _SERVO_JOINT_CFG})
    # 25 times lighter than the stand-up task's, and that gap is deliberate: the roll *is* trunk
    # angular velocity, so this regularizer has to stay near zero here or it prices the manoeuvre
    # itself (addendum section 7.23).
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.002, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.001)
    # Inherited silently by upstream: it is simply not in this task's deletion list (addendum
    # section 7.13), so it is stated here rather than left to be discovered.
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Weight 0.0, ramped to -0.05 from iteration 3500. The height-and-tilt gate is what makes it
    # safe on a task built out of rotation: it is shut for the whole roll and only damps the
    # overshoot once the robot is back up.
    arrival_damping = RewTerm(
        func=mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low": 0.09,
            "height_high": 0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg": _TRUNK_BODY_CFG,
        },
    )
    # Self-negating, so the weight is positive; ramped up to 0.005 from iteration 2500. Unlike every
    # other polish term this one is active from step 0, because discovery is easy on this task and
    # style is the scarce resource: upstream's first run found a violent solution under zero impact
    # cost and locked it in.
    gentle_landing = RewTerm(func=mdp.trunk_vertical_accel_penalty, weight=0.002)
    # Ten times lighter than the stand-up task's -1.0, and for the opposite reason to the usual: a
    # tucked roll *needs* body-on-body contact, so upstream prices it rather than forbidding it.
    # ``saturate`` keeps the many-to-many sensor on upstream's 0/1 scale, so the weight is the
    # penalty for touching yourself at all rather than a per-collider tariff. It matters more here
    # than on the stand-up task: a roulade is a deep tuck, so the robot spends the roll near the
    # poses that fire this term.
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG, "saturate": True},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (addendum section 4.6).

    **There is no failure termination at all.** The velocity task's tilt check is deleted here for a
    stronger reason than on the stand-up task: falling over *is* the task. Every episode therefore
    runs its full 250 steps unless the state stops being finite, and upstream's inherited
    terrain-bounds termination -- all-false on a ground plane, on a flat-only task -- is not carried
    over (addendum section 7.24).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Catches a *broken* robot rather than a fallen one.
    #
    # Deviation from upstream, deliberately: upstream leaves this term's sensor list empty here and
    # names its foot contact sensor only on the stand-up task, which the extraction reads as drift
    # rather than design and recommends closing everywhere in the port (addendum section 7.9). This
    # is the task that deliberately slams a head into the floor at 3.5 to 5.5 rad/s, so it is the
    # last place to leave a degenerate contact force unguarded. The guard only changes behaviour in
    # states that are already broken, and the matching half of it is the pair of NaN-safe critic
    # terms in :class:`ObservationsCfg`.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (addendum section 4.8).

    Two groups:

    * **The reverse curriculum**, :attr:`roulade_spawn_mix`, which starts half the episodes part-way
      through the roll and walks that down to a fifth as the flip gets discovered from a standing
      start. Mid-roll never goes to zero -- it keeps the landing practised, and it is realistic
      domain randomization anyway. Upstream pushed both boundaries out by a factor of two after a
      run that shifted away from mid-roll before standing-spawn rolls were mastered.
    * **Smoothness polish**, from iteration 2500. The motion penalties are introduced only once the
      roll exists, which is the stand-up task's timing lesson: an attempt tax during discovery stops
      the manoeuvre being found at all, and the fix is timing rather than magnitude.

    Upstream's schedules are written in PPO iterations; :func:`_iterations` converts them to the
    global environment-step count these terms compare against.
    """

    roulade_spawn_mix = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "set_roulade_state",
            "param_stages": [
                {"step": _iterations(0), "params": {"standing_prob": 0.50, "midroll_prob": 0.50}},
                {"step": _iterations(3000), "params": {"standing_prob": 0.65, "midroll_prob": 0.35}},
                {"step": _iterations(6000), "params": {"standing_prob": 0.80, "midroll_prob": 0.20}},
            ],
        },
    )

    ##
    # Domain randomization ramps, matched to the stand-up and velocity tasks.
    ##

    com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
                {"step": _iterations(1500), "range": 0.015},
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
    # Smoothness polish.
    ##

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -0.1},
                {"step": _iterations(1500), "weight": -0.2},
                {"step": _iterations(3000), "weight": -0.4},
            ],
        },
    )

    arrival_damping_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "arrival_damping",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(2500), "weight": -0.025},
                {"step": _iterations(3500), "weight": -0.05},
            ],
        },
    )

    torque_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(2500), "weight": -5e-4},
                {"step": _iterations(3500), "weight": -1e-3},
            ],
        },
    )

    # Positive weights: the term is self-negating.
    gentle_landing_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "gentle_landing",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.002},
                {"step": _iterations(2500), "weight": 0.005},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckRouladeFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck forward-roll environment on flat ground."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckRouladePhysicsCfg())
    # ``env_spacing`` is upstream's scene extent (reference section 1).
    scene: MicroDuckRouladeSceneCfg = MicroDuckRouladeSceneCfg(num_envs=4096, env_spacing=2.0)
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
        # at. Episodes are 5 s -- 250 control steps -- which upstream sized as a paced roll of about
        # 2 s, a rise of about 1.5 s and a moment to settle; at 4 s the rise had nowhere to happen.
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        # Run the BAM servos through the backend-native path, as the velocity and stand-up tasks do
        # and as upstream does. The decimation above is even, which is what lets the stateful servo
        # delay line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        for sensor in (
            self.scene.contact_forces,
            self.scene.self_collision,
            self.scene.head_ground_contact,
            self.scene.robot_ground_contact,
        ):
            if sensor is not None:
                sensor.update_period = self.sim.dt
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
