# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stand-up environment for the Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks2 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the shared mjlab base template live in the companion
``historical original source extraction`` and are cited as "reference section N".

The task is the other half of a sit/stand pair. An episode starts the robot on the ground -- face
down, face up, folded into the sitting keyframe a sit policy hands off, or already standing -- and
asks it to reach and hold the standing keyframe. There is no trajectory and no waypoint gating: one
fixed target is rewarded from the first step, and the policy discovers its own rise path.

Two structural differences from :mod:`..velocity` are worth stating up front, because everything
else follows from them:

* The robot is :data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_ALLCOLLISIONS_CFG`, not the walking model. A
robot
  that lies on its back and pushes itself up needs a trunk, hips, shins and head that touch the
  floor, and the walking model only has soles.
* The reward set is a **rise** recipe, not a locomotion one: every walking term is gone and the
  weights are upstream's after its 2026-07 division by four, which brought the task mass to about
  12 so the shared sim-to-real regularizers act at the same relative strength as in the
  well-transferring velocity recipe.

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
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

##
# Keyframe heights and poses (addendum section 3.1)
##

MICRODUCK_STAND_HEIGHT = 0.115
"""Trunk height [m] of the standing keyframe, the single target this task rewards.

Upstream measured it off a velocity policy holding the robot still at zero command. It is 2 mm
below the 0.11718 m the model reaches geometrically at the stand pose with the soles on the ground
(addendum section 2.2), which is the servos sagging under load.
"""

MICRODUCK_SIT_HEIGHT = 0.060
"""Trunk height [m] of the seated equilibrium, where the ``upright_sharp`` height gate opens.

This is upstream's *measured* rest height for :data:`MICRODUCK_SITTING_JOINT_POS`, the sitting
keyframe defined below -- the height the trunk settles at once the robot's weight is on its folded
legs. Resting that keyframe on the floor geometrically, with no dynamics, puts the trunk 2 cm higher
at 0.0813 m; upstream's gate was tuned against the settled value, so that is the one used here.
"""

MICRODUCK_SITTING_JOINT_POS = {
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.4079,
    "left_knee": 1.35,
    "left_ankle": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.4079,
    "right_knee": -1.35,
    "right_ankle": 0.0,
}
"""The sitting keyframe [rad], as joint deltas from nothing -- these are absolute joint positions.

Knees folded to about 77 degrees, hips leaning 0.05 rad forward of the stand pose, ankles flat. It
is the *end state of the sit policy*, not a pose designed here, so it is the hand-off this task must
start from; upstream keeps the two in sync by hand.

The neck and head are deliberately absent and therefore stay at the stand pose: the sit policy
leaves the head where it is, and the head-pose command steers it from there.

Upstream keys this by servo index (addendum section 3.4); the port keys it by name, because the
converted asset resolves joints in Newton's order rather than the MJCF's.
"""

##
# Body-pose command envelope (addendum section 3.1)
##

_BODY_CMD_MAX_Z_DOWN = 0.04
_BODY_CMD_MAX_Z_UP = 0.030
"""Final crouch and extension range [m] of the trunk-height command, asymmetric on purpose.

:data:`MICRODUCK_STAND_HEIGHT` is the natural equilibrium at the stand pose, so there is plenty of
room to crouch below it and only about a centimetre of leg extension above it.
"""

_BODY_CMD_MAX_ANGLE = math.radians(15.0)
"""Final trunk roll and pitch command range [rad]. Upstream capped it at 15 degrees after a
velocity body-control run at 20 degrees trained twitchy, overdriven tilting."""

_BODY_CMD_ALIVE_XY = 0.005
_BODY_CMD_ALIVE_ANGLE = 0.05
"""Permanent range of the three *untracked* command axes, ``x``, ``y`` and ``yaw``.

They are held at a tiny non-zero range forever rather than pinned at zero so that the input neurons
of the deployed 61-wide vector stay alive; the policy learns to ignore them because nothing rewards
them.
"""

_BODY_CMD_ZERO_PROB = 0.3
"""Probability that a body-pose resample yields the exact zero command.

Uniform sampling of six dimensions never produces it, so without this bucket the deployment idle
case -- stand at nominal, no request -- would be absent from training. This task is the first in the
family to use it.
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
# no ``preserve_order``: the material randomization resolves body IDs into backend shape ranges and
# documents that callers must not pre-swizzle them
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")

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
class MicroDuckStandUpPhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck stand-up environment.

    MicroDuck is trained on MuJoCo upstream, so MJWarp is the reference backend and the only one
    offered. Upstream inherits the mjlab base template's solver limits unchanged and never revisits
    them for this task (addendum section 7.4); this port measures them instead, because the
    all-collisions robot spends most of every episode lying on the floor and upstream's inherited
    ``nconmax`` of 35 was sized for a walking robot on two soles.

    MJWarp is also the only backend that can run this task as configured: the environment sets
    ``sim.use_newton_actuators = True`` and the solver-hosted BAM model is rejected on the PhysX
    family's host adapter. See
    :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.MicroDuckPhysicsCfg`.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Measured, not inherited. Profiling under random actions -- the regime where the
            # robots flail on the floor and grind every collider into it, with the pushes forced to
            # full magnitude -- peaks at **27 contacts and 82 constraints** per environment. The
            # peak is not structural the way the walking task's is (there the same three geometries
            # touch every step); it moves with the pose, so the profile was run at both 256 and 2048
            # environments and the two agree to one contact, which is what says the tail has been
            # sampled. Logs:
            # ``artifacts/microduck/profile_microduck_contacts_standup_{256,2048}envs.log``, from
            # ``artifacts/microduck/profile_microduck_contacts.py``.
            #
            # That is 2.7x the walking task's contact peak, which is what ten world colliders on the
            # floor instead of two buys. ``njmax`` is a hard per-environment cap and carries the
            # margin; ``nconmax`` is a per-environment share of one shared buffer and cannot
            # overflow at the measured peak, so it sits just above it.
            njmax=128,
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
class MicroDuckStandUpSceneCfg(InteractiveSceneCfg):
    """Scene with the all-collisions MicroDuck robot on a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
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

    # The all-collisions model. It is the walking robot with six more colliders -- a second trunk
    # shell, the two hip cheeks and the three head shells -- and with both shins promoted from
    # self-collision-only to world contact, so ten of its eleven colliders reach the ground
    # (``source/isaaclab_tasks/test/contrib/microduck/test_microduck_variant_assets.py``). A robot that starts every
    # episode lying down needs all of them.
    robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Feet and trunk, as the velocity task does: upstream tracks ground contact on the two soles
    # (addendum section 3.2) and drops the base template's terrain and foot-height scanners
    # wholesale. The ankle bodies carry exactly one collider each -- the sole -- on this model too.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    # Self-collision sensor. Upstream filters the trunk subtree against itself and reports whether
    # its single contact slot found anything, i.e. a 0/1 "is the robot touching itself" signal.
    #
    # This is that sensor: the model's ten enabled colliders against each other, many-to-many, on
    # the Newton backend's shape-level expressions. It sees every self-contact the robot can make,
    # including the ones between two limbs, where neither end is the trunk. Both families are
    # reachable inside the joint limits: a deep knee fold drives each shin into the hip shell on its
    # own side, and the sagittal chain -- hip pitch, knee and ankle all +/-1.571 rad -- swings the
    # feet across the midline, where the narrower hip yaw and roll travel is enough to overlap them
    # sole against sole.
    #
    # ``prim_path`` is ignored for the sensing objects once ``sensor_shape_prim_expr`` is set, but
    # the base sensor still requires one; the trunk is the cheapest expression that resolves.
    #
    # The reward saturates (``saturate=True`` below). Sensing both sides of a pair reports one
    # contact twice, so counting would put a leg fold at 2 where upstream's contact slot reports 1.
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
    """Command specifications for the MDP (addendum section 3.5)."""

    # Kept only so the deployed 61-wide observation keeps its three-wide twist slot. No reward reads
    # it, the ranges are a hundredth of the velocity task's, and it resamples at most once per
    # episode. ``rel_forward_envs = 0.2`` is inherited from upstream's base template and never
    # overridden there (addendum section 7.22): a fifth of the resamples get a commanded surge of
    # 0.3 m/s that nothing acts on, which is a quirk of the observation distribution rather than of
    # the behaviour, and is reproduced rather than quietly cleaned up.
    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 12.0),
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

    # Joint-position deltas from the stand pose for (neck_pitch, head_pitch, head_yaw, head_roll).
    # Identical to the velocity task's, including the ``head_pose_range`` curriculum that opens it:
    # a robot that has just stood up is still expected to look around on command.
    head_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )

    # A trunk-pose delta, ``(x, y, z)`` [m] then ``(roll, pitch, yaw)`` [rad]. Unlike the velocity
    # task, which keeps this slot alive at zero weight, the stand-up task *trains* it: the tracking
    # weight and the z/roll/pitch ranges both ramp in from iteration 2500, once the recovery skills
    # exist. The three untracked axes stay at their alive ranges forever.
    body_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        zero_command_prob=_BODY_CMD_ZERO_PROB,
        ranges=(
            (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
            (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
            (-0.005, 0.005),
            (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
            (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
            (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Identical to the velocity task's: the same 14 servos in the same deploy order, at the same unit
    scale, closing the same encoder-bias loop.
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

    The actor group is byte-for-byte the velocity task's 61-wide deploy contract, which is the whole
    point: one runtime on the robot feeds every MicroDuck policy from the same buffer, so a stand-up
    policy and a walking policy have to read the same vector.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract.

        See :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.ObservationsCfg` for
        what each corruption models; the terms, their order, their noise and their delays are
        upstream's velocity values, which the stand-up task copies deliberately for sim-to-real
        parity.
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
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function (addendum section 6.2).

        The actor's terms with every corruption removed, plus the base linear velocity and the three
        foot terms the robot has no sensor for. It is narrower than the velocity task's critic by
        the two ``foot_height`` columns, which upstream deletes here because the stand-up scene
        carries no height sensor and no foot-height reward to justify one.

        The two sensor-derived terms are the NaN-guarded variants. Upstream applies them on this
        task and not on its siblings, and says why in its own configuration: a robot that lands and
        flips constantly produces degenerate contacts far more often than a walking one, and a
        single non-finite value here reaches the learner without passing the state-based NaN
        termination.
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
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events (addendum section 3.7).

    The domain-randomization suite is deliberately the velocity task's, term for term and range for
    range, so a policy trained here meets the same hardware spread as one trained to walk. What this
    task adds is :attr:`set_ground_state`, which replaces the root reset's height and orientation
    with one of four ground keyframes.

    **Declaration order is behaviour.** Isaac Lab fires reset events in the order they are declared,
    and :attr:`set_ground_state` overwrites what :attr:`reset_base` and :attr:`reset_robot_joints`
    wrote. Only the root reset's horizontal spread survives, so a port that reorders these three
    changes the spawn distribution outright (addendum section 7.11).
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

    # Upstream also randomizes the base height and yaw here, and ``set_ground_state`` then
    # overwrites both (addendum section 7.11). Only the horizontal spread is live, so only the
    # horizontal spread is configured.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )

    # Zero-width on purpose: upstream resets every joint exactly to the stand pose, and the sitting
    # bucket of ``set_ground_state`` is what folds the legs afterwards.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # The episode distribution. The four probabilities here are the ``ground_state_mix``
    # curriculum's first stage, which ramps them toward the prone poses as the run progresses; the
    # height bands and the sitting keyframe are left alone by that curriculum.
    set_ground_state = EventTerm(
        func=mdp.reset_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.20,
            "face_up_prob": 0.00,
            "sitting_prob": 0.40,
            "standing_prob": 0.40,
            # the trunk rests at about 0.044 m face down, so spawning it in this band drops it a
            # centimetre rather than the 15 cm the base template's prone default would
            "prone_z_range": (0.05, 0.09),
            "sitting_z_range": (0.05, 0.09),
            "standing_z_range": (0.11, 0.12),
            "sitting_joint_pos": MICRODUCK_SITTING_JOINT_POS,
            # about 7 degrees per joint. Without it the policy overfits the exact keyframe and does
            # not survive the hand-off from a real sit policy.
            "sitting_joint_noise_std": 0.12,
            "sitting_tilt_max": math.radians(10.0),
            # a reverse curriculum for back recovery: flat on the back has no reward gradient until
            # the roll completes, so some episodes start part-way along it
            "face_up_roll_max": math.radians(90.0),
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

    ##
    # Interval.
    ##

    # Ramped from nothing by the ``push_magnitude`` curriculum, unlike the velocity task which
    # pushes at full strength from the first step. A robot that starts seated on the floor cannot
    # be shoved around while it is still learning to rise.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (addendum section 3.3).

    One fixed goal -- the stand pose at :data:`MICRODUCK_STAND_HEIGHT` -- rewarded from the first
    step of every episode, plus the shared sim-to-real regularizers. Three shapes recur and explain
    the apparent duplication:

    * **Two-layer attractors.** ``height_stand`` / ``height_stand_sharp`` and ``upright_linear`` /
      ``upright_sharp`` are the same quantity at a wide and a narrow width. The wide layer is
      already saturated by the time the robot is close, so on its own it leaves no gradient for the
      last centimetre or the last few degrees; the narrow layer supplies exactly that and nothing
      else.
    * **L1 bootstraps.** ``pose_stand_l1`` and ``height_stand_l1`` carry a constant gradient where
      the Gaussians are flat, which is what makes sitting still a net cost rather than a comfortable
      local optimum.
    * **Late-phase penalties.** ``arrival_damping``, ``joint_torque_rate_l2``, ``head_pose_bias``
      and ``body_pose_tracking`` all start at weight zero and are introduced at iteration 2500-3000
      by curricula. Upstream established across two failed runs that any motion tax active during
      the recovery-discovery phase stops the flips being discovered at all -- the fix is timing, not
      magnitude, so the weights below are the *initial* ones and the schedules are load-bearing.

    Sign convention: ``pose_stand_l1``, ``height_stand_l1``, ``gentle_rise`` and ``head_pose_bias``
    negate themselves and therefore take **positive** weights.

    Every weight is upstream's after its 2026-07 division of the whole task stack by four, which
    brought the task mass to about 12 to match the velocity recipe's; upstream's own per-term
    comments quote the pre-division numbers.
    """

    ##
    # Posture: hold the legs at the stand pose.
    ##

    pose_stand_legs = RewTerm(
        func=mdp.joint_pose_gaussian,
        weight=2.0,
        params={"std": 0.5, "asset_cfg": _LEG_JOINT_CFG},
    )
    pose_stand_l1 = RewTerm(func=mdp.joint_pose_l1, weight=1.25, params={"asset_cfg": _LEG_JOINT_CFG})

    ##
    # Head: steered by command, not pinned to the stand pose.
    ##

    head_pose_tracking = RewTerm(
        func=mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
    )
    # Weight 0.0, ramped to +1.5 from iteration 3000. The upright gate is mandatory here and is what
    # makes the term safe on a recovery task: without it the sustained head-down error of a prone
    # robot would be charged in full, taxing the head-pivot the flip needs.
    head_pose_bias = RewTerm(
        func=mdp.head_pose_bias_penalty,
        weight=0.0,
        params={
            "command_name": "head_pose",
            "tau_s": 1.0,
            "asset_cfg": _HEAD_JOINT_CFG,
            "gate_height_low": 0.09,
            "gate_height_high": 0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": 45.0,
        },
    )

    ##
    # Height: reach and hold the standing trunk height.
    ##

    height_stand = RewTerm(
        func=mdp.root_height_gaussian,
        weight=1.0,
        params={"std": 0.04, "target_height": MICRODUCK_STAND_HEIGHT},
    )
    # Ramped down to 0.2 from iteration 3000, when the body-pose command takes over the job of
    # pinning the nominal stand and this layer would otherwise out-bid a commanded crouch.
    height_stand_sharp = RewTerm(
        func=mdp.root_height_gaussian,
        weight=1.0,
        params={"std": 0.015, "target_height": MICRODUCK_STAND_HEIGHT},
    )
    height_stand_l1 = RewTerm(
        func=mdp.root_height_l1,
        weight=7.5,
        params={"target_height": MICRODUCK_STAND_HEIGHT},
    )

    ##
    # Rising: pay for the motion, then damp it on arrival.
    ##

    com_upward_velocity = RewTerm(func=mdp.com_upward_velocity, weight=0.75, params={"max_height": 0.125})
    gentle_rise = RewTerm(func=mdp.trunk_vertical_accel_penalty, weight=0.005)
    # Weight 0.0, ramped to -0.05 from iteration 3000.
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

    ##
    # Orientation and the composite goal state.
    ##

    upright_linear = RewTerm(func=mdp.body_upright_linear, weight=1.5)
    # Ramped down to 0.5 from iteration 3000, with the other two sharp attractors.
    upright_sharp = RewTerm(
        func=mdp.upright_gaussian_at_height,
        weight=1.5,
        params={"std": 0.3, "height_low": MICRODUCK_SIT_HEIGHT, "height_high": MICRODUCK_STAND_HEIGHT},
    )
    # Ramped down to 1.5 from iteration 3000.
    standing_composite = RewTerm(
        func=mdp.standing_composite_score,
        weight=3.75,
        params={
            "target_height": MICRODUCK_STAND_HEIGHT,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )

    ##
    # Commanded trunk pose. Weight 0.0, ramped to 4.0 from iteration 2500.
    ##

    # Only z, roll and pitch are weighted -- the three axes the deployed runtime exposes. The other
    # three ride along at weight zero so the command keeps its six-wide observation slot. Upstream
    # selects its feet-relative kernel here; with x, y and yaw at zero weight the two kernels are
    # identical, and the shared one is used. See
    # :func:`~isaaclab_tasks.contrib.microduck.pollen_mdp.rewards.body_pose_tracking_6d`.
    #
    # The widths are tight on purpose: at 1 cm of height error a ``z_std`` of 0.01 drops the axis
    # reward to 0.37, which is a real gradient, where 0.02 would leave it at 0.78.
    body_pose_tracking = RewTerm(
        func=mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": MICRODUCK_STAND_HEIGHT,
            "xy_std": 0.02,
            "z_std": 0.01,
            "angle_std": math.radians(5.0),
            "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
        },
    )

    ##
    # Sim-to-real regularizers, matched to the velocity task.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    # Inherited silently by upstream: it is simply not in the stand-up task's deletion list
    # (addendum section 7.13), so it is stated here rather than left to be discovered.
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Ramped to -1.0 by iteration 1500.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # Weight 0.0, ramped to -1e-3 from iteration 3000. Penalizes torque *change*, not magnitude.
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=0.0, params={"asset_cfg": _SERVO_JOINT_CFG})
    # ``saturate`` keeps the many-to-many sensor on upstream's 0/1 scale, so the weight is the
    # penalty for touching yourself at all rather than a per-collider tariff.
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG, "saturate": True},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (addendum section 3.6).

    The tilt termination the velocity task ends a fall with is **absent**: the robot starts on the
    ground here, so terminating on tilt would end most episodes on their first step.

    Upstream also inherits a terrain-bounds termination, which returns all-false on a ground plane
    and is therefore dead on this flat-only task (addendum section 7.24); it is not carried over.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Catches a *broken* robot rather than a fallen one. Upstream additionally names its foot
    # contact sensor here -- it is the one task in its family that does, and it says why: a robot
    # that lands and flips constantly produces degenerate contacts, and a non-finite contact force
    # would otherwise reach the learner unchecked.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (addendum section 3.8).

    Three groups, and the boundaries between them are the design:

    * **Recovery discovery**, iterations 0-2500. ``ground_state_mix`` walks the reset distribution
      from mostly standing and sitting to mostly prone, introducing face-up last because it is the
      hardest recovery. Nothing taxes motion in this window.
    * **Body control**, from iteration 2500. The trunk-pose tracking weight and its command ranges
      ramp in together, and the three sharp fixed-stand attractors are relaxed to make room --
      otherwise they out-bid every commanded deviation.
    * **Smoothness polish**, from iteration 3000. The motion penalties are introduced only once the
      flips already exist and are being exercised by the prone resets.

    Upstream's schedules are written in PPO iterations; :func:`_iterations` converts them to the
    global environment-step count these terms compare against.
    """

    ##
    # Recovery discovery.
    ##

    ground_state_mix = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {
                    "step": _iterations(0),
                    "params": {
                        "standing_prob": 0.40,
                        "sitting_prob": 0.40,
                        "face_down_prob": 0.20,
                        "face_up_prob": 0.00,
                    },
                },
                {
                    "step": _iterations(600),
                    "params": {
                        "standing_prob": 0.25,
                        "sitting_prob": 0.30,
                        "face_down_prob": 0.35,
                        "face_up_prob": 0.10,
                    },
                },
                {
                    "step": _iterations(1500),
                    "params": {
                        "standing_prob": 0.20,
                        "sitting_prob": 0.25,
                        "face_down_prob": 0.30,
                        "face_up_prob": 0.25,
                    },
                },
                {
                    "step": _iterations(2500),
                    "params": {
                        "standing_prob": 0.15,
                        "sitting_prob": 0.20,
                        "face_down_prob": 0.30,
                        "face_up_prob": 0.35,
                    },
                },
            ],
        },
    )

    ##
    # Domain randomization ramps, matched to the velocity task.
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

    # Upstream drives this one with an exclusive step comparison where ``ground_state_mix`` uses an
    # inclusive one (addendum section 7.6). The inconsistency is reproduced rather than smoothed
    # over, so a stage table transcribed from upstream schedules identically here.
    push_magnitude = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "push_robot",
            "inclusive": False,
            "param_stages": [
                {"step": _iterations(0), "params": {"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}},
                {"step": _iterations(500), "params": {"velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}}},
                {"step": _iterations(1000), "params": {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}},
            ],
        },
    )

    head_pose_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": _iterations(0), "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": _iterations(500), "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))},
                {"step": _iterations(1000), "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))},
                {"step": _iterations(1500), "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))},
                {"step": _iterations(2000), "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -0.1},
                {"step": _iterations(500), "weight": -0.2},
                {"step": _iterations(750), "weight": -0.4},
                {"step": _iterations(1000), "weight": -0.6},
                {"step": _iterations(1250), "weight": -0.8},
                {"step": _iterations(1500), "weight": -1.0},
            ],
        },
    )

    ##
    # Body control, from iteration 2500.
    ##

    body_pose_tracking_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "body_pose_tracking",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(2500), "weight": 1.5},
                {"step": _iterations(3000), "weight": 3.0},
                {"step": _iterations(4000), "weight": 4.0},
            ],
        },
    )

    body_pose_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {
                    "step": _iterations(0),
                    "ranges": (
                        (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
                        (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
                        (-0.005, 0.005),
                        (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
                        (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
                        (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
                    ),
                },
                {
                    "step": _iterations(2500),
                    "ranges": (
                        (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
                        (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
                        (-0.010, 0.005),
                        (-math.radians(8.0), math.radians(8.0)),
                        (-math.radians(8.0), math.radians(8.0)),
                        (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
                    ),
                },
                {
                    "step": _iterations(3000),
                    "ranges": (
                        (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
                        (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
                        (-0.018, 0.008),
                        (-math.radians(12.0), math.radians(12.0)),
                        (-math.radians(12.0), math.radians(12.0)),
                        (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
                    ),
                },
                {
                    "step": _iterations(4000),
                    "ranges": (
                        (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
                        (-_BODY_CMD_ALIVE_XY, _BODY_CMD_ALIVE_XY),
                        (-_BODY_CMD_MAX_Z_DOWN, _BODY_CMD_MAX_Z_UP),
                        (-_BODY_CMD_MAX_ANGLE, _BODY_CMD_MAX_ANGLE),
                        (-_BODY_CMD_MAX_ANGLE, _BODY_CMD_MAX_ANGLE),
                        (-_BODY_CMD_ALIVE_ANGLE, _BODY_CMD_ALIVE_ANGLE),
                    ),
                },
            ],
        },
    )

    # The three relax schedules below make room for the commanded trunk pose. At a 2 cm crouch and
    # 15 degrees of tilt the sharp fixed-stand layers oppose it by about 3.5 per step at their
    # initial weights, which no achievable tracking weight can out-bid; their bootstrap job is done
    # by iteration 3000, and the tracking term's own tighter widths take over the role of pinning
    # the nominal stand.
    height_stand_sharp_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "height_stand_sharp",
            "weight_stages": [
                {"step": _iterations(0), "weight": 1.0},
                {"step": _iterations(3000), "weight": 0.5},
                {"step": _iterations(4000), "weight": 0.2},
            ],
        },
    )

    upright_sharp_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "upright_sharp",
            "weight_stages": [
                {"step": _iterations(0), "weight": 1.5},
                {"step": _iterations(3000), "weight": 1.0},
                {"step": _iterations(4000), "weight": 0.5},
            ],
        },
    )

    standing_composite_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "standing_composite",
            "weight_stages": [
                {"step": _iterations(0), "weight": 3.75},
                {"step": _iterations(3000), "weight": 2.5},
                {"step": _iterations(4000), "weight": 1.5},
            ],
        },
    )

    ##
    # Smoothness polish, from iteration 3000.
    ##

    arrival_damping_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "arrival_damping",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(3000), "weight": -0.025},
                {"step": _iterations(4000), "weight": -0.05},
            ],
        },
    )

    head_pose_bias_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(3000), "weight": 0.5},
                {"step": _iterations(4000), "weight": 1.5},
            ],
        },
    )

    torque_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(3000), "weight": -1e-3},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckStandUpFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck stand-up environment on flat ground."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckStandUpPhysicsCfg())
    # ``env_spacing`` is upstream's scene extent (reference section 1).
    scene: MicroDuckStandUpSceneCfg = MicroDuckStandUpSceneCfg(num_envs=4096, env_spacing=2.0)
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
        # at. Episodes are 6 s -- 300 control steps -- which is a gentle rise plus a moment to
        # stabilize, a third of the velocity task's window.
        self.decimation = 4
        self.episode_length_s = 6.0
        self.sim.dt = 0.005
        # Run the BAM servos through the backend-native path, as the velocity task does and as
        # upstream does -- see :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.
        # MicroDuckVelocityRoughEnvCfg`. The decimation above is even, which is what lets the
        # stateful servo delay line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.self_collision is not None:
            self.scene.self_collision.update_period = self.sim.dt
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
