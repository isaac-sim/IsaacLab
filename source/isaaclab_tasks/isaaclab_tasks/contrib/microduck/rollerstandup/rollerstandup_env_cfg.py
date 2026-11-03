# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stand-up environment for the roller-skating Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks3 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the roller recipe live in the companion
``historical tasks2 source extraction`` and are cited as "addendum section N".

This is the stand-up recipe on skates: an episode starts the robot face down, face up or already
standing, and asks it to reach and hold the standing station within six seconds. There is no
trajectory and no waypoint gating -- one fixed target is rewarded from the first step, and the policy
discovers its own rise path, exactly as on
:mod:`~isaaclab_tasks.contrib.microduck.standup.standup_env_cfg`.

What the wheels change is the *bootstrap*. A rolling contact has no longitudinal grip, so a robot on
skates has nothing to push against, and upstream's answer is a wheel-friction curriculum that runs
**backwards**: it starts with the bearings almost locked -- which makes them behave like feet -- and
relaxes them over four thousand iterations toward the physics of a real bearing. That has a
deployment consequence, stated in :attr:`CurriculumCfg.wheel_friction` and worth repeating here:
**only checkpoints from after the last stage are candidates for the robot.**

.. warning::

    Upstream's version of this environment **cannot run at the pinned commit**. Three of its reward
    terms index a fourteen-wide servo view with indices drawn from the eighteen-joint model layout,
    which raises ``IndexError`` on the first reward evaluation (section 13.2, upstream issue draft
    016). The port therefore diverges to function: every joint selection here is resolved **by name**,
    which is what the rest of this package does anyway and what makes the interleaved wheel hinges a
    non-issue. Because upstream cannot be run, this task has **no accuracy gate against it**; what
    stands in its place is an internal acceptance test that evaluates the task's own reward from a
    fallen-on-skates start under physics (``test_microduck_rollerstandup_env.py``).

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import (
    MICRODUCK_WHEEL_JOINT_NAMES,
    MicroDuckVelocityRollersFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import CurriculumCfg as RollersCurriculumCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)

##
# Keyframe heights (section 9.1)
##

MICRODUCK_ROLLER_STAND_HEIGHT = 0.138
"""Trunk height [m] of the standing keyframe on skates, the single target this task rewards.

It is :data:`~isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg.MICRODUCK_ROLLERS_STANDING_HEIGHT`
less about 2.7 mm of servo sag under load, which is the same allowance the wheel-less stand-up task
makes between its geometric 0.11718 m and its measured 0.115 m. Upstream states the derivation and
this port re-measured both halves of it: the geometric heights reproduce exactly (section 9.1).

The height also sits inside the roller task's reset band ``(0.1335, 0.1435)``, which is why the
"already standing" spawn bucket below can reuse it unchanged.
"""

MICRODUCK_ROLLER_PRONE_HEIGHT = 0.075
"""Trunk height [m] at rest face down on the skates, measured kinematically (section 9.1).

Quoted because :attr:`RewardsCfg.upright_sharp`'s height gate opens from it. Face *up* the robot
rests 28 mm lower still, at 0.0475 m, and the two prone poses nonetheless share one spawn band --
see :attr:`EventsCfg.set_ground_state`.
"""

MICRODUCK_ROLLER_RISE_CEILING = 0.148
"""Trunk height [m] above which the upward-velocity reward is switched off.

A centimetre above the standing station, so the term pays through the whole rise and stops paying
once the robot would be jumping rather than standing.
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
"""The ten leg joints the three pose-scoring terms read, resolved by name.

This selection is the port's divergence from upstream, and it is a divergence to *function*: upstream
hard-codes ``[0, 1, 2, 3, 4, 11, 12, 13, 14, 15]``, which are the legs' positions in the roller
model's eighteen-joint layout, and then feeds them to helpers that have already collapsed to the
fourteen-wide servo view -- so the last two indices are out of bounds and the environment raises on
its first reward evaluation (section 13.2). Naming the joints removes the class of bug rather than
the instance: there is no layout for the indices to be drawn from the wrong one of.
"""
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_WHEEL_JOINT_NAMES, preserve_order=True)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")


def _iterations(count: int) -> int:
    """Convert an upstream PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP (section 9.3).

    One command, kept only so the deployed 61-wide observation keeps its three-wide twist slot. No
    reward reads it and its ranges are a hundredth of the skating task's, which is the same
    neutralization the wheel-less stand-up and ball-kick tasks apply. It also **downgrades** the
    roller task's relative-heading command back to the plain velocity one: a live heading error in an
    observation slot nothing tracks would be a distractor rather than a shape placeholder.
    """

    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 12.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        # inherited from upstream's base template and never overridden there; inert here, because
        # nothing reads the surge slot
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
class RewardsCfg:
    """Reward terms for the MDP (sections 9.2 and 9.5).

    Nineteen terms: eight regularizers carried over from the skating recipe, and the eleven-term
    rise stack that replaces everything else.

    **What the skating recipe loses, and why.** ``feet_flat`` goes because the blades are *not* flat
    during a rise and the term would fight the gesture; ``hip_roll_neutral`` because standing up
    needs the legs spread; ``pose`` and ``com_height_target`` because the rise stack supplies its own
    pose and height targets; ``upright`` because :attr:`upright_linear` and :attr:`upright_sharp`
    replace it; and the whole stroke -- wheel speed, braking, air time, glide, single support, gait
    symmetry, forward lean and heading hold -- because there is no stroke.

    **Three shapes recur in what replaces it**, and they are the wheel-less stand-up task's:

    * **Two-layer attractors.** :attr:`height_stand` / :attr:`height_stand_sharp` and
      :attr:`upright_linear` / :attr:`upright_sharp` are the same quantity at a wide and a narrow
      width. The wide layer is saturated by the time the robot is close and leaves no gradient for
      the last centimetre; the narrow layer supplies exactly that.
    * **L1 bootstraps.** :attr:`pose_stand_l1` and :attr:`height_stand_l1` carry a constant gradient
      where the Gaussians are flat, which is what makes lying still a net cost rather than a
      comfortable local optimum.
    * **A composite goal.** :attr:`standing_composite` is the product of the height, upright and pose
      scores, so it pays only when all three are right at once -- and at weight 15.0 it is the
      largest single term in the task.

    Sign convention: :attr:`pose_stand_l1`, :attr:`height_stand_l1` and :attr:`gentle_rise` negate
    themselves and therefore take **positive** weights. :attr:`gentle_rise` is the one worth
    checking twice -- upstream inherited it at ``-0.02`` from the wheel-less task, measured it
    *rewarding* vertical acceleration, and fixed the sign here; its own comment calls it the cause of
    a "very violent" rise and explains the failed damping attempts it had produced (section 9.6).
    """

    ##
    # Regularizers carried over from the skating recipe (section 9.2).
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    # Upstream declares -0.6 here and -0.4 at stage 0 of the curriculum that owns this weight, and the
    # curriculum manager runs before the first reward evaluation, so the declared literal is dead
    # (section 13.12). The live value is the one stated. The ramp itself is the wheel-less stand-up
    # task's rather than the skating task's, deliberately: the skating ramp reaches -2.0 for a calm
    # gait, which is a movement blocker on a task whose whole point is a fast rise from the back.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.4)
    self_collisions = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG}
    )
    neck_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2,
        weight=-0.5,
        params={"action_name": "joint_pos", "asset_cfg": _HEAD_JOINT_CFG},
    )
    neck_joint_pos_l2 = RewTerm(func=mdp.joint_pose_l2, weight=-0.5, params={"asset_cfg": _HEAD_JOINT_CFG})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3, params={"asset_cfg": _SERVO_JOINT_CFG})
    action_over_limit = RewTerm(
        func=mdp.action_over_limit_penalty,
        weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )

    ##
    # Posture: hold the legs at the stand pose.
    ##

    pose_stand_legs = RewTerm(
        func=mdp.joint_pose_gaussian, weight=8.0, params={"std": 0.5, "asset_cfg": _LEG_JOINT_CFG}
    )
    # Negates itself, hence the positive weight.
    pose_stand_l1 = RewTerm(func=mdp.joint_pose_l1, weight=5.0, params={"asset_cfg": _LEG_JOINT_CFG})

    ##
    # Height: reach and hold the standing trunk height.
    ##

    height_stand = RewTerm(
        func=mdp.root_height_gaussian,
        weight=4.0,
        params={"std": 0.04, "target_height": MICRODUCK_ROLLER_STAND_HEIGHT},
    )
    height_stand_sharp = RewTerm(
        func=mdp.root_height_gaussian,
        weight=4.0,
        params={"std": 0.015, "target_height": MICRODUCK_ROLLER_STAND_HEIGHT},
    )
    # Negates itself, hence the positive weight -- and at 30.0 it is the strongest gradient in the
    # task while the robot is still on the floor.
    height_stand_l1 = RewTerm(
        func=mdp.root_height_l1, weight=30.0, params={"target_height": MICRODUCK_ROLLER_STAND_HEIGHT}
    )

    ##
    # Rising: pay for the motion, and damp its violence.
    ##

    # Ungated, unlike the walking-and-recovery task's: it pays for any upward trunk velocity below
    # the ceiling, at any tilt, which is what a robot pivoting on its head and shoulders needs.
    com_upward_velocity = RewTerm(
        func=mdp.com_upward_velocity, weight=3.0, params={"max_height": MICRODUCK_ROLLER_RISE_CEILING}
    )
    # POSITIVE WEIGHT, AND NOT A TYPO. The kernel already returns ``-|a_z|``, so the -0.02 inherited
    # from the wheel-less task was a double negative that *rewarded* vertical acceleration -- upstream
    # measured it logging as the only penalty term with a positive episode return, identified it as
    # the cause of the violent rise, and flipped the sign (section 9.6).
    gentle_rise = RewTerm(func=mdp.trunk_vertical_accel_penalty, weight=0.02, params={"asset_cfg": _TRUNK_BODY_CFG})

    ##
    # Orientation and the composite goal state.
    ##

    upright_linear = RewTerm(func=mdp.body_upright_linear, weight=6.0, params={"asset_cfg": _TRUNK_BODY_CFG})
    upright_sharp = RewTerm(
        func=mdp.upright_gaussian_at_height,
        weight=6.0,
        params={
            "std": 0.3,
            "height_low": MICRODUCK_ROLLER_PRONE_HEIGHT,
            "height_high": MICRODUCK_ROLLER_STAND_HEIGHT,
            "asset_cfg": _TRUNK_BODY_CFG,
        },
    )
    # The product of the height, upright and pose scores, so it pays only when all three are right at
    # once. The largest single weight in the task.
    standing_composite = RewTerm(
        func=mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height": MICRODUCK_ROLLER_STAND_HEIGHT,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )

    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=-0.2, params={"asset_cfg": _SERVO_JOINT_CFG})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (section 9.7).

    The skating task's tilt termination is **absent**: the robot starts on the ground here, so
    terminating on tilt would end most episodes on their first step.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Catches a *broken* robot rather than a fallen one, with the foot sensor named -- the family's
    # NaN-guard norm, carried over from the skating task. Upstream reaches for a different mechanism
    # here, an observation-level ``nan_policy = "sanitize"`` that zeroes the offending columns and
    # keeps running (section 9.4); this port keeps the termination, because an environment quietly
    # training on zeroed observations is harder to notice than a spike in resets.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class EventsCfg:
    """Configuration for events (section 9.7).

    The skating task's randomization suite plus :attr:`set_ground_state`, which replaces the root
    reset's height and orientation with one of three keyframes and *is* this task's episode
    distribution.

    **Declaration order is behaviour.** Isaac Lab fires reset events in the order they are declared,
    and :attr:`set_ground_state` overwrites what :attr:`reset_base` and :attr:`reset_robot_joints`
    wrote, so only the root reset's horizontal spread survives. The suite is spelled out here rather
    than inherited for exactly that reason: an appended term would land after the randomizations
    instead of between them and the root reset, which is a different -- if here equivalent -- chain.
    """

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

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

    # Upstream also randomizes the base height and yaw here, and ``set_ground_state`` then overwrites
    # both. Only the horizontal spread is live, so only the horizontal spread is configured.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )

    # Zero-width on purpose: upstream resets every joint exactly to the stand pose, wheels included.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # The episode distribution. These four probabilities are :attr:`CurriculumCfg.ground_state_mix`'s
    # first stage; the height bands are left alone by that curriculum.
    #
    # The "already standing" bucket never disappears from the schedule, and upstream says why: without
    # it the policy learns to rise but not to *hold*, and falls over again immediately afterwards.
    set_ground_state = EventTerm(
        func=mdp.reset_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,
            # the hardest recovery, introduced only from iteration 600
            "face_up_prob": 0.00,
            "sitting_prob": 0.00,
            "standing_prob": 0.50,
            # One band for two poses whose contacts have nothing in common: face down the robot rests
            # at 0.0752 m and face up at 0.0475 m. Upstream picks the floor that eliminates
            # interpenetration on the belly -- measured, a 0.05 m spawn buries it 25 mm in the floor
            # -- at the cost of a back spawn that starts 28 to 42 mm above its own rest height.
            "prone_z_range": (0.076, 0.09),
            "standing_z_range": (0.134, 0.144),
            "sitting_joint_pos": None,
            # In ``reset_ground_state`` the standing bucket reuses the sitting bucket's orientation
            # sampler, so this tilt noise applies to standing spawns as well. That is upstream's
            # intent, stated in its own comment, and not an accident of the port.
            "sitting_tilt_max": math.radians(10.0),
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )

    # Bearing drag, shipped almost locked and *relaxed* by :attr:`CurriculumCfg.wheel_friction`.
    randomize_wheel_friction = EventTerm(
        func=mdp.randomize_joint_dry_friction,
        mode="reset",
        params={"asset_cfg": _WHEEL_JOINT_CFG, "friction_range": (0.05, 0.05)},
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

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    # Servos only: the wheels' armature is the bearing model, and the friction event above is what
    # randomizes it.
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": _SERVO_JOINT_CFG,
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    ##
    # Interval.
    ##

    # Ramped from nothing by :attr:`CurriculumCfg.push_magnitude`, unlike the skating task which
    # pushes at full strength from the first step: a robot that starts lying on the floor cannot be
    # shoved around while it is still learning to rise.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


@configclass
class CurriculumCfg(RollersCurriculumCfg):
    """Curriculum terms for the MDP (sections 9.7, 9.8 and 9.9).

    The skating task's two centre-of-mass ramps are inherited; its action-rate ramp is **replaced**
    and its wheel-friction ramp is **inverted**. Two more schedules are added.
    """

    # Walks the reset distribution from the easy mix toward the hard one, introducing face-up last
    # because recovering from the back is the hardest case. Every stage sums to 1.0, and the standing
    # bucket never vanishes.
    ground_state_mix = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {
                    "step": _iterations(0),
                    "params": {
                        "standing_prob": 0.50,
                        "sitting_prob": 0.00,
                        "face_down_prob": 0.50,
                        "face_up_prob": 0.00,
                    },
                },
                {
                    "step": _iterations(600),
                    "params": {
                        "standing_prob": 0.35,
                        "sitting_prob": 0.00,
                        "face_down_prob": 0.45,
                        "face_up_prob": 0.20,
                    },
                },
                {
                    "step": _iterations(1500),
                    "params": {
                        "standing_prob": 0.25,
                        "sitting_prob": 0.00,
                        "face_down_prob": 0.40,
                        "face_up_prob": 0.35,
                    },
                },
                {
                    "step": _iterations(2500),
                    "params": {
                        "standing_prob": 0.20,
                        "sitting_prob": 0.00,
                        "face_down_prob": 0.40,
                        "face_up_prob": 0.40,
                    },
                },
            ],
        },
    )

    # INVERTED relative to the skating task's, which ramps the bearing drag *up* from zero. Here it
    # starts almost locked and is relaxed toward the same 0.0015 the skating task converges on, and
    # upstream states the reason: rolling wheels give no longitudinal grip at all, so there is nothing
    # to push against, and the gesture is bootstrapped on a nearly-footed problem before the real
    # rolling physics is imposed.
    #
    # Two consequences upstream flags and this port carries:
    #
    # * **Deployment gate.** Only checkpoints from after the last stage -- iteration 4000 and later --
    #   are candidates for the robot. Before that the policy is leaning on a rolling friction the
    #   hardware does not have.
    # * **Diagnostic.** If ``standing_composite`` collapses at one of the stage boundaries, the
    #   "sticky feet" technique has not transferred to free wheels and a skater's technique has to be
    #   guided instead.
    #
    # ``event_param_stages`` with the exclusive comparison, matching upstream's dedicated
    # wheel-friction curriculum (addendum section 7.6).
    wheel_friction = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "randomize_wheel_friction",
            "inclusive": False,
            "param_stages": [
                {"step": _iterations(0), "params": {"friction_range": (0.05, 0.05)}},
                {"step": _iterations(1000), "params": {"friction_range": (0.02, 0.02)}},
                {"step": _iterations(2000), "params": {"friction_range": (0.008, 0.008)}},
                {"step": _iterations(3000), "params": {"friction_range": (0.003, 0.003)}},
                {"step": _iterations(4000), "params": {"friction_range": (0.0015, 0.0015)}},
            ],
        },
    )

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -0.4},
                {"step": _iterations(250), "weight": -0.8},
                {"step": _iterations(500), "weight": -1.0},
            ],
        },
    )

    # Ramped from nothing, unlike the skating task which pushes at full strength from the first step:
    # a robot that starts lying on the floor cannot be shoved around while it is still learning to
    # rise. Upstream drives this one with the exclusive step comparison.
    push_magnitude = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "push_robot",
            "inclusive": False,
            "param_stages": [
                {"step": _iterations(0), "params": {"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}},
                {"step": _iterations(500), "params": {"velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}}},
                {"step": _iterations(1000), "params": {"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}}},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckRollerStandUpFlatEnvCfg(MicroDuckVelocityRollersFlatEnvCfg):
    """MicroDuck stand-up-on-skates environment on flat ground.

    The scene, the sensors, the action space and both observation groups are the skating task's,
    which upstream inherits the same way -- this task is derived from the roller factory rather than
    rebuilt from the mjlab template (section 9).
    """

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Six seconds -- 300 control steps -- which is a rise plus a moment to hold it, and a third of
        # the skating task's window (section 9.1).
        self.episode_length_s = 6.0

        # Measured, not inherited: this robot spends most of every episode lying on the floor with
        # its head, shoulders and hips against it, which the skating profile does not cover. Profiled
        # under random actions with the pushes forced to full magnitude, at 256, 2048 and 4096
        # environments: **98 constraints and 28 contacts** per environment at the peak, against the
        # skating task's 83 and 26 on the same model. Logs:
        # ``artifacts/microduck/profile_microduck_contacts_rollerstandup_{256,2048,4096}envs.log``,
        # from ``artifacts/microduck/profile_microduck_contacts.py``.
        #
        # ``njmax`` is a hard per-environment cap and carries the wider margin; ``nconmax`` is a
        # per-environment share of one shared buffer and cannot overflow at the measured peak, so it
        # sits just above it, at the same 1.2x the rest of the family uses.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 192
        newton_mjwarp.solver_cfg.nconmax = 34
        self.sim.physics.default = newton_mjwarp
