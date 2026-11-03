# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whole-body recovery with passive roller skates and roller-specific reset heights."""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.microduck.mdp as mdp
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
# Keyframe heights
##

MICRODUCK_ROLLER_STAND_HEIGHT = 0.138
"""Trunk height [m] of the standing keyframe on skates, the single target this task rewards.

The height also sits inside the roller task's reset band ``(0.1335, 0.1435)``, which is why the
"already standing" spawn bucket below can reuse it unchanged.
"""

MICRODUCK_ROLLER_PRONE_HEIGHT = 0.075
"""Trunk height [m] at rest face down on the skates, measured kinematically.

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
"""The ten leg joints the three pose-scoring terms read, resolved by name."""
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_WHEEL_JOINT_NAMES, preserve_order=True)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")


def _iterations(count: int) -> int:
    """Convert a PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    One command, kept only so the deployed 61-wide observation keeps its three-wide twist slot. No
    reward reads it and its ranges are a hundredth of the skating task's, which is the same
    neutralization the wheel-less stand-up and ball-kick tasks apply. It also **downgrades** the
    roller task's relative-heading command back to the plain velocity one: a live heading error in an
    observation slot nothing tracks would be a distractor rather than a shape placeholder.
    """

    base_velocity = mdp.ForwardMaskVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 12.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.ForwardMaskVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.01, 0.01),
            lin_vel_y=(-0.01, 0.01),
            ang_vel_z=(-0.05, 0.05),
            heading=None,
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

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
    """

    ##
    # Regularizers carried over from the skating recipe.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2_raw, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2_raw, weight=-0.02)
    # The live value is the one stated. The ramp itself is the wheel-less stand-up task's rather than the skating
    # task's, deliberately: the skating ramp reaches -2.0 for a calm gait, which is a movement blocker on a task
    # whose whole point is a fast rise from the back.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.4)
    self_collisions = RewTerm(
        func=mdp.self_collision_cost_saturable, weight=-1.0, params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG}
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
    # POSITIVE WEIGHT, AND NOT A TYPO.
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
    """Termination terms for the MDP.

    The skating task's tilt termination is **absent**: the robot starts on the ground here, so
    terminating on tilt would end most episodes on their first step.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Catches a *broken* robot rather than a fallen one, with the foot sensor named -- the family's NaN-guard norm,
    # carried over from the skating task.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan_optional_sensors,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class EventsCfg:
    """Configuration for events.

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

    # Only the horizontal spread is live, so only the horizontal spread is configured.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # The episode distribution. These four probabilities are :attr:`CurriculumCfg.ground_state_mix`'s first stage;
    # the height bands are left alone by that curriculum.
    set_ground_state = EventTerm(
        func=mdp.reset_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,
            # the hardest recovery, introduced only from iteration 600
            "face_up_prob": 0.00,
            "sitting_prob": 0.00,
            "standing_prob": 0.50,
            # One band for two poses whose contacts have nothing in common: face down the robot rests at 0.0752 m
            # and face up at 0.0475 m.
            "prone_z_range": (0.076, 0.09),
            "standing_z_range": (0.134, 0.144),
            "sitting_joint_pos": None,
            # In ``reset_ground_state`` the standing bucket reuses the sitting bucket's orientation sampler, so this
            # tilt noise applies to standing spawns as well.
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
    """Curriculum terms for the MDP.

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

    # INVERTED relative to the skating task's, which ramps the bearing drag *up* from zero. Before that the policy
    # is leaning on a rolling friction the hardware does not have. * **Diagnostic.** If ``standing_composite``
    # collapses at one of the stage boundaries, the "sticky feet" technique has not transferred to free wheels
    # and a skater's technique has to be guided instead.
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

    # Ramped from nothing, unlike the skating task which pushes at full strength from the first step: a robot that
    # starts lying on the floor cannot be shoved around while it is still learning to rise.
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
    """MicroDuck stand-up-on-skates environment on flat ground."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Six seconds -- 300 control steps -- which is a rise plus a moment to hold it, and a third of
        # the skating task's window.
        self.episode_length_s = 6.0

        # Reserve contact and constraint headroom for simultaneous body-ground contact.

        # Constraint limits are per environment; contact storage is shared across environments.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 192
        newton_mjwarp.solver_cfg.nconmax = 34
        self.sim.physics.default = newton_mjwarp
