# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke and recipe-parity tests for the contributed MicroDuck ball-kick environment.

Simulator-backed checks retrieve the prebuilt USD through Isaac Lab's native asset cache.
Recipe checks do not need an asset; source fidelity comparisons use optional read-only fixtures.

The parity tests need neither the asset nor the simulator: they read the assembled configuration and
compare it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference_tasks3.md`` section 6. The expected values are spelled out
rather than imported from the configuration under test, so that a drifting value fails rather than
agrees with itself.
"""

import math
import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ALLCOLLISIONS_USD_PATH, MICRODUCK_BALL_RADIUS
from isaaclab_tasks.contrib.microduck.ballkick.agents.rsl_rl_ppo_cfg import MicroDuckBallKickPPORunnerCfg
from isaaclab_tasks.contrib.microduck.ballkick.ballkick_env_cfg import (
    MICRODUCK_BALL_OFFSET_ABS_Y,
    MICRODUCK_BALL_OFFSET_X,
    MICRODUCK_BALL_POS_NOISE_XY,
    MICRODUCK_BALL_TARGET_SPEED,
    MICRODUCK_KICK_FOOT,
    MICRODUCK_STAND_HEIGHT,
    MICRODUCK_SUPPORT_FOOT,
    MicroDuckBallKickFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_allcollisions_usd = pytest.mark.usefixtures("microduck_allcollisions_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-BallKick-Flat-MicroDuck"

ACTOR_OBSERVATION_TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
]
"""The deployed MicroDuck observation layout, which is shared across the whole policy family.

Addendum section 11.1: all eight of upstream's MicroDuck tasks present the same 61-wide vector,
because one runtime on the robot feeds every policy from the same buffer. The head and body slots
are zero padding on this task, and the table is deliberately a copy of the velocity task's rather
than a reference to it -- if the two drift apart, both suites fail.
"""

ACTOR_OBSERVATION_DIM = 61
"""Actor observation width the deployed MicroDuck policy expects."""

CRITIC_OBSERVATION_TERMS = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("foot_air_time", 2),
    ("foot_contact", 2),
    ("foot_contact_forces", 6),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
    ("ball_position", 3),
    ("ball_velocity", 3),
]
"""The privileged critic layout (addendum section 11.2), which is not a deploy contract.

It is the stand-up task's critic plus the two ball terms, appended after the command block -- the
widest critic in the family, and the only one that sees a second entity.
"""

CRITIC_OBSERVATION_DIM = 80
"""Critic observation width, measured from the assembled group.

The extraction derives 80 by hand from upstream's term list (addendum section 11.2); this pins the
number the port actually produces against it.
"""

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, upstream's ``num_steps_per_env``."""

MEASURED_TOE_TIP_X = 0.0340
"""Forward reach [m] of a foot collision mesh at the stand pose, measured on the MJCF.

Upstream sizes the ball offset against it and claims a worst-case clearance of 6 mm; the extraction
reproduced both numbers (addendum section 2.3). It is the quantity a re-tune has to preserve, so it
is pinned here rather than left implicit in the 0.09.
"""

MEASURED_FOOT_SITE_ABS_Y = 0.0418
"""Half-spacing [m] of the two foot sites at the stand pose, measured on the MJCF."""

EXPECTED_SERVO_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 14 servos in upstream's MJCF actuator order: 0-4 left leg, 5-8 neck/head, 9-13 right leg."""

EXPECTED_LEG_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 10 leg joints -- upstream's ``_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]``."""

EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The 4 neck and head servos -- upstream's ``_NECK_JOINTS = [5, 6, 7, 8]``."""

EXPECTED_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot bodies in upstream's ``[left, right]`` site order."""

EXPECTED_TRUNK_BODY_NAMES = ["trunk_base"]
"""The body upstream measures the trunk height, tilt and angular velocity on, and randomizes."""

EXPECTED_HEAD_BODY_COUNT = 4
"""Bodies the head centre-of-mass randomization perturbs; upstream's fifth is a hip link it lists in
error, and the velocity port drops it."""

EXPECTED_ENTITY_SELECTIONS = {
    # "<manager>.<term>.<param>": (kind, expected names, preserve_order)
    "rewards.ball_forward_velocity.asset_cfg": ("asset", "ball", False),
    "rewards.ball_speed_overshoot.asset_cfg": ("asset", "ball", False),
    "rewards.support_foot_grounded.sensor_cfg": ("sensor", "support_foot_ground_contact", False),
    "rewards.pose_stand_legs.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.pose_stand_neck.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.upright.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.body_ang_vel.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.self_collisions.sensor_cfg": ("sensor", "self_collision", False),
    "events.foot_friction.asset_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, False),
    "events.mass_inertia.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_com.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_armature.asset_cfg": ("joint", [".*"], False),
    "events.set_ground_state.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "events.reset_ball.asset_cfg": ("asset", "ball", False),
}
"""Every entity selection the ball-kick recipe makes, outside the observation groups.

These are as load-bearing as the scalar parameters ``_scalar_params`` compares: a posture term that
silently selected all 14 joints instead of the 10 legs would fight the neck term, and a kick reward
that read the *robot's* velocity instead of the ball's would pay for walking. Isaac Lab resolves
joints and bodies in USD order, which is neither upstream's nor this table's, so ``preserve_order``
is part of the contract wherever a term indexes a block positionally.

``events.randomize_head_com`` is absent because its selection is four *patterns* rather than four
names; ``test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link`` pins it instead.
``events.randomize_joint_friction`` is absent because the term reads only the articulation name.
"""

EXPECTED_OBSERVATION_SELECTIONS = {
    # "<group>.<term>.<param>": (kind, expected names, preserve_order)
    "policy.joint_pos.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "policy.joint_vel.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.joint_pos.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.joint_vel.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.foot_air_time.sensor_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, True),
    "critic.foot_contact.sensor_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, True),
    "critic.foot_contact_forces.sensor_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, True),
    "critic.ball_position.asset_cfg": ("asset", "ball", False),
    "critic.ball_velocity.asset_cfg": ("asset", "ball", False),
}
"""Entity selections inside the two observation groups, all of which are ordering contracts.

The joint blocks are the deployed vector's own layout, which the runtime on the robot rebuilds by
hand from its sensor reads; the foot blocks are three consecutive critic terms that must agree on
which column is the left foot. The two ball entries are what keeps this task's privileged pair
pointed at the prop rather than at the robot.
"""


def _entity_cfg_of(term_cfg, key: str) -> SceneEntityCfg:
    """Fetch a term's entity selection, looking inside a delayed term's wrapped parameters."""
    if key in term_cfg.params:
        return term_cfg.params[key]
    return term_cfg.params["term_params"][key]


def _observation_entity_cfgs(term_cfg) -> dict[str, SceneEntityCfg]:
    """Every entity selection a single observation term carries, wrapped or not.

    The delayed actor terms hold theirs inside the wrapped term's parameters, so a walk that only
    looked at ``params`` would miss ``policy.joint_vel`` -- which is exactly one of the deploy-order
    contracts this file exists to pin.
    """
    selections = {key: value for key, value in term_cfg.params.items() if isinstance(value, SceneEntityCfg)}
    for key, value in term_cfg.params.get("term_params", {}).items():
        if isinstance(value, SceneEntityCfg):
            selections[key] = value
    return selections


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which carry no upstream scalar to compare against."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


##
# Recipe parity against upstream (addendum section 6)
##

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "ball_forward_velocity": (12.0, {"max_speed": 1.0}),
    "ball_speed_overshoot": (-4.0, {"target_speed": 1.0, "max_penalty": 5.0}),
    "support_foot_grounded": (2.0, {}),
    "pose_stand_legs": (2.0, {"std": 0.5}),
    "pose_stand_neck": (1.0, {"std": 0.3}),
    "height_stand": (1.0, {"std": 0.04, "target_height": 0.115}),
    "upright": (2.0, {"std": math.sqrt(0.05)}),
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    "dof_pos_limits": (-1.0, {}),
    "action_rate_l2": (-0.1, {}),
    "self_collisions": (-1.0, {"saturate": True}),
}
"""Upstream's reward recipe (addendum section 6.3), keyed by term name, at its initial weights.

``action_rate_l2`` is ramped by a curriculum and is listed at the weight the configuration ships,
not the -1.0 it reaches. ``dof_pos_limits`` is upstream's silently inherited regularizer -- it is
never mentioned in the ball-kick configuration and survives only because it is not in the deletion
list (addendum section 13.18), so a port that enumerated the terms upstream *writes* would miss it.

The two kick weights are transcribed from upstream's code, **not** from upstream's comments, which
describe a landscape calibrated for a target speed the file no longer uses (addendum section 13.5).
"""

EXPECTED_EVENTS = {
    # name: (mode, scalar params)
    "foot_friction": (
        "startup",
        {
            "static_friction_range": (0.7, 1.3),
            # Newton samples one coefficient per shape, so the dynamic range is supplied only
            # because the signature requires it; upstream sets the friction outright
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    ),
    "encoder_bias": ("startup", {"bias_range": (-0.015, 0.015)}),
    "mass_inertia": (
        "startup",
        {"mass_distribution_params": (0.95, 1.05), "operation": "scale", "recompute_inertia": True},
    ),
    "reset_base": ("reset", {"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}}),
    # the only task in the batch that adds joint noise to the base reset
    "reset_robot_joints": ("reset", {"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)}),
    "set_ground_state": (
        "reset",
        {
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "standing_z_range": (0.11, 0.12),
            "sitting_tilt_max": math.radians(5.0),
        },
    ),
    "reset_ball": (
        "reset",
        {"offset": (0.09, -0.042), "noise_xy": 0.015, "ball_radius": 0.035},
    ),
    "randomize_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_head_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_armature": ("reset", {"armature_distribution_params": (0.9, 1.1), "operation": "scale"}),
    "randomize_joint_friction": ("reset", {"scale_range": (0.9, 1.1)}),
    "push_robot": ("interval", {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}),
}
"""Upstream's event suite (addendum section 6.6), keyed by term name.

Upstream's ``base_com`` is absent because it selects zero bodies there; its
``expand_bam_friction_fields`` and ``reset_action_history`` are absent because Isaac Lab's actuator
storage and action manager already do their jobs; and ``randomize_motor_gains`` is absent because
upstream ships it disabled.
"""

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "set_ground_state",
    "reset_ball",
    "randomize_com",
    "randomize_head_com",
    "randomize_armature",
    "randomize_joint_friction",
]
"""Upstream's reset chain, in the order it fires (addendum sections 2.3 and 6.6).

This is behaviour, not housekeeping. ``set_ground_state`` overwrites the root height and orientation
that ``reset_base`` wrote -- including a uniformly random yaw -- and ``reset_ball`` then places the
ball in *that* yaw's frame. Running the ball placement earlier aims it at a heading the robot no
longer has, which upstream's own comment calls out as a silent failure.
"""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params)
    "time_out": (True, {}),
    "fell_over": (False, {"limit_angle": math.radians(70.0)}),
    "nan_state": (False, {"sensor_names": ("contact_forces",)}),
}
"""Upstream's terminations (addendum section 6.7), keyed by term name.

The tilt termination is **kept** here where the stand-up and forward-roll tasks delete it: this robot
starts standing and is supposed to finish that way. Upstream's inherited terrain-bounds termination
is not carried over: it returns all-false on a ground plane and this task has no rough variant.
"""

EXPECTED_CURRICULUM_TERMS = {
    "action_rate_weight",
    "com_range",
    "head_com_range",
    "push_magnitude",
}
"""Upstream's curriculum term names (addendum section 6.8).

``terrain_levels`` and ``command_vel`` are absent because upstream deletes both. **Nothing schedules
the kick**, which is upstream's documented intent rather than an omission.
"""

EXPECTED_WEIGHT_STAGES = {
    "action_rate_weight": ([-0.1, -0.2, -0.4, -0.6, -0.8, -1.0], [0, 500, 750, 1000, 1250, 1500]),
}
"""Upstream's only reward-weight ramp: payload and PPO-iteration boundaries."""

EXPECTED_RANGE_STAGES = {
    "com_range": ([0.003, 0.005, 0.01, 0.015], [0, 500, 1000, 1500]),
    "head_com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
}
"""Upstream's centre-of-mass range ramps, matched to the velocity task's."""

EXPECTED_PUSH_STAGES = [
    (0, {"x": (0.0, 0.0), "y": (0.0, 0.0)}),
    (500, {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}),
    (1000, {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
]
"""Upstream's push ramp, matched to the stand-up task's."""

EXPECTED_ACTOR_NOISE = {
    "base_ang_vel": 0.03,
    "projected_gravity": 0.01,
    "joint_pos": 0.001,
    "joint_vel": 0.25,
}
"""Half-width of the uniform noise on each corrupted actor term, matched to the velocity task."""


@pytest.mark.unit
def test_the_reward_recipe_matches_upstream_term_for_term():
    """Every reward slot upstream trains with is present, with its weight and its parameters."""
    rewards = MicroDuckBallKickFlatEnvCfg().rewards

    # two-sided, so a term left behind by the port also fails
    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_walking_and_gait_reward_terms_are_gone():
    """Upstream deletes every locomotion term: a one-shot kick has no gait to shape."""
    rewards = MicroDuckBallKickFlatEnvCfg().rewards

    for name in (
        "track_lin_vel",
        "track_ang_vel",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "soft_landing",
    ):
        assert not hasattr(rewards, name), name


@pytest.mark.unit
def test_the_kick_landscape_is_the_shipped_constants_not_the_upstream_prose():
    """The kick pays ``12*min(v,1) - 4*max(v-1,0)``: +12 at target, zero at 4 m/s, floored at -8.

    Upstream's comment block describes a peak of about +3/step and a zero crossing at 1.0 m/s, which
    are the numbers for a target speed of 0.25 that the file no longer sets (addendum section 13.5).
    The adjudication for this port is to reproduce the code, so this test pins the *arithmetic the
    shipped constants produce* -- deliberately spelled out, so that "fixing" the weights to match
    upstream's prose fails here rather than passing quietly.
    """
    rewards = MicroDuckBallKickFlatEnvCfg().rewards
    gain = rewards.ball_forward_velocity.weight
    tax = -rewards.ball_speed_overshoot.weight
    target = rewards.ball_forward_velocity.params["max_speed"]

    # both terms read the same target: a plateau split between two speeds is not a plateau
    assert rewards.ball_speed_overshoot.params["target_speed"] == pytest.approx(target)
    assert target == pytest.approx(MICRODUCK_BALL_TARGET_SPEED)

    def payout(speed: float) -> float:
        return gain * min(speed, target) - tax * max(speed - target, 0.0)

    assert payout(0.0) == pytest.approx(0.0)
    assert payout(0.5 * target) == pytest.approx(0.5 * gain)
    assert payout(target) == pytest.approx(12.0)
    assert payout(4.0 * target) == pytest.approx(0.0)
    # past the overshoot clamp the penalty stops growing, so the worst step is bounded
    clamp = rewards.ball_speed_overshoot.params["max_penalty"]
    assert payout(target + clamp) == pytest.approx(gain - tax * clamp)
    assert gain - tax * clamp == pytest.approx(-8.0)


@pytest.mark.unit
def test_the_kick_and_stand_halves_are_weighted_as_upstream_leaves_them():
    """The standing stack is 8 against a kick worth up to 12, which is what section 13.5 measures."""
    rewards = MicroDuckBallKickFlatEnvCfg().rewards

    standing_mass = sum(
        getattr(rewards, name).weight
        for name in ("support_foot_grounded", "pose_stand_legs", "pose_stand_neck", "upright", "height_stand")
    )
    assert standing_mass == pytest.approx(8.0)
    assert rewards.ball_forward_velocity.weight > standing_mass


@pytest.mark.unit
def test_the_event_suite_matches_upstream_term_for_term():
    """Every event upstream fires is present, in its mode, with its ranges."""
    events = MicroDuckBallKickFlatEnvCfg().events

    assert set(vars(events)) == set(EXPECTED_EVENTS)
    for name, (mode, params) in EXPECTED_EVENTS.items():
        term = getattr(events, name)
        assert term.mode == mode, name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == value, f"{name}.{key}"
    assert events.push_robot.interval_range_s == (3.0, 6.0)


@pytest.mark.unit
def test_the_ball_placement_runs_after_the_pose_it_is_measured_from():
    """Isaac Lab fires reset events in declaration order, and this order is the spawn geometry."""
    events = MicroDuckBallKickFlatEnvCfg().events

    reset_terms = [name for name, term in vars(events).items() if getattr(term, "mode", None) == "reset"]
    assert reset_terms == EXPECTED_RESET_EVENT_ORDER
    assert reset_terms.index("reset_ball") > reset_terms.index("set_ground_state")


@pytest.mark.unit
def test_the_ball_spawns_clear_of_the_toe_at_the_kicking_foot():
    """The offset is sized against the model, and 6 mm of worst-case clearance is what it buys.

    Upstream records that an offset of ``0.08 +/- 0.02`` penetrated the toe at reset and the solver
    ejected the ball, paying a kick reward for no kick. The margin, not the literal 0.09, is what a
    re-tune has to preserve, so it is what this test measures.
    """
    params = MicroDuckBallKickFlatEnvCfg().events.reset_ball.params
    forward, lateral = params["offset"]
    noise = params["noise_xy"]

    assert forward == pytest.approx(MICRODUCK_BALL_OFFSET_X)
    assert abs(lateral) == pytest.approx(MICRODUCK_BALL_OFFSET_ABS_Y)
    assert noise == pytest.approx(MICRODUCK_BALL_POS_NOISE_XY)
    # the ball sits on the kicking foot's own line, not in front of the robot's centre
    assert (lateral < 0.0) == (MICRODUCK_KICK_FOOT == "right")
    assert abs(lateral) == pytest.approx(MEASURED_FOOT_SITE_ABS_Y, abs=1e-3)
    # worst case: the ball as far back as the noise allows, and its rear surface still clear
    rear_surface_x = forward - noise - params["ball_radius"]
    assert rear_surface_x - MEASURED_TOE_TIP_X == pytest.approx(0.006, abs=1e-4)
    assert params["ball_radius"] == pytest.approx(MICRODUCK_BALL_RADIUS)


@pytest.mark.unit
def test_the_ground_state_reset_is_standing_only():
    """The kick starts from a stand, so the three ground buckets are switched off entirely."""
    params = MicroDuckBallKickFlatEnvCfg().events.set_ground_state.params

    assert params["standing_prob"] == 1.0
    for bucket in ("face_down_prob", "face_up_prob", "sitting_prob"):
        assert params[bucket] == 0.0, bucket
    # the bands and the keyframe those buckets would sample are left unconfigured rather than
    # filled with values nothing draws
    for unused in ("prone_z_range", "sitting_z_range", "sitting_joint_pos"):
        assert unused not in params, unused
    # the standing band brackets the height the whole standing stack is built around
    low, high = params["standing_z_range"]
    assert low <= MICRODUCK_STAND_HEIGHT <= high
    # upstream spells the stand's tilt bound ``sitting_tilt_max``; the two buckets share one sampler
    assert params["sitting_tilt_max"] == pytest.approx(math.radians(5.0))


@pytest.mark.unit
def test_the_terms_select_the_joints_bodies_sensors_and_assets_upstream_measures():
    """A term that measures the wrong entity is as wrong as one carrying the wrong weight."""
    cfg = MicroDuckBallKickFlatEnvCfg()

    # two-sided over the terms that carry a selection at all, so a term that gains or loses one
    # fails rather than going unchecked
    measured = {
        f"{manager}.{term_name}.{key}"
        for manager in ("rewards", "events")
        for term_name, term in vars(getattr(cfg, manager)).items()
        for key, value in term.params.items()
        if isinstance(value, SceneEntityCfg)
    }
    # the one selection pinned by its own test rather than by the table: it names four body
    # *patterns*, which the table's name equality cannot express. The second assertion is what stops
    # the exemption outliving the selection it excuses.
    exempt = {"events.randomize_head_com.asset_cfg"}
    assert measured - exempt == set(EXPECTED_ENTITY_SELECTIONS)
    assert exempt <= measured, "the exemption names a selection the recipe no longer makes"

    for path, (kind, expected, preserve_order) in EXPECTED_ENTITY_SELECTIONS.items():
        manager, term_name, key = path.split(".")
        entity_cfg = getattr(getattr(cfg, manager), term_name).params[key]
        if kind in ("sensor", "asset"):
            assert entity_cfg.name == expected, path
            assert entity_cfg.joint_names is None and entity_cfg.body_names is None, path
            continue
        assert entity_cfg.name == "robot", path
        if kind == "joint":
            assert entity_cfg.joint_names == expected, path
            assert entity_cfg.body_names is None, path
        else:
            assert entity_cfg.body_names == expected, path
            assert entity_cfg.joint_names is None, path
        assert entity_cfg.preserve_order is preserve_order, path


@pytest.mark.unit
def test_both_observation_groups_read_the_servos_the_feet_and_the_ball_in_the_deploy_order():
    """Isaac Lab resolves joints and bodies in USD order; the deployed vector is in MJCF order.

    Widths are order-blind, so the integration test that measures 61 and 80 cannot see a joint block
    that lost ``preserve_order`` and now reports the servos in the articulation's order -- which
    would produce a policy that runs on the robot against the wrong columns. This is the assertion
    that catches it.
    """
    observations = MicroDuckBallKickFlatEnvCfg().observations
    groups = {"policy": _observation_terms(observations.policy), "critic": _observation_terms(observations.critic)}

    # two-sided over the observation terms that carry a selection at all, so a term that gains or
    # loses one -- an actor term that started reading the ball, say -- fails rather than going
    # unchecked
    measured = {
        f"{group}.{term_name}.{key}"
        for group, terms in groups.items()
        for term_name, term in terms.items()
        for key in _observation_entity_cfgs(term)
    }
    assert measured == set(EXPECTED_OBSERVATION_SELECTIONS)

    for path, (kind, expected, preserve_order) in EXPECTED_OBSERVATION_SELECTIONS.items():
        group, term_name, key = path.split(".")
        entity_cfg = _entity_cfg_of(groups[group][term_name], key)
        if kind == "asset":
            assert entity_cfg.name == expected, path
            assert entity_cfg.joint_names is None and entity_cfg.body_names is None, path
        elif kind == "joint":
            assert entity_cfg.name == "robot", path
            assert entity_cfg.joint_names == expected, path
        else:
            assert entity_cfg.name == "contact_forces", path
            assert entity_cfg.body_names == expected, path
        assert entity_cfg.preserve_order is preserve_order, path

    # the neck reward indexes its four servos out of the same 14-wide block the observation reports,
    # so the two orders are one contract rather than two
    assert EXPECTED_SERVO_JOINT_NAMES[5:9] == EXPECTED_HEAD_JOINT_NAMES


@pytest.mark.unit
def test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link():
    """``bearing_roll`` is the right hip-yaw link; upstream lists it among the head bodies in error."""
    body_names = MicroDuckBallKickFlatEnvCfg().events.randomize_head_com.params["asset_cfg"].body_names

    assert "bearing_roll" not in body_names
    assert len(body_names) == EXPECTED_HEAD_BODY_COUNT


@pytest.mark.unit
def test_the_terminations_keep_the_tilt_check_this_task_needs():
    """A task that starts and should end standing terminates on falling over."""
    terminations = MicroDuckBallKickFlatEnvCfg().terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        assert _scalar_params(term) == params, name


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """The staged schedules carry upstream's payloads at upstream's iteration boundaries."""
    curriculum = MicroDuckBallKickFlatEnvCfg().curriculum

    # two-sided, so an unexpected extra schedule quietly rewriting a weight or a range also fails
    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS

    for name, (weights, iterations) in EXPECTED_WEIGHT_STAGES.items():
        stages = getattr(curriculum, name).params["weight_stages"]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage["weight"] for stage in stages] == pytest.approx(weights), name

    for name, (ranges, iterations) in EXPECTED_RANGE_STAGES.items():
        stages = getattr(curriculum, name).params["range_stages"]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage["range"] for stage in stages] == pytest.approx(ranges), name


@pytest.mark.unit
def test_nothing_schedules_the_kick_reward():
    """The kick is live from step 0, which is upstream's stated design and not an oversight."""
    curriculum = MicroDuckBallKickFlatEnvCfg().curriculum

    scheduled = {
        term.params["reward_name"] for term in vars(curriculum).values() if "reward_name" in getattr(term, "params", {})
    }
    assert scheduled == {"action_rate_l2"}


@pytest.mark.unit
def test_the_push_curriculum_ramps_from_nothing_on_upstreams_exclusive_comparison():
    """A robot still learning to swing one leg cannot be shoved; upstream also switches operator."""
    push = MicroDuckBallKickFlatEnvCfg().curriculum.push_magnitude

    stages = push.params["param_stages"]
    assert len(stages) == len(EXPECTED_PUSH_STAGES)
    for stage, (iteration, velocity_range) in zip(stages, EXPECTED_PUSH_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        assert stage["params"] == {"velocity_range": velocity_range}
    # upstream's push helper compares with ``>`` where its event-parameter helper uses ``>=``
    assert push.params["inclusive"] is False


@pytest.mark.unit
def test_the_command_survives_only_to_keep_the_observation_slot_alive():
    """No reward reads the twist; it exists so the deployed vector keeps its three-wide slot."""
    cfg = MicroDuckBallKickFlatEnvCfg()
    twist = cfg.commands.base_velocity

    # the twist is the only command: upstream declares no head or body pose command here
    assert set(vars(cfg.commands)) == {"base_velocity"}
    assert twist.ranges.lin_vel_x == (-0.01, 0.01)
    assert twist.ranges.lin_vel_y == (-0.01, 0.01)
    assert twist.ranges.ang_vel_z == (-0.05, 0.05)
    assert twist.resampling_time_range[0] >= cfg.episode_length_s
    assert twist.heading_command is False
    assert twist.rel_standing_envs == 0.0
    assert twist.rel_turn_in_place_envs == 0.0
    # inherited from upstream's base template and never overridden there (addendum section 7.22)
    assert twist.rel_forward_envs == pytest.approx(0.2)


@pytest.mark.unit
def test_the_actor_observation_layout_is_the_shared_deploy_contract_and_is_blind_to_the_ball():
    """The kick policy reads the same 61-wide vector every other MicroDuck policy does, and no more.

    The blindness is the design, not an omission: the real robot has no ball sensor, so a policy
    trained on ball state could not be deployed. It is asserted here by name rather than only by
    width, because a ball term that replaced a zero pad would keep the width at 61.
    """
    observations = MicroDuckBallKickFlatEnvCfg().observations
    terms = _observation_terms(observations.policy)

    assert list(terms) == [name for name, _ in ACTOR_OBSERVATION_TERMS]
    assert sum(width for _, width in ACTOR_OBSERVATION_TERMS) == ACTOR_OBSERVATION_DIM
    assert observations.policy.concatenate_terms
    assert observations.policy.enable_corruption
    for name, term in terms.items():
        magnitude = EXPECTED_ACTOR_NOISE.get(name)
        if magnitude is None:
            assert term.noise is None, name
        else:
            assert (term.noise.n_min, term.noise.n_max) == (-magnitude, magnitude), name
    # no actor term reads the ball, by function or by asset selection
    for name, term in terms.items():
        assert term.func not in (mdp.ball_pos_in_base, mdp.ball_vel_in_base), name
        assert "ball" not in {value.name for value in term.params.values() if isinstance(value, SceneEntityCfg)}, name
    # the two command slots the task does not use are constant zero, not a live tiny range
    for name in ("head_pose_commands", "body_pose_commands"):
        assert terms[name].func is mdp.zero_command_padding, name
    # distinct configuration objects: an alias would share one delay term's buffer between them
    assert terms["base_ang_vel"] is not terms["projected_gravity"]


@pytest.mark.unit
def test_the_critic_sees_the_ball_and_guards_its_sensor_reads():
    """The asymmetric half: 80 wide, with the ball appended after the command block."""
    observations = MicroDuckBallKickFlatEnvCfg().observations
    terms = _observation_terms(observations.critic)

    assert list(terms) == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert sum(width for _, width in CRITIC_OBSERVATION_TERMS) == CRITIC_OBSERVATION_DIM
    assert not observations.critic.enable_corruption
    assert terms["ball_position"].func is mdp.ball_pos_in_base
    assert terms["ball_velocity"].func is mdp.ball_vel_in_base
    # upstream deletes foot_height here: no height sensor, and no foot-height reward to need one
    assert "foot_height" not in terms
    # the NaN guarding this port applies where upstream leaves it off (addendum section 14.4)
    assert terms["foot_air_time"].func is mdp.foot_air_time_safe
    assert terms["foot_contact_forces"].func is mdp.foot_contact_forces_safe
    # a privileged group the runner never reads is dead weight
    assert MicroDuckBallKickPPORunnerCfg().obs_groups == {"actor": ["policy"], "critic": ["critic"]}


@pytest.mark.unit
def test_the_scene_carries_the_ball_next_to_the_all_collisions_robot():
    """Two entities, on a plane, with the ball declared after the robot that places it."""
    scene = MicroDuckBallKickFlatEnvCfg().scene

    assert scene.robot.spawn.usd_path == MICRODUCK_ALLCOLLISIONS_USD_PATH
    assert scene.ball.spawn.radius == pytest.approx(MICRODUCK_BALL_RADIUS)
    assert scene.ball.spawn.mass_props[0].mass == pytest.approx(0.015)
    assert scene.terrain.terrain_type == "plane"
    assert scene.terrain.terrain_generator is None
    # the ball's contacts on top of a full-collision robot's; measured, not inherited
    solver = MicroDuckBallKickFlatEnvCfg().sim.physics.default.solver_cfg
    assert solver.nconmax >= 30
    assert solver.njmax >= 86


@pytest.mark.unit
def test_the_support_foot_sensor_watches_the_foot_that_does_not_kick():
    """The sensor names the *other* foot, and filters against the terrain rather than everything.

    Reading a net contact force would be wrong here in a way it is not on the family's other foot
    sensors: the ball rolls along the ground into the robot's feet, so an unfiltered support sole
    would report "grounded" while airborne and merely touching the ball.
    """
    scene = MicroDuckBallKickFlatEnvCfg().scene

    assert MICRODUCK_SUPPORT_FOOT != MICRODUCK_KICK_FOOT
    (sensor_expr,) = scene.support_foot_ground_contact.sensor_shape_prim_expr
    assert f"{MICRODUCK_SUPPORT_FOOT}_foot_collision" in sensor_expr
    assert f"{MICRODUCK_KICK_FOOT}_foot_collision" not in sensor_expr
    # shape expressions full-match shape paths, so the trailing token is what reaches the collider
    assert sensor_expr.endswith("/[^/]*")
    # filtered against the ground plane specifically, which is a single shape shared by every
    # environment and therefore addressed absolutely
    (filter_expr,) = scene.support_foot_ground_contact.filter_shape_prim_expr
    assert filter_expr.startswith("/World/ground")
    assert not scene.support_foot_ground_contact.filter_prim_paths_expr


@pytest.mark.unit
def test_the_self_collision_sensor_is_the_familys_many_to_many_one():
    """Upstream's sensor is the trunk subtree against itself, which is many-to-many."""
    cfg = MicroDuckBallKickFlatEnvCfg()

    assert cfg.scene.self_collision.sensor_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    assert cfg.scene.self_collision.filter_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    # sensing both sides of a contact reports it twice, so the reward has to saturate
    assert cfg.rewards.self_collisions.params["saturate"] is True


@pytest.mark.unit
def test_the_runner_names_the_kicking_foot_and_carries_upstreams_budget():
    """The mirrored kick is a different policy, so the two must not share a run directory."""
    runner = MicroDuckBallKickPPORunnerCfg()

    assert runner.experiment_name.endswith(MICRODUCK_KICK_FOOT)
    assert runner.max_iterations == 10000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    assert runner.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert runner.algorithm.desired_kl == pytest.approx(0.01)
    # left-right symmetry is off, because the kick is inherently one-footed
    assert getattr(runner.algorithm, "symmetry_cfg", None) is None


@pytest.mark.unit
def test_the_episode_and_simulation_rates_match_upstream():
    """A 5 s episode is a kick plus several seconds of rolling, the shortest window in the family."""
    cfg = MicroDuckBallKickFlatEnvCfg()

    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.episode_length_s == pytest.approx(5.0)
    # the BAM delay line is actuator state, and an odd decimation cannot be graph-captured
    assert cfg.sim.use_newton_actuators is True
    assert cfg.decimation % 2 == 0


##
# Environment smoke tests
##


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_observation_groups_are_the_widths_their_contracts_name():
    """The actor group is the deployed 61-vector, term for term, and the critic measures 80."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        obs, _ = env.reset()

        assert obs["policy"].shape[-1] == ACTOR_OBSERVATION_DIM
        assert obs["critic"].shape[-1] == CRITIC_OBSERVATION_DIM
        # per-term widths as well as the total, so two compensating drifts cannot agree on 61
        manager = env.unwrapped.observation_manager
        for group, expected in (("policy", ACTOR_OBSERVATION_TERMS), ("critic", CRITIC_OBSERVATION_TERMS)):
            measured = [
                (name, dim[0]) for name, dim in zip(manager.active_terms[group], manager.group_obs_term_dim[group])
            ]
            assert measured == expected, group
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_reset_puts_the_ball_in_front_of_the_kicking_foot_at_every_heading():
    """The ball lands in the robot's *reset* yaw frame, resting on the ground, at rest.

    This is the acceptance test for the whole placement chain, and the random yaw is the point: the
    ground-state reset draws a heading uniformly over the circle, so a world-axis placement would
    put the ball behind the robot on half the episodes and pass any check that only measured a
    distance.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=8)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        ball = unwrapped.scene["ball"]
        forward, lateral = env_cfg.events.reset_ball.params["offset"]
        noise = env_cfg.events.reset_ball.params["noise_xy"]

        headings, in_frame, heights = [], [], []
        for _ in range(6):
            unwrapped.reset()
            quat = robot.data.root_link_quat_w.torch
            # Isaac Lab quaternions are (x, y, z, w)
            qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            offset_w = (ball.data.root_link_pos_w.torch - robot.data.root_link_pos_w.torch)[:, :2]
            cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
            # rotate the world offset back into the robot's yaw frame
            in_frame.append(
                torch.stack(
                    (
                        cos_yaw * offset_w[:, 0] + sin_yaw * offset_w[:, 1],
                        -sin_yaw * offset_w[:, 0] + cos_yaw * offset_w[:, 1],
                    ),
                    dim=-1,
                ).clone()
            )
            headings.append(yaw.clone())
            heights.append((ball.data.root_link_pos_w.torch[:, 2] - unwrapped.scene.env_origins[:, 2]).clone())
            # the ball is set down, not thrown
            torch.testing.assert_close(
                ball.data.root_link_lin_vel_w.torch,
                torch.zeros_like(ball.data.root_link_lin_vel_w.torch),
                atol=1e-5,
                rtol=0.0,
            )

        heading = torch.cat(headings)
        measured = torch.cat(in_frame)
        height = torch.cat(heights)

        # the headings really do cover the circle, so the frame check below has something to prove
        assert heading.max().item() - heading.min().item() > math.pi

        # every ball inside the configured offset box, in the robot's own frame
        assert (measured[:, 0] - forward).abs().max().item() <= noise + 1e-4
        assert (measured[:, 1] - lateral).abs().max().item() <= noise + 1e-4
        # and never behind the robot, nor on the support foot's side
        assert measured[:, 0].min().item() > 0.0
        assert (measured[:, 1] < 0.0).all() == (MICRODUCK_KICK_FOOT == "right")
        # resting exactly on the plane: no penetration to eject it, no drop to bounce it
        torch.testing.assert_close(height, torch.full_like(height, MICRODUCK_BALL_RADIUS), atol=1e-4, rtol=0.0)

        # and the frozen kick direction is the reset heading, which is what "forward" means to the
        # two kick rewards for the rest of the episode
        direction = mdp.ball_kick_direction(unwrapped)
        torch.testing.assert_close(direction[:, 0], torch.cos(headings[-1]), atol=1e-5, rtol=0.0)
        torch.testing.assert_close(direction[:, 1], torch.sin(headings[-1]), atol=1e-5, rtol=0.0)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_kick_rewards_pay_a_ball_pushed_forward_and_nothing_for_one_pushed_back():
    """The acceptance test for the reward the whole task is about, measured on a moving ball.

    The ball is given a velocity by hand along and against the frozen kick direction, which
    exercises the projection rather than the physics: a term that read the ball's *speed* instead of
    its forward component would pay both cases equally, and one that read a live heading instead of
    the frozen one would drift as the robot turned.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        ball = unwrapped.scene["ball"]
        direction = mdp.ball_kick_direction(unwrapped)
        target = env_cfg.rewards.ball_forward_velocity.params["max_speed"]
        # env 0 at rest, env 1 at half the target, env 2 exactly at it, env 3 backwards at it
        speeds = torch.tensor([0.0, 0.5 * target, target, -target], device=unwrapped.device)

        velocity = torch.zeros((4, 6), device=unwrapped.device)
        velocity[:, :2] = direction * speeds.unsqueeze(-1)
        with torch.inference_mode():
            ball.write_root_com_velocity_to_sim_index(root_velocity=velocity)

            gain = mdp.ball_forward_velocity(unwrapped, max_speed=target, asset_cfg=SceneEntityCfg("ball"))
            tax = mdp.ball_speed_overshoot_penalty(
                unwrapped, target_speed=target, max_penalty=5.0, asset_cfg=SceneEntityCfg("ball")
            )

        torch.testing.assert_close(gain, speeds.clamp(min=0.0), atol=1e-4, rtol=0.0)
        # nothing is charged below the target, including for the ball travelling backwards
        torch.testing.assert_close(tax, torch.zeros_like(tax), atol=1e-5, rtol=0.0)

        # and past the target the tax opens while the gain stays capped
        velocity[:, :2] = direction * (3.0 * target)
        with torch.inference_mode():
            ball.write_root_com_velocity_to_sim_index(root_velocity=velocity)
            gain = mdp.ball_forward_velocity(unwrapped, max_speed=target, asset_cfg=SceneEntityCfg("ball"))
            tax = mdp.ball_speed_overshoot_penalty(
                unwrapped, target_speed=target, max_penalty=5.0, asset_cfg=SceneEntityCfg("ball")
            )
        torch.testing.assert_close(gain, torch.full_like(gain, target), atol=1e-4, rtol=0.0)
        torch.testing.assert_close(tax, torch.full_like(tax, 2.0 * target), atol=1e-4, rtol=0.0)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_support_foot_reward_reads_the_floor_rather_than_the_ball():
    """A planted support foot pays; an airborne one does not, even resting against the ball.

    The second half is the regression for the terrain filter: the ball is a rolling object at
    exactly sole height, so the unfiltered net contact force the family's other foot terms read
    cannot tell it from the ground.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        ball = unwrapped.scene["ball"]
        sensor = unwrapped.scene.sensors["support_foot_ground_contact"]
        # exactly the support foot's sole against the ground plane, and nothing else
        assert sensor.num_sensors == 1, sensor.sensor_names
        assert sensor.num_filter_objects == 1, sensor.filter_object_names
        assert MICRODUCK_SUPPORT_FOOT in sensor.sensor_names[0]

        # env 0 stands on the floor at the settled standing height; env 1 is held a metre up with
        # the ball pressed against its support sole, which is the only contact it has
        pose = torch.zeros((2, 7), device=unwrapped.device)
        pose[:, :3] = unwrapped.scene.env_origins
        pose[0, 2] += MICRODUCK_STAND_HEIGHT
        pose[1, 2] += 1.0
        pose[:, 6] = 1.0  # (x, y, z, w) identity
        joint_pos = robot.data.default_joint_pos.torch.clone()
        joint_vel = torch.zeros_like(joint_pos)
        velocity = torch.zeros((2, 6), device=unwrapped.device)

        ball_pose = torch.zeros((2, 7), device=unwrapped.device)
        ball_pose[:, :3] = unwrapped.scene.env_origins
        ball_pose[0, 0] += 5.0  # env 0's ball parked well out of the way
        ball_pose[0, 2] += MICRODUCK_BALL_RADIUS
        # env 1's ball lifted with the robot, just under the support sole
        ball_pose[1, 1] += (
            MICRODUCK_BALL_OFFSET_ABS_Y if MICRODUCK_SUPPORT_FOOT == "left" else -MICRODUCK_BALL_OFFSET_ABS_Y
        )
        ball_pose[1, 2] += 1.0 - MICRODUCK_BALL_RADIUS
        ball_pose[:, 6] = 1.0

        action = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
        with torch.inference_mode():
            for _ in range(12):
                robot.write_root_link_pose_to_sim_index(root_pose=pose)
                robot.write_root_com_velocity_to_sim_index(root_velocity=velocity)
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                ball.write_root_link_pose_to_sim_index(root_pose=ball_pose)
                ball.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((2, 6), device=unwrapped.device))
                env.step(action)

            grounded = mdp.single_foot_grounded_reward(
                unwrapped, sensor_cfg=SceneEntityCfg("support_foot_ground_contact")
            )

        assert grounded[0].item() == 1.0, "a standing robot was not credited with a planted foot"
        assert grounded[1].item() == 0.0, "an airborne robot was credited through its ball contact"
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
