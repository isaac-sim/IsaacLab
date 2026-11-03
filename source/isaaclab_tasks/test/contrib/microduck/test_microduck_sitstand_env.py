# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke and recipe-parity tests for the contributed MicroDuck sit-stand environment.

Simulator-backed checks retrieve the prebuilt USD through Isaac Lab's native asset cache.
Recipe checks do not need an asset; source fidelity comparisons use optional read-only fixtures.

The parity tests need neither the asset nor the simulator: they read the assembled configuration and
compare it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference_tasks3.md`` section 4. Transcribed rather than imported --
a table that read the configuration it checks would agree with itself.
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
from isaaclab_tasks.contrib.microduck.sitstand.agents.rsl_rl_ppo_cfg import MicroDuckSitStandPPORunnerCfg
from isaaclab_tasks.contrib.microduck.sitstand.sitstand_env_cfg import (
    MICRODUCK_MAX_DESCENT_SPEED,
    MICRODUCK_MAX_RISE_SPEED,
    MICRODUCK_POSTURE_RAMP_S,
    MicroDuckSitStandFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.standup.standup_env_cfg import (
    MICRODUCK_SITTING_JOINT_POS,
    MicroDuckStandUpFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_allcollisions_usd = pytest.mark.usefixtures("microduck_allcollisions_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-SitStand-Flat-MicroDuck"

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, upstream's ``num_steps_per_env``."""

STAND_HEIGHT = 0.115
SIT_HEIGHT = 0.060
"""The two measured rest heights (addendum section 4.1), transcribed from upstream rather than
imported from the stand-up task the configuration shares them with."""

SIT_JOINT_POS = {
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.4079,
    "left_knee": 1.35,
    "left_ankle": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.4079,
    "right_knee": -1.35,
    "right_ankle": 0.0,
}
"""Upstream's ``SITTING_TARGET_OVERRIDES`` (addendum section 4.1), keyed by name where upstream keys
by servo index. Stability-verified upstream on 2026-07-27; the neck and head are absent on purpose,
because the head is command-steered in both postures."""

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
"""The deployed MicroDuck observation layout, shared across the whole policy family.

Addendum section 11.1: every upstream MicroDuck task presents the same 61-wide vector, because one
runtime on the robot feeds every policy from the same buffer. Here the three-wide ``velocity_commands``
slot carries the posture flag rather than a twist, and the six-wide body slot is zero padding.
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
]
"""The privileged critic layout (addendum section 11.2), which is not a deploy contract.

Narrower than the velocity task's by the two ``foot_height`` columns, which upstream deletes here
because this scene carries no height sensor.
"""

CRITIC_OBSERVATION_DIM = 74
"""Critic observation width, upstream's own figure for this task."""

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
"""The 10 leg joints the posture terms score, upstream's ``_LEG_JOINTS``."""

EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The 4 head servos, in the order the head-pose command indexes its columns."""

EXPECTED_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot bodies in upstream's ``[left, right]`` site order."""

EXPECTED_TRUNK_BODY_NAMES = ["trunk_base"]
"""The body every posture term measures the height, tilt and vertical speed on."""

EXPECTED_REWARDS = {
    "neck_strain_leaned": (
        -0.5,
        {"command_name": "base_velocity", "tilt_start_deg": 12.0, "tilt_full_deg": 25.0, "torque_scale": 0.1},
    ),
    # name: (stage-0 weight, scalar params)
    "posture_pose_legs": (4.0, {"command_name": "base_velocity", "sit_joint_pos": SIT_JOINT_POS, "std": 0.5}),
    "posture_pose_l1": (1.0, {"command_name": "base_velocity", "sit_joint_pos": SIT_JOINT_POS}),
    "head_pose_tracking": (0.75, {"command_name": "head_pose", "std": 0.5}),
    "posture_height": (
        1.0,
        {"command_name": "base_velocity", "sit_height": SIT_HEIGHT, "stand_height": STAND_HEIGHT, "std": 0.04},
    ),
    "posture_height_sharp": (
        1.0,
        {"command_name": "base_velocity", "sit_height": SIT_HEIGHT, "stand_height": STAND_HEIGHT, "std": 0.015},
    ),
    "posture_height_l1": (
        6.0,
        {"command_name": "base_velocity", "sit_height": SIT_HEIGHT, "stand_height": STAND_HEIGHT},
    ),
    "rise_bootstrap": (0.75, {"command_name": "base_velocity", "max_height": 0.125, "max_vz": 0.08}),
    "descent_speed": (10.0, {"max_down_vel": 0.05}),
    "rise_speed": (0.0, {"max_up_vel": 0.08}),
    "gentle_motion": (0.05, {}),
    "upright_linear": (2.5, {}),
    "upright_while_tall": (1.5, {"height_low": 0.075, "height_high": 0.10}),
    "posture_stillness": (
        2.0,
        {
            "command_name": "base_velocity",
            "sit_height": SIT_HEIGHT,
            "stand_height": STAND_HEIGHT,
            "band_full": 0.012,
            "band_zero": 0.03,
            "vel_std": 0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    ),
    "posture_composite": (
        3.0,
        {
            "command_name": "base_velocity",
            "sit_joint_pos": SIT_JOINT_POS,
            "sit_height": SIT_HEIGHT,
            "stand_height": STAND_HEIGHT,
            "height_std": 0.03,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "head_std": 0.40,
            "head_command_name": "head_pose",
        },
    ),
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    "dof_pos_limits": (-1.0, {}),
    "action_rate_l2": (-0.1, {}),
    "joint_torque_rate_l2": (0.0, {}),
    "self_collisions": (-1.0, {"saturate": True}),
}
"""Upstream's twenty reward terms with their stage-0 weights (addendum section 4.4).

Four of them ship at a weight a curriculum then moves, so these are the *initial* values and the
stage tables below are as load-bearing as they are. ``dof_pos_limits`` is upstream's silently
inherited regularizer -- it survives only because it is not in the deletion list -- and
``self_collisions`` carries the port's own ``saturate``, which keeps the wide sensor on upstream's
0-or-1 scale.
"""

SELF_NEGATING_REWARDS = ["posture_pose_l1", "posture_height_l1", "descent_speed", "rise_speed", "gentle_motion"]
"""The five terms whose kernels already return a value at or below zero (addendum section 4.4).

They therefore carry **positive** weights, and upstream lost a full run to getting that backwards:
at negative weights the double negative made the three speed and shock penalties the largest
positive terms in the stack and trained a butt-hopping, crash-sitting policy.
"""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params)
    "time_out": (True, {}),
    "nan_state": (False, {"sensor_names": ("contact_forces",)}),
}
"""Upstream's terminations (addendum section 4.7).

There is no fall termination at all: a wobble during a transition has to play out so the policy pays
the impact and uprightness costs. ``nan_state``'s sensor list is the port's Ruling-2 deviation.
"""

EXPECTED_POSTURE_COMMAND = {
    "sit_prob": 0.5,
    "ramp_s": 2.0,
    "sit_height": SIT_HEIGHT,
    "stand_height": STAND_HEIGHT,
    "resampling_time_range": (3.5, 6.5),
    "heading_command": False,
}
"""The posture command's configuration (addendum section 4.3)."""

EXPECTED_GROUND_STATE_PARAMS = {
    "face_down_prob": 0.0,
    "face_up_prob": 0.0,
    "sitting_prob": 0.5,
    "standing_prob": 0.5,
    "sitting_z_range": (0.06, 0.075),
    "standing_z_range": (0.11, 0.12),
    "sitting_joint_pos": SIT_JOINT_POS,
    "sitting_joint_noise_std": 0.10,
    "sitting_tilt_max": math.radians(8.0),
}
"""The reset mixture (addendum section 4.6): half standing, half already seated, and no curriculum
touches it. Drawn independently of the posture request, which is what gives all four
(start state x request) combinations equal coverage."""

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "set_ground_state",
    "randomize_com",
    "randomize_head_com",
    "randomize_armature",
    "randomize_joint_friction",
]
"""The reset chain, in the order it fires.

This is behaviour, not housekeeping: ``set_ground_state`` overwrites the root height and orientation
``reset_base`` wrote, so it has to run after it, and only the horizontal spread survives.
"""

EXPECTED_CURRICULUM_TERMS = {
    "seated_head_jerk_prob",
    "seated_tip_range",
    "descent_speed_weight",
    "rise_speed_weight",
    "action_rate_weight",
    "torque_rate_weight",
    "com_range",
    "head_com_range",
    "head_pose_range",
    "push_magnitude",
}
"""Every curriculum term (addendum section 4.9). Nothing schedules the task itself -- both
transitions are rewarded from step 0 and the reset mixture never moves."""

EXPECTED_WEIGHT_STAGES = {
    # name: (weights, PPO iteration boundaries)
    "descent_speed_weight": ([10.0, 20.0], [0, 500]),
    "rise_speed_weight": ([0.0, 5.0, 10.0], [0, 1500, 2500]),
    "action_rate_weight": ([-0.1, -0.2, -0.4, -0.6, -0.8, -1.0], [0, 500, 750, 1000, 1250, 1500]),
    "torque_rate_weight": ([0.0, -5e-4, -1e-3], [0, 750, 1250]),
}
"""The four reward-weight ramps (addendum section 4.9)."""

EXPECTED_PUSH_STAGES = [
    (0, 0.0),
    (1000, 0.05),
    (1500, 0.10),
    (2000, 0.20),
    (2500, 0.30),
]
"""The push ramp: PPO iteration and the symmetric magnitude [m/s] it reaches there.

The latest in the family by a wide margin. Upstream's reason is measured: a push mid-descent before
the transition motions have consolidated made the policy unlearn sitting entirely.
"""

EXPECTED_ENTITY_SELECTIONS = {
    "rewards.neck_strain_leaned.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    # "<manager>.<term>.<param>": (kind, expected names, preserve_order)
    "rewards.posture_pose_legs.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.posture_pose_l1.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.head_pose_tracking.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.posture_composite.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.posture_composite.head_asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.descent_speed.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.rise_speed.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.gentle_motion.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.upright_linear.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.upright_while_tall.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.body_ang_vel.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.joint_torque_rate_l2.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "rewards.self_collisions.sensor_cfg": ("sensor", "self_collision", False),
    "events.foot_friction.asset_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, False),
    "events.mass_inertia.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_com.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_armature.asset_cfg": ("joint", [".*"], False),
    "events.set_ground_state.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
}
"""Every entity selection the recipe makes, outside the observation groups.

These are as load-bearing as the scalar parameters: the composite's two selections have to stay apart
-- one scores ten leg joints against the posture keyframe and the other four head joints against a
four-wide command -- and the torque-rate penalty sizes its state from the selection it is handed.
Isaac Lab resolves joints and bodies in USD order, which is neither upstream's nor this table's, so
``preserve_order`` is part of the contract wherever a term indexes a block positionally.

``events.randomize_head_com`` is absent because its selection is four *patterns* rather than four
names; ``events.randomize_joint_friction`` is absent because the term reads only the articulation
name.
"""

EXPECTED_OBSERVATION_SELECTIONS = {
    # "<group>.<term>.<param>": (kind, expected names, preserve_order)
    "policy.joint_pos.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "policy.joint_vel.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.joint_pos.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.joint_vel.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "critic.foot_air_time.sensor_cfg": ("sensor", EXPECTED_FOOT_BODY_NAMES, True),
    "critic.foot_contact.sensor_cfg": ("sensor", EXPECTED_FOOT_BODY_NAMES, True),
    "critic.foot_contact_forces.sensor_cfg": ("sensor", EXPECTED_FOOT_BODY_NAMES, True),
}
"""Entity selections inside the two observation groups, all of which are ordering contracts.

The joint blocks are the deployed vector's own layout, which the runtime on the robot rebuilds by
hand from its sensor reads; the foot blocks are three consecutive critic terms that must agree on
which column is the left foot. The table is *consumed* two-sidedly below, so an observation term that
gains or loses a selection fails rather than going unchecked.
"""

EXPECTED_NJMAX = 128
EXPECTED_NCONMAX = 32
"""The measured solver budget: 82 constraints and 28 contacts per environment at the peak."""

MEASURED_PEAK_CONSTRAINTS = 82
MEASURED_PEAK_CONTACTS = 28
"""The worst profiled peaks under random actions with the pushes forced to full magnitude.

Profiled at 256, 2048 and 4096 environments -- the last is this task's own training default. The
peak saturates from 2048 upward (28 then 27 contacts, 82 constraints at both), so these are the
maxima across all three sizes rather than any single run's. Logs:
``artifacts/microduck/profile_microduck_contacts_sitstand_{256,2048,4096}envs.log``.
"""

EXPECTED_SOLVER_ITERATIONS = 30
EXPECTED_SOLVER_LS_ITERATIONS = 50
"""Upstream's raised solver profile (addendum section 4.8), the one place in the family where the
iteration counts are not the template's 10 and 20."""


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
    """Drop the entity selections, which carry no upstream scalar to compare against.

    They are not left unchecked: :data:`EXPECTED_ENTITY_SELECTIONS` and
    :data:`EXPECTED_OBSERVATION_SELECTIONS` pin them by name, and the two selection tests below are
    what consume those tables.
    """
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


##
# The recipe
##


@pytest.mark.unit
def test_the_rewards_match_upstream_term_for_term():
    """Every slot upstream trains with is present, with its stage-0 weight and its parameters."""
    rewards = MicroDuckSitStandFlatEnvCfg().rewards

    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_five_self_negating_kernels_carry_positive_weights():
    """Upstream's most transferable lesson, and a full lost run: a double negative here pays for
    violence. The kernels return values at or below zero, so the weights have to be positive."""
    cfg = MicroDuckSitStandFlatEnvCfg()

    for name in SELF_NEGATING_REWARDS:
        weight = getattr(cfg.rewards, name).weight
        assert weight >= 0.0, name
    # ``rise_speed`` ships at exactly zero and is ramped up, so its schedule is what carries the sign
    assert all(stage["weight"] >= 0.0 for stage in cfg.curriculum.rise_speed_weight.params["weight_stages"])
    assert all(stage["weight"] >= 0.0 for stage in cfg.curriculum.descent_speed_weight.params["weight_stages"])
    # and the ordinary penalties keep the ordinary sign, so the test cannot pass by weighting
    # everything positively
    for name in ("body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2", "self_collisions"):
        assert getattr(cfg.rewards, name).weight <= 0.0, name


@pytest.mark.unit
def test_the_posture_stack_tracks_the_slewed_blend_except_for_the_rise_bootstrap():
    """The slew is the anti-crash mechanism; the bootstrap is upstream's one deliberate exception.

    Every posture kernel reads the command term's ``alpha``, which moves at a fixed rate, so being
    ahead of the ramp scores nothing. ``posture_rise_bootstrap`` reads the raw flag instead, so it
    switches off the instant a sit is requested and can never bid against the descent.
    """
    rewards = MicroDuckSitStandFlatEnvCfg().rewards
    posture_terms = [
        "posture_pose_legs",
        "posture_pose_l1",
        "posture_height",
        "posture_height_sharp",
        "posture_height_l1",
        "posture_stillness",
        "posture_composite",
        "rise_bootstrap",
    ]

    for name in posture_terms:
        assert getattr(rewards, name).params["command_name"] == "base_velocity", name
    assert rewards.rise_bootstrap.func is mdp.posture_rise_bootstrap
    # the bootstrap's ceiling sits above the standing rest, so the final centimetre still pays
    assert rewards.rise_bootstrap.params["max_height"] > STAND_HEIGHT
    # and its speed cap is the same one the rise penalty charges past, so the two cannot disagree
    assert rewards.rise_bootstrap.params["max_vz"] == pytest.approx(MICRODUCK_MAX_RISE_SPEED)
    assert rewards.rise_speed.params["max_up_vel"] == pytest.approx(MICRODUCK_MAX_RISE_SPEED)


@pytest.mark.unit
def test_the_ramp_is_slow_enough_to_stay_under_both_speed_caps():
    """Upstream's own sanity check on the ramp duration, and the reason the caps are backstops.

    A blend that traversed the two rest heights faster than the caps allow would make the tracked
    setpoint itself punishable, which would put the two halves of the recipe in opposition.
    """
    ramp_speed = (STAND_HEIGHT - SIT_HEIGHT) / MICRODUCK_POSTURE_RAMP_S

    assert ramp_speed < MICRODUCK_MAX_DESCENT_SPEED
    assert ramp_speed < MICRODUCK_MAX_RISE_SPEED


@pytest.mark.unit
def test_the_upright_gate_window_sits_between_the_two_rest_heights():
    """It has to: above it the tip-backward descent is denied, below it a seated trunk is fine."""
    params = MicroDuckSitStandFlatEnvCfg().rewards.upright_while_tall.params

    assert SIT_HEIGHT < params["height_low"] < params["height_high"] < STAND_HEIGHT


@pytest.mark.unit
def test_the_posture_command_is_the_sit_stand_flag():
    """The twist slot carries a binary posture request, resampled on the dwell time."""
    commands = MicroDuckSitStandFlatEnvCfg().commands

    command = commands.base_velocity
    assert command.class_type is mdp.SitStandCommand
    for key, value in EXPECTED_POSTURE_COMMAND.items():
        assert getattr(command, key) == pytest.approx(value), key
    # the flag's rest heights have to be the ones the rewards track, or the blend a spawn is seeded
    # with would not be the blend the rewards read
    assert command.sit_height == pytest.approx(SIT_HEIGHT)
    assert command.stand_height == pytest.approx(STAND_HEIGHT)
    # the head is commandable in both postures, and there is no body-pose command at all
    assert set(vars(commands)) == {"base_velocity", "head_pose"}


@pytest.mark.unit
def test_the_sitting_keyframe_is_the_stand_up_tasks_own():
    """Upstream keeps its three sit/stand environments' keyframes in sync by hand and says so in each
    of them, so the port imports one rather than carrying a second copy to drift."""
    assert pytest.approx(SIT_JOINT_POS) == MICRODUCK_SITTING_JOINT_POS
    # the keyframe folds the legs only: the neck and head stay at the stand pose, where the head-pose
    # command steers them
    assert set(SIT_JOINT_POS) < set(EXPECTED_LEG_JOINT_NAMES)
    # and every angle is inside the +/-1.5708 rad sagittal limits the MJCF authors on these joints,
    # so the target is reachable. Upstream's RollerCrouch keyframe is *not* -- it overshoots the
    # roller model's neck-pitch and knee ranges, which upstream-issue draft 018 records -- so this is
    # a bound worth asserting rather than assuming on any keyframe in this family.
    #
    # The limit is transcribed from the model rather than read out of it: this is a sim-free parity
    # test and the articulation is not loaded here. It therefore catches an out-of-range *sagittal*
    # angle but would not catch a joint whose own compiled range is narrower than the sagittal one --
    # which is the exact shape of the 018 defect. A model-backed check belongs in the integration
    # tier and is owed by whichever task actually reproduces that defect, not by this one.
    assert all(abs(angle) < 1.5708 for angle in SIT_JOINT_POS.values())


@pytest.mark.unit
def test_the_reset_spawns_both_postures_with_equal_probability():
    """Drawn independently of the posture request, which is what trains all four combinations."""
    params = _scalar_params(MicroDuckSitStandFlatEnvCfg().events.set_ground_state)

    assert set(params) == set(EXPECTED_GROUND_STATE_PARAMS)
    for key, value in EXPECTED_GROUND_STATE_PARAMS.items():
        assert params[key] == pytest.approx(value), key
    # the two live buckets partition the resets; the three ground poses are switched off, so their
    # bands are left unconfigured rather than filled with values nothing samples
    assert params["sitting_prob"] + params["standing_prob"] == pytest.approx(1.0)
    assert "prone_z_range" not in params and "crouch_z_range" not in params
    # the seated band starts at the measured rest height, so a seated spawn settles rather than drops
    assert params["sitting_z_range"][0] == pytest.approx(SIT_HEIGHT)


@pytest.mark.unit
def test_the_ground_state_reset_runs_after_the_resets_it_overwrites():
    """Isaac Lab fires reset events in declaration order, and this order is the spawn distribution."""
    events = MicroDuckSitStandFlatEnvCfg().events

    reset_terms = [name for name, term in vars(events).items() if getattr(term, "mode", None) == "reset"]
    assert reset_terms == EXPECTED_RESET_EVENT_ORDER


@pytest.mark.unit
def test_there_is_no_fall_termination():
    """A wobble during a transition has to play out, so the policy pays the impact and uprightness
    costs instead of having the episode truncated under it."""
    terminations = MicroDuckSitStandFlatEnvCfg().terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        assert _scalar_params(term) == params, name


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """The schedule is what makes the rise discoverable; a boundary that moved would be another run."""
    curriculum = MicroDuckSitStandFlatEnvCfg().curriculum

    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS
    for name, (weights, iterations) in EXPECTED_WEIGHT_STAGES.items():
        stages = getattr(curriculum, name).params["weight_stages"]
        assert [stage["weight"] for stage in stages] == pytest.approx(weights), name
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name


@pytest.mark.unit
def test_the_descent_cap_is_live_from_the_start_and_the_rise_cap_is_not():
    """Upstream's attempt-tax lesson, and the ordering *is* the design.

    The sit is the easy direction and a crash-sit has to be net-negative before it can be discovered,
    so its cap is at full strength from step 0. The rise cap waits: a motion tax live while the rise
    is still being explored makes every attempt net-negative and the skill is never found. Upstream
    moved it from 750/1250 to 1500/2500 after a run stalled in a half-finished head-down fold.
    """
    curriculum = MicroDuckSitStandFlatEnvCfg().curriculum

    descent = curriculum.descent_speed_weight.params["weight_stages"]
    rise = curriculum.rise_speed_weight.params["weight_stages"]
    assert descent[0]["weight"] > 0.0
    assert rise[0]["weight"] == pytest.approx(0.0)
    # the rise cap arrives after the descent cap has already tightened
    assert rise[1]["step"] > descent[-1]["step"]
    # and after the action-rate ramp has finished, so the two taxes do not land together
    assert rise[1]["step"] >= curriculum.action_rate_weight.params["weight_stages"][-1]["step"]


@pytest.mark.unit
def test_the_push_ramp_is_the_latest_in_the_family():
    """A push mid-descent before the transitions have consolidated made upstream's policy unlearn
    sitting entirely, so nothing is pushed at all until iteration 1000."""
    curriculum = MicroDuckSitStandFlatEnvCfg().curriculum

    stages = curriculum.push_magnitude.params["param_stages"]
    assert curriculum.push_magnitude.params["event_name"] == "push_robot"
    # upstream drives this one with an exclusive step comparison where the event-parameter helper
    # uses an inclusive one, and the inconsistency is reproduced rather than smoothed over
    assert curriculum.push_magnitude.params["inclusive"] is False
    assert len(stages) == len(EXPECTED_PUSH_STAGES)
    for stage, (iteration, magnitude) in zip(stages, EXPECTED_PUSH_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        for axis in ("x", "y"):
            assert stage["params"]["velocity_range"][axis] == pytest.approx((-magnitude, magnitude))
    # nothing is pushed while the transitions are being discovered. The walking task pushes at full
    # strength from step 0 and carries no ramp at all, so the stand-up task -- the other one that
    # starts its robot on the floor -- is what "latest" is measured against: it is already at full
    # magnitude by iteration 1000, where this ramp has not yet left zero.
    assert EXPECTED_PUSH_STAGES[0][1] == pytest.approx(0.0)
    standup_end = MicroDuckStandUpFlatEnvCfg().curriculum.push_magnitude.params["param_stages"][-1]["step"]
    assert stages[1]["step"] >= standup_end
    assert stages[-1]["step"] > standup_end


@pytest.mark.unit
def test_the_terms_select_the_joints_bodies_and_sensors_upstream_measures():
    """A term that measures the wrong entity is as wrong as one carrying the wrong weight."""
    cfg = MicroDuckSitStandFlatEnvCfg()

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
        if kind == "sensor":
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
def test_both_observation_groups_read_the_servos_and_the_feet_in_the_deploy_order():
    """Isaac Lab resolves joints and bodies in USD order; the deployed vector is in MJCF order.

    Widths are order-blind, so the integration test that measures 61 and 74 cannot see a joint block
    that lost ``preserve_order`` and now reports the servos in the articulation's order -- which
    would produce a policy that runs on the robot against the wrong columns. This is the assertion
    that catches it.
    """
    observations = MicroDuckSitStandFlatEnvCfg().observations
    groups = {"policy": _observation_terms(observations.policy), "critic": _observation_terms(observations.critic)}

    # two-sided over the observation terms that carry a selection at all, so a term that gains or
    # loses one fails rather than going unchecked
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
        if kind == "joint":
            assert entity_cfg.name == "robot", path
            assert entity_cfg.joint_names == expected, path
        else:
            assert entity_cfg.name == "contact_forces", path
            assert entity_cfg.body_names == expected, path
        assert entity_cfg.preserve_order is preserve_order, path

    # the head rewards index their command's columns positionally, so their joint order is the
    # command's; the observation joint order is the deployed vector's. Different contracts, same
    # articulation, and the head block of one is the other's.
    assert EXPECTED_SERVO_JOINT_NAMES[5:9] == EXPECTED_HEAD_JOINT_NAMES


@pytest.mark.unit
def test_the_actor_observation_layout_is_the_shared_deploy_contract():
    """One runtime on the robot feeds every MicroDuck policy from the same 61-wide buffer."""
    terms = _observation_terms(MicroDuckSitStandFlatEnvCfg().observations.policy)

    assert list(terms) == [name for name, _ in ACTOR_OBSERVATION_TERMS]
    assert sum(width for _, width in ACTOR_OBSERVATION_TERMS) == ACTOR_OBSERVATION_DIM
    # the body slot is a *shape* placeholder for a runtime hot-swap, deliberately constant zero
    # rather than a live tiny range, and the head slot is a real command
    assert terms["body_pose_commands"].func is mdp.zero_command_padding
    assert terms["head_pose_commands"].params["command_name"] == "head_pose"
    # the posture flag rides in the twist slot, which keeps its three columns
    assert terms["velocity_commands"].params["command_name"] == "base_velocity"


@pytest.mark.unit
def test_the_critic_group_drops_the_foot_height_columns():
    """Upstream deletes them here: no height sensor in this scene and no foot-height reward."""
    terms = _observation_terms(MicroDuckSitStandFlatEnvCfg().observations.critic)

    assert list(terms) == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert sum(width for _, width in CRITIC_OBSERVATION_TERMS) == CRITIC_OBSERVATION_DIM
    assert "foot_height" not in terms
    # the NaN-guarded variants, which is the port's family norm rather than upstream's choice here
    assert terms["foot_air_time"].func is mdp.foot_air_time_safe
    assert terms["foot_contact_forces"].func is mdp.foot_contact_forces_safe


@pytest.mark.unit
def test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link():
    """``bearing_roll`` is the right hip-yaw link; upstream lists it among the head bodies in error."""
    body_names = MicroDuckSitStandFlatEnvCfg().events.randomize_head_com.params["asset_cfg"].body_names

    assert "bearing_roll" not in body_names


@pytest.mark.unit
def test_the_task_runs_the_all_collisions_robot_on_a_plane():
    """The seated pose rests the trunk on the floor, which the walking model has no collider for."""
    scene = MicroDuckSitStandFlatEnvCfg().scene

    assert scene.robot.spawn.usd_path.endswith("microduck_allcollisions.usd")
    assert scene.terrain.terrain_type == "plane"
    assert scene.terrain.terrain_generator is None


@pytest.mark.unit
def test_the_self_collision_sensor_senses_every_collider_against_every_other():
    """Upstream's trunk-subtree-against-itself sensor, and there is deliberately no head-impact one."""
    cfg = MicroDuckSitStandFlatEnvCfg()

    sensor = cfg.scene.self_collision
    assert sensor.sensor_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    assert sensor.filter_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    # sensing both sides of a pair reports one contact twice, so the cost saturates to upstream's
    # single-slot 0-or-1 signal rather than becoming a per-collider tariff
    assert cfg.rewards.self_collisions.params["saturate"] is True
    # using the head as a third support point is allowed here, so there is no head sensor to add one
    assert set(vars(cfg.scene)) & {"head_contact", "head_ground_contact"} == set()


@pytest.mark.unit
def test_the_solver_profile_is_upstreams_and_the_budget_is_measured():
    """Upstream raises the iteration counts here and nowhere else, and it says why: the seated pose
    put trunk, legs and head in close contact at once and the contact solve diverged into NaN."""
    solver = MicroDuckSitStandFlatEnvCfg().sim.physics.newton_mjwarp.solver_cfg
    velocity_solver = MicroDuckVelocityFlatEnvCfg().sim.physics.newton_mjwarp.solver_cfg

    assert solver.iterations == EXPECTED_SOLVER_ITERATIONS
    assert solver.ls_iterations == EXPECTED_SOLVER_LS_ITERATIONS
    assert solver.iterations > velocity_solver.iterations
    assert solver.njmax == EXPECTED_NJMAX
    assert solver.nconmax == EXPECTED_NCONMAX
    # a floor under the measurement, so a later retune cannot silently drop below it
    assert solver.njmax >= MEASURED_PEAK_CONSTRAINTS
    assert solver.nconmax >= MEASURED_PEAK_CONTACTS
    # and above the walking task's, which is what says the model swap was accounted for
    assert solver.nconmax > velocity_solver.nconmax


@pytest.mark.unit
def test_the_episode_is_sized_for_two_or_three_posture_segments():
    """Upstream sizes the episode from the dwell time, so an episode carries a transition rather
    than being one long hold."""
    cfg = MicroDuckSitStandFlatEnvCfg()

    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.episode_length_s == pytest.approx(12.0)
    shortest_dwell, longest_dwell = EXPECTED_POSTURE_COMMAND["resampling_time_range"]
    # every dwell clears a gentle transition plus a stretch of rest, so "arrive, then hold still" is
    # trained on every segment rather than only on the lucky long ones
    assert shortest_dwell > MICRODUCK_POSTURE_RAMP_S
    # two segments fit even at the longest dwell, and three at the shortest -- upstream's "2-3"
    assert 2.0 * longest_dwell >= cfg.episode_length_s >= 3.0 * shortest_dwell
    # 600 control steps at the 50 Hz the deployed policy runs at
    assert round(cfg.episode_length_s / (cfg.decimation * cfg.sim.dt)) == 600


@pytest.mark.unit
def test_the_runner_differs_from_the_velocity_one_in_two_fields():
    """Upstream shares the network, the optimizer and the rollout across the whole family."""
    from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg

    runner, velocity = MicroDuckSitStandPPORunnerCfg(), MicroDuckPPORunnerCfg()

    assert runner.experiment_name == "microduck_sitstand"
    assert runner.max_iterations == 15000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION == velocity.num_steps_per_env
    assert runner.save_interval == velocity.save_interval == 250
    assert runner.actor.hidden_dims == velocity.actor.hidden_dims == [512, 256, 128]
    assert runner.critic.hidden_dims == velocity.critic.hidden_dims
    assert runner.obs_groups == velocity.obs_groups
    for field in ("clip_param", "entropy_coef", "learning_rate", "gamma", "lam", "desired_kl"):
        assert getattr(runner.algorithm, field) == getattr(velocity.algorithm, field), field
    # symmetry stays off, as it does on every task in this batch
    assert getattr(runner.algorithm, "symmetry_cfg", None) is None


##
# Simulator-backed acceptance
##


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_observation_groups_are_the_widths_their_contracts_name():
    """The actor width is a deploy contract; the critic width is measured against the term table."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        observations, _ = env.reset()

        assert observations["policy"].shape[1] == ACTOR_OBSERVATION_DIM
        assert observations["critic"].shape[1] == CRITIC_OBSERVATION_DIM
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_reset_seeds_the_posture_blend_from_the_pose_the_robot_actually_spawned_in():
    """The one behaviour upstream could not get and had to work around.

    Its command manager resets *before* the event that teleports the robot, so it re-initializes the
    blend from inside ``compute`` guarded on the episode counter, and its own comment says a port
    that moved this into a reset hook would drag seated spawns upward. Isaac Lab fires reset events
    first, so the hook is correct here -- and this is the test that says so, because getting it wrong
    is silent: the episode still runs, the seated half is just rewarded toward standing.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=32)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        command = unwrapped.command_manager.get_term("base_velocity")
        height = robot.data.root_link_pos_w.torch[:, 2] - unwrapped.scene.env_origins[:, 2]
        blend = command.alpha

        # both buckets are sampled, and each spawn lands inside its own configured band
        seated = height < 0.09
        assert bool(seated.any()) and bool((~seated).any()), "the reset collapsed onto one posture"
        assert float(height[seated].max()) <= EXPECTED_GROUND_STATE_PARAMS["sitting_z_range"][1] + 1e-3
        assert float(height[~seated].min()) >= EXPECTED_GROUND_STATE_PARAMS["standing_z_range"][0] - 1e-3

        # The blend is seeded from the height rather than from the flag, and this is the whole
        # relation rather than its two ends: the expected value is worked out here from the two rest
        # heights, so a seed that read the flag, or the pre-teleport height, fails.
        expected = ((STAND_HEIGHT - height) / (STAND_HEIGHT - SIT_HEIGHT)).clamp(0.0, 1.0)
        torch.testing.assert_close(blend, expected, atol=1e-5, rtol=0.0)
        # which puts seated spawns at the sit end and standing ones essentially at the stand end
        assert float(blend[seated].min()) > 0.7
        assert float(blend[~seated].max()) < 0.1
        # and it is independent of the request, which is what trains "hold what you are doing"
        flag = unwrapped.command_manager.get_command("base_velocity")[:, 0]
        assert bool((flag[seated] == 0.0).any()), "no seated spawn was asked to stand up"
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_posture_blend_slews_at_the_configured_rate_and_the_observation_does_not():
    """The policy sees the raw flip; the rewards see a two-second glide. That gap is the task."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        command = unwrapped.command_manager.get_term("base_velocity")
        # force a full stand-to-sit request on every environment, from a settled stand blend
        command.vel_command_b[:] = 0.0
        command.vel_command_b[:, 0] = 1.0
        command._alpha[:] = 0.0

        action = torch.zeros((4, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
        observations, *_ = env.step(action)
        per_step = command.alpha.clone()

        assert float(per_step.max()) == pytest.approx(unwrapped.step_dt / MICRODUCK_POSTURE_RAMP_S, rel=1e-4)
        # the observation carries the flag, not the blend, so it is already at the requested end
        flag_column = sum(width for _, width in ACTOR_OBSERVATION_TERMS[:5])
        assert observations["policy"][:, flag_column].min().item() == pytest.approx(1.0)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
