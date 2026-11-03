# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke, recipe-parity and acceptance tests for the contributed MicroDuck forward-roll environment.

Simulator-backed checks retrieve the prebuilt USD through Isaac Lab's native asset cache.
Recipe checks do not need an asset; source fidelity comparisons use optional read-only fixtures.

The parity tests need neither the asset nor the simulator: they read the assembled configuration and
compare it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference_tasks2.md`` section 4. The expected values are spelled out
rather than imported from the configuration under test, so that a drifting value fails rather than
agrees with itself.

The two integration tests at the end are the **acceptance tests** the extraction asks for
(``upstream_reference_tasks2.md`` section 7.2). Upstream's ``FULL_COLLISION`` configuration reads as
if it disabled every unnamed collider and in fact disables none, so a port that implemented it
faithfully would silently delete the three head shells -- and the failure would be almost invisible,
because the mid-roll spawn bucket is *granted* the head latch and would keep collecting the
completion-gated rewards while the standing bucket could never earn one. The tests therefore assert
the two links of that chain directly: that a head plant produces head-ground contact, and that a
standing spawn can earn the latch and open the completion gate.
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
from tensordict import TensorDict

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ALLCOLLISIONS_USD_PATH
from isaaclab_tasks.contrib.microduck.mdp.symmetry import compute_roulade_symmetric_states
from isaaclab_tasks.contrib.microduck.roulade.agents.rsl_rl_ppo_cfg import MicroDuckRouladePPORunnerCfg
from isaaclab_tasks.contrib.microduck.roulade.roulade_env_cfg import (
    MICRODUCK_LANDING_GATE,
    MICRODUCK_RISE_GATE,
    MICRODUCK_STAND_HEIGHT,
    MICRODUCK_TUCK_JOINT_POS,
    MicroDuckRouladeFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_allcollisions_usd = pytest.mark.usefixtures("microduck_allcollisions_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-Roulade-Flat-MicroDuck"

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

Addendum section 6.1: all of upstream's MicroDuck tasks present the same 61-wide vector, because one
runtime on the robot feeds every policy from the same buffer. This task has neither a head-pose nor
a body-pose command, so the last two blocks are zero padding rather than commands -- but they are
still there, and still that wide.
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
"""The privileged critic layout (addendum section 6.2), which is not a deploy contract."""

CRITIC_OBSERVATION_DIM = 74
"""Critic observation width, measured from the assembled group.

The extraction derives 74 by hand from upstream's term list (addendum section 6.2); this pins the
number the port actually produces against it.
"""

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, upstream's ``num_steps_per_env``."""

EPISODE_CONTROL_STEPS = 250
"""Control steps in an episode: 5 s at 50 Hz. Every one of them is run -- there is no failure
termination on this task, because falling over is the task."""

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
"""The 10 leg joints -- upstream's ``_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]``, which is
what the landing composite scores against the stand pose."""

EXPECTED_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot bodies in upstream's ``[left, right]`` site order."""

EXPECTED_TRUNK_BODY_NAMES = ["trunk_base"]
"""The body upstream measures the trunk height, tilt and angular velocity on, and randomizes."""

EXPECTED_HEAD_SHELL_BODY_NAMES = ["jaw_soft"]
"""The body carrying the three head collision shells, which the roll pivots on."""

EXPECTED_HEAD_BODY_COUNT = 4
"""Bodies the head centre-of-mass randomization perturbs. Upstream lists five, the fifth being
``bearing_roll`` -- the right hip-yaw link, which its own comment admits is listed here in error."""

EXPECTED_ENTITY_SELECTIONS = {
    # "<manager>.<term>.<param>": (kind, expected names, preserve_order)
    "rewards.roulade_progress.asset_cfg": ("body", EXPECTED_HEAD_SHELL_BODY_NAMES, False),
    "rewards.roulade_progress.support_sensor_cfg": ("sensor", "robot_ground_contact", False),
    "rewards.roulade_progress.head_sensor_cfg": ("sensor", "head_ground_contact", False),
    "rewards.roulade_head_pivot.asset_cfg": ("body", EXPECTED_HEAD_SHELL_BODY_NAMES, False),
    "rewards.roulade_head_pivot.sensor_cfg": ("sensor", "head_ground_contact", False),
    "rewards.roulade_landing_composite.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.joint_torque_rate_l2.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "rewards.body_ang_vel.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.arrival_damping.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.self_collisions.sensor_cfg": ("sensor", "self_collision", False),
    "events.foot_friction.asset_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, False),
    "events.mass_inertia.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_com.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_armature.asset_cfg": ("joint", [".*"], False),
    "events.set_roulade_state.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
}
"""Every entity selection the roll recipe makes, outside the observation groups.

These are as load-bearing as the scalar parameters ``_scalar_params`` compares, and two of them are
the task itself: the head terms measure the tuck on ``jaw_soft``, and the two contact sensors are
what the rotation accumulator gates on. Upstream hard-codes both sensor names in its ``mdp.py``;
naming them here is the port's equivalent, so a renamed sensor fails loudly rather than silently
switching a gate off.

``events.randomize_head_com`` is absent because its selection is four *patterns* rather than four
names; ``test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link`` pins it instead.
``events.randomize_joint_friction`` is absent because the term reads only the articulation name.
"""

EXPECTED_OBSERVATION_SELECTIONS = {
    # "<group>.<term>": (kind, expected names)
    "policy.joint_pos": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "policy.joint_vel": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "critic.joint_pos": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "critic.joint_vel": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "critic.foot_air_time": ("body", EXPECTED_FOOT_BODY_NAMES),
    "critic.foot_contact": ("body", EXPECTED_FOOT_BODY_NAMES),
    "critic.foot_contact_forces": ("body", EXPECTED_FOOT_BODY_NAMES),
}
"""Entity selections inside the two observation groups, all of which are ordering contracts."""


def _entity_cfg_of(term_cfg, key: str) -> SceneEntityCfg:
    """Fetch a term's entity selection, looking inside a delayed term's wrapped parameters."""
    if key in term_cfg.params:
        return term_cfg.params[key]
    return term_cfg.params["term_params"][key]


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which carry no upstream scalar to compare against.

    They are not left unchecked: :data:`EXPECTED_ENTITY_SELECTIONS` and
    :data:`EXPECTED_OBSERVATION_SELECTIONS` pin every one of them by name.
    """
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


##
# Recipe parity against upstream (addendum section 4)
##

_LANDING_GATE_PARAMS = {"gate_lo": math.radians(260.0), "gate_hi": math.radians(330.0)}
_RISE_GATE_PARAMS = {"gate_lo": math.radians(180.0), "gate_hi": math.radians(260.0)}

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "roulade_progress": (8.0, {"target_angle": 2.0 * math.pi, "max_paid_rate": 5.0}),
    "roulade_overspeed": (-0.1, {"omega_max": 7.0}),
    "roulade_head_pivot": (
        0.5,
        {"angle_lo": math.radians(30.0), "angle_hi": math.radians(240.0), "rate_norm": 2.0},
    ),
    "roulade_landing_composite": (
        4.0,
        {
            "target_height": 0.115,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            **_LANDING_GATE_PARAMS,
        },
    ),
    "roulade_upright_after_roll": (1.5, dict(_LANDING_GATE_PARAMS)),
    "roulade_height_after_roll": (1.0, {"target_height": 0.115, "std": 0.04, **_LANDING_GATE_PARAMS}),
    "roulade_landing_sharp": (
        2.0,
        {"target_height": 0.115, "height_std": 0.015, "upright_std": 0.3, **_LANDING_GATE_PARAMS},
    ),
    "roulade_stand_tax": (5.0, {"target_height": 0.115, **_LANDING_GATE_PARAMS}),
    "roulade_rise_velocity": (0.75, {"max_height": 0.125, **_RISE_GATE_PARAMS}),
    "roulade_sagittal": (-0.1, {}),
    "roulade_lateral_vel": (-0.5, {}),
    "roulade_flatness": (-0.5, {}),
    "action_rate_l2": (-0.1, {}),
    "joint_torque_rate_l2": (0.0, {}),
    "body_ang_vel": (-0.002, {}),
    "angular_momentum": (-0.001, {}),
    "dof_pos_limits": (-1.0, {}),
    "arrival_damping": (
        0.0,
        {"height_low": 0.09, "height_high": 0.11, "tilt_full_deg": 20.0, "tilt_zero_deg": 45.0},
    ),
    "gentle_landing": (0.002, {}),
    "self_collisions": (-0.1, {"saturate": True}),
}
"""Upstream's reward recipe (addendum section 4.4), keyed by term name, at its initial weights.

Four of these are ramped by curricula and are listed at the weight the configuration ships, not at
the one they reach: ``action_rate_l2``, ``arrival_damping``, ``joint_torque_rate_l2`` and
``gentle_landing``.

``dof_pos_limits`` is upstream's silently inherited regularizer -- never mentioned in the roll
configuration, surviving only because it is not in the deletion list (addendum section 7.13).
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
    "reset_robot_joints": ("reset", {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}),
    "set_roulade_state": (
        "reset",
        {
            "standing_prob": 0.5,
            "midroll_prob": 0.5,
            "standing_z_range": (0.11, 0.12),
            "standing_tilt_max": math.radians(5.0),
            "forward_vel_range": (0.0, 0.0),
            "midroll_pitch_range": (math.radians(50.0), math.radians(340.0)),
            "midroll_z_range": (0.05, 0.10),
            "midroll_omega_range": (0.0, 3.0),
            "tuck_joint_pos": MICRODUCK_TUCK_JOINT_POS,
            "tuck_factor_range": (0.3, 1.0),
            "joint_noise_std": 0.08,
        },
    ),
    "randomize_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_head_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_armature": ("reset", {"armature_distribution_params": (0.9, 1.1), "operation": "scale"}),
    "randomize_joint_friction": ("reset", {"scale_range": (0.9, 1.1)}),
}
"""Upstream's event suite (addendum section 4.7), keyed by term name.

It is the stand-up task's minus ``push_robot``, which upstream deletes here because *a push mid-roll
is incoherent*. Upstream's ``base_com`` is absent because it selects zero bodies there; its
``expand_bam_friction_fields`` and ``reset_action_history`` are absent because Isaac Lab's actuator
storage and action manager already do their jobs; ``randomize_motor_gains`` is absent because
upstream ships it disabled.
"""

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "set_roulade_state",
    "randomize_com",
    "randomize_head_com",
    "randomize_armature",
    "randomize_joint_friction",
]
"""Upstream's reset chain, in the order it fires (addendum section 4.5).

This is behaviour, not housekeeping: ``set_roulade_state`` overwrites the root pose and velocity
``reset_base`` wrote, and its mid-roll tuck lerps *from* the pose the joint reset wrote. Upstream's
own configuration comment states the second dependency explicitly.
"""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params)
    "time_out": (True, {}),
    "nan_state": (False, {"sensor_names": ("contact_forces",)}),
}
"""Upstream's terminations (addendum section 4.6), keyed by term name.

There is no failure termination: the velocity task's tilt check is deleted because falling over is
the task, and the inherited terrain-bounds check is all-false on a ground plane (section 7.24).

The sensor list is a **deliberate deviation**: upstream leaves it empty here and names its foot
sensor only on the stand-up task, which the extraction reads as drift rather than design and
recommends closing everywhere in the port (section 7.9).
"""

EXPECTED_CURRICULUM_TERMS = {
    "roulade_spawn_mix",
    "com_range",
    "head_com_range",
    "action_rate_weight",
    "arrival_damping_weight",
    "torque_rate_weight",
    "gentle_landing_weight",
}
"""Upstream's curriculum term names (addendum section 4.8).

``terrain_levels`` and ``command_vel`` are absent because upstream deletes both, and there is no
head- or body-pose range schedule because there are no such commands.
"""

EXPECTED_WEIGHT_STAGES = {
    "action_rate_weight": ([-0.1, -0.2, -0.4], [0, 1500, 3000]),
    "arrival_damping_weight": ([0.0, -0.025, -0.05], [0, 2500, 3500]),
    "torque_rate_weight": ([0.0, -5e-4, -1e-3], [0, 2500, 3500]),
    "gentle_landing_weight": ([0.002, 0.005], [0, 2500]),
}
"""Upstream's reward-weight ramps: payloads and PPO-iteration boundaries."""

EXPECTED_RANGE_STAGES = {
    "com_range": ([0.003, 0.005, 0.01, 0.015], [0, 500, 1000, 1500]),
    "head_com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
}
"""Upstream's centre-of-mass range ramps, matched to the stand-up and velocity tasks'."""

EXPECTED_SPAWN_MIX_STAGES = [
    (0, {"standing_prob": 0.50, "midroll_prob": 0.50}),
    (3000, {"standing_prob": 0.65, "midroll_prob": 0.35}),
    (6000, {"standing_prob": 0.80, "midroll_prob": 0.20}),
]
"""Upstream's reverse curriculum: half the episodes start mid-roll, falling to a fifth."""

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
    rewards = MicroDuckRouladeFlatEnvCfg().rewards

    # two-sided, so a stand-up or walking term left behind by the port also fails
    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_progress_term_is_evaluated_before_the_terms_that_read_its_frontier():
    """Only this term advances the accumulator, and Isaac Lab evaluates rewards in declaration order."""
    rewards = MicroDuckRouladeFlatEnvCfg().rewards

    names = list(vars(rewards))
    readers = [
        "roulade_head_pivot",
        "roulade_landing_composite",
        "roulade_upright_after_roll",
        "roulade_height_after_roll",
        "roulade_landing_sharp",
        "roulade_stand_tax",
        "roulade_rise_velocity",
    ]
    assert names[0] == "roulade_progress"
    for name in readers:
        assert names.index(name) > names.index("roulade_progress"), name
    # and it is the only term wired to both sensors, which is what makes it the one that integrates
    assert set(rewards.roulade_progress.params) >= {"support_sensor_cfg", "head_sensor_cfg"}


@pytest.mark.unit
def test_the_self_negating_reward_terms_carry_positive_weights():
    """Two terms return a non-positive value, so a negative weight would pay for what it prices."""
    cfg = MicroDuckRouladeFlatEnvCfg()

    assert cfg.rewards.roulade_stand_tax.weight > 0.0
    assert cfg.rewards.gentle_landing.weight > 0.0
    stages = cfg.curriculum.gentle_landing_weight.params["weight_stages"]
    assert all(stage["weight"] >= 0.0 for stage in stages)


@pytest.mark.unit
def test_nothing_in_the_recipe_opposes_the_flip():
    """Upstream's core lesson on this task: an always-on upright or motion tax prevents discovery."""
    cfg = MicroDuckRouladeFlatEnvCfg()

    # the walking and stand-up attractors that would fight a deliberate fall
    for name in ("upright", "upright_linear", "upright_sharp", "height_stand", "height_stand_l1", "pose_stand_legs"):
        assert not hasattr(cfg.rewards, name), name
    # the trunk rotation regularizer survives, but 25 times lighter than the stand-up task's -0.05,
    # because on this task the trunk rotation *is* the manoeuvre (addendum section 7.23)
    assert cfg.rewards.body_ang_vel.weight == pytest.approx(-0.002)
    # every landing reward is gated, and the gate cannot open before 260 degrees of rotation
    for name in (
        "roulade_landing_composite",
        "roulade_upright_after_roll",
        "roulade_height_after_roll",
        "roulade_landing_sharp",
        "roulade_stand_tax",
    ):
        assert getattr(cfg.rewards, name).params["gate_lo"] == pytest.approx(MICRODUCK_LANDING_GATE[0]), name
    # the exit rise opens one quadrant earlier, on the back
    assert cfg.rewards.roulade_rise_velocity.params["gate_lo"] == pytest.approx(MICRODUCK_RISE_GATE[0])
    assert MICRODUCK_RISE_GATE[1] == pytest.approx(MICRODUCK_LANDING_GATE[0])
    # a tucked roll needs body-on-body contact, so self-collision is priced, not forbidden
    assert cfg.rewards.self_collisions.weight == pytest.approx(-0.1)


@pytest.mark.unit
def test_the_event_suite_matches_upstream_term_for_term():
    """Every event upstream fires is present, in its mode, with its ranges -- and the push is gone."""
    events = MicroDuckRouladeFlatEnvCfg().events

    assert set(vars(events)) == set(EXPECTED_EVENTS)
    assert not hasattr(events, "push_robot")
    for name, (mode, params) in EXPECTED_EVENTS.items():
        term = getattr(events, name)
        assert term.mode == mode, name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == value, f"{name}.{key}"


@pytest.mark.unit
def test_the_roll_state_reset_runs_after_the_resets_it_overwrites():
    """Isaac Lab fires reset events in declaration order, and this order is the spawn distribution."""
    events = MicroDuckRouladeFlatEnvCfg().events

    reset_terms = [name for name, term in vars(events).items() if getattr(term, "mode", None) == "reset"]
    assert reset_terms == EXPECTED_RESET_EVENT_ORDER


@pytest.mark.unit
def test_the_tuck_anchor_folds_the_legs_and_tucks_the_chin():
    """The chin tuck is what puts the flat top of the head on the floor, so the latch can fire."""
    params = MicroDuckRouladeFlatEnvCfg().events.set_roulade_state.params

    # the configuration deep-copies its mutable defaults, so this is equality rather than identity
    assert params["tuck_joint_pos"] == MICRODUCK_TUCK_JOINT_POS
    assert set(MICRODUCK_TUCK_JOINT_POS) == {
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "neck_pitch",
        "head_pitch",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    }
    # mirrored left and right, as the two legs' joint axes are
    for joint in ("hip_pitch", "knee", "ankle"):
        left = MICRODUCK_TUCK_JOINT_POS[f"left_{joint}"]
        right = MICRODUCK_TUCK_JOINT_POS[f"right_{joint}"]
        assert left == pytest.approx(-right), joint
    # the chin tuck: the neck folds one way and the head the other, which rolls the flat top down
    assert MICRODUCK_TUCK_JOINT_POS["neck_pitch"] == pytest.approx(-1.0)
    assert MICRODUCK_TUCK_JOINT_POS["head_pitch"] == pytest.approx(1.0)
    # never fully absent, so every mid-roll spawn demonstrates some of the tucked configuration
    assert params["tuck_factor_range"][0] > 0.0
    assert params["tuck_factor_range"][1] == pytest.approx(1.0)


@pytest.mark.unit
def test_the_mid_roll_spawn_covers_the_whole_second_half_of_the_roll():
    """Upstream widened the band to 340 degrees so the crouch-to-stand last mile is spawned into."""
    params = MicroDuckRouladeFlatEnvCfg().events.set_roulade_state.params

    pitch_lo, pitch_hi = params["midroll_pitch_range"]
    assert pitch_lo == pytest.approx(math.radians(50.0))
    assert pitch_hi == pytest.approx(math.radians(340.0))
    # spawns past the landing gate open it at birth, which is where the dense landing data comes from
    assert pitch_hi > MICRODUCK_LANDING_GATE[1]
    # low and rotating, unlike the standing bucket
    assert params["midroll_z_range"] == (0.05, 0.10)
    assert params["midroll_omega_range"] == (0.0, 3.0)
    # the standing band brackets the height the landing rewards target
    low, high = params["standing_z_range"]
    assert low <= MICRODUCK_STAND_HEIGHT <= high
    # upstream's élan hook, shipped disabled
    assert params["forward_vel_range"] == (0.0, 0.0)


@pytest.mark.unit
def test_the_terms_select_the_joints_bodies_and_sensors_upstream_measures():
    """A term that measures the wrong body is as wrong as one carrying the wrong weight."""
    cfg = MicroDuckRouladeFlatEnvCfg()

    # two-sided over the terms that carry a selection at all, so a term that gains or loses one fails
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
    """Isaac Lab resolves joints and bodies in USD order; the deployed vector is in MJCF order."""
    observations = MicroDuckRouladeFlatEnvCfg().observations
    groups = {"policy": _observation_terms(observations.policy), "critic": _observation_terms(observations.critic)}

    for path, (kind, expected) in EXPECTED_OBSERVATION_SELECTIONS.items():
        group, term_name = path.split(".")
        term = groups[group][term_name]
        # the delayed actor terms hold their selection inside the wrapped term's parameters
        entity_cfg = _entity_cfg_of(term, "asset_cfg" if kind == "joint" else "sensor_cfg")
        if kind == "joint":
            assert entity_cfg.name == "robot", path
            assert entity_cfg.joint_names == expected, path
        else:
            assert entity_cfg.name == "contact_forces", path
            assert entity_cfg.body_names == expected, path
        assert entity_cfg.preserve_order, path


@pytest.mark.unit
def test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link():
    """``bearing_roll`` is the right hip-yaw link; upstream lists it among the head bodies in error."""
    body_names = MicroDuckRouladeFlatEnvCfg().events.randomize_head_com.params["asset_cfg"].body_names

    assert "bearing_roll" not in body_names
    assert len(body_names) == EXPECTED_HEAD_BODY_COUNT


@pytest.mark.unit
def test_the_terminations_leave_the_robot_free_to_fall_over():
    """Falling over is the task, so the only non-time-out termination catches a broken robot."""
    terminations = MicroDuckRouladeFlatEnvCfg().terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        assert _scalar_params(term) == params, name
    assert not hasattr(terminations, "fell_over")


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """The staged schedules carry upstream's payloads at upstream's iteration boundaries."""
    curriculum = MicroDuckRouladeFlatEnvCfg().curriculum

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
def test_the_spawn_curriculum_walks_the_reverse_curriculum_back_toward_a_standing_start():
    """The mid-roll bucket teaches the landing; it shrinks as the flip itself gets discovered."""
    curriculum = MicroDuckRouladeFlatEnvCfg().curriculum
    stages = curriculum.roulade_spawn_mix.params["param_stages"]

    assert len(stages) == len(EXPECTED_SPAWN_MIX_STAGES)
    for stage, (iteration, params) in zip(stages, EXPECTED_SPAWN_MIX_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        assert stage["params"] == params
    # it never reaches zero: the landing stays practised and it is realistic randomization anyway
    assert stages[-1]["params"]["midroll_prob"] > 0.0
    # upstream's event-parameter helper compares inclusively, unlike its reward-weight one
    # (addendum section 7.6); the inconsistency is reproduced rather than smoothed over
    assert "inclusive" not in curriculum.roulade_spawn_mix.params


@pytest.mark.unit
def test_the_task_takes_no_command_and_pads_the_two_it_does_not_have():
    """The roll is triggered by the policy switch, but the deployed vector keeps all three slots."""
    cfg = MicroDuckRouladeFlatEnvCfg()

    assert set(vars(cfg.commands)) == {"base_velocity"}
    twist = cfg.commands.base_velocity
    assert twist.ranges.lin_vel_x == (-0.01, 0.01)
    assert twist.ranges.lin_vel_y == (-0.01, 0.01)
    assert twist.ranges.ang_vel_z == (-0.05, 0.05)
    # upstream derives the resampling window from the episode length, so it resamples at most once
    assert twist.resampling_time_range == (cfg.episode_length_s, 2.0 * cfg.episode_length_s)
    assert twist.heading_command is False
    assert twist.rel_standing_envs == 0.0
    assert twist.rel_turn_in_place_envs == 0.0
    # inherited from upstream's base template and never overridden there (addendum section 7.22)
    assert twist.rel_forward_envs == pytest.approx(0.2)

    for group in (cfg.observations.policy, cfg.observations.critic):
        terms = _observation_terms(group)
        for name, dim in (("head_pose_commands", 4), ("body_pose_commands", 6)):
            assert terms[name].func is mdp.zero_command_padding
            assert terms[name].params == {"dim": dim}


@pytest.mark.unit
def test_the_actor_observation_layout_is_the_shared_deploy_contract():
    """The roll policy reads the same 61-wide vector every other MicroDuck policy does."""
    observations = MicroDuckRouladeFlatEnvCfg().observations
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
    # distinct configuration objects: an alias would share one delay term's buffer between them
    assert terms["base_ang_vel"] is not terms["projected_gravity"]


@pytest.mark.unit
def test_the_critic_group_guards_its_sensor_reads():
    """A deliberate deviation: upstream guards them on the stand-up task only (section 7.9)."""
    observations = MicroDuckRouladeFlatEnvCfg().observations
    terms = _observation_terms(observations.critic)

    assert list(terms) == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert not observations.critic.enable_corruption
    assert "foot_height" not in terms
    assert terms["foot_air_time"].func is mdp.foot_group_air_time_safe
    assert terms["foot_contact_forces"].func is mdp.foot_group_contact_forces_safe
    # a privileged group the runner never reads is dead weight
    assert MicroDuckRouladePPORunnerCfg().obs_groups == {"actor": ["policy"], "critic": ["critic"]}


@pytest.mark.unit
def test_the_scene_carries_the_two_sensors_the_accumulator_gates_on():
    """The head sensor is one body and the support sensor is every body that can reach the floor."""
    scene = MicroDuckRouladeFlatEnvCfg().scene

    assert scene.robot.spawn.usd_path == MICRODUCK_ALLCOLLISIONS_USD_PATH
    assert scene.terrain.terrain_type == "plane"
    assert scene.terrain.terrain_generator is None
    # the head shells live on ``jaw_soft``, and this sensor is what reports them touching the ground
    assert scene.head_ground_contact.prim_path.endswith("jaw_soft")
    # the head sensor is unfiltered: it asks "is this body loaded", which is upstream's ``found``
    assert not scene.head_ground_contact.filter_prim_paths_expr
    assert not scene.head_ground_contact.filter_shape_prim_expr
    # the support sensor is not, because the anti-breakdance gate has to tell the floor from the
    # robot's own shin. It senses every collider and filters them against the terrain alone.
    assert scene.robot_ground_contact.sensor_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    assert scene.robot_ground_contact.filter_shape_prim_expr == [scene.terrain.prim_path + "/.*"]
    assert not scene.robot_ground_contact.filter_prim_paths_expr
    # the head slams into the floor here, so the contact budget is measured rather than inherited
    solver = MicroDuckRouladeFlatEnvCfg().sim.physics.default.solver_cfg
    assert solver.nconmax >= 26
    assert solver.njmax >= 86


@pytest.mark.unit
def test_the_runner_turns_the_family_symmetry_machinery_on():
    """The roll is the one task in the family with symmetry enabled, and it uses the mirror loss."""
    runner = MicroDuckRouladePPORunnerCfg()

    assert runner.experiment_name == "microduck_roulade"
    assert runner.max_iterations == 10000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    symmetry = runner.algorithm.symmetry_cfg
    assert symmetry is not None
    assert symmetry.use_mirror_loss is True
    # data augmentation would train the critic on mirrored transitions labelled with unmirrored
    # privileged observations, which the family's tables do not produce
    assert symmetry.use_data_augmentation is False
    assert symmetry.mirror_loss_coeff == pytest.approx(0.5)
    assert symmetry.data_augmentation_func is compute_roulade_symmetric_states


@pytest.mark.unit
def test_the_symmetry_tables_mirror_the_deploy_vector_and_undo_themselves():
    """A mirror applied twice is the identity, which is what makes the tables self-checking."""
    torch.manual_seed(0)
    observation = torch.randn(3, ACTOR_OBSERVATION_DIM)
    critic = torch.randn(3, CRITIC_OBSERVATION_DIM)
    actions = torch.randn(3, 14)
    obs = TensorDict({"policy": observation, "critic": critic}, batch_size=[3])

    mirrored_obs, mirrored_actions = compute_roulade_symmetric_states(None, obs, actions)
    twice_obs, twice_actions = compute_roulade_symmetric_states(
        None, TensorDict({"policy": mirrored_obs["policy"][3:], "critic": critic}, batch_size=[3]), mirrored_actions[3:]
    )

    assert mirrored_obs.batch_size[0] == 6
    torch.testing.assert_close(mirrored_obs["policy"][:3], observation)
    torch.testing.assert_close(mirrored_actions[:3], actions)
    # the mirror is an involution
    torch.testing.assert_close(twice_obs["policy"][3:], observation)
    torch.testing.assert_close(twice_actions[3:], actions)
    # the critic is repeated unmirrored, which is only sound for the mirror loss
    torch.testing.assert_close(mirrored_obs["critic"][3:], critic)
    # the left leg block lands on the right leg block, negated
    torch.testing.assert_close(mirrored_obs["policy"][3:, 6:11], -observation[:, 15:20])
    # the two sagittal head servos keep their sign and their place
    torch.testing.assert_close(mirrored_obs["policy"][3:, 11:13], observation[:, 11:13])


@pytest.mark.unit
def test_the_symmetry_tables_refuse_an_observation_they_do_not_describe():
    """The tables are a fixed layout, so a widened observation group has to fail loudly."""
    obs = TensorDict({"policy": torch.zeros(2, 51), "critic": torch.zeros(2, 74)}, batch_size=[2])

    with pytest.raises(ValueError, match="61-wide"):
        compute_roulade_symmetric_states(None, obs, None)
    with pytest.raises(ValueError, match="14 servos"):
        compute_roulade_symmetric_states(None, None, torch.zeros(2, 12))


@pytest.mark.unit
def test_the_episode_and_simulation_rates_match_upstream():
    """A 5 s episode is a paced roll, a rise and a moment to settle -- 4 s left no room for the rise."""
    cfg = MicroDuckRouladeFlatEnvCfg()

    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.episode_length_s == pytest.approx(5.0)
    assert cfg.episode_length_s / (cfg.sim.dt * cfg.decimation) == pytest.approx(EPISODE_CONTROL_STEPS)


@pytest.mark.unit
def test_the_servos_run_on_the_backend_native_path():
    """The roll task runs the BAM servos where the walking and stand-up tasks do.

    The three tasks share one robot and one servo deployment, so a policy trained on one plant and
    evaluated against the other would silently compare different robots. The decimation is checked
    alongside because it is a precondition: the BAM delay line is actuator state, and
    :class:`~isaaclab_newton.physics.NewtonManager` refuses to CUDA-graph-capture stateful Newton
    actuators at a decimation of one and warns at an odd one.
    """
    cfg = MicroDuckRouladeFlatEnvCfg()

    assert cfg.sim.use_newton_actuators is True
    assert cfg.decimation % 2 == 0


##
# Environment smoke tests
##


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_observation_groups_are_the_widths_their_contracts_name():
    """The actor group is the deployed 61-vector, term for term, and the critic measures 74."""
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
def test_the_mid_roll_spawn_seeds_the_rotation_bookkeeping_it_is_born_with():
    """The pre-seed and the granted latch are what make the reverse curriculum coherent.

    Without the pre-seed a 340-degree spawn would be paid for a full roll it never performed; without
    the granted latch its landing gate could never open, because it had no chance to earn a latch.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        # the curriculum rewrites these probabilities at step 0, so it has to be switched off
        env_cfg.curriculum.roulade_spawn_mix = None
        params = env_cfg.events.set_roulade_state.params
        params["standing_prob"], params["midroll_prob"] = 0.0, 1.0
        params["midroll_pitch_range"] = (math.radians(100.0), math.radians(100.0))
        params["midroll_omega_range"] = (3.0, 3.0)
        params["midroll_z_range"] = (0.08, 0.08)
        params["tuck_factor_range"] = (1.0, 1.0)
        params["joint_noise_std"] = 0.0

        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        env.reset()
        robot = env.unwrapped.scene["robot"]
        state = mdp.roulade_roll_state(env.unwrapped)

        spawn_angle = torch.full_like(state.accumulated_angle, math.radians(100.0))
        torch.testing.assert_close(state.accumulated_angle, spawn_angle)
        torch.testing.assert_close(state.frontier, spawn_angle)
        torch.testing.assert_close(state.paid, spawn_angle)
        assert bool(state.head_latch.all())
        # upstream writes the free joint's body-frame angular velocity, so a yawed spawn still rolls
        # straight ahead in its own frame; this port rotates it into the world frame to match
        forward_rate = robot.data.root_link_ang_vel_b.torch[:, 1]
        torch.testing.assert_close(forward_rate, torch.full_like(forward_rate, 3.0), atol=1e-3, rtol=0.0)
        height = robot.data.root_link_pos_w.torch[:, 2] - env.unwrapped.scene.env_origins[:, 2]
        torch.testing.assert_close(height, torch.full_like(height, 0.08), atol=1e-4, rtol=0.0)
        # the tuck is lerped from the pose the joint reset wrote, so at factor 1 it is written exactly
        names = list(robot.joint_names)
        for joint, angle in MICRODUCK_TUCK_JOINT_POS.items():
            measured = robot.data.joint_pos.torch[:, names.index(joint)]
            torch.testing.assert_close(measured, torch.full_like(measured, angle), atol=1e-4, rtol=0.0)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_a_standing_spawn_plants_its_head_and_can_reach_the_landing_gate():
    """The acceptance test for the head shells, run end to end from a standing start.

    Two links are checked, because breaking either one silently kills the standing bucket while the
    mid-roll bucket keeps every training curve looking healthy (addendum section 7.2):

    1. **The head shells are collidable and the latch is earnable.** A standing spawn is tipped
       forward tucked, which is the manoeuvre the task is about; the head-ground sensor has to report
       a real force, and the latch -- head contact, head top pointing at the floor, rotation inside
       the first-quadrant window -- has to be earned by an episode that was *not* granted one.
    2. **The completion gate then opens on rotation.** Driving the rotation frontier to the gate is
       the same write ``reset_roulade_state`` performs for every mid-roll spawn, so it is the task's
       own mechanism rather than a stand-in for one; with the earned latch it opens the gate fully
       and the gated rewards start paying.

    A passive rollout cannot *complete* a roll -- coming back up off the head needs a policy, which
    is what the task trains -- so what is asserted is reachability, not completion.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        env_cfg.curriculum.roulade_spawn_mix = None
        params = env_cfg.events.set_roulade_state.params
        params["standing_prob"], params["midroll_prob"] = 1.0, 0.0
        params["standing_tilt_max"] = 0.0

        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()
        robot = unwrapped.scene["robot"]
        state = mdp.roulade_roll_state(unwrapped)

        # a standing spawn starts at zero rotation and without the latch, unlike a mid-roll one
        torch.testing.assert_close(state.frontier, torch.zeros_like(state.frontier))
        assert not bool(state.head_latch.any())

        # hold the tuck through the servos and tip the robot forward with one nudge, which is the
        # entry a trained policy performs
        names = list(robot.joint_names)
        tuck = robot.data.default_joint_pos.torch.clone()
        for joint, angle in MICRODUCK_TUCK_JOINT_POS.items():
            tuck[:, names.index(joint)] = angle
        action_ids = unwrapped.action_manager._terms["joint_pos"]._joint_ids
        action = (tuck - robot.data.default_joint_pos.torch)[:, action_ids].clone()
        nudge = torch.zeros((unwrapped.num_envs, 6), device=unwrapped.device)
        nudge[:, 4] = 2.0
        robot.write_root_link_velocity_to_sim_index(root_velocity=nudge)

        head_sensor = unwrapped.scene.sensors["head_ground_contact"]
        peak_head_force = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        with torch.inference_mode():
            for _ in range(60):
                env.step(action)
                force = head_sensor.data.net_forces_w.torch.norm(dim=-1).squeeze(-1)
                peak_head_force = torch.maximum(peak_head_force, force)

        # 1. the head really does reach the floor, and the latch really is earnable from standing
        assert bool((peak_head_force > 1.0).all()), f"head never loaded: {peak_head_force.tolist()}"
        assert bool(state.head_latch.all()), "a standing spawn could not earn the over-the-head latch"

        # 2. with the latch earned, the frontier is all that stands between it and the annuity
        assert bool((mdp.roulade_completion_gate(unwrapped, *MICRODUCK_LANDING_GATE) == 0.0).all())
        state.frontier[:] = MICRODUCK_LANDING_GATE[1]
        gate = mdp.roulade_completion_gate(unwrapped, *MICRODUCK_LANDING_GATE)
        torch.testing.assert_close(gate, torch.ones_like(gate))
        # and a gated reward that is non-zero for any pose below standing height starts paying
        tax = mdp.roulade_stand_tax(
            unwrapped,
            target_height=MICRODUCK_STAND_HEIGHT,
            gate_lo=MICRODUCK_LANDING_GATE[0],
            gate_hi=MICRODUCK_LANDING_GATE[1],
        )
        assert bool((tax < 0.0).all()), f"the gated landing terms stayed zero: {tax.tolist()}"
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_support_gate_stays_shut_for_a_robot_touching_only_itself():
    """The anti-breakdance gate must read the floor, not any contact at all.

    Upstream's first run found that without a support gate the optimal policy is a ballistic whip
    that earns the rotation without ever rolling. An unfiltered net contact force reopens exactly
    that hole from the other side: it cannot tell the floor from the robot's own shin, so a tucked
    robot in mid-air reads as supported and collects the rotation the gate exists to deny.

    Two environments are spun at the same rate about the same axis and differ only in what they are
    touching, so the rotation frontier is a direct measurement of the gate rather than of the pose.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        # both environments must start at zero rotation, which the mid-roll bucket is granted
        env_cfg.curriculum.roulade_spawn_mix = None
        params = env_cfg.events.set_roulade_state.params
        params["standing_prob"], params["midroll_prob"] = 1.0, 0.0
        params["standing_tilt_max"] = 0.0

        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        names = list(robot.joint_names)
        limits = robot.data.joint_pos_limits.torch[0]
        joint_pos = robot.data.default_joint_pos.torch.clone()
        # env 1 folds both legs, which drives each shin into the hip shell on its own side
        for joint, angle in {
            "left_hip_pitch": -1.5,
            "left_knee": 1.5,
            "right_hip_pitch": 1.5,
            "right_knee": -1.5,
        }.items():
            index = names.index(joint)
            joint_pos[1, index] = min(max(angle, limits[index, 0].item()), limits[index, 1].item())

        # env 0 rests on the floor, env 1 hangs a metre above it; both are level, so the sagittal
        # gate is fully open and the only difference between them is what they touch
        pose = torch.zeros((2, 7), device=unwrapped.device)
        pose[:, :3] = unwrapped.scene.env_origins
        pose[0, 2] += MICRODUCK_STAND_HEIGHT
        pose[1, 2] += 1.0
        pose[:, 6] = 1.0  # xyzw identity
        # a forward roll is a positive body-frame pitch rate, and level means body frame is world
        velocity = torch.zeros((2, 6), device=unwrapped.device)
        velocity[:, 4] = 3.0
        joint_vel = torch.zeros_like(joint_pos)
        action = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)

        support = unwrapped.scene.sensors["robot_ground_contact"]
        steps = 10
        with torch.inference_mode():
            for _ in range(steps):
                robot.write_root_link_pose_to_sim_index(root_pose=pose)
                robot.write_root_com_velocity_to_sim_index(root_velocity=velocity)
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                env.step(action)

            ground = support.data.force_matrix_w.torch.norm(dim=-1).flatten(1).max(dim=-1).values
            net = support.data.net_forces_w.torch.norm(dim=-1).max(dim=-1).values
            frontier = mdp.roulade_roll_state(unwrapped).frontier.clone()

        # the sensor tells the two apart: only the resting robot is loaded *by the terrain*
        assert ground[0].item() > 1.0, "the resting robot never reached the floor"
        assert ground[1].item() == 0.0, "an airborne robot was reported as touching the ground"
        # and the airborne one is loaded all the same, which is what the old unfiltered read saw
        assert net[1].item() > 1.0, "the airborne pose produced no self-contact to be fooled by"
        # so the gate opens for the supported roll and stays shut for the self-contacting one
        assert frontier[0].item() > 0.0, "a supported roll earned no rotation"
        assert frontier[1].item() == 0.0, "an airborne robot earned rotation off its own knee"
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
