# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test stand-up configuration, reset mixtures and recovery rewards.

Configuration checks use independent expectations; scene checks cover reset states and contacts.
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ALLCOLLISIONS_USD_PATH
from isaaclab_tasks.contrib.microduck.standup.agents.rsl_rl_ppo_cfg import MicroDuckStandUpPPORunnerCfg
from isaaclab_tasks.contrib.microduck.standup.standup_env_cfg import (
    MICRODUCK_SIT_HEIGHT,
    MICRODUCK_SITTING_JOINT_POS,
    MICRODUCK_STAND_HEIGHT,
    MicroDuckStandUpFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_allcollisions_usd = pytest.mark.usefixtures("microduck_allcollisions_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-StandUp-Flat-MicroDuck"

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
"""The shared 61-component deployed actor observation layout, in concatenation order."""

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
"""Privileged critic observation layout, including task-only state outside the deployment interface."""

CRITIC_OBSERVATION_DIM = 74
"""Expected width of the assembled critic observation group."""

STEPS_PER_ITERATION = 24
"""Environment control steps per PPO iteration."""

_SIT_TO_STAND_HEIGHT_GAP = 0.055
"""Distance [m] the trunk has to travel from the seated equilibrium to the standing keyframe."""


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
"""Fourteen controlled servos in action order: left leg 0-4, neck/head 5-8 and right leg 9-13."""

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
"""The ten physical leg joints, excluding the four command-driven neck/head joints."""

EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The 4 head servos, in the order the head-pose command indexes its columns."""

EXPECTED_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot body names in left/right order."""

EXPECTED_TRUNK_BODY_NAMES = ["trunk_base"]
"""Trunk body used for height, tilt, angular velocity and mass-property randomization."""

EXPECTED_HEAD_BODY_COUNT = 4
"""Expected body count for head center-of-mass randomization."""

EXPECTED_ENTITY_SELECTIONS = {
    # "<manager>.<term>.<param>": (kind, expected names, preserve_order)
    "rewards.pose_stand_legs.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.pose_stand_l1.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.standing_composite.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.head_pose_tracking.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.head_pose_bias.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.joint_torque_rate_l2.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "rewards.arrival_damping.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.body_ang_vel.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.self_collisions.sensor_cfg": ("sensor", "self_collision", False),
    "events.foot_friction.asset_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, False),
    "events.mass_inertia.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_com.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_armature.asset_cfg": ("joint", [".*"], False),
    "events.set_ground_state.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
}
"""Entity selections used by rewards, events and terminations outside the observation groups."""

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
"""Entity selections inside the two observation groups, all of which are ordering contracts.

The joint blocks are the deployed vector's own layout, which the runtime on the robot rebuilds by
hand from its sensor reads; the foot blocks are three consecutive critic terms that must agree on
which column is the left foot.
"""


def _entity_cfg_of(term_cfg, key: str) -> SceneEntityCfg:
    """Fetch a term's entity selection, looking inside a delayed term's wrapped parameters."""
    if key in term_cfg.params:
        return term_cfg.params[key]
    return term_cfg.params["term_params"][key]


def _scalar_params(term_cfg) -> dict:
    """Return scalar parameters; entity selections are checked separately by name and order."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


##
# Configuration expectations
##

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "pose_stand_legs": (2.0, {"std": 0.5}),
    "pose_stand_l1": (1.25, {}),
    "head_pose_tracking": (0.75, {"command_name": "head_pose", "std": 0.5}),
    "head_pose_bias": (
        0.0,
        {
            "command_name": "head_pose",
            "tau_s": 1.0,
            "gate_height_low": 0.09,
            "gate_height_high": 0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": 45.0,
        },
    ),
    "height_stand": (1.0, {"std": 0.04, "target_height": 0.115}),
    "height_stand_sharp": (1.0, {"std": 0.015, "target_height": 0.115}),
    "height_stand_l1": (7.5, {"target_height": 0.115}),
    "com_upward_velocity": (0.75, {"max_height": 0.125}),
    "gentle_rise": (0.005, {}),
    "arrival_damping": (
        0.0,
        {"height_low": 0.09, "height_high": 0.11, "tilt_full_deg": 20.0, "tilt_zero_deg": 45.0},
    ),
    "upright_linear": (1.5, {}),
    "upright_sharp": (1.5, {"std": 0.3, "height_low": 0.060, "height_high": 0.115}),
    "standing_composite": (
        3.75,
        {"target_height": 0.115, "height_std": 0.04, "upright_std": 0.40, "pose_std": 0.40},
    ),
    "body_pose_tracking": (
        0.0,
        {
            "command_name": "body_pose",
            "nominal_height": 0.115,
            "z_std": 0.01,
            "angle_std": math.radians(5.0),
            "xy_std": 0.02,
            "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
        },
    ),
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    "dof_pos_limits": (-1.0, {}),
    "action_rate_l2": (-0.1, {}),
    "joint_torque_rate_l2": (0.0, {}),
    "self_collisions": (-1.0, {"saturate": True}),
}
"""Initial reward terms, weights and scalar parameters, including inherited terms."""

EXPECTED_EVENTS = {
    # name: (mode, scalar params)
    "foot_friction": (
        "startup",
        {
            "static_friction_range": (0.7, 1.3),
            # The event overwrites friction; its required scalar parameters remain explicit.
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    ),
    "encoder_bias": ("startup", {"bias_range": (-0.015, 0.015)}),
    "mass_inertia": (
        "startup",
        # Recomputing inertia couples the sampled mass scale to the inertia tensor.
        {"mass_distribution_params": (0.95, 1.05), "operation": "scale", "recompute_inertia": True},
    ),
    "reset_base": ("reset", {"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}}),
    "reset_robot_joints": ("reset", {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}),
    "set_ground_state": (
        "reset",
        {
            "face_down_prob": 0.20,
            "face_up_prob": 0.00,
            "sitting_prob": 0.40,
            "standing_prob": 0.40,
            "prone_z_range": (0.05, 0.09),
            "sitting_z_range": (0.05, 0.09),
            "standing_z_range": (0.11, 0.12),
            "sitting_joint_pos": MICRODUCK_SITTING_JOINT_POS,
            "sitting_joint_noise_std": 0.12,
            "sitting_tilt_max": math.radians(10.0),
            "face_up_roll_max": math.radians(90.0),
        },
    ),
    "randomize_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_head_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_armature": ("reset", {"armature_distribution_params": (0.9, 1.1), "operation": "scale"}),
    "randomize_joint_friction": ("reset", {"scale_range": (0.9, 1.1)}),
    "push_robot": ("interval", {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}),
}
"""Configured event modes and scalar parameters."""

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "set_ground_state",
    "randomize_com",
    "randomize_head_com",
    "randomize_armature",
    "randomize_joint_friction",
]
"""Reset events in their required execution order."""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params)
    "time_out": (True, {}),
    "nan_state": (False, {"sensor_names": ("contact_forces",)}),
}
"""Configured termination names, timeout flags and thresholds."""

EXPECTED_CURRICULUM_TERMS = {
    "ground_state_mix",
    "com_range",
    "head_com_range",
    "push_magnitude",
    "head_pose_range",
    "action_rate_weight",
    "body_pose_tracking_weight",
    "body_pose_range",
    "height_stand_sharp_weight",
    "upright_sharp_weight",
    "standing_composite_weight",
    "arrival_damping_weight",
    "head_pose_bias_weight",
    "torque_rate_weight",
}
"""Configured curriculum term names."""

EXPECTED_WEIGHT_STAGES = {
    "action_rate_weight": ([-0.1, -0.2, -0.4, -0.6, -0.8, -1.0], [0, 500, 750, 1000, 1250, 1500]),
    "arrival_damping_weight": ([0.0, -0.025, -0.05], [0, 3000, 4000]),
    "head_pose_bias_weight": ([0.0, 0.5, 1.5], [0, 3000, 4000]),
    "torque_rate_weight": ([0.0, -1e-3], [0, 3000]),
    "body_pose_tracking_weight": ([0.0, 1.5, 3.0, 4.0], [0, 2500, 3000, 4000]),
    "height_stand_sharp_weight": ([1.0, 0.5, 0.2], [0, 3000, 4000]),
    "upright_sharp_weight": ([1.5, 1.0, 0.5], [0, 3000, 4000]),
    "standing_composite_weight": ([3.75, 2.5, 1.5], [0, 3000, 4000]),
}
"""Reward weights at each curriculum boundary."""

EXPECTED_RANGE_STAGES = {
    "com_range": ([0.003, 0.005, 0.01, 0.015], [0, 500, 1000, 1500]),
    "head_com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
}
"""Center-of-mass randomization ranges at each curriculum boundary."""

EXPECTED_GROUND_STATE_STAGES = [
    (0, {"standing_prob": 0.40, "sitting_prob": 0.40, "face_down_prob": 0.20, "face_up_prob": 0.00}),
    (600, {"standing_prob": 0.25, "sitting_prob": 0.30, "face_down_prob": 0.35, "face_up_prob": 0.10}),
    (1500, {"standing_prob": 0.20, "sitting_prob": 0.25, "face_down_prob": 0.30, "face_up_prob": 0.25}),
    (2500, {"standing_prob": 0.15, "sitting_prob": 0.20, "face_down_prob": 0.30, "face_up_prob": 0.35}),
]
"""Reset-distribution stages, progressing from easier to harder mixtures."""

EXPECTED_PUSH_STAGES = [
    (0, {"x": (0.0, 0.0), "y": (0.0, 0.0)}),
    (500, {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}),
    (1000, {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
]
"""Push velocity ranges [m/s] at each curriculum boundary."""

EXPECTED_HEAD_POSE_STAGES = [
    (0, ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (500, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
    (1000, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
    (1500, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))),
    (2000, ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))),
]
"""Head command ranges [rad], ordered as neck pitch, head pitch, yaw and roll."""

_ALIVE_XY = (-0.005, 0.005)
_ALIVE_ANGLE = (-0.05, 0.05)

EXPECTED_BODY_POSE_STAGES = [
    (0, (_ALIVE_XY, _ALIVE_XY, (-0.005, 0.005), _ALIVE_ANGLE, _ALIVE_ANGLE, _ALIVE_ANGLE)),
    (
        2500,
        (
            _ALIVE_XY,
            _ALIVE_XY,
            (-0.010, 0.005),
            (-math.radians(8.0), math.radians(8.0)),
            (-math.radians(8.0), math.radians(8.0)),
            _ALIVE_ANGLE,
        ),
    ),
    (
        3000,
        (
            _ALIVE_XY,
            _ALIVE_XY,
            (-0.018, 0.008),
            (-math.radians(12.0), math.radians(12.0)),
            (-math.radians(12.0), math.radians(12.0)),
            _ALIVE_ANGLE,
        ),
    ),
    (
        4000,
        (
            _ALIVE_XY,
            _ALIVE_XY,
            (-0.04, 0.030),
            (-math.radians(15.0), math.radians(15.0)),
            (-math.radians(15.0), math.radians(15.0)),
            _ALIVE_ANGLE,
        ),
    ),
]
"""Body command range stages: translation [m] and orientation [rad]. The asymmetric height range allows more
crouch travel than upward extension.
"""

EXPECTED_ACTOR_NOISE = {
    "base_ang_vel": 0.03,
    "projected_gravity": 0.01,
    "joint_pos": 0.001,
    "joint_vel": 0.25,
}
"""Half-width of the uniform noise on each corrupted actor term, matched to the velocity task."""


@pytest.mark.unit
def test_the_reward_recipe_matches_upstream_term_for_term():
    """Reward terms must match the expected set, weights and scalar parameters."""
    rewards = MicroDuckStandUpFlatEnvCfg().rewards

    # Check for missing and extra reward terms.
    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_self_negating_reward_terms_carry_positive_weights():
    """Four terms return a non-positive value, so a negative weight would pay for what it prices."""
    rewards = MicroDuckStandUpFlatEnvCfg().rewards

    for name in ("pose_stand_l1", "height_stand_l1", "gentle_rise"):
        assert getattr(rewards, name).weight > 0.0, name
    # head_pose_bias ships at zero and is ramped, so its schedule is what has to stay positive
    stages = MicroDuckStandUpFlatEnvCfg().curriculum.head_pose_bias_weight.params["weight_stages"]
    assert all(stage["weight"] >= 0.0 for stage in stages)


@pytest.mark.unit
def test_the_walking_reward_terms_are_gone():
    """Stand-up recovery must not retain locomotion or gait reward terms."""
    rewards = MicroDuckStandUpFlatEnvCfg().rewards

    for name in ("track_lin_vel", "track_ang_vel", "air_time", "foot_clearance", "foot_swing_height", "foot_slip"):
        assert not hasattr(rewards, name), name
    # the base template's own Gaussian upright is replaced by the two-layer pair
    assert not hasattr(rewards, "upright")
    assert hasattr(rewards, "upright_linear") and hasattr(rewards, "upright_sharp")


@pytest.mark.unit
def test_the_two_layer_attractors_are_a_wide_and_a_narrow_copy_of_one_target():
    """The sharp layer only earns its slot by being sharper than the layer it stacks on."""
    rewards = MicroDuckStandUpFlatEnvCfg().rewards

    assert rewards.height_stand.func is rewards.height_stand_sharp.func
    assert rewards.height_stand.params["target_height"] == rewards.height_stand_sharp.params["target_height"]
    assert rewards.height_stand_sharp.params["std"] < rewards.height_stand.params["std"]
    # the sharp height layer must still resolve the last centimetre of the rise
    assert rewards.height_stand_sharp.params["std"] < 0.02
    # the upright gate opens across exactly the seated-to-standing travel
    assert rewards.upright_sharp.params["height_low"] == pytest.approx(MICRODUCK_SIT_HEIGHT)
    assert rewards.upright_sharp.params["height_high"] == pytest.approx(MICRODUCK_STAND_HEIGHT)
    assert pytest.approx(_SIT_TO_STAND_HEIGHT_GAP) == MICRODUCK_STAND_HEIGHT - MICRODUCK_SIT_HEIGHT


@pytest.mark.unit
def test_the_body_pose_command_is_trained_here_rather_than_kept_alive():
    """The velocity task zero-weights this slot forever; the stand-up task ramps it to 4.0."""
    cfg = MicroDuckStandUpFlatEnvCfg()

    # only the three axes the deployed runtime exposes are weighted
    assert cfg.rewards.body_pose_tracking.params["axis_weights"] == (0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
    stages = cfg.curriculum.body_pose_tracking_weight.params["weight_stages"]
    assert stages[-1]["weight"] == pytest.approx(4.0)
    # the command widens at the same iteration the weight first becomes non-zero
    weight_start = next(stage["step"] for stage in stages if stage["weight"] > 0.0)
    range_stages = cfg.curriculum.body_pose_range.params["range_stages"]
    assert range_stages[1]["step"] == weight_start
    # the untracked axes never widen; the tracked ones do
    for index in (0, 1, 5):
        assert {stage["ranges"][index] for stage in range_stages} == {range_stages[0]["ranges"][index]}, index
    for index in (2, 3, 4):
        assert range_stages[-1]["ranges"][index] != range_stages[0]["ranges"][index], index
    # the exact-zero bucket, which this task is the first in the family to use
    assert cfg.commands.body_pose.zero_command_prob == pytest.approx(0.3)


@pytest.mark.unit
def test_the_event_suite_matches_upstream_term_for_term():
    """Event modes and parameter ranges must match the expected configuration."""
    events = MicroDuckStandUpFlatEnvCfg().events

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
def test_the_ground_state_reset_runs_after_the_resets_it_overwrites():
    """Isaac Lab fires reset events in declaration order, and this order is the spawn distribution."""
    events = MicroDuckStandUpFlatEnvCfg().events

    reset_terms = [name for name, term in vars(events).items() if getattr(term, "mode", None) == "reset"]
    assert reset_terms == EXPECTED_RESET_EVENT_ORDER


@pytest.mark.unit
def test_the_terms_select_the_joints_bodies_and_sensors_upstream_measures():
    """A term that measures the wrong joints is as wrong as one carrying the wrong weight."""
    cfg = MicroDuckStandUpFlatEnvCfg()

    # two-sided over the terms that carry a selection at all, so a term that gains or loses one
    # -- say a height reward that starts reading a body -- fails rather than going unchecked
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
    observations = MicroDuckStandUpFlatEnvCfg().observations
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

    # the two head rewards index their command's columns positionally, so their joint order is the
    # command's; the observation joint order is the deployed vector's. They are different contracts
    # over the same articulation, and both are pinned above.
    assert EXPECTED_SERVO_JOINT_NAMES[5:9] == EXPECTED_HEAD_JOINT_NAMES


@pytest.mark.unit
def test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link():
    """Head center-of-mass randomization must select the intended head bodies without a hip link."""
    body_names = MicroDuckStandUpFlatEnvCfg().events.randomize_head_com.params["asset_cfg"].body_names

    assert "bearing_roll" not in body_names
    assert len(body_names) == EXPECTED_HEAD_BODY_COUNT


@pytest.mark.unit
def test_the_sitting_keyframe_is_the_sit_policy_hand_off_pose():
    """The keyframe folds the legs and leaves the head alone, and is sampled with noise."""
    params = MicroDuckStandUpFlatEnvCfg().events.set_ground_state.params

    # legs only: the neck and head stay at the stand pose so the head-pose command steers them
    assert set(MICRODUCK_SITTING_JOINT_POS) == {
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    }
    # mirrored left and right, as the two legs' joint axes are
    for joint in ("hip_roll", "hip_pitch", "knee", "ankle"):
        left = MICRODUCK_SITTING_JOINT_POS[f"left_{joint}"]
        right = MICRODUCK_SITTING_JOINT_POS[f"right_{joint}"]
        assert left == pytest.approx(-right), joint
    # about 77 degrees of knee fold is what puts the trunk at the seated height
    assert abs(MICRODUCK_SITTING_JOINT_POS["left_knee"]) == pytest.approx(1.35)
    # the hand-off from a real sit policy never reproduces the keyframe exactly
    assert params["sitting_joint_noise_std"] > 0.0
    assert params["sitting_tilt_max"] > 0.0
    # the sitting height band brackets the seated equilibrium
    low, high = params["sitting_z_range"]
    assert low <= MICRODUCK_SIT_HEIGHT <= high
    # the standing band brackets the target the whole recipe is built around
    low, high = params["standing_z_range"]
    assert low <= MICRODUCK_STAND_HEIGHT <= high


@pytest.mark.unit
def test_the_terminations_match_upstream_and_drop_the_tilt_check():
    """A task that starts on the ground cannot end an episode on tilt."""
    terminations = MicroDuckStandUpFlatEnvCfg().terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        assert _scalar_params(term) == params, name


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """Scheduled values must change at the expected control-step boundaries."""
    curriculum = MicroDuckStandUpFlatEnvCfg().curriculum

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
def test_the_ground_state_curriculum_walks_the_reset_mix_from_easy_to_hard():
    """Recovery is discovered in the order the reset distribution introduces it."""
    stages = MicroDuckStandUpFlatEnvCfg().curriculum.ground_state_mix.params["param_stages"]

    assert len(stages) == len(EXPECTED_GROUND_STATE_STAGES)
    for stage, (iteration, params) in zip(stages, EXPECTED_GROUND_STATE_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        assert stage["params"] == params
    # face-up is the hardest recovery and is withheld until the second stage
    assert stages[0]["params"]["face_up_prob"] == 0.0
    # and the mix ends up dominated by the two prone poses
    final = stages[-1]["params"]
    assert final["face_down_prob"] + final["face_up_prob"] > final["standing_prob"] + final["sitting_prob"]


@pytest.mark.unit
def test_the_push_curriculum_ramps_from_nothing_on_upstreams_exclusive_comparison():
    """Pushes start at zero and advance only after each strict curriculum boundary."""
    push = MicroDuckStandUpFlatEnvCfg().curriculum.push_magnitude

    stages = push.params["param_stages"]
    assert len(stages) == len(EXPECTED_PUSH_STAGES)
    for stage, (iteration, velocity_range) in zip(stages, EXPECTED_PUSH_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        assert stage["params"] == {"velocity_range": velocity_range}
    # Push stages use strict boundaries; event-parameter stages use inclusive boundaries.
    assert push.params["inclusive"] is False
    assert "inclusive" not in MicroDuckStandUpFlatEnvCfg().curriculum.ground_state_mix.params


@pytest.mark.unit
def test_the_pose_command_curricula_reproduce_upstream_stage_tables():
    """The head command opens as the velocity task's does; the body command opens later and less."""
    curriculum = MicroDuckStandUpFlatEnvCfg().curriculum

    for name, expected in (
        ("head_pose_range", EXPECTED_HEAD_POSE_STAGES),
        ("body_pose_range", EXPECTED_BODY_POSE_STAGES),
    ):
        stages = getattr(curriculum, name).params["range_stages"]
        assert len(stages) == len(expected), name
        for stage, (iteration, ranges) in zip(stages, expected):
            assert stage["step"] == iteration * STEPS_PER_ITERATION, name
            assert stage["ranges"] == ranges, name


@pytest.mark.unit
def test_the_commands_are_registered_at_their_initial_ranges():
    """Both pose commands start at their schedules' first stage, and the twist is inert."""
    commands = MicroDuckStandUpFlatEnvCfg().commands

    assert commands.head_pose.ranges == EXPECTED_HEAD_POSE_STAGES[0][1]
    assert commands.head_pose.resampling_time_range == (2.0, 5.0)
    assert commands.head_pose.zero_command_prob == 0.0
    assert commands.body_pose.ranges == EXPECTED_BODY_POSE_STAGES[0][1]
    assert commands.body_pose.resampling_time_range == (2.0, 5.0)

    # the twist survives only so the deployed vector keeps its three-wide slot: no reward reads it,
    # its ranges are a hundredth of the velocity task's, and it resamples at most once per episode
    twist = commands.base_velocity
    assert twist.ranges.lin_vel_x == (-0.01, 0.01)
    assert twist.ranges.lin_vel_y == (-0.01, 0.01)
    assert twist.ranges.ang_vel_z == (-0.05, 0.05)
    assert twist.resampling_time_range[0] >= MicroDuckStandUpFlatEnvCfg().episode_length_s
    assert twist.heading_command is False
    assert twist.rel_standing_envs == 0.0
    assert twist.rel_turn_in_place_envs == 0.0
    # Forward-only resampling retains the configured 0.3 m/s lower bound.
    assert twist.rel_forward_envs == pytest.approx(0.2)


@pytest.mark.unit
def test_the_actor_observation_layout_is_the_shared_deploy_contract():
    """The stand-up policy reads the same 61-wide vector every other MicroDuck policy does."""
    observations = MicroDuckStandUpFlatEnvCfg().observations
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
def test_the_critic_group_drops_foot_height_and_guards_its_sensor_reads():
    """The critic omits foot height and guards its selected contact observations against nonfinite values."""
    observations = MicroDuckStandUpFlatEnvCfg().observations
    terms = _observation_terms(observations.critic)

    assert list(terms) == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert not observations.critic.enable_corruption
    # No height sensor or foot-height reward is configured.
    assert "foot_height" not in terms
    assert terms["foot_air_time"].func is mdp.foot_group_air_time_safe
    assert terms["foot_contact_forces"].func is mdp.foot_group_contact_forces_safe
    # a privileged group the runner never reads is dead weight
    assert MicroDuckStandUpPPORunnerCfg().obs_groups == {"actor": ["policy"], "critic": ["critic"]}


@pytest.mark.unit
def test_the_task_runs_the_all_collisions_robot_on_a_plane():
    """A robot that pushes itself off the floor needs colliders the walking model does not have."""
    scene = MicroDuckStandUpFlatEnvCfg().scene

    assert scene.robot.spawn.usd_path == MICRODUCK_ALLCOLLISIONS_USD_PATH
    assert scene.terrain.terrain_type == "plane"
    assert scene.terrain.terrain_generator is None
    # ten world colliders instead of two need a wider contact budget than the walking task's ten
    solver = MicroDuckStandUpFlatEnvCfg().sim.physics.default.solver_cfg
    assert solver.nconmax >= 27
    assert solver.njmax >= 82


@pytest.mark.unit
def test_the_self_collision_sensor_senses_every_collider_against_every_other():
    """Self-collision sensing must cover all selected collider pairs and saturate the pair cost at one."""
    cfg = MicroDuckStandUpFlatEnvCfg()
    scene = cfg.scene

    assert scene.self_collision.sensor_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    assert scene.self_collision.filter_shape_prim_expr == [MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR]
    # the two are mutually exclusive on the backend, and the shape pair is the one that is wired
    assert not scene.self_collision.filter_prim_paths_expr
    assert not scene.contact_forces.filter_prim_paths_expr
    # Saturate the contact-pair cost at zero or one.
    assert cfg.rewards.self_collisions.params["saturate"] is True


@pytest.mark.unit
def test_the_collider_expression_names_the_models_ten_enabled_colliders():
    """The collider expression must select the model's ten enabled contact meshes without visual geometry."""
    for xform in (
        "left_foot_collision",
        "right_foot_collision",
        "leg_1",
        "hip_l_1",
        "top_head_shell_1",
        "jaw_1",
        "bottom_head_shell_1",
        "np_f970_1",
    ):
        assert xform in MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR, xform
    # ``power_support`` is the trunk's self-collision-only shell, which the conversion disables
    # outright; a sensor cannot see a collider that is not there
    assert "power_support" not in MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR
    # shape expressions full-match shape paths, so the trailing token is what reaches the collider
    # prim below its Xform -- an expression stopping at the Xform selects nothing
    assert MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR.endswith(")/[^/]*")
    assert MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR.startswith("{ENV_REGEX_NS}/Robot/Geometry/trunk_base/")


@pytest.mark.unit
def test_the_runner_differs_from_the_velocity_one_in_two_fields():
    """The runner shares the velocity learner settings and changes only its experiment name and budget."""
    runner = MicroDuckStandUpPPORunnerCfg()

    assert runner.experiment_name == "microduck_stand"
    assert runner.max_iterations == 15000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    assert runner.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert runner.algorithm.desired_kl == pytest.approx(0.01)


@pytest.mark.unit
def test_the_episode_and_simulation_rates_match_upstream():
    """A 6 s episode is a gentle rise plus a moment to stabilize, a third of the walking window."""
    cfg = MicroDuckStandUpFlatEnvCfg()

    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.episode_length_s == pytest.approx(6.0)


@pytest.mark.unit
def test_the_servos_run_on_the_backend_native_path():
    """The stand-up task runs the BAM servos where the walking task does, on the Newton-native path.

    The two tasks share one robot and one servo deployment, so a policy trained on one plant and
    evaluated against the other would silently compare different robots. The decimation is checked
    alongside because it is a precondition: the BAM delay line is actuator state, and
    :class:`~isaaclab_newton.physics.NewtonManager` refuses to CUDA-graph-capture stateful Newton
    actuators at a decimation of one and warns at an odd one.
    """
    cfg = MicroDuckStandUpFlatEnvCfg()

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
def test_the_reset_spawns_every_ground_pose_the_mix_asks_for():
    """The four buckets are what the task trains on, so a reset that collapses to one is a failure."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        # every bucket equally likely, so four environments are very unlikely to miss all but one,
        # and the curriculum is switched off because it rewrites these probabilities at step 0
        env_cfg.curriculum.ground_state_mix = None
        for name in ("face_down_prob", "face_up_prob", "sitting_prob", "standing_prob"):
            env_cfg.events.set_ground_state.params[name] = 0.25

        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        env.reset()
        robot = env.unwrapped.scene["robot"]
        origins = env.unwrapped.scene.env_origins

        heights, tilts, knees = [], [], []
        knee_ids, _ = robot.find_joints(["left_knee"], preserve_order=True)
        # cloned, not referenced: the articulation's data tensors are written in place on every
        # reset, so keeping the views would collapse all twelve samples onto the last one
        for _ in range(12):
            env.unwrapped.reset()
            heights.append((robot.data.root_link_pos_w.torch[:, 2] - origins[:, 2]).clone())
            quat = robot.data.root_link_quat_w.torch
            tilts.append((1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)).clone())
            knees.append(robot.data.joint_pos.torch[:, knee_ids[0]].clone())
        height = torch.cat(heights)
        tilt = torch.cat(tilts)
        knee = torch.cat(knees)

        # every spawn lands inside one of the three configured height bands
        params = env_cfg.events.set_ground_state.params
        assert height.min().item() >= min(params["prone_z_range"][0], params["sitting_z_range"][0]) - 1e-4
        assert height.max().item() <= params["standing_z_range"][1] + 1e-4
        # prone spawns are the ones lying down: cos(tilt) near zero rather than near one
        assert (tilt.abs() < 0.2).any(), "no face-down or face-up spawn"
        # upright spawns cover both the standing band and the lower sitting one
        upright = tilt > 0.9
        assert (height[upright] > params["standing_z_range"][0] - 1e-4).any(), "no standing spawn"
        assert (height[upright] < params["sitting_z_range"][1] + 1e-4).any(), "no sitting spawn"
        # only the sitting bucket folds the knees, and it folds them far past the stand pose
        default_knee = robot.data.default_joint_pos.torch[0, knee_ids[0]].item()
        assert (knee - default_knee).abs().max().item() > 1.0, "no sitting spawn folded its knees"
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_the_self_collision_sensor_fires_on_a_fold_and_stays_quiet_on_a_stand():
    """Folded or crossed configurations must incur self-collision cost while a standing configuration remains quiet."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        # the sensor this replaced, kept alongside for the comparison below
        env_cfg.scene.narrow_self_collision = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/(jaw_soft|hip_l|hip_l_2|leg|leg_2|ankle_left|ankle_right)"
            ],
            update_period=env_cfg.sim.dt,
        )
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.scene["robot"]
        names = list(robot.joint_names)
        limits = robot.data.joint_pos_limits.torch[0]
        joint_pos = robot.data.default_joint_pos.torch.clone()

        # env 0 stays at the stand pose; env 1 folds both legs, env 2 folds only the left and swings
        # it inward as far as the hip allows, env 3 crosses the legs. Clamped to the joint limits, so
        # a pose the servos could not hold is not what the sensor is being shown -- the cross in
        # particular is asked for beyond the limits and lands wherever the clamp puts it, which is
        # the point: it stays commandable.
        folds = {
            1: {"left_hip_pitch": -1.5, "left_knee": 1.5, "right_hip_pitch": 1.5, "right_knee": -1.5},
            2: {"left_hip_pitch": -1.5, "left_knee": 1.5, "left_hip_roll": 0.4, "left_hip_yaw": 0.6},
            3: {
                "left_hip_yaw": 0.524,
                "left_hip_roll": -0.384,
                "left_hip_pitch": -1.571,
                "left_knee": -1.571,
                "left_ankle": -1.571,
                "right_hip_yaw": -0.524,
                "right_hip_roll": 0.384,
                "right_hip_pitch": 1.571,
                "right_knee": 1.571,
                "right_ankle": 1.571,
            },
        }
        for env_id, pose_angles in folds.items():
            for joint, angle in pose_angles.items():
                index = names.index(joint)
                joint_pos[env_id, index] = min(max(angle, limits[index, 0].item()), limits[index, 1].item())

        # env 0 stands on the floor at the settled standing height -- the *spawn* height is 1 cm
        # above it and would leave the robot hovering, since holding the pose skips the drop -- and
        # the other three are held well clear of the floor
        pose = torch.zeros((4, 7), device=unwrapped.device)
        pose[:, :3] = unwrapped.scene.env_origins
        pose[0, 2] += MICRODUCK_STAND_HEIGHT
        pose[1:, 2] += 1.0
        pose[:, 6] = 1.0  # xyzw identity
        velocity = torch.zeros((4, 6), device=unwrapped.device)
        joint_vel = torch.zeros_like(joint_pos)
        action = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)

        wide = unwrapped.scene.sensors["self_collision"]
        narrow = unwrapped.scene.sensors["narrow_self_collision"]
        assert wide.num_sensors == 10, wide.sensor_names
        assert wide.num_filter_objects == 10, wide.filter_object_names
        # the narrow sensor resolved too, so the zero it reports below is a miss and not a
        # mis-resolved expression that could never have fired at all
        assert (narrow.num_sensors, narrow.num_filter_objects) == (1, 7), narrow.filter_object_names

        # hold each pose against the solver rather than letting it fall, and read the settled contact
        with torch.inference_mode():
            for _ in range(12):
                robot.write_root_link_pose_to_sim_index(root_pose=pose)
                robot.write_root_com_velocity_to_sim_index(root_velocity=velocity)
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                env.step(action)

            sensor_cfg = SceneEntityCfg("self_collision")
            cost = mdp.self_collision_cost_saturable(unwrapped, sensor_cfg=sensor_cfg, saturate=True)
            narrow_cost = mdp.self_collision_cost_saturable(
                unwrapped, sensor_cfg=SceneEntityCfg("narrow_self_collision"), saturate=True
            )
            net_force = wide.data.net_forces_w.torch.norm(dim=-1).max(dim=-1).values
            pair_force = wide.data.force_matrix_w.torch.norm(dim=-1)

        # standing costs nothing even though it is very much in contact with the floor
        assert cost[0].item() == 0.0, "a standing robot was charged for touching itself"
        assert net_force[0].item() > 1.0, "the standing robot never reached the floor"
        # every folded and crossed pose is seen, and none of them by the sensor this replaced
        torch.testing.assert_close(cost[1:], torch.ones(3, device=cost.device))
        torch.testing.assert_close(narrow_cost, torch.zeros(4, device=cost.device))
        # those three are airborne, so what the sensor found is self-contact and nothing else
        assert net_force[1:].min().item() > 1.0
        # the cross is a left-against-right contact specifically, not just any self-contact: read the
        # sole-vs-sole entry by name so a reordered shape list cannot make this pass by accident
        left = wide.sensor_names.index("sole_left")
        right = wide.filter_object_names.index("sole_right")
        assert pair_force[3, left, right].item() > 1.0, (
            "the crossed pose produced no sole-against-sole contact:"
            f" {pair_force[3, left].tolist()} against {wide.filter_object_names}"
        )
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_allcollisions_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
