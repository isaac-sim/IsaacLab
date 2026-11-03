# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test roller configuration, deployment interfaces and physical wheel rolling.

Configuration checks use independent expectations; scene checks distinguish free and braked bearings.
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
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_CFG, MICRODUCK_ROLLERS_USD_PATH
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckRollersPPORunnerCfg
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import (
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_FOOT_NORMAL_AXIS,
    MICRODUCK_ROLLERS_JOINT_NAMES,
    MICRODUCK_ROLLERS_SPAWN_HEIGHT,
    MICRODUCK_ROLLERS_STANDING_HEIGHT,
    MICRODUCK_TIRE_BODY_NAMES,
    MICRODUCK_TIRES_PER_FOOT,
    MicroDuckVelocityRollersFlatEnvCfg,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_rollers_usd = pytest.mark.usefixtures("microduck_rollers_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-Velocity-Flat-MicroDuck-Rollers"

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
    ("wheel_vel", 4),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
]
"""Privileged critic observation layout, including task-only state outside the deployment interface."""

CRITIC_OBSERVATION_DIM = 78
"""Expected width of the assembled critic observation group."""

STEPS_PER_ITERATION = 24
"""Environment control steps per PPO iteration."""

EPISODE_CONTROL_STEPS = 1000
"""Control steps in an episode: 20 s at 50 Hz, the velocity task's, inherited untouched."""

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
"""The four neck/head servos in command order."""

EXPECTED_WHEEL_JOINT_NAMES = ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]
"""The 4 passive wheel hinges, the configured ``^passive_.*wheel``."""

EXPECTED_TIRE_BODY_NAMES = ["tire", "tire_2", "tire_3", "tire_4"]
"""The 4 tire bodies in per-foot order: the left pair first, as the configured foot slots are."""

EXPECTED_TRUNK_BODY_NAMES = ["trunk_base"]
"""Trunk body used for height, tilt, angular velocity and mass-property randomization."""

EXPECTED_HEAD_BODY_COUNT = 4
"""Expected body count for head center-of-mass randomization."""

EXPECTED_ENTITY_SELECTIONS = {
    # "<manager>.<term>.<param>": (kind, expected names, preserve_order)
    "rewards.upright.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.pose.asset_cfg": ("joint", MICRODUCK_ROLLERS_JOINT_NAMES, True),
    "rewards.body_ang_vel.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.self_collisions.sensor_cfg": ("sensor", "self_collision", False),
    "rewards.feet_flat.asset_cfg": ("body", MICRODUCK_FOOT_BODY_NAMES, True),
    "rewards.feet_flat.sensor_cfg": ("sensor_bodies", EXPECTED_TIRE_BODY_NAMES, True),
    "rewards.neck_action_rate_l2.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.neck_joint_pos_l2.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.joint_torques_l2.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "rewards.hip_roll_neutral.asset_cfg": ("joint", ["left_hip_roll", "right_hip_roll"], True),
    "rewards.wheel_speed.asset_cfg": ("joint", EXPECTED_WHEEL_JOINT_NAMES, True),
    "rewards.skating_air_time.sensor_cfg": ("sensor_bodies", EXPECTED_TIRE_BODY_NAMES, True),
    "rewards.glide.sensor_cfg": ("sensor_bodies", EXPECTED_TIRE_BODY_NAMES, True),
    "rewards.glide.asset_cfg": ("joint", EXPECTED_LEG_JOINT_NAMES, True),
    "rewards.single_support.sensor_cfg": ("sensor_bodies", EXPECTED_TIRE_BODY_NAMES, True),
    "rewards.gait_symmetry.sensor_cfg": ("sensor_bodies", EXPECTED_TIRE_BODY_NAMES, True),
    "events.mass_inertia.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_wheel_friction.asset_cfg": ("joint", EXPECTED_WHEEL_JOINT_NAMES, True),
    "events.randomize_com.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "events.randomize_armature.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
}
"""Every entity selection the roller recipe makes, outside the observation groups.

Three of them are the task itself. ``rewards.wheel_speed.asset_cfg`` is the only positive task
reward's measurement; ``events.randomize_armature.asset_cfg`` is where the wheels have to be
*excluded*, because their armature is the bearing model rather than a servo's rotor; and every
``sensor_bodies`` entry pins the tire order the per-foot folding depends on.
"""

EXPECTED_OBSERVATION_SELECTIONS = {
    # "<group>.<term>": (kind, expected names)
    "policy.joint_pos": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "policy.joint_vel": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "critic.joint_pos": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "critic.joint_vel": ("joint", EXPECTED_SERVO_JOINT_NAMES),
    "critic.wheel_vel": ("joint", EXPECTED_WHEEL_JOINT_NAMES),
    "critic.foot_air_time": ("sensor", EXPECTED_TIRE_BODY_NAMES),
    "critic.foot_contact": ("sensor", EXPECTED_TIRE_BODY_NAMES),
    "critic.foot_contact_forces": ("sensor", EXPECTED_TIRE_BODY_NAMES),
}
"""Entity selections inside the two observation groups, all of which are ordering contracts."""


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

_STD_STANDING = {
    ".*hip_yaw.*": 0.05,
    ".*hip_roll.*": 0.05,
    ".*hip_pitch.*": 0.05,
    ".*knee.*": 0.05,
    ".*ankle.*": 0.05,
    ".*neck.*": 0.05,
    ".*head.*": 0.05,
    ".*passive_.*": 999.0,
}
_STD_WALKING = {
    ".*hip_yaw.*": 0.3,
    ".*hip_roll.*": 0.6,
    ".*hip_pitch.*": 0.4,
    ".*knee.*": 0.4,
    ".*ankle.*": 0.25,
    ".*neck.*": 0.05,
    ".*head.*": 0.05,
    ".*passive_.*": 999.0,
}
_STD_RUNNING = {
    ".*hip_yaw.*": 0.5,
    ".*hip_roll.*": 0.8,
    ".*hip_pitch.*": 0.8,
    ".*knee.*": 0.8,
    ".*ankle.*": 0.5,
    ".*neck.*": 0.05,
    ".*head.*": 0.05,
    ".*passive_.*": 999.0,
}

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "upright": (2.0, {"std": math.sqrt(0.2)}),
    "pose": (
        2.0,
        {
            "command_name": "base_velocity",
            "std_standing": _STD_STANDING,
            "std_walking": _STD_WALKING,
            "std_running": _STD_RUNNING,
            "walking_threshold": 0.01,
            "running_threshold": 0.5,
        },
    ),
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    "action_rate_l2": (-1.0, {}),
    "com_height_target": (2.0, {"target_height_min": 0.0935, "target_height_max": 0.1235}),
    "self_collisions": (-1.0, {}),
    "feet_flat": (-2.0, {"normal_axis": MICRODUCK_FOOT_NORMAL_AXIS, "bodies_per_foot": 2}),
    "neck_action_rate_l2": (-0.5, {"action_name": "joint_pos"}),
    "neck_joint_pos_l2": (-0.5, {}),
    "joint_torques_l2": (-1e-3, {}),
    "action_over_limit": (-0.5, {"action_name": "joint_pos", "overshoot": 0.3}),
    "hip_roll_neutral": (-2.0, {}),
    "wheel_speed": (10.0, {"command_name": "base_velocity", "vel_scale": 0.3, "wheel_radius": 0.0175}),
    "braking": (1.0, {"command_name": "base_velocity", "vel_std": 0.3}),
    "skating_air_time": (
        1.5,
        {
            "command_name": "base_velocity",
            "threshold_min": 0.15,
            "threshold_max": 0.45,
            "vel_gate_ref": 0.2,
            "bodies_per_foot": 2,
        },
    ),
    "glide": (4.0, {"command_name": "base_velocity", "vel_ref": 0.2, "bodies_per_foot": 2}),
    "single_support": (3.0, {"command_name": "base_velocity", "vel_gate_ref": 0.2, "bodies_per_foot": 2}),
    "gait_symmetry": (-1.0, {"bodies_per_foot": 2}),
    "forward_lean": (1.5, {"command_name": "base_velocity", "target_pitch": 0.262, "std": 0.1}),
    "heading_hold": (1.0, {"std": 0.4}),
}
"""Initial reward terms, weights and scalar parameters, including inherited terms."""

EXPECTED_DELETED_REWARDS = [
    # the two velocity-tracking terms: ``cmd_x`` is a throttle here, not a speed to track
    "track_lin_vel",
    "track_ang_vel",
    # the walking gait terms, replaced by the skating ones
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    # The roller task omits this regularizer and uses action_over_limit for over-command costs.
    "dof_pos_limits",
    # no head-pose or body-pose command to track
    "head_pose_tracking",
    "head_pose_bias",
    "body_pose_tracking",
]
"""Velocity reward terms absent from the roller task."""

EXPECTED_EVENTS = {
    # name: (mode, scalar params)
    "encoder_bias": ("startup", {"bias_range": (-0.015, 0.015)}),
    "mass_inertia": (
        "startup",
        {"mass_distribution_params": (0.95, 1.05), "operation": "scale", "recompute_inertia": True},
    ),
    "reset_base": (
        "reset",
        {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    ),
    "reset_robot_joints": ("reset", {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}),
    "randomize_wheel_friction": ("reset", {"friction_range": (0.0, 0.0)}),
    "randomize_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_head_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_joint_friction": ("reset", {"scale_range": (0.9, 1.1)}),
    "randomize_armature": ("reset", {"armature_distribution_params": (0.9, 1.1), "operation": "scale"}),
    "push_robot": ("interval", {"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}}),
}
"""Configured event modes and scalar parameters."""

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "randomize_wheel_friction",
    "randomize_com",
    "randomize_head_com",
    "randomize_joint_friction",
    "randomize_armature",
]
"""Reset events in their required execution order."""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params)
    "time_out": (True, {}),
    "fell_over": (False, {"limit_angle": math.radians(70.0)}),
    "nan_state": (False, {"sensor_names": ("contact_forces",)}),
}
"""Configured termination names, timeout flags and thresholds."""

EXPECTED_CURRICULUM_TERMS = {"action_rate_weight", "wheel_friction", "com_range", "head_com_range"}
"""Configured curriculum term names."""

EXPECTED_WEIGHT_STAGES = {"action_rate_weight": ([-1.0, -1.5, -2.0], [0, 250, 500])}
"""Reward weights at each curriculum boundary."""

EXPECTED_RANGE_STAGES = {
    "com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
    "head_com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
}
"""Center-of-mass randomization ranges at each curriculum boundary."""

EXPECTED_WHEEL_FRICTION_STAGES = [
    (0, {"friction_range": (0.0, 0.0)}),
    (2000, {"friction_range": (5e-4, 5e-4)}),
    (3500, {"friction_range": (1e-3, 1e-3)}),
    (5000, {"friction_range": (1.5e-3, 1.5e-3)}),
]
"""The configured bearing-drag ramp: free wheels until iteration 2000, then a gentle, realistic drag."""

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
    rewards = MicroDuckVelocityRollersFlatEnvCfg().rewards

    # two-sided, so a walking term the ``keep`` set should have removed also fails
    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name in EXPECTED_DELETED_REWARDS:
        assert not hasattr(rewards, name), name
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_wheel_speed_reward_is_the_only_positive_task_reward():
    """Nothing else pays for going anywhere, which is why the runner triples the entropy bonus."""
    rewards = MicroDuckVelocityRollersFlatEnvCfg().rewards

    assert rewards.wheel_speed.weight == pytest.approx(10.0)
    assert rewards.wheel_speed.params["asset_cfg"].joint_names == EXPECTED_WHEEL_JOINT_NAMES
    # the stroke terms all pay zero at a zero throttle or a stationary robot, so none of them is a
    # substitute: each is gated on the command, on forward progress, or on both
    for name in ("skating_air_time", "single_support", "glide"):
        assert "command_name" in getattr(rewards, name).params, name
    for name, key in (("skating_air_time", "vel_gate_ref"), ("single_support", "vel_gate_ref"), ("glide", "vel_ref")):
        assert getattr(rewards, name).params[key] > 0.0, name


@pytest.mark.unit
def test_the_two_known_stale_upstream_numbers_are_reproduced_verbatim():
    """The configured COM-height band remains below the roller standing height and requests a crouch.
    The 0.0175 m reward radius differs from the 0.0150 m geometric wheel radius.
    """
    rewards = MicroDuckVelocityRollersFlatEnvCfg().rewards

    band = (
        rewards.com_height_target.params["target_height_min"],
        rewards.com_height_target.params["target_height_max"],
    )
    assert band == (0.0935, 0.1235)
    # The target COM-height band remains below the roller standing height.
    assert band[1] < MICRODUCK_ROLLERS_STANDING_HEIGHT
    assert rewards.wheel_speed.params["wheel_radius"] == pytest.approx(0.0175)


@pytest.mark.unit
def test_the_spawn_height_puts_the_tires_on_the_ground_rather_than_under_it():
    """The asset ships the walking model's 0.125 m and warns that a roller task must override it."""
    cfg = MicroDuckVelocityRollersFlatEnvCfg()

    spawn_z = cfg.scene.robot.init_state.pos[2]
    assert spawn_z == pytest.approx(MICRODUCK_ROLLERS_SPAWN_HEIGHT)
    # the asset's own default is untouched: this is a task-level override, not an asset fix
    assert MICRODUCK_CFG.init_state.pos[2] == pytest.approx(0.125)
    # and the reset band around it is the configured absolute (0.1335, 0.1435)
    low, high = cfg.events.reset_base.params["pose_range"]["z"]
    assert (spawn_z + low, spawn_z + high) == pytest.approx((0.1335, 0.1435))
    # which brackets the height the robot actually stands at
    assert spawn_z + low <= MICRODUCK_ROLLERS_STANDING_HEIGHT <= spawn_z + high


@pytest.mark.unit
def test_the_posture_reward_still_selects_the_wheels_and_neutralizes_them_by_tolerance():
    """Wheel joints remain in the posture mean and are neutralized by tolerance; dropping them changes its scale."""
    pose = MicroDuckVelocityRollersFlatEnvCfg().rewards.pose

    joint_names = pose.params["asset_cfg"].joint_names
    assert joint_names == MICRODUCK_ROLLERS_JOINT_NAMES
    assert len(joint_names) == 18
    assert set(EXPECTED_WHEEL_JOINT_NAMES).issubset(joint_names)
    for key in ("std_standing", "std_walking", "std_running"):
        assert pose.params[key][".*passive_.*"] == pytest.approx(999.0), key
    # the running regime is genuinely reachable here, unlike on the walking task
    assert pose.params["running_threshold"] == pytest.approx(0.5)
    # and the hip-roll tolerance is loosened for the lateral push
    assert pose.params["std_walking"][".*hip_roll.*"] > pose.params["std_standing"][".*hip_roll.*"]


@pytest.mark.unit
def test_the_event_suite_matches_upstream_term_for_term():
    """Event modes and parameter ranges must match the expected configuration."""
    events = MicroDuckVelocityRollersFlatEnvCfg().events

    assert set(vars(events)) == set(EXPECTED_EVENTS)
    assert not hasattr(events, "foot_friction")
    for name, (mode, params) in EXPECTED_EVENTS.items():
        term = getattr(events, name)
        assert term.mode == mode, name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == value, f"{name}.{key}"
    assert events.push_robot.interval_range_s == (3.0, 6.0)


@pytest.mark.unit
def test_the_reset_events_fire_in_upstream_order():
    """Isaac Lab fires reset events in declaration order, and this order is the configured reset chain."""
    events = MicroDuckVelocityRollersFlatEnvCfg().events

    reset_terms = [name for name, term in vars(events).items() if getattr(term, "mode", None) == "reset"]
    assert reset_terms == EXPECTED_RESET_EVENT_ORDER


@pytest.mark.unit
def test_the_armature_randomization_leaves_the_wheel_bearings_alone():
    """Wheel-bearing armature remains 1e-4 while servo armature is randomized."""
    events = MicroDuckVelocityRollersFlatEnvCfg().events

    randomized = events.randomize_armature.params["asset_cfg"].joint_names
    assert randomized == EXPECTED_SERVO_JOINT_NAMES
    assert not set(randomized) & set(EXPECTED_WHEEL_JOINT_NAMES)
    # the bearings' own randomization is the dry-friction one, and it selects exactly the wheels
    assert events.randomize_wheel_friction.params["asset_cfg"].joint_names == EXPECTED_WHEEL_JOINT_NAMES
    assert events.randomize_wheel_friction.func is mdp.randomize_joint_dry_friction


@pytest.mark.unit
def test_the_terms_select_the_joints_bodies_and_sensors_upstream_measures():
    """A term that measures the wrong body is as wrong as one carrying the wrong weight."""
    cfg = MicroDuckVelocityRollersFlatEnvCfg()

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
        if kind == "sensor_bodies":
            assert entity_cfg.name == "contact_forces", path
            assert entity_cfg.body_names == expected, path
            assert entity_cfg.preserve_order is preserve_order, path
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
def test_both_observation_groups_read_the_servos_the_wheels_and_the_tires_in_a_pinned_order():
    """Isaac Lab resolves joints and bodies in USD order; every one of these orders is a contract."""
    observations = MicroDuckVelocityRollersFlatEnvCfg().observations
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
    """Head center-of-mass randomization must select the intended head bodies without a hip link."""
    body_names = MicroDuckVelocityRollersFlatEnvCfg().events.randomize_head_com.params["asset_cfg"].body_names

    assert "bearing_roll" not in body_names
    assert len(body_names) == EXPECTED_HEAD_BODY_COUNT


@pytest.mark.unit
def test_the_terminations_keep_a_real_failure_check():
    """Falling over is a failure on skates, unlike on the stand-up and roll tasks."""
    terminations = MicroDuckVelocityRollersFlatEnvCfg().terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        assert _scalar_params(term) == params, name
    assert not hasattr(terminations, "out_of_terrain_bounds")


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """Scheduled values must change at the expected control-step boundaries."""
    curriculum = MicroDuckVelocityRollersFlatEnvCfg().curriculum

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
def test_the_wheel_friction_curriculum_keeps_the_bearings_free_until_skating_is_robust():
    """The configured earlier schedule added drag exactly when the wheel-speed reward peaked, and broke it."""
    curriculum = MicroDuckVelocityRollersFlatEnvCfg().curriculum
    stages = curriculum.wheel_friction.params["param_stages"]

    assert len(stages) == len(EXPECTED_WHEEL_FRICTION_STAGES)
    for stage, (iteration, params) in zip(stages, EXPECTED_WHEEL_FRICTION_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        assert stage["params"] == params
    # Wheel-friction stages use strict boundaries; event-parameter stages use inclusive boundaries.
    assert curriculum.wheel_friction.params["inclusive"] is False


@pytest.mark.unit
def test_the_command_is_a_throttle_with_its_turning_clamped_away():
    """``cmd_x`` is 0 to coast, positive to push, negative to brake -- not a velocity to track."""
    cfg = MicroDuckVelocityRollersFlatEnvCfg()

    assert set(vars(cfg.commands)) == {"base_velocity"}
    command = cfg.commands.base_velocity
    assert isinstance(command, mdp.RelativeHeadingVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (-0.5, 0.6)
    # a skate cannot translate sideways, and the yaw slot's range is the clamp on a heading *error*
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.ranges.heading is None
    assert command.heading_command is False
    assert command.resampling_time_range == (3.0, 8.0)
    assert command.rel_standing_envs == 0.0
    assert command.rel_turn_in_place_envs == 0.0
    # Forward-only resampling retains the configured 0.3 m/s lower bound.
    # Six roller reward terms read this throttle command.
    assert command.rel_forward_envs == pytest.approx(0.2)

    for group in (cfg.observations.policy, cfg.observations.critic):
        terms = _observation_terms(group)
        for name, dim in (("head_pose_commands", 4), ("body_pose_commands", 6)):
            assert terms[name].func is mdp.zero_command_padding
            assert terms[name].params == {"dim": dim}


@pytest.mark.unit
def test_the_actor_observation_layout_is_the_shared_deploy_contract():
    """The skating policy reads the same 61-wide vector every other MicroDuck policy does."""
    observations = MicroDuckVelocityRollersFlatEnvCfg().observations
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
def test_the_critic_group_adds_the_wheel_speeds_and_guards_its_sensor_reads():
    """The wheel speeds are privileged: no sensor on the robot reports them."""
    observations = MicroDuckVelocityRollersFlatEnvCfg().observations
    terms = _observation_terms(observations.critic)

    assert list(terms) == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert not observations.critic.enable_corruption
    assert "foot_height" not in terms
    assert terms["wheel_vel"].func is mdp.joint_vel_rel
    # a deliberate deviation: the task guards these on the stand-up task only
    assert terms["foot_air_time"].func is mdp.foot_group_air_time_safe
    assert terms["foot_contact_forces"].func is mdp.foot_group_contact_forces_safe
    # a privileged group the runner never reads is dead weight
    assert MicroDuckRollersPPORunnerCfg().obs_groups == {"actor": ["policy"], "critic": ["critic"]}


@pytest.mark.unit
def test_the_foot_terms_fold_the_two_tires_of_a_bogie_into_one_foot():
    """Fold both tire contacts of each bogie into one left/right foot state."""
    cfg = MicroDuckVelocityRollersFlatEnvCfg()

    assert MICRODUCK_TIRES_PER_FOOT == 2
    assert len(MICRODUCK_TIRE_BODY_NAMES) == 2 * len(MICRODUCK_FOOT_BODY_NAMES)
    for term in (
        cfg.rewards.feet_flat,
        cfg.rewards.skating_air_time,
        cfg.rewards.glide,
        cfg.rewards.single_support,
        cfg.rewards.gait_symmetry,
        _observation_terms(cfg.observations.critic)["foot_air_time"],
        _observation_terms(cfg.observations.critic)["foot_contact"],
        _observation_terms(cfg.observations.critic)["foot_contact_forces"],
    ):
        assert term.params["bodies_per_foot"] == MICRODUCK_TIRES_PER_FOOT
    # the ankle bodies are the *frame* the blade flatness is measured in; they carry no collider
    assert cfg.rewards.feet_flat.params["asset_cfg"].body_names == MICRODUCK_FOOT_BODY_NAMES
    assert cfg.rewards.feet_flat.params["sensor_cfg"].body_names == MICRODUCK_TIRE_BODY_NAMES


@pytest.mark.unit
def test_the_scene_watches_the_tires_because_the_ankles_carry_no_collider():
    """A sensor on the ankle bodies would report zero force forever on this model."""
    scene = MicroDuckVelocityRollersFlatEnvCfg().scene

    assert scene.robot.spawn.usd_path == MICRODUCK_ROLLERS_USD_PATH
    assert scene.terrain.terrain_type == "plane"
    assert scene.terrain.terrain_generator is None
    for tire in MICRODUCK_TIRE_BODY_NAMES:
        assert tire in scene.contact_forces.prim_path, tire
    for ankle in MICRODUCK_FOOT_BODY_NAMES:
        assert ankle not in scene.contact_forces.prim_path, ankle
    assert scene.contact_forces.track_air_time
    # the self-collision sensor needs a filter, or no force matrix is produced at all
    assert scene.self_collision.filter_prim_paths_expr
    # Contact and constraint capacities use measured bounds.
    solver = MicroDuckVelocityRollersFlatEnvCfg().sim.physics.default.solver_cfg
    assert solver.nconmax >= 26
    assert solver.njmax >= 83


@pytest.mark.unit
def test_the_runner_raises_the_entropy_bonus_and_keeps_the_family_hyper_parameters():
    """The only task in the family that raises exploration, because the task reward starts at zero."""
    runner = MicroDuckRollersPPORunnerCfg()

    assert runner.experiment_name == "microduck_velocity_rollers"
    assert runner.max_iterations == 50000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    assert runner.algorithm.entropy_coef == pytest.approx(0.03)
    # symmetry is off on this task, unlike the roll one
    assert runner.algorithm.symmetry_cfg is None
    assert runner.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert runner.actor.hidden_dims == [512, 256, 128]


@pytest.mark.unit
def test_the_episode_and_simulation_rates_match_upstream():
    """The task never overrides the velocity template's episode here, so it is 20 s of skating."""
    cfg = MicroDuckVelocityRollersFlatEnvCfg()

    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.episode_length_s == pytest.approx(20.0)
    assert cfg.episode_length_s / (cfg.sim.dt * cfg.decimation) == pytest.approx(EPISODE_CONTROL_STEPS)


@pytest.mark.unit
def test_the_servos_run_on_the_backend_native_path():
    """The skating task runs the BAM servos where every other MicroDuck task does.

    The tasks share one robot and one servo deployment, so a policy trained on one plant and evaluated
    against the other would silently compare different robots. The decimation is checked alongside
    because it is a precondition: the BAM delay line is actuator state, and
    :class:`~isaaclab_newton.physics.NewtonManager` refuses to CUDA-graph-capture stateful Newton
    actuators at a decimation of one and warns at an odd one.
    """
    cfg = MicroDuckVelocityRollersFlatEnvCfg()

    assert cfg.sim.use_newton_actuators is True
    assert cfg.decimation % 2 == 0


##
# Environment smoke tests
##


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_observation_and_action_widths_are_the_ones_their_contracts_name():
    """The actor group is the deployed 61-vector, the critic measures 78, and the action stays 14."""
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

        # the robot has 18 hinges and the action space is 14 of them: the wheels are not driven
        robot = env.unwrapped.scene["robot"]
        assert robot.num_joints == 18
        assert env.unwrapped.action_manager.total_action_dim == 14
        action_joints = [robot.joint_names[int(i)] for i in env.unwrapped.action_manager._terms["joint_pos"]._joint_ids]
        assert action_joints == EXPECTED_SERVO_JOINT_NAMES
        assert not set(action_joints) & set(EXPECTED_WHEEL_JOINT_NAMES)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_blade_normal_axis_is_the_one_the_flatness_reward_measures():
    """The selected body-frame axis must match the blade normal used by the flatness reward."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()
        robot = unwrapped.scene["robot"]

        # upright at the stand pose, so a flat blade's normal is vertical by construction
        pose = torch.zeros((unwrapped.num_envs, 7), device=unwrapped.device)
        pose[:, 0:3] = unwrapped.scene.env_origins
        pose[:, 2] += MICRODUCK_ROLLERS_STANDING_HEIGHT
        pose[:, 6] = 1.0
        robot.write_root_link_pose_to_sim_index(root_pose=pose)
        joint_pos = robot.data.default_joint_pos.torch.clone()
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
        unwrapped.sim.forward()

        body_ids = [list(robot.body_names).index(name) for name in MICRODUCK_FOOT_BODY_NAMES]
        foot_quat_w = robot.data.body_link_quat_w.torch[:, body_ids]
        gravity_dir_w = torch.nn.functional.normalize(robot.data.GRAVITY_VEC_W.torch, dim=-1)
        gravity_dir_b = math_utils.quat_apply_inverse(
            foot_quat_w, gravity_dir_w.unsqueeze(1).expand(-1, len(body_ids), -1)
        )
        normal = torch.tensor(MICRODUCK_FOOT_NORMAL_AXIS, device=unwrapped.device)
        alignment = torch.abs(torch.sum(gravity_dir_b * normal, dim=-1))

        # gravity lies within a few degrees of the configured normal on both feet...
        assert bool((alignment > 0.99).all()), f"blade normal is not vertical at the stand pose: {alignment.tolist()}"
        # ... and nowhere near either other body axis, so the choice of axis is not a coincidence
        for axis in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
            other = torch.abs(torch.sum(gravity_dir_b * torch.tensor(axis, device=unwrapped.device), dim=-1))
            assert bool((other < 0.2).all()), f"axis {axis} is also vertical: {other.tolist()}"
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_a_scripted_push_rolls_the_wheels_and_glides_further_than_braked_bearings():
    """A scripted push must rotate free wheel joints and glide farther than the braked-bearing control."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        # every randomization off: this measures the plant, not the reset distribution
        for name in list(vars(env_cfg.events)):
            if not name.startswith("_"):
                setattr(env_cfg.events, name, None)
        for name in list(vars(env_cfg.curriculum)):
            if not name.startswith("_"):
                setattr(env_cfg.curriculum, name, None)
        env_cfg.terminations.fell_over = None

        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()
        robot = unwrapped.scene["robot"]
        wheel_ids = [list(robot.joint_names).index(name) for name in EXPECTED_WHEEL_JOINT_NAMES]

        # environments 2 and 3 get braked bearings, which is the control
        braked = torch.tensor([2, 3], device=unwrapped.device)
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.full((len(braked), len(wheel_ids)), 0.5, device=unwrapped.device),
            joint_ids=wheel_ids,
            env_ids=braked,
        )

        # standing, at rest, then pushed forward at 0.3 m/s
        pose = torch.zeros((unwrapped.num_envs, 7), device=unwrapped.device)
        pose[:, 0:3] = unwrapped.scene.env_origins
        pose[:, 2] += MICRODUCK_ROLLERS_STANDING_HEIGHT
        pose[:, 6] = 1.0
        robot.write_root_link_pose_to_sim_index(root_pose=pose)
        push = torch.zeros((unwrapped.num_envs, 6), device=unwrapped.device)
        push[:, 0] = 0.3
        robot.write_root_link_velocity_to_sim_index(root_velocity=push)
        joint_pos = robot.data.default_joint_pos.torch.clone()
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
        unwrapped.sim.forward()

        start_x = robot.data.root_link_pos_w.torch[:, 0].clone()
        action = torch.zeros((unwrapped.num_envs, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
        peak_rate = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        with torch.inference_mode():
            for _ in range(20):
                env.step(action)
                peak_rate = torch.maximum(peak_rate, robot.data.joint_vel.torch[:, wheel_ids].abs().mean(dim=1))
        travelled = robot.data.root_link_pos_w.torch[:, 0] - start_x

        free_rate, braked_rate = peak_rate[:2], peak_rate[2:]
        free_travel, braked_travel = travelled[:2], travelled[2:]

        # 1. free bearings roll: the wheels are spun up by the ground alone, since nothing drives them
        assert bool((free_rate > 1.0).all()), f"the wheels never turned: {free_rate.tolist()}"
        # 2. braked bearings do not, by an order of magnitude. The bound is not zero because the
        #    first physics step of the push happens before the friction constraint is solved, which
        #    leaves the braked wheels a tenth of a radian per second of transient. 0.25 is a
        #    generous margin above that transient, not itself the contract: the assertion below
        #    (>=10x separation from the free bearings) is what actually pins the behavior, so a
        #    solver friction-staging change that shifts the transient does not need this bound
        #    hand-tuned to match.
        assert bool((braked_rate < 0.25).all()), f"the braked wheels turned anyway: {braked_rate.tolist()}"
        assert float(free_rate.min()) > 10.0 * float(braked_rate.max())
        # 3. and the rolling robot glides further than the skidding one
        assert float(free_travel.min()) > 1.2 * float(braked_travel.max()), (
            f"free {free_travel.tolist()} did not out-glide braked {braked_travel.tolist()}"
        )
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
