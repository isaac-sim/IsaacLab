# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe-parity and smoke tests for the contributed MicroDuck roller-slope environment.

The parity tests need neither the asset nor the simulator: they read the assembled configuration and
compare it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference_tasks4.md``. The expected values are spelled out rather
than imported from the configuration under test, so that a drifting value fails rather than agrees
with itself.

This task is built as a **delta on the skating recipe**, which is what upstream does too: its
factory calls the roller factory and then edits. So the tests come in two halves. The first pins
what is inherited -- the scene robot, the sensors, the action space and both observation groups --
and the *deletion list*, because upstream keeps exactly one inherited reward and a port that
silently kept a second one would still pass a term-by-term check of the nine it declares. The second
pins the slope layer: the terrain, the neutralized command, the rolling entry, the void floor and
the terrain-level ladder.

Two of the simulator-backed tests are the ones the kernel unit tests cannot cover. The rolling entry
is checked by **reading the wheel velocities back off the articulation** after a reset rather than by
asserting the write happened, and the environment origins are checked against the surface of the
terrain that was actually built.

Simulator-backed checks retrieve the prebuilt USD through Isaac Lab's native asset cache.
Recipe checks do not need an asset; source fidelity comparisons use optional read-only fixtures."""

import copy
import dataclasses
import math
import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym
import numpy as np
import pytest
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
from isaaclab_tasks.contrib.microduck.rollerslope.agents.rsl_rl_ppo_cfg import MicroDuckRollerSlopePPORunnerCfg
from isaaclab_tasks.contrib.microduck.rollerslope.rollerslope_env_cfg import (
    MICRODUCK_SLOPE_ENTRY_SPEED_X,
    MICRODUCK_SLOPE_FELL_OVER_LIMIT,
    MICRODUCK_SLOPE_NUM_ROWS,
    MICRODUCK_SLOPE_TERRAIN_CFG,
    MICRODUCK_SLOPE_TILE_SIZE,
    MICRODUCK_SLOPE_UPRIGHT_STD,
    MICRODUCK_SLOPE_VOID_FLOOR,
    MicroDuckRollerSlopeFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.rollerslope.terrain import (
    MICRODUCK_RAMP_DEG_MAX,
    MICRODUCK_RAMP_DEG_MIN,
    FlatRampTerrainCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import MicroDuckVelocityRollersFlatEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_rollers_usd = pytest.mark.usefixtures("microduck_rollers_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-RollerSlope-Flat-MicroDuck"

STEPS_PER_ITERATION = 24
ACTOR_OBSERVATION_DIM = 61
CRITIC_OBSERVATION_DIM = 78

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
EXPECTED_WHEEL_JOINT_NAMES = ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]
EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
EXPECTED_FOOT_BODY_NAMES = ["ankle_l_v1", "ankle_r_v1"]
EXPECTED_TIRE_BODY_NAMES = ["tire", "tire_2", "tire_3", "tire_4"]
EXPECTED_TRUNK_BODY_NAMES = ["trunk_base"]

EXPECTED_REWARDS = {
    # name: (weight, scalar params) -- addendum section 5.3, the nine terms upstream trains with
    "action_rate_l2": (-1.0, {}),
    "upright": (3.0, {"std": math.sqrt(0.08)}),
    "alive": (1.0, {}),
    "wheel_glide": (2.0, {"cap_speed": 0.35}),
    "heading_hold": (1.5, {"std": 0.4}),
    "feet_flat": (-2.0, {"normal_axis": (0.0, 1.0, 0.0), "bodies_per_foot": 2}),
    "neck_action_rate_l2": (-0.5, {"action_name": "joint_pos"}),
    "neck_joint_pos_l2": (-0.75, {}),
    "joint_torques_l2": (-1e-3, {}),
}
"""Upstream's nine-term reward recipe (addendum section 5.3), keyed by term name.

No curriculum touches any of these weights, so unlike four of the tasks in the previous batch the
declared literals **are** the live values -- ``action_rate_l2`` really is -1.0 from step 0, where the
skating recipe ramps it to -2.0.
"""

EXPECTED_ABSENT_REWARDS = {
    # the skating recipe's terms that upstream's ``keep`` set deletes and never declares again
    "pose",
    "body_ang_vel",
    "angular_momentum",
    "com_height_target",
    "self_collisions",
    "action_over_limit",
    "hip_roll_neutral",
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
}
"""The fourteen skating terms that do not survive into this task (addendum section 5.3).

Upstream's ``keep = {"action_rate_l2"}`` deletes **twenty** inherited terms and then declares six of
them again with fresh parameters, so "deleted" is not the same as "absent" and only these fourteen
are gone. The deletion is the design: no fixed-pose reward, so the robot is free to fold and lean into
the slope instead of being told to hold the flat-ground stand.
"""

EXPECTED_ROLLER_VERBATIM_REWARDS = ("feet_flat", "neck_action_rate_l2", "joint_torques_l2")
"""The three redeclared terms that restate the skating recipe character for character.

They are compared against the skating task rather than against a transcribed literal, so that a
change to the proven roller recipe reaches this task's test instead of the two drifting apart.
"""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params) -- addendum section 5.4
    "time_out": (True, {}),
    "fell_over": (False, {"limit_angle": 1.0}),
    "nan_state": (False, {"sensor_names": ("contact_forces",)}),
    "fell_into_void": (False, {"minimum_height": -3.4117618741296187}),
}

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "randomize_wheel_friction",
    "randomize_com",
    "randomize_head_com",
    "randomize_joint_friction",
    "randomize_armature",
    "reset_rolling_entry",
]
"""The reset chain, in the order it fires.

This is behaviour, not housekeeping: :func:`~isaaclab_tasks.contrib.microduck.pollen_mdp.events.reset_rolling_entry`
writes the whole root velocity six-vector, so it has to run after ``reset_base`` places the base.
Upstream appends it last too, and guarantees the order the same way -- by declaration order.
"""

EXPECTED_CURRICULUM_TERMS = {"terrain_levels"}
"""One term, with no parameters (addendum section 5.6). Every skating schedule is deleted."""

EXPECTED_ENTITY_SELECTIONS = {
    # "<manager>.<term>.<param>": (kind, expected names, preserve_order)
    "rewards.upright.asset_cfg": ("body", EXPECTED_TRUNK_BODY_NAMES, False),
    "rewards.wheel_glide.asset_cfg": ("joint", EXPECTED_WHEEL_JOINT_NAMES, True),
    "rewards.feet_flat.asset_cfg": ("body", EXPECTED_FOOT_BODY_NAMES, True),
    "rewards.feet_flat.sensor_cfg": ("sensor", EXPECTED_TIRE_BODY_NAMES, True),
    "rewards.neck_action_rate_l2.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.neck_joint_pos_l2.asset_cfg": ("joint", EXPECTED_HEAD_JOINT_NAMES, True),
    "rewards.joint_torques_l2.asset_cfg": ("joint", EXPECTED_SERVO_JOINT_NAMES, True),
    "events.reset_rolling_entry.asset_cfg": ("joint", EXPECTED_WHEEL_JOINT_NAMES, True),
}
"""Every entity selection the slope layer makes.

``events.reset_rolling_entry.asset_cfg`` is the load-bearing one: the event spins *whatever it is
given* at ``v / r``, and it carries no guard against a slice selection, so a default
``SceneEntityCfg("robot")`` would launch all eighteen hinges -- servos included -- at 20 rad/s.
Naming the four wheels is what makes that unexpressible.
"""

EXPECTED_WHEEL_SELECTING_TERMS = {
    # the only places on this task where a passive hinge is legitimately selected
    "rewards.wheel_glide.asset_cfg",
    "events.randomize_wheel_friction.asset_cfg",
    "events.reset_rolling_entry.asset_cfg",
    "observations.critic.wheel_vel.asset_cfg",
}
"""Where a ``passive_*`` hinge may appear. Everywhere else it would silently change a mean or a sum."""

RAMP_LENGTH_PIN = 5.0
"""Ramp length [m] the accuracy-gate regimes pin, used here to make a generated tile reproducible."""


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which are pinned by name in their own test."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _entity_selections(cfg) -> dict[str, SceneEntityCfg]:
    """Every ``SceneEntityCfg`` reachable from the reward, event and observation managers, by path."""
    found: dict[str, SceneEntityCfg] = {}
    for manager in ("rewards", "events", "terminations", "curriculum"):
        for term_name, term in vars(getattr(cfg, manager)).items():
            if term is None:
                continue
            for key, value in getattr(term, "params", {}).items():
                if isinstance(value, SceneEntityCfg):
                    found[f"{manager}.{term_name}.{key}"] = value
    for group in ("policy", "critic"):
        for term_name, term in vars(getattr(cfg.observations, group)).items():
            params = getattr(term, "params", None)
            if not isinstance(params, dict):
                continue
            for key, value in list(params.items()) + list(params.get("term_params", {}).items()):
                if isinstance(value, SceneEntityCfg):
                    found[f"observations.{group}.{term_name}.{key}"] = value
    return found


def _selection_names(entity_cfg: SceneEntityCfg, kind: str) -> list[str] | None:
    return {"joint": entity_cfg.joint_names, "body": entity_cfg.body_names, "sensor": entity_cfg.body_names}[kind]


def _built_terrain(difficulty_range: tuple[float, float] = (0.0, 1.0)) -> TerrainGenerator:
    """Run the shipped terrain generator on the CPU, with the only random draw pinned.

    The ramp length is drawn from the *global* numpy state, which the generator does not seed, so the
    range is pinned rather than the seed -- the same pin the accuracy-gate regimes use.
    """
    cfg = copy.deepcopy(MICRODUCK_SLOPE_TERRAIN_CFG)
    cfg.difficulty_range = difficulty_range
    cfg.sub_terrains["flat_ramp"].ramp_length_range = (RAMP_LENGTH_PIN, RAMP_LENGTH_PIN)
    cfg.seed = 0
    return TerrainGenerator(cfg, device="cpu")


def _surface_height_under(mesh: trimesh.Trimesh, x: float, y: float) -> float:
    """Height [m] of the highest terrain surface under a point, measured by a downward ray."""
    hits, _, _ = mesh.ray.intersects_location(
        ray_origins=np.array([[x, y, 10.0]]), ray_directions=np.array([[0.0, 0.0, -1.0]])
    )
    if len(hits) == 0:
        return float("nan")
    return float(np.max(hits[:, 2]))


##
# The skating layer: inherited verbatim
##


@pytest.mark.unit
def test_the_scene_actions_and_observations_are_the_skating_task_untouched():
    """Everything but the terrain is inherited, and the actor never learns it is on a slope.

    Upstream edits the terrain, the commands, two events, the whole reward dict, the terminations,
    the observation NaN policy and the curriculum -- and **no** observation term, sensor,
    randomization term, actuator or asset (addendum sections 1 and 5.5). The 61-wide actor therefore
    sees the incline only through ``projected_gravity`` and proprioception, which is half of what
    made this task portable without a terrain-height sensor.
    """
    slope = MicroDuckRollerSlopeFlatEnvCfg()
    rollers = MicroDuckVelocityRollersFlatEnvCfg()

    assert slope.observations.to_dict() == rollers.observations.to_dict()
    assert slope.actions.to_dict() == rollers.actions.to_dict()
    assert slope.episode_length_s == pytest.approx(rollers.episode_length_s) == pytest.approx(20.0)
    assert slope.decimation == rollers.decimation
    assert slope.sim.dt == pytest.approx(rollers.sim.dt)
    assert slope.sim.use_newton_actuators is True

    # the scene differs in the terrain and in nothing else -- same robot, same two contact sensors
    slope_scene, roller_scene = slope.scene.to_dict(), rollers.scene.to_dict()
    assert [name for name in slope_scene if slope_scene[name] != roller_scene[name]] == ["terrain"]


@pytest.mark.unit
def test_the_randomization_suite_is_the_skating_one_with_a_rolling_entry_appended():
    """Upstream edits two reset events and adds one; every other randomization is inherited."""
    slope = MicroDuckRollerSlopeFlatEnvCfg()
    rollers = MicroDuckVelocityRollersFlatEnvCfg()

    slope_events, roller_events = slope.events.to_dict(), rollers.events.to_dict()
    assert set(slope_events) == set(roller_events) | {"reset_rolling_entry"}
    changed = [name for name in roller_events if slope_events[name] != roller_events[name]]
    assert changed == ["reset_base"]

    # the whole reset chain, in the order the manager fires it
    reset_events = [name for name, term in vars(slope.events).items() if term is not None and term.mode == "reset"]
    assert reset_events == EXPECTED_RESET_EVENT_ORDER


##
# The slope layer
##


@pytest.mark.unit
def test_the_reward_recipe_matches_upstream_term_for_term():
    """Every reward slot upstream trains with is present, with its weight and its parameters."""
    rewards = MicroDuckRollerSlopeFlatEnvCfg().rewards

    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_skating_reward_recipe_is_deleted_rather_than_inherited():
    """The deletion list is the design, and a derived configuration is where it can silently rot.

    Upstream keeps exactly one inherited term and rebuilds the rest, so a port that quietly left a
    second skating term in place would still pass a term-by-term check of the nine it declares. This
    is the test that catches it.
    """
    slope = MicroDuckRollerSlopeFlatEnvCfg().rewards
    rollers = MicroDuckVelocityRollersFlatEnvCfg().rewards

    inherited = set(vars(rollers))
    # twenty terms are deleted; six of them come back with fresh parameters
    assert len(inherited - {"action_rate_l2"}) == 20
    assert inherited - set(vars(slope)) == EXPECTED_ABSENT_REWARDS
    assert len(EXPECTED_ABSENT_REWARDS) == 14
    # nothing measuring a fixed pose or a commanded speed survives
    assert not {"pose", "com_height_target", "wheel_speed", "braking"} & set(vars(slope))

    # the self-collision *sensor* is still in the scene, unread: upstream keeps it too, because the
    # deletion is of the reward rather than of the sensor
    assert MicroDuckRollerSlopeFlatEnvCfg().scene.self_collision is not None


@pytest.mark.unit
def test_the_three_restated_skating_terms_are_that_task_verbatim_and_the_neck_one_is_not():
    """One weight breaks a copy-the-block port, and it is the easiest thing in this task to miss.

    ``feet_flat``, ``neck_action_rate_l2`` and ``joint_torques_l2`` restate the skating declarations
    character for character; ``neck_joint_pos_l2`` goes from -0.5 to **-0.75**, a fifty percent
    increase (addendum section 5.3).
    """
    slope = MicroDuckRollerSlopeFlatEnvCfg().rewards
    rollers = MicroDuckVelocityRollersFlatEnvCfg().rewards

    for name in EXPECTED_ROLLER_VERBATIM_REWARDS:
        ours, theirs = getattr(slope, name), getattr(rollers, name)
        assert ours.func is theirs.func, name
        assert ours.weight == pytest.approx(theirs.weight), name
        assert ours.to_dict() == theirs.to_dict(), name

    assert slope.neck_joint_pos_l2.func is rollers.neck_joint_pos_l2.func
    assert rollers.neck_joint_pos_l2.weight == pytest.approx(-0.5)
    assert slope.neck_joint_pos_l2.weight == pytest.approx(-0.75)
    assert slope.neck_joint_pos_l2.weight == pytest.approx(1.5 * rollers.neck_joint_pos_l2.weight)


@pytest.mark.unit
def test_the_upright_standard_deviation_matches_upstreams_kernel_at_vertical():
    """The two kernels measure different tilts, so the width is converted rather than copied.

    Upstream's ``body_upright_gaussian`` scores ``exp(-(1 - cos t) / std^2)`` at ``std = 0.2`` where
    ``mdp.upright`` scores ``exp(-sin^2(t) / std^2)`` (addendum section 4.6). Matching the exponents
    needs ``std_ours^2 = std_up^2 * (1 + cos t)``, which is exact only at one tilt; normalizing at
    vertical -- where the reward and its gradient matter most -- gives ``sqrt(2) * 0.2``.

    The derivation is redone here rather than transcribed, and the residual is measured, so the
    "slightly sharper off vertical" claim in the configuration is checked rather than asserted.
    """
    upstream_std = 0.2
    assert pytest.approx(math.sqrt(2.0) * upstream_std) == MICRODUCK_SLOPE_UPRIGHT_STD
    assert pytest.approx(math.sqrt(0.08)) == MICRODUCK_SLOPE_UPRIGHT_STD
    assert MicroDuckRollerSlopeFlatEnvCfg().rewards.upright.params["std"] == pytest.approx(MICRODUCK_SLOPE_UPRIGHT_STD)

    def upstream_score(tilt: float) -> float:
        return math.exp(-(1.0 - math.cos(tilt)) / upstream_std**2)

    def our_score(tilt: float) -> float:
        return math.exp(-(math.sin(tilt) ** 2) / MICRODUCK_SLOPE_UPRIGHT_STD**2)

    assert our_score(0.0) == pytest.approx(upstream_score(0.0)) == pytest.approx(1.0)
    # one-signed and bounded: ours decays *faster* off vertical, and by the tilt termination both
    # are numerically irrelevant because the episode has ended
    for degrees in (5.0, 15.0, 30.0):
        assert our_score(math.radians(degrees)) > upstream_score(math.radians(degrees)) - 1e-12
    assert our_score(math.radians(30.0)) == pytest.approx(0.0439, abs=1e-4)
    assert upstream_score(math.radians(30.0)) == pytest.approx(0.0351, abs=1e-4)
    # by the tilt termination both are four orders of magnitude below the vertical value, so the
    # residual cannot decide anything: the episode ends there
    assert max(our_score(MICRODUCK_SLOPE_FELL_OVER_LIMIT), upstream_score(MICRODUCK_SLOPE_FELL_OVER_LIMIT)) < 2e-4


@pytest.mark.unit
def test_the_twist_command_is_kept_alive_and_neutralized():
    """The term stays so the 61-wide observation keeps its three twist slots, and reads zero.

    Four of the five edits are upstream's (addendum section 5.1). The fifth --
    ``rel_forward_envs = 0.0`` -- is this port's, and it is the one worth pinning: upstream's forced
    forward bucket clamps the surge slot to at least 0.3 at resample time, and upstream's own command
    override drops the standing-environment zeroing that ``rel_standing_envs = 1.0`` was set to
    trigger, so on upstream's stack a fifth of the environments carry a 0.3 throttle for a whole
    resampling interval -- in an observation slot that no reward on this task reads.
    """
    slope = MicroDuckRollerSlopeFlatEnvCfg()
    rollers = MicroDuckVelocityRollersFlatEnvCfg()
    command = slope.commands.base_velocity

    assert type(command) is type(rollers.commands.base_velocity)
    assert command.rel_standing_envs == pytest.approx(1.0)
    assert command.rel_heading_envs == pytest.approx(0.0)
    assert command.ranges.lin_vel_x == (0.0, 0.0)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    # the skating task's throttle range is what is being neutralized, so it was not already zero
    assert rollers.commands.base_velocity.ranges.lin_vel_x == (-0.5, 0.6)

    # the deliberate deviation, and the value it deviates from
    assert command.rel_forward_envs == pytest.approx(0.0)
    assert rollers.commands.base_velocity.rel_forward_envs == pytest.approx(0.2)
    # everything else about the term is the skating task's
    assert command.resampling_time_range == rollers.commands.base_velocity.resampling_time_range
    assert command.heading_command is False

    # the observation slot is untouched, which is what keeps the actor at 61
    assert slope.observations.policy.velocity_commands.to_dict() == (
        rollers.observations.policy.velocity_commands.to_dict()
    )


@pytest.mark.unit
def test_the_rolling_entry_is_wired_after_the_base_reset_and_names_the_four_wheels():
    """The event has no guard against a slice selection, so the configuration is the guard.

    With a default ``SceneEntityCfg("robot")`` the joint selection is a slice and every hinge would
    be spun at ``v / r`` -- 14 to 26 rad/s on the servos. The four wheels are named instead, in the
    model's own order, with ``preserve_order`` so the selection cannot be silently reordered.
    """
    events = MicroDuckRollerSlopeFlatEnvCfg().events
    entry = events.reset_rolling_entry

    assert entry.func is mdp.reset_rolling_entry
    assert entry.mode == "reset"
    assert entry.params["speed_range"] == MICRODUCK_SLOPE_ENTRY_SPEED_X == (0.25, 0.45)
    assert entry.params["asset_cfg"].joint_names == EXPECTED_WHEEL_JOINT_NAMES
    assert entry.params["asset_cfg"].preserve_order is True
    # upstream takes the wheel radius default, which is the known-stale 0.0175 m (addendum 9.3)
    assert "wheel_radius" not in entry.params

    # the base reset faces down the slope and injects no velocity of its own, so the entry owns the
    # whole root velocity six-vector
    reset_base = events.reset_base
    assert reset_base.params["pose_range"]["yaw"] == (0.0, 0.0)
    assert reset_base.params["velocity_range"] == {}
    # the inherited horizontal jitter is upstream's and is deliberately *not* narrowed
    assert reset_base.params["pose_range"]["x"] == (-0.5, 0.5)
    assert reset_base.params["pose_range"]["y"] == (-0.5, 0.5)


@pytest.mark.unit
def test_the_terminations_tighten_the_tilt_limit_and_add_an_absolute_void_floor():
    """The void floor is the only height quantity in the task, and it is a world-frame one."""
    slope = MicroDuckRollerSlopeFlatEnvCfg()
    terminations = slope.terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"

    # 57.30 degrees, tightened from the skating task's 70
    assert pytest.approx(1.0) == MICRODUCK_SLOPE_FELL_OVER_LIMIT
    assert math.degrees(MICRODUCK_SLOPE_FELL_OVER_LIMIT) == pytest.approx(57.2958, abs=1e-4)
    assert MicroDuckVelocityRollersFlatEnvCfg().terminations.fell_over.params["limit_angle"] == pytest.approx(
        math.radians(70.0)
    )

    # the constant is derived, not transcribed: half a metre below the deepest reachable runout
    longest_ramp = MICRODUCK_SLOPE_TERRAIN_CFG.sub_terrains["flat_ramp"].ramp_length_range[1]
    deepest = -longest_ramp * math.tan(math.radians(MICRODUCK_RAMP_DEG_MAX))
    assert pytest.approx(deepest - 0.5) == MICRODUCK_SLOPE_VOID_FLOOR
    assert pytest.approx(-3.4117618741296187, abs=1e-12) == MICRODUCK_SLOPE_VOID_FLOOR
    assert terminations.fell_into_void.func is mdp.root_height_below_minimum


@pytest.mark.unit
def test_the_curriculum_is_the_slope_ladder_alone():
    """Every skating schedule is deleted, so the reward weights are static for the whole run."""
    slope = MicroDuckRollerSlopeFlatEnvCfg()

    assert set(vars(slope.curriculum)) == EXPECTED_CURRICULUM_TERMS
    assert slope.curriculum.terrain_levels.func is mdp.terrain_levels_slope
    assert slope.curriculum.terrain_levels.params == {}
    # the skating recipe's four schedules are gone, including the wheel-friction ramp
    assert set(vars(MicroDuckVelocityRollersFlatEnvCfg().curriculum)) == {
        "action_rate_weight",
        "wheel_friction",
        "com_range",
        "head_com_range",
    }
    # so the bearings stay at whatever the inherited reset event writes, which is zero
    assert slope.events.randomize_wheel_friction.params["friction_range"] == (0.0, 0.0)
    # and the generator has to lay its rows out by difficulty for the ladder to mean anything
    assert slope.scene.terrain.terrain_generator.curriculum is True


@pytest.mark.unit
def test_the_terrain_generator_reproduces_upstreams_tile_and_ladder():
    """The ten-row ladder, the tile that fits the longest ramp, and starting on the gentlest row."""
    terrain = MicroDuckRollerSlopeFlatEnvCfg().scene.terrain
    generator = terrain.terrain_generator

    assert terrain.terrain_type == "generator"
    assert terrain.max_init_terrain_level == 0
    assert generator.size == MICRODUCK_SLOPE_TILE_SIZE == (15.0, 4.0)
    assert generator.num_rows == MICRODUCK_SLOPE_NUM_ROWS == 10
    assert generator.num_cols == 1
    assert generator.difficulty_range == (0.0, 1.0)

    sub_terrain = generator.sub_terrains["flat_ramp"]
    assert isinstance(sub_terrain, FlatRampTerrainCfg)
    assert (sub_terrain.flat_length, sub_terrain.runout_length, sub_terrain.spawn_on_ramp) == (2.0, 4.0, 0.3)
    assert sub_terrain.ramp_length_range == (3.0, 8.0)
    assert (sub_terrain.deg_min, sub_terrain.deg_max) == (MICRODUCK_RAMP_DEG_MIN, MICRODUCK_RAMP_DEG_MAX)
    # the tile budget: the longest ramp plus its platforms fits with a metre to spare
    assert sub_terrain.flat_length + sub_terrain.ramp_length_range[1] + sub_terrain.runout_length == 14.0
    assert MICRODUCK_SLOPE_TILE_SIZE[0] >= 14.0

    # the live promotion gates, which upstream's own docstring miscalibrates against an 8 m tile
    assert MICRODUCK_SLOPE_TILE_SIZE[0] * 0.4 == pytest.approx(6.0)
    assert MICRODUCK_SLOPE_TILE_SIZE[0] * 0.2 == pytest.approx(3.0)


@pytest.mark.unit
def test_the_terrain_is_collided_against_as_a_heightfield():
    """Load-bearing, not a detail: against the raw mesh this robot's tires do not carry it.

    On the triangle mesh the generator emits, the roller model rides for about 0.1 s, sinks 45 mm
    and stops; on the heightfield it rides like it does on an analytic ground plane. The conversion
    is lossless here because the tile is piecewise planar with no overhangs, and it is what every
    stock Newton-validated terrain configuration does. Traces:
    ``artifacts/microduck/golden_trajectories/rollerslope/controls/``.
    """
    generator = MicroDuckRollerSlopeFlatEnvCfg().scene.terrain.terrain_generator

    assert generator.sub_terrains["flat_ramp"].convert_to_heightfield is True
    # the conversion is all-or-nothing across a generator's sub-terrains, so a second sub-terrain
    # that left the flag unset would silently drop the whole tile back to the mesh
    assert all(sub_cfg.convert_to_heightfield for sub_cfg in generator.sub_terrains.values())
    # the stock default is off, so this is a deliberate override rather than an inherited value
    stock_default = {field.name: field for field in dataclasses.fields(SubTerrainBaseCfg)}[
        "convert_to_heightfield"
    ].default_factory()
    assert stock_default is False
    # rasterization resolution: the generator's own horizontal_scale, left at the stock value
    assert generator.horizontal_scale == pytest.approx(0.1)


@pytest.mark.unit
def test_the_generated_environment_origins_lie_on_the_ramp_surface():
    """The whole reason this task needed new terrain, measured on the mesh the generator built.

    Nothing in the task senses terrain-relative height, so the *only* thing that has to be right
    about the slope is that the per-environment origin the reset events place the robot at sits on
    the incline. The origin is compared against a downward ray cast onto the assembled terrain, which
    exercises the generator's two frame shifts -- the sub-terrain re-centring and the grid offset --
    rather than the formula that produced it.
    """
    generator = _built_terrain()
    origins = generator.terrain_origins

    assert origins.shape == (MICRODUCK_SLOPE_NUM_ROWS, 1, 3)
    implied_degrees = []
    for row in range(MICRODUCK_SLOPE_NUM_ROWS):
        x, y, z = (float(value) for value in origins[row, 0])
        assert _surface_height_under(generator.terrain_mesh, x, y) == pytest.approx(z, abs=1e-6), f"row {row}"
        # on the incline rather than on the starting platform, so gravity starts the descent
        assert z < 0.0
        implied_degrees.append(math.degrees(math.atan(-z / 0.3)))

    # row r draws its difficulty in [r/10, (r+1)/10), which is an angle in [2 + 1.8r, 3.8 + 1.8r]
    for row, degrees in enumerate(implied_degrees):
        assert MICRODUCK_RAMP_DEG_MIN + 1.8 * row <= degrees <= MICRODUCK_RAMP_DEG_MIN + 1.8 * (row + 1)
    assert implied_degrees == sorted(implied_degrees)


@pytest.mark.unit
def test_a_pinned_difficulty_makes_every_row_the_same_ramp():
    """What the accuracy gate stands on: pinning the range removes the per-row difficulty draw."""
    for difficulty in (0.5, 1.0):
        generator = _built_terrain(difficulty_range=(difficulty, difficulty))
        expected_z = -0.3 * math.tan(
            math.radians(MICRODUCK_RAMP_DEG_MIN + difficulty * (MICRODUCK_RAMP_DEG_MAX - MICRODUCK_RAMP_DEG_MIN))
        )
        for row in range(MICRODUCK_SLOPE_NUM_ROWS):
            assert float(generator.terrain_origins[row, 0, 2]) == pytest.approx(expected_z, abs=1e-9)


@pytest.mark.unit
def test_the_entity_selections_name_the_entities_they_measure():
    """A term that measures the wrong joints is as wrong as one carrying the wrong weight."""
    cfg = MicroDuckRollerSlopeFlatEnvCfg()
    selections = _entity_selections(cfg)

    for path, (kind, expected, preserve_order) in EXPECTED_ENTITY_SELECTIONS.items():
        entity_cfg = selections[path]
        assert _selection_names(entity_cfg, kind) == expected, path
        assert bool(entity_cfg.preserve_order) is preserve_order, path


@pytest.mark.unit
def test_no_passive_hinge_reaches_a_selection_that_is_not_about_the_wheels():
    """On this model the wheels interleave with the servos, so a wide selector is not harmless.

    The reward terms are means and sums over their selections: a stray wheel in the torque penalty or
    the neck penalty would change the value of every sample, and a stray wheel in the action term
    would widen the deployed 14-vector.
    """
    cfg = MicroDuckRollerSlopeFlatEnvCfg()
    selections = _entity_selections(cfg)

    # the exemption list is only a guard if every path on it still exists
    assert set(selections) >= EXPECTED_WHEEL_SELECTING_TERMS
    for path, entity_cfg in selections.items():
        names = entity_cfg.joint_names or []
        if isinstance(names, str):
            names = [names]
        passive = [name for name in names if "passive" in name]
        if path in EXPECTED_WHEEL_SELECTING_TERMS:
            assert passive == EXPECTED_WHEEL_JOINT_NAMES, path
        else:
            assert passive == [], path

    # and the action term names the fourteen servos, which cannot pick up a wheel
    assert cfg.actions.joint_pos.joint_names == EXPECTED_SERVO_JOINT_NAMES
    assert not set(cfg.actions.joint_pos.joint_names) & set(EXPECTED_WHEEL_JOINT_NAMES)


@pytest.mark.unit
def test_the_contact_budget_is_measured_rather_than_inherited():
    """The skating budget is sized for one analytic contact patch per tire; a heightfield is not that.

    Rasterized at 0.1 m, a sprawled robot's colliders straddle cell boundaries and pick up two
    triangles per cell across several cells, so *both* inherited buffers sit below the measured peaks
    and this is a live fix rather than a cosmetic one.
    """
    solver = MicroDuckRollerSlopeFlatEnvCfg().sim.physics.default.solver_cfg
    inherited = MicroDuckVelocityRollersFlatEnvCfg().sim.physics.default.solver_cfg

    # measured peaks under random actions with the tilt termination dropped: 295 and 92
    assert solver.njmax >= 295
    assert solver.nconmax >= 92
    assert inherited.njmax < 295
    assert inherited.nconmax < 92
    # upstream's flat solver profile is inherited unchanged; only the buffers are resized
    assert (solver.iterations, solver.ls_iterations) == (inherited.iterations, inherited.ls_iterations) == (10, 20)


@pytest.mark.unit
def test_the_runner_keeps_the_family_hyper_parameters_under_its_own_log_tree():
    """Upstream's runner differs from the *velocity* one in two fields (addendum section 7).

    Deriving from the skating runner instead would silently triple the exploration noise: that class
    raises ``entropy_coef`` to 0.03 for the stroke, and this task declares the family's 0.01.
    """
    runner = MicroDuckRollerSlopePPORunnerCfg()

    assert runner.experiment_name == "microduck_rollerslope"
    assert runner.max_iterations == 8000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    assert runner.algorithm.entropy_coef == pytest.approx(0.01)
    assert runner.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert runner.algorithm.schedule == "adaptive"
    assert runner.algorithm.num_learning_epochs == 5
    assert runner.algorithm.num_mini_batches == 4
    assert runner.algorithm.symmetry_cfg is None
    assert runner.actor.hidden_dims == [512, 256, 128]
    assert runner.critic.hidden_dims == [512, 256, 128]
    assert runner.actor.obs_normalization and runner.critic.obs_normalization


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
        robot = env.unwrapped.scene["robot"]
        assert robot.num_joints == 18
        assert env.unwrapped.action_manager.total_action_dim == 14
        action_joints = [robot.joint_names[int(i)] for i in env.unwrapped.action_manager._terms["joint_pos"]._joint_ids]
        assert action_joints == EXPECTED_SERVO_JOINT_NAMES
        assert not set(action_joints) & set(EXPECTED_WHEEL_JOINT_NAMES)
        # The twist slots are alive and reading zero, which is what the neutralized command means --
        # checked straight out of the reset as well as after a step, because a command term is
        # resampled on reset and only zeroed on the following update, so the two are different reads.
        command = env.unwrapped.command_manager.get_command("base_velocity")
        assert command.shape[-1] == 3
        assert float(command.abs().max()) == pytest.approx(0.0, abs=1e-6)
        env.step(torch.zeros((2, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device))
        assert float(env.unwrapped.command_manager.get_command("base_velocity").abs().max()) == pytest.approx(
            0.0, abs=1e-6
        )
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_rolling_entry_leaves_the_wheels_turning_after_a_reset():
    """Read back from the articulation, not asserted from the write: the state is what matters.

    The event's unit tests run against a CPU double, which cannot see the interaction with the
    articulation's cached state -- whether the write survives the rest of the reset chain, and
    whether the buffers the rest of the environment reads report it. So the four wheel velocities and
    the root velocity are read *after* a full reset, and the rolling-without-slipping relation
    ``omega * r == v`` is checked per environment on values that came back out of the simulator.
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

        # prime the derived buffers *before* the reset, so a stale cache would be visible
        _ = robot.data.joint_vel.torch
        _ = robot.data.root_link_lin_vel_w.torch
        unwrapped.reset()

        wheel_ids, wheel_names = robot.find_joints(EXPECTED_WHEEL_JOINT_NAMES, preserve_order=True)
        assert wheel_names == EXPECTED_WHEEL_JOINT_NAMES
        wheel_vel = robot.data.joint_vel.torch[:, wheel_ids]
        root_vel_x = robot.data.root_link_lin_vel_w.torch[:, 0]

        speed_min, speed_max = MICRODUCK_SLOPE_ENTRY_SPEED_X
        wheel_radius = 0.0175  # upstream's default, which the configuration deliberately takes
        assert torch.isfinite(wheel_vel).all()
        # every wheel of an environment turns at the same rate, and that rate is v / r
        assert torch.allclose(wheel_vel, wheel_vel[:, :1].expand_as(wheel_vel), atol=1e-4)
        assert torch.allclose(wheel_vel[:, 0] * wheel_radius, root_vel_x, atol=1e-4)
        assert float(root_vel_x.min()) >= speed_min - 1e-4
        assert float(root_vel_x.max()) <= speed_max + 1e-4
        # a draw per environment rather than one shared draw
        assert float(root_vel_x.max() - root_vel_x.min()) > 1e-3
        # the servos are left alone, which is what naming the four wheels buys
        servo_ids, _ = robot.find_joints(EXPECTED_SERVO_JOINT_NAMES, preserve_order=True)
        assert float(robot.data.joint_vel.torch[:, servo_ids].abs().max()) < 1.0
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_environments_start_on_the_ramp_and_the_void_guard_stays_quiet_through_a_descent():
    """The sloped-terrain path end to end: the origin is on the incline and the robot rides it down.

    The difficulty is pinned to the steepest ramp, which is where the origin is furthest below the
    platform, where the descent is fastest and where a wrong void floor would be most likely to fire.
    Two things are asserted about the same rollout: the environment origins sit on the analytic ramp
    surface, and a robot released there **descends** -- so the on-ramp origin, the incline contact
    and the rolling entry are all exercised together rather than one at a time.

    ``fell_over`` is removed for the rollout, because it is what would otherwise end it: an untrained
    MicroDuck on skates topples within a second of leaving the reset, on a slope as on the flat. The
    guard under test is the void floor, which sits half a metre below the deepest reachable run-out
    and must never fire while the robot is anywhere on the terrain.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        # the steepest ramp, with the only random draw in the terrain pinned
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        env_cfg.scene.terrain.terrain_generator.sub_terrains["flat_ramp"].ramp_length_range = (
            RAMP_LENGTH_PIN,
            RAMP_LENGTH_PIN,
        )
        # spawn at the origin rather than in the inherited +/-0.5 m jitter, which on a 20 degree ramp
        # would start the trunk up to 182 mm above the surface or 109 mm inside the platform
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (-0.005, 0.005),
            "yaw": (0.0, 0.0),
        }
        # pin the entry draw to the band's midpoint, which is the accuracy gate's regime, so the
        # distance below is not also sampling the (0.25, 0.45) spread
        entry_speed = sum(MICRODUCK_SLOPE_ENTRY_SPEED_X) / 2.0
        env_cfg.events.reset_rolling_entry.params["speed_range"] = (entry_speed, entry_speed)
        env_cfg.terminations.fell_over = None
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()
        # The first reset scores the curriculum against a root pose the reset events have not written
        # yet -- the robot is still on the cloner's spacing grid, tens of metres from its terrain
        # origin -- so every environment is promoted once and demoted back on the next reset. That is
        # upstream's behaviour too: the same curriculum function against the same centred grid.
        unwrapped.reset()

        terrain = unwrapped.scene.terrain
        assert int(terrain.terrain_levels.max()) == 0
        origins = unwrapped.scene.env_origins
        row_zero_z = float(terrain.terrain_origins[0, 0, 2])
        assert torch.allclose(origins[:, 2], torch.full_like(origins[:, 2], row_zero_z))
        # every row is the same pinned 20 degree ramp, so the origin height is exact
        spawn_on_ramp = MICRODUCK_SLOPE_TERRAIN_CFG.sub_terrains["flat_ramp"].spawn_on_ramp
        assert row_zero_z == pytest.approx(-spawn_on_ramp * math.tan(math.radians(MICRODUCK_RAMP_DEG_MAX)), abs=1e-6)

        robot = unwrapped.scene["robot"]
        start = (robot.data.root_pos_w.torch - origins).clone()
        action = torch.zeros((unwrapped.num_envs, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
        void_term = unwrapped.termination_manager.get_term_cfg("fell_into_void")
        travelled_at_60 = None
        for step in range(200):
            env.step(action)
            assert torch.isfinite(robot.data.root_pos_w.torch).all()
            assert not bool(mdp.root_height_below_minimum(unwrapped, **void_term.params).any())
            if step == 59:
                travelled_at_60 = (robot.data.root_pos_w.torch - origins - start).clone()

        # It rode the ramp rather than sinking onto it, and that is the discriminator this test
        # exists for. A robot whose tires stop carrying it sinks and stalls within the first
        # quarter second, which is what the raw triangle-mesh collider does to this model: measured
        # on this configuration with the flag flipped off, the best environment reaches 0.23 m by
        # step 60 where the heightfield reaches 0.51-0.54. The thresholds sit between those, and on
        # the spread rather than on the worst environment -- the domain randomization is live here,
        # so an environment that topples early is a normal draw rather than a broken plant.
        distance_at_60 = travelled_at_60[:, 0]
        assert float(distance_at_60.max()) > 0.35
        assert float(distance_at_60.median()) > 0.2

        # and it went down the hill as well as along it, which is what "on the ramp" means
        travelled = robot.data.root_pos_w.torch - origins - start
        assert float(travelled[:, 0].max()) > 0.35
        assert float(travelled[:, 2].max()) < -0.03
        assert float(robot.data.root_pos_w.torch[:, 2].min()) > MICRODUCK_SLOPE_VOID_FLOOR
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_a_diverged_environment_does_not_poison_the_reward_buffer():
    """The failure that killed two 4096-environment training runs, reproduced deterministically.

    A rare MuJoCo Warp divergence leaves one environment's whole joint state and every body
    orientation non-finite for a single step while the root quaternion stays normalized. The
    ``nan_state`` termination exists to catch exactly that, and it does -- but detection is not what
    is missing. ``ManagerBasedRLEnv.step`` computes the terminations and then the rewards from the
    same post-physics buffers, and only *repairs* the flagged environments afterwards, in
    ``_reset_idx``; so the reward for the step the divergence happened on is computed on the poisoned
    state whichever manager runs first, and RSL-RL aborts the run on it. Measured rate on this task:
    about one step-environment in sixteen million, which is the order upstream reports for the same
    divergence.

    The state written below is the shape captured from a live rollout, not a guess:
    ``artifacts/microduck/reward_nan/``. Two things are asserted, and the second is the one that
    keeps the guard honest -- the divergence must still be *detected*, only kept out of the reward.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()
        robot = unwrapped.scene["robot"]

        joint_pos = robot.data.joint_pos.torch.clone()
        joint_vel = robot.data.joint_vel.torch.clone()
        joint_pos[0] = float("nan")
        joint_vel[0] = float("nan")
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
        unwrapped.sim.forward()

        # the divergence really did reach the quantities the reward terms read
        assert not bool(torch.isfinite(robot.data.joint_pos.torch[0]).all())
        assert not bool(torch.isfinite(robot.data.body_link_quat_w.torch[0]).all())
        # ...and not the root orientation, which is what leaves ``upright`` and ``heading_hold``
        # scoring normally and is why only two terms were ever implicated
        assert bool(torch.isfinite(robot.data.root_link_quat_w.torch[0]).all())

        rewards = unwrapped.reward_manager.compute(unwrapped.step_dt)
        per_term = unwrapped.reward_manager._step_reward
        terms = list(unwrapped.reward_manager.active_terms)

        assert torch.isfinite(per_term).all(), {name: float(per_term[0, index]) for index, name in enumerate(terms)}
        assert torch.isfinite(rewards).all()
        # the two guarded terms charge nothing for a state that has no pose and no orientation
        for name in ("feet_flat", "neck_joint_pos_l2"):
            assert float(per_term[0, terms.index(name)]) == pytest.approx(0.0, abs=1e-9), name
        # the healthy environments are untouched
        assert torch.isfinite(per_term[1:]).all()

        # the guard keeps the divergence out of the reward; it does not hide it. The environment is
        # still recycled by the termination that owns this failure.
        broken = mdp.robot_state_is_nan(unwrapped, sensor_names=("contact_forces",))
        assert bool(broken[0])
        assert not bool(broken[1:].any())

        # Second shape, which the capture did *not* contain: the root link's orientation going with
        # it. That the root quaternion survived is empirical from one event rather than an invariant,
        # and it is the only reason ``upright`` and ``heading_hold`` scored normally above -- so the
        # buffer has to stay finite when it does not survive either.
        root_pose = robot.data.root_link_pose_w.torch.clone()
        root_pose[0] = float("nan")
        robot.write_root_link_pose_to_sim_index(root_pose=root_pose)
        unwrapped.sim.forward()
        assert not bool(torch.isfinite(robot.data.root_link_quat_w.torch[0]).all())

        unwrapped.reward_manager.compute(unwrapped.step_dt)
        per_term = unwrapped.reward_manager._step_reward
        assert torch.isfinite(per_term).all(), {name: float(per_term[0, index]) for index, name in enumerate(terms)}
        # both root-quaternion readers pay a broken environment nothing rather than full marks
        for name in ("upright", "heading_hold"):
            assert float(per_term[0, terms.index(name)]) == pytest.approx(0.0, abs=1e-4), name
        assert torch.isfinite(per_term[1:]).all()
        assert bool(mdp.robot_state_is_nan(unwrapped, sensor_names=("contact_forces",))[0])
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
