# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test recovery on rollers and valid leg reward selections.

Leg rewards must resolve ten named physical joints within the fourteen-servo view.
Scene checks compare the rise reward stack for standing and fallen configurations.
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.rollerstandup.agents.rsl_rl_ppo_cfg import MicroDuckRollerStandUpPPORunnerCfg
from isaaclab_tasks.contrib.microduck.rollerstandup.rollerstandup_env_cfg import (
    MICRODUCK_ROLLER_PRONE_HEIGHT,
    MICRODUCK_ROLLER_RISE_CEILING,
    MICRODUCK_ROLLER_STAND_HEIGHT,
    MicroDuckRollerStandUpFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import (
    MICRODUCK_ROLLERS_STANDING_HEIGHT,
    MicroDuckVelocityRollersFlatEnvCfg,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_rollers_usd = pytest.mark.usefixtures("microduck_rollers_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-RollerStandUp-Flat-MicroDuck"

STEPS_PER_ITERATION = 24
ACTOR_OBSERVATION_DIM = 61
CRITIC_OBSERVATION_DIM = 78

EPISODE_CONTROL_STEPS = 300
"""Control steps in an episode: 6 s at 50 Hz, a third of the skating task's window."""

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
EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
EXPECTED_WHEEL_JOINT_NAMES = ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]

UPSTREAM_LEG_JOINT_INDICES = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
"""Indices from the eighteen-joint layout used to check bounds against a fourteen-servo view.
The valid reward selection resolves the ten physical leg joints by name.
"""

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    # the task declares -0.6 and its curriculum's stage 0 is -0.4, which is what is ever live
    "action_rate_l2": (-0.4, {}),
    "self_collisions": (-1.0, {}),
    "neck_action_rate_l2": (-0.5, {"action_name": "joint_pos"}),
    "neck_joint_pos_l2": (-0.5, {}),
    "joint_torques_l2": (-1e-3, {}),
    "action_over_limit": (-0.5, {"action_name": "joint_pos", "overshoot": 0.3}),
    # the eleven-term rise stack
    "pose_stand_legs": (8.0, {"std": 0.5}),
    "pose_stand_l1": (5.0, {}),
    "height_stand": (4.0, {"std": 0.04, "target_height": 0.138}),
    "height_stand_sharp": (4.0, {"std": 0.015, "target_height": 0.138}),
    "height_stand_l1": (30.0, {"target_height": 0.138}),
    "com_upward_velocity": (3.0, {"max_height": 0.148}),
    "gentle_rise": (0.02, {}),
    "upright_linear": (6.0, {}),
    "upright_sharp": (6.0, {"std": 0.3, "height_low": 0.075, "height_high": 0.138}),
    "standing_composite": (
        15.0,
        {"target_height": 0.138, "height_std": 0.04, "upright_std": 0.40, "pose_std": 0.40},
    ),
    "joint_torque_rate_l2": (-0.2, {}),
}
"""Initial reward terms, weights and scalar parameters, including inherited terms."""

EXPECTED_REMOVED_SKATING_REWARDS = [
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
]
"""The thirteen skating terms the task pops. All thirteen exist in the roller dict."""

EXPECTED_POSITIVE_WEIGHT_SELF_NEGATING = ["pose_stand_l1", "height_stand_l1", "gentle_rise"]
"""Terms whose kernels return a non-positive value and which therefore take positive weights."""

EXPECTED_CURRICULUM_TERMS = {
    "ground_state_mix",
    "wheel_friction",
    "action_rate_weight",
    "push_magnitude",
    # inherited from the skating recipe
    "com_range",
    "head_com_range",
}

EXPECTED_GROUND_STATE_STAGES = [
    (0, {"standing_prob": 0.50, "sitting_prob": 0.00, "face_down_prob": 0.50, "face_up_prob": 0.00}),
    (600, {"standing_prob": 0.35, "sitting_prob": 0.00, "face_down_prob": 0.45, "face_up_prob": 0.20}),
    (1500, {"standing_prob": 0.25, "sitting_prob": 0.00, "face_down_prob": 0.40, "face_up_prob": 0.35}),
    (2500, {"standing_prob": 0.20, "sitting_prob": 0.00, "face_down_prob": 0.40, "face_up_prob": 0.40}),
]

EXPECTED_WHEEL_FRICTION_STAGES = [
    (0, {"friction_range": (0.05, 0.05)}),
    (1000, {"friction_range": (0.02, 0.02)}),
    (2000, {"friction_range": (0.008, 0.008)}),
    (3000, {"friction_range": (0.003, 0.003)}),
    (4000, {"friction_range": (0.0015, 0.0015)}),
]
"""The configured **inverted** bearing-drag ramp: almost locked, relaxed toward free."""

EXPECTED_RESET_EVENT_ORDER = [
    "reset_base",
    "reset_robot_joints",
    "set_ground_state",
    "randomize_wheel_friction",
    "randomize_com",
    "randomize_head_com",
    "randomize_joint_friction",
    "randomize_armature",
]
"""The reset chain, in the order it fires. ``set_ground_state`` has to come after the root reset."""


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which are pinned by name in their own tests."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


##
# Configuration expectations
##


@pytest.mark.unit
def test_the_reward_recipe_matches_upstream_term_for_term():
    """Reward terms must match the expected set, weights and scalar parameters."""
    rewards = MicroDuckRollerStandUpFlatEnvCfg().rewards
    skating = MicroDuckVelocityRollersFlatEnvCfg().rewards

    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    assert len(EXPECTED_REWARDS) == 19
    for name in EXPECTED_REMOVED_SKATING_REWARDS:
        assert not hasattr(rewards, name), name
        # each of them is a real skating term, so none of the removals is a no-op
        assert getattr(skating, name) is not None, name
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_three_self_negating_terms_take_positive_weights():
    """``gentle_rise`` above all: at a negative weight it *rewards* vertical acceleration."""
    rewards = MicroDuckRollerStandUpFlatEnvCfg().rewards

    for name in EXPECTED_POSITIVE_WEIGHT_SELF_NEGATING:
        assert getattr(rewards, name).weight > 0.0, name
    # the wheel-less stand-up task uses the same kernel and the same sign, which is what makes
    # the configured -0.02 on this task a copy that was never re-derived
    assert rewards.gentle_rise.func is mdp.trunk_vertical_accel_penalty
    assert rewards.gentle_rise.weight == pytest.approx(0.02)


@pytest.mark.unit
def test_the_pose_terms_resolve_the_legs_by_name_rather_than_by_index():
    """Resolve ten physical leg joints by name within the fourteen-servo reward view."""
    rewards = MicroDuckRollerStandUpFlatEnvCfg().rewards

    for name in ("pose_stand_legs", "pose_stand_l1", "standing_composite"):
        asset_cfg = getattr(rewards, name).params["asset_cfg"]
        assert asset_cfg.joint_names == EXPECTED_LEG_JOINT_NAMES, name
        assert asset_cfg.preserve_order, name
        assert not set(asset_cfg.joint_names) & set(EXPECTED_WHEEL_JOINT_NAMES), name
    # the head terms are scoped to the head, and the torque terms to all fourteen servos
    assert rewards.neck_joint_pos_l2.params["asset_cfg"].joint_names == EXPECTED_HEAD_JOINT_NAMES
    assert rewards.joint_torques_l2.params["asset_cfg"].joint_names == EXPECTED_SERVO_JOINT_NAMES


@pytest.mark.unit
def test_the_keyframe_heights_are_the_roller_models_measured_ones():
    """0.138 m is the geometric standing height less about 2.7 mm of servo sag."""
    assert pytest.approx(0.138) == MICRODUCK_ROLLER_STAND_HEIGHT
    assert pytest.approx(0.14070) == MICRODUCK_ROLLERS_STANDING_HEIGHT
    assert pytest.approx(0.0027, abs=1e-4) == MICRODUCK_ROLLERS_STANDING_HEIGHT - MICRODUCK_ROLLER_STAND_HEIGHT
    # the rise ceiling is a centimetre above the station, so the upward-velocity term pays throughout
    assert MICRODUCK_ROLLER_RISE_CEILING > MICRODUCK_ROLLER_STAND_HEIGHT
    assert pytest.approx(0.010, abs=1e-6) == MICRODUCK_ROLLER_RISE_CEILING - MICRODUCK_ROLLER_STAND_HEIGHT
    # and the upright gate opens from the face-down rest height
    upright_sharp = MicroDuckRollerStandUpFlatEnvCfg().rewards.upright_sharp
    assert upright_sharp.params["height_low"] == pytest.approx(MICRODUCK_ROLLER_PRONE_HEIGHT)
    assert upright_sharp.params["height_high"] == pytest.approx(MICRODUCK_ROLLER_STAND_HEIGHT)
    # the standing spawn band brackets the station
    ground_state = MicroDuckRollerStandUpFlatEnvCfg().events.set_ground_state
    low, high = ground_state.params["standing_z_range"]
    assert low <= MICRODUCK_ROLLER_STAND_HEIGHT <= high


@pytest.mark.unit
def test_the_ground_state_spawn_reproduces_upstream_bands_and_probabilities():
    """One height band for two prone poses whose contacts have nothing in common."""
    events = MicroDuckRollerStandUpFlatEnvCfg().events

    params = events.set_ground_state.params
    assert events.set_ground_state.func is mdp.reset_ground_state
    assert params["prone_z_range"] == (0.076, 0.09)
    assert params["sitting_prob"] == 0.0
    assert params["sitting_joint_pos"] is None
    assert params["sitting_tilt_max"] == pytest.approx(math.radians(10.0))
    # stage 0 of the curriculum and the declared probabilities agree
    for key, value in EXPECTED_GROUND_STATE_STAGES[0][1].items():
        assert params[key] == pytest.approx(value), key
    # the floor eliminates belly interpenetration: it is above the face-down rest height
    assert params["prone_z_range"][0] > MICRODUCK_ROLLER_PRONE_HEIGHT


@pytest.mark.unit
def test_the_reset_events_fire_in_an_order_that_lets_the_ground_state_win():
    """``set_ground_state`` overwrites the root reset's height and orientation, so it must follow it."""
    events = MicroDuckRollerStandUpFlatEnvCfg().events

    reset_terms = [name for name, term in vars(events).items() if getattr(term, "mode", None) == "reset"]
    assert reset_terms == EXPECTED_RESET_EVENT_ORDER
    assert reset_terms.index("set_ground_state") > reset_terms.index("reset_base")
    assert reset_terms.index("set_ground_state") > reset_terms.index("reset_robot_joints")


@pytest.mark.unit
def test_the_wheel_friction_curriculum_runs_backwards():
    """The task's one genuinely new piece of machinery: locked bearings, relaxed toward free."""
    curriculum = MicroDuckRollerStandUpFlatEnvCfg().curriculum
    events = MicroDuckRollerStandUpFlatEnvCfg().events
    stages = curriculum.wheel_friction.params["param_stages"]

    assert len(stages) == len(EXPECTED_WHEEL_FRICTION_STAGES)
    for stage, (iteration, params) in zip(stages, EXPECTED_WHEEL_FRICTION_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        assert stage["params"] == params
    # monotonically decreasing, which is the inversion
    values = [stage["params"]["friction_range"][0] for stage in stages]
    assert values == sorted(values, reverse=True)
    # the skating task's ramp on the same event goes the other way
    skating = MicroDuckVelocityRollersFlatEnvCfg().curriculum.wheel_friction.params["param_stages"]
    skating_values = [stage["params"]["friction_range"][0] for stage in skating]
    assert skating_values == sorted(skating_values)
    # and the event ships stage 0's value, so nothing is silently dead
    assert events.randomize_wheel_friction.params["friction_range"] == stages[0]["params"]["friction_range"]
    assert events.randomize_wheel_friction.params["asset_cfg"].joint_names == EXPECTED_WHEEL_JOINT_NAMES
    # Wheel-friction stages use strict boundaries; event-parameter stages use inclusive boundaries.
    assert curriculum.wheel_friction.params["inclusive"] is False


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """Six schedules, two of them replacing the skating task's and two of them new."""
    curriculum = MicroDuckRollerStandUpFlatEnvCfg().curriculum

    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS

    stages = curriculum.ground_state_mix.params["param_stages"]
    assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it, _ in EXPECTED_GROUND_STATE_STAGES]
    for stage, (_, params) in zip(stages, EXPECTED_GROUND_STATE_STAGES):
        assert stage["params"] == params
        # every stage is a distribution, and the standing bucket never vanishes
        assert sum(params.values()) == pytest.approx(1.0)
        assert params["standing_prob"] > 0.0

    # the action-rate ramp is the wheel-less stand-up task's, not the skating task's: the skating
    # ramp reaches -2.0 for a calm gait, which blocks the fast rise this task needs
    action_rate = [stage["weight"] for stage in curriculum.action_rate_weight.params["weight_stages"]]
    assert action_rate == pytest.approx([-0.4, -0.8, -1.0])
    assert min(action_rate) > -2.0
    assert MicroDuckRollerStandUpFlatEnvCfg().rewards.action_rate_l2.weight == pytest.approx(action_rate[0])

    # the pushes ramp from nothing, unlike the skating task's full-strength-from-step-one shove
    push_stages = curriculum.push_magnitude.params["param_stages"]
    assert push_stages[0]["params"]["velocity_range"] == {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    assert push_stages[-1]["params"]["velocity_range"] == {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
    assert curriculum.push_magnitude.params["inclusive"] is False
    assert (
        MicroDuckRollerStandUpFlatEnvCfg().events.push_robot.params["velocity_range"]
        == (push_stages[0]["params"]["velocity_range"])
    )


@pytest.mark.unit
def test_the_tilt_termination_is_dropped_and_the_nan_guard_is_kept():
    """The robot starts on the ground, so a tilt termination would end every episode at step 0."""
    terminations = MicroDuckRollerStandUpFlatEnvCfg().terminations

    assert set(vars(terminations)) == {"time_out", "nan_state"}
    assert not hasattr(terminations, "fell_over")
    # the family's NaN-guard norm, where the task reaches for an observation-level sanitizer instead
    assert terminations.nan_state.params["sensor_names"] == ("contact_forces",)
    assert terminations.nan_state.time_out is False


@pytest.mark.unit
def test_the_command_is_a_neutralized_shape_placeholder():
    """No reward reads it; it exists so the deployed 61-wide vector keeps its twist slot."""
    cfg = MicroDuckRollerStandUpFlatEnvCfg()
    command = cfg.commands.base_velocity

    assert set(vars(cfg.commands)) == {"base_velocity"}
    # downgraded from the skating task's relative-heading command: a live heading error in a slot
    # nothing tracks would be a distractor rather than a placeholder
    assert isinstance(command, mdp.ForwardMaskVelocityCommandCfg)
    assert not isinstance(command, mdp.RelativeHeadingVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (-0.01, 0.01)
    assert command.ranges.lin_vel_y == (-0.01, 0.01)
    assert command.ranges.ang_vel_z == (-0.05, 0.05)
    assert command.heading_command is False
    # at most one resample in a six-second episode
    assert min(command.resampling_time_range) >= cfg.episode_length_s


@pytest.mark.unit
def test_the_scene_actions_and_observations_are_the_skating_task_untouched():
    """The slope task changes terrain while preserving the roller scene, actions and observations."""
    standup = MicroDuckRollerStandUpFlatEnvCfg()
    skating = MicroDuckVelocityRollersFlatEnvCfg()

    assert standup.observations.to_dict() == skating.observations.to_dict()
    assert standup.actions.to_dict() == skating.actions.to_dict()
    assert standup.scene.to_dict() == skating.scene.to_dict()
    assert standup.episode_length_s == pytest.approx(6.0)
    assert standup.episode_length_s / (standup.sim.dt * standup.decimation) == pytest.approx(EPISODE_CONTROL_STEPS)


@pytest.mark.unit
def test_the_contact_budget_is_measured_rather_than_inherited():
    """Contact and constraint capacities must cover the task's measured contact load."""
    solver = MicroDuckRollerStandUpFlatEnvCfg().sim.physics.default.solver_cfg
    skating = MicroDuckVelocityRollersFlatEnvCfg().sim.physics.default.solver_cfg

    # measured peaks: 98 constraints and 28 contacts per environment
    assert solver.njmax >= 98
    assert solver.nconmax >= 28
    assert solver.nconmax == pytest.approx(round(1.2 * 28), abs=1)
    # heavier than the skating profile on the same model, which is why it was re-measured
    assert solver.njmax > skating.njmax
    assert solver.nconmax > skating.nconmax
    # the solver iteration counts are the configured template values, untouched
    assert (solver.iterations, solver.ls_iterations) == (10, 20)


@pytest.mark.unit
def test_the_runner_carries_the_deployment_gate_in_its_iteration_budget():
    """The wheel-friction ramp finishes at iteration 4000, well inside the configured 15 000 budget."""
    runner = MicroDuckRollerStandUpPPORunnerCfg()
    curriculum = MicroDuckRollerStandUpFlatEnvCfg().curriculum

    assert runner.experiment_name == "microduck_rollerstandup"
    assert runner.max_iterations == 15000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    assert runner.algorithm.entropy_coef == pytest.approx(0.01)
    assert runner.algorithm.symmetry_cfg is None
    # the deployment gate is reachable: the last friction stage lands with room to train past it
    last_stage = curriculum.wheel_friction.params["param_stages"][-1]["step"] / STEPS_PER_ITERATION
    assert last_stage < runner.max_iterations
    # and checkpoints are written often enough that one exists after it
    assert runner.save_interval < runner.max_iterations - last_stage


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
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_upstream_leg_indices_would_have_indexed_past_the_servo_view():
    """Eighteen-joint indices can exceed the fourteen-servo view; named selection must resolve valid leg columns."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()
        robot = unwrapped.scene["robot"]

        # the model layout the configured comment describes, re-measured here
        assert list(robot.joint_names)[:1] != []
        wheel_positions = [list(robot.joint_names).index(name) for name in EXPECTED_WHEEL_JOINT_NAMES]
        assert robot.num_joints == 18
        assert len(wheel_positions) == 4

        # the servo view every pose helper works in
        servo_ids = [list(robot.joint_names).index(name) for name in EXPECTED_SERVO_JOINT_NAMES]
        servo_view = robot.data.joint_pos.torch[:, servo_ids]
        assert servo_view.shape[1] == 14

        # the configured indices against that view: out of bounds by two. The indexing is done on a host
        # copy on purpose -- an out-of-range gather on the device raises a device-side assert that
        # poisons the CUDA context for every test after it, where the host copy raises cleanly.
        assert max(UPSTREAM_LEG_JOINT_INDICES) >= servo_view.shape[1]
        host_view = servo_view.cpu()
        with pytest.raises(IndexError):
            host_view[:, torch.tensor(UPSTREAM_LEG_JOINT_INDICES)]

        # The named selection resolves the ten physical leg joints to valid servo columns.
        term_cfg = unwrapped.reward_manager.get_term_cfg("pose_stand_legs")
        resolved = [robot.joint_names[int(i)] for i in term_cfg.params["asset_cfg"].joint_ids]
        assert resolved == EXPECTED_LEG_JOINT_NAMES
        assert torch.isfinite(mdp.joint_pose_gaussian(unwrapped, **term_cfg.params)).all()
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_rise_stack_scores_a_stand_far_above_a_fallen_start():
    """The internal acceptance test that stands in for an accuracy gate against the task."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        # the reset distribution is not what is under test: the poses are placed by hand below. The
        # curricula that drive those events go with them, since each names the term it rewrites.
        for name in ("set_ground_state", "randomize_com", "randomize_head_com", "push_robot"):
            setattr(env_cfg.events, name, None)
        for name in ("ground_state_mix", "com_range", "head_com_range", "push_magnitude"):
            setattr(env_cfg.curriculum, name, None)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()
        robot = unwrapped.scene["robot"]

        def place(height: float, pitch: float) -> None:
            """Put the robot at a trunk height with a pitch about the body ``y`` axis, and refresh the
            buffers.
            """
            pose = torch.zeros((unwrapped.num_envs, 7), device=unwrapped.device)
            pose[:, 0:3] = unwrapped.scene.env_origins
            pose[:, 2] += height
            pose[:, 4] = math.sin(0.5 * pitch)
            pose[:, 6] = math.cos(0.5 * pitch)
            robot.write_root_link_pose_to_sim_index(root_pose=pose)
            robot.write_root_link_velocity_to_sim_index(
                root_velocity=torch.zeros((unwrapped.num_envs, 6), device=unwrapped.device)
            )
            joint_pos = robot.data.default_joint_pos.torch.clone()
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
            unwrapped.sim.forward()

        def score() -> tuple[float, float]:
            """The weighted reward sum this step, and the composite term's own contribution."""
            total = unwrapped.reward_manager.compute(dt=unwrapped.step_dt)
            composite = unwrapped.reward_manager._episode_sums["standing_composite"].clone()
            unwrapped.reward_manager.reset()
            return float(total.mean()), float(composite.mean())

        # the three states the curriculum mixes: standing, face down (+90 deg pitch), face up (-90)
        place(MICRODUCK_ROLLER_STAND_HEIGHT, 0.0)
        standing_total, standing_composite = score()
        place(MICRODUCK_ROLLER_PRONE_HEIGHT + 0.001, 0.5 * math.pi)
        face_down_total, face_down_composite = score()
        place(0.0475 + 0.001, -0.5 * math.pi)
        face_up_total, face_up_composite = score()

        # 1. the task's own reward orders the three states the way the task intends
        assert standing_total > face_down_total, (standing_total, face_down_total)
        assert face_down_total > face_up_total, (face_down_total, face_up_total)

        # 2. the composite is what separates them, and it is a real number rather than a zero
        assert standing_composite > 0.0
        assert standing_composite > 10.0 * max(face_down_composite, face_up_composite, 1e-6)

        # 3. the margin is large rather than marginal. The sums are the manager's, so they carry the
        #    0.02 s control step; at the station the stack pays several times what it pays face down,
        #    which is the gradient the rise is meant to climb.
        assert standing_total > 4.0 * face_down_total, (standing_total, face_down_total)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
