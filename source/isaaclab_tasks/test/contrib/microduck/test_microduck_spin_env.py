# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe-parity and smoke tests for the contributed MicroDuck spin trick environment.

The parity tests need neither the asset nor the simulator: they read the assembled configuration and
compare it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference_tasks3.md`` section 10. The expected values are spelled out
rather than imported from the configuration under test, so that a drifting value fails rather than
agrees with itself.

The task is structurally the crouch-glide trick with a rate objective in place of a pose one, so the
tests that matter here are the ones that pin the *differences*: the dropped angular-momentum term,
the envelope arithmetic, the scoped neck penalty that leaves the head yaw free, and the decaying
scissor curriculum. The wheel-friction interlock the two tricks share is tested once, on the
crouch-glide task; this file asserts only that this environment carries the same degenerate event.

Simulator-backed checks retrieve the prebuilt USD through Isaac Lab's native asset cache.
Recipe checks do not need an asset; source fidelity comparisons use optional read-only fixtures."""

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
from isaaclab_tasks.contrib.microduck.rollercrouch.rollercrouch_env_cfg import MicroDuckRollerCrouchFlatEnvCfg
from isaaclab_tasks.contrib.microduck.spin.agents.rsl_rl_ppo_cfg import MicroDuckSpinPPORunnerCfg
from isaaclab_tasks.contrib.microduck.spin.spin_env_cfg import (
    MICRODUCK_ENTRY_VELOCITY_X,
    MICRODUCK_SPIN_ACCEL_END,
    MICRODUCK_SPIN_BRAKE_END,
    MICRODUCK_SPIN_HOLD_END,
    MICRODUCK_SPIN_LAUNCH_DRIFT_SCALE,
    MICRODUCK_SPIN_NECK_JOINT_NAMES,
    MICRODUCK_SPIN_PERIOD,
    MICRODUCK_SPIN_RATE_MAX,
    MICRODUCK_SPIN_WHEEL_OMEGA_SCALE,
    MicroDuckSpinFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import MicroDuckVelocityRollersFlatEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_rollers_usd = pytest.mark.usefixtures("microduck_rollers_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-Spin-Flat-MicroDuck"

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

_ENVELOPE = {"rate_max": 3.0, "accel_end": 0.125, "hold_end": 0.525, "brake_end": 0.650}

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "upright": (2.0, {"std": math.sqrt(0.2)}),
    "body_ang_vel": (-0.05, {}),
    # upstream declares -1.0 and its curriculum's stage 0 is -0.5, which is what is ever live
    "action_rate_l2": (-0.5, {}),
    "spin_rate_track": (6.0, {"command_name": "base_velocity", "std": 1.5, **_ENVELOPE}),
    "spin_rate_l1": (0.5, {"command_name": "base_velocity", **_ENVELOPE}),
    "spin_stay_in_place": (-3.0, {"command_name": "base_velocity", "launch_scale": 0.2, "accel_end": 0.125}),
    "spin_wheel_differential": (1.0, {"command_name": "base_velocity", "omega_scale": 17.0, **_ENVELOPE}),
    "leg_antisymmetry": (1.0, {"command_name": "base_velocity", **_ENVELOPE}),
    "spin_grounded": (0.5, {"command_name": "base_velocity", "bodies_per_foot": 2, **_ENVELOPE}),
    "feet_flat": (-2.0, {"normal_axis": (0.0, 1.0, 0.0), "bodies_per_foot": 2}),
    "self_collisions": (-1.0, {}),
    "neck_action_rate_l2": (-0.5, {"action_name": "joint_pos"}),
    "neck_joint_pos_l2": (-0.2, {}),
    "joint_torques_l2": (-1e-3, {}),
}
"""Upstream's fourteen-term reward recipe (section 10.2), keyed by term name, at its live weights."""

EXPECTED_CURRICULUM_TERMS = {"action_rate_weight", "leg_antisym_weight", "com_range", "head_com_range"}

EXPECTED_WEIGHT_STAGES = {
    "action_rate_weight": ([-0.5, -0.8, -1.0], [0, 250, 500]),
    # the family's only *decaying* reward weight: a training wheel, then removed
    "leg_antisym_weight": ([1.0, 0.5, 0.25], [0, 1500, 3000]),
}
EXPECTED_RANGE_STAGES = {
    "com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
    "head_com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
}


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which are pinned by name in their own tests."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


##
# Recipe parity against upstream (section 10)
##


@pytest.mark.unit
def test_the_reward_recipe_matches_upstream_term_for_term():
    """Every reward slot upstream trains with is present, with its weight and its parameters."""
    rewards = MicroDuckSpinFlatEnvCfg().rewards

    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_angular_momentum_penalty_is_the_one_term_this_task_drops():
    """It charges the norm of the whole angular-momentum vector, so it would fight the spin head-on.

    ``body_ang_vel`` survives in its place because it charges roll and pitch only, which damps the
    wobble without touching the rotation. This is the sharpest reward-design delta against the
    crouch-glide task, which keeps both (section 10.2).
    """
    spin = MicroDuckSpinFlatEnvCfg().rewards
    crouch = MicroDuckRollerCrouchFlatEnvCfg().rewards

    assert not hasattr(spin, "angular_momentum")
    assert crouch.angular_momentum.weight == pytest.approx(-0.02)
    assert spin.body_ang_vel.weight == pytest.approx(crouch.body_ang_vel.weight)
    assert spin.body_ang_vel.func is mdp.body_ang_vel_xy_l2_raw


@pytest.mark.unit
def test_the_envelope_is_one_counter_clockwise_turn_per_cycle():
    """The trapezoid's area is 6.30 rad, which is 1.0027 turns -- upstream's own arithmetic."""
    cfg = MicroDuckSpinFlatEnvCfg()

    assert pytest.approx(4.0) == MICRODUCK_SPIN_PERIOD
    assert pytest.approx(3.0) == MICRODUCK_SPIN_RATE_MAX
    assert (MICRODUCK_SPIN_ACCEL_END, MICRODUCK_SPIN_HOLD_END, MICRODUCK_SPIN_BRAKE_END) == (0.125, 0.525, 0.650)
    durations = [
        MICRODUCK_SPIN_ACCEL_END,
        MICRODUCK_SPIN_HOLD_END - MICRODUCK_SPIN_ACCEL_END,
        MICRODUCK_SPIN_BRAKE_END - MICRODUCK_SPIN_HOLD_END,
        1.0 - MICRODUCK_SPIN_BRAKE_END,
    ]
    assert [d * MICRODUCK_SPIN_PERIOD for d in durations] == pytest.approx([0.5, 1.6, 0.5, 1.4])

    # the area, derived here rather than transcribed: two half-triangles plus the hold
    area_fraction = 0.5 * durations[0] + durations[1] + 0.5 * durations[2]
    turns = area_fraction * MICRODUCK_SPIN_PERIOD * MICRODUCK_SPIN_RATE_MAX / (2.0 * math.pi)
    assert turns == pytest.approx(1.0, abs=0.005)
    # and a twenty-second episode is five cycles
    assert cfg.episode_length_s / MICRODUCK_SPIN_PERIOD == pytest.approx(5.0)


@pytest.mark.unit
def test_the_neck_position_penalty_leaves_the_head_yaw_free_as_a_flywheel():
    """The only place in the family where this penalty is scoped, and the reason is task-level.

    Upstream frees ``head_yaw`` so it can serve as an inertia flywheel for launching the rotation
    (section 10.2). The *rate* penalty is deliberately left over all four head joints, which is the
    asymmetry upstream ships.
    """
    rewards = MicroDuckSpinFlatEnvCfg().rewards

    scoped = rewards.neck_joint_pos_l2.params["asset_cfg"].joint_names
    assert scoped == MICRODUCK_SPIN_NECK_JOINT_NAMES == ["neck_pitch", "head_pitch", "head_roll"]
    assert "head_yaw" not in scoped
    assert rewards.neck_joint_pos_l2.weight == pytest.approx(-0.2)
    # the rate penalty still covers all four, including the yaw
    assert rewards.neck_action_rate_l2.params["asset_cfg"].joint_names == [
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
    ]
    # the skating task charges the position penalty over all four at -0.5, so this is a real narrowing
    rollers = MicroDuckVelocityRollersFlatEnvCfg().rewards
    assert len(rollers.neck_joint_pos_l2.params["asset_cfg"].joint_names) == 4
    assert rollers.neck_joint_pos_l2.weight == pytest.approx(-0.5)


@pytest.mark.unit
def test_the_wheel_and_leg_selections_are_named_rather_than_pattern_matched():
    """Upstream's looser ``^passive_.*`` pattern here is unexpressible when the hinges are named.

    Upstream selects the wheels with ``^passive_.*`` on this task where every sibling uses
    ``^passive_.*wheel``, which its own conventions forbid: on a model carrying backlash hinges the
    looser pattern would pick those up too and widen the critic (section 13.4). On the plain roller
    model the two are equivalent, so there is no live defect to reproduce -- and naming the four
    hinges, as the rest of this port does, removes the latent one rather than carrying it.
    """
    rewards = MicroDuckSpinFlatEnvCfg().rewards
    differential = rewards.spin_wheel_differential

    left = differential.params["left_wheel_cfg"].joint_names
    right = differential.params["right_wheel_cfg"].joint_names
    assert left == ["passive_LF_wheel", "passive_LR_wheel"]
    assert right == ["passive_RF_wheel", "passive_RR_wheel"]
    assert sorted(left + right) == sorted(EXPECTED_WHEEL_JOINT_NAMES)
    assert differential.params["left_wheel_cfg"].preserve_order
    assert differential.params["right_wheel_cfg"].preserve_order

    scissor = rewards.leg_antisymmetry
    assert scissor.params["left_joint_cfg"].joint_names == ["left_hip_pitch", "left_knee"]
    assert scissor.params["right_joint_cfg"].joint_names == ["right_hip_pitch", "right_knee"]
    # and the critic still reads exactly the four wheels, inherited from the skating task
    critic_wheel_cfg = MicroDuckSpinFlatEnvCfg().observations.critic.wheel_vel.params["asset_cfg"]
    assert critic_wheel_cfg.joint_names == EXPECTED_WHEEL_JOINT_NAMES


@pytest.mark.unit
def test_the_known_unreproducible_wheel_scale_is_kept_verbatim():
    """Parity over correctness: upstream's derivation does not reproduce, but its policy trained on it.

    Upstream derives 17.0 rad/s from a half-track it states as 0.0499 m and a 0.0175 m tire radius.
    Measured on the pinned model the half-track is 0.03925 m at the foot sites and the tire radius is
    0.0150 m, so neither input reproduces (section 13.9). The constant is kept and only the
    arithmetic is dropped.
    """
    rewards = MicroDuckSpinFlatEnvCfg().rewards

    assert rewards.spin_wheel_differential.params["omega_scale"] == pytest.approx(MICRODUCK_SPIN_WHEEL_OMEGA_SCALE)
    assert pytest.approx(17.0) == MICRODUCK_SPIN_WHEEL_OMEGA_SCALE
    # upstream's own arithmetic, reproduced here to show it does not land on 17.0 with measured inputs
    measured_half_track, measured_radius = 0.03925, 0.0150
    assert 2.0 * MICRODUCK_SPIN_RATE_MAX * measured_half_track / measured_radius != pytest.approx(17.0, rel=0.05)


@pytest.mark.unit
def test_the_drift_cost_is_discounted_at_the_launch_and_full_during_the_rest():
    """It is what makes the trick a spin rather than a pivot around one skate (section 10.3)."""
    rewards = MicroDuckSpinFlatEnvCfg().rewards
    term = rewards.spin_stay_in_place

    assert term.weight == pytest.approx(-3.0)
    assert term.params["launch_scale"] == pytest.approx(MICRODUCK_SPIN_LAUNCH_DRIFT_SCALE) == pytest.approx(0.2)
    assert term.params["accel_end"] == pytest.approx(MICRODUCK_SPIN_ACCEL_END)
    # deliberately *not* gated by the envelope: the standing rest is exactly when stillness matters
    assert "rate_max" not in term.params
    assert "brake_end" not in term.params


@pytest.mark.unit
def test_the_phase_command_and_the_entry_momentum_match_upstream():
    """The clock starts standing at phase 0, and the entry band reaches zero unlike the crouch's."""
    cfg = MicroDuckSpinFlatEnvCfg()

    command = cfg.commands.base_velocity
    assert isinstance(command, mdp.GroundPickPhaseCommandCfg)
    assert command.period == pytest.approx(MICRODUCK_SPIN_PERIOD)
    assert command.randomize_phase is False

    assert cfg.events.reset_base.params["velocity_range"] == {"x": MICRODUCK_ENTRY_VELOCITY_X}
    assert MICRODUCK_ENTRY_VELOCITY_X == (0.0, 0.3)
    # the button can be pressed from a standstill, which the crouch-glide task's band cannot express
    crouch_low, _ = MicroDuckRollerCrouchFlatEnvCfg().events.reset_base.params["velocity_range"]["x"]
    assert MICRODUCK_ENTRY_VELOCITY_X[0] < crouch_low


@pytest.mark.unit
def test_the_degenerate_wheel_event_is_carried_here_too():
    """The section 13.1 / 13.3 interlock is tested once, on the crouch task; this is its other half.

    Upstream ships the same frictionless-bearing event on both tricks and a ramping curriculum on
    neither, so both train on free bearings forever (issue draft 017). The event is kept for the same
    reason it is kept there.
    """
    cfg = MicroDuckSpinFlatEnvCfg()

    assert cfg.events.randomize_wheel_friction is not None
    assert cfg.events.randomize_wheel_friction.params["friction_range"] == (0.0, 0.0)
    assert cfg.events.randomize_wheel_friction.params["asset_cfg"].joint_names == EXPECTED_WHEEL_JOINT_NAMES
    assert "wheel_friction" not in vars(cfg.curriculum)
    assert not hasattr(cfg.events, "expand_bam_friction_fields")


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """Four schedules, one of which is the family's only decaying reward weight."""
    curriculum = MicroDuckSpinFlatEnvCfg().curriculum

    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS
    for name, (weights, iterations) in EXPECTED_WEIGHT_STAGES.items():
        stages = getattr(curriculum, name).params["weight_stages"]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage["weight"] for stage in stages] == pytest.approx(weights), name
    for name, (ranges, iterations) in EXPECTED_RANGE_STAGES.items():
        stages = getattr(curriculum, name).params["range_stages"]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage["range"] for stage in stages] == pytest.approx(ranges), name

    # the scissor schedule only ever decreases, which no other reward ramp in the family does
    scissor = [stage["weight"] for stage in curriculum.leg_antisym_weight.params["weight_stages"]]
    assert scissor == sorted(scissor, reverse=True)
    # stage 0 of each agrees with the weight the term ships
    rewards = MicroDuckSpinFlatEnvCfg().rewards
    for name, reward_name in (("action_rate_weight", "action_rate_l2"), ("leg_antisym_weight", "leg_antisymmetry")):
        stage_zero = getattr(curriculum, name).params["weight_stages"][0]["weight"]
        assert getattr(rewards, reward_name).weight == pytest.approx(stage_zero), name


@pytest.mark.unit
def test_the_scene_actions_observations_and_terminations_are_the_roller_task_untouched():
    """Upstream rebuilds these from the mjlab template and arrives at the roller recipe (section 10.5)."""
    spin = MicroDuckSpinFlatEnvCfg()
    rollers = MicroDuckVelocityRollersFlatEnvCfg()

    assert spin.observations.to_dict() == rollers.observations.to_dict()
    assert spin.actions.to_dict() == rollers.actions.to_dict()
    assert spin.terminations.to_dict() == rollers.terminations.to_dict()
    assert spin.scene.to_dict() == rollers.scene.to_dict()
    assert spin.episode_length_s == pytest.approx(20.0)
    spin_events, roller_events = spin.events.to_dict(), rollers.events.to_dict()
    assert [name for name in spin_events if spin_events[name] != roller_events[name]] == ["reset_base"]


@pytest.mark.unit
def test_the_contact_budget_is_measured_rather_than_inherited():
    """A spin drags four small tire patches with rotating normals, which upstream never profiled."""
    solver = MicroDuckSpinFlatEnvCfg().sim.physics.default.solver_cfg

    # measured peaks: 90 constraints and 30 contacts per environment
    assert solver.njmax >= 90
    assert solver.nconmax >= 30
    # and above the mjlab template's inherited 35, which upstream leaves untouched here
    assert solver.nconmax >= 35
    assert (solver.iterations, solver.ls_iterations) == (10, 20)


@pytest.mark.unit
def test_the_runner_keeps_the_family_hyper_parameters_under_its_own_log_tree():
    """Upstream's runner differs from the velocity one in two fields (section 10.8)."""
    runner = MicroDuckSpinPPORunnerCfg()

    assert runner.experiment_name == "microduck_spin"
    assert runner.max_iterations == 8000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    assert runner.algorithm.entropy_coef == pytest.approx(0.01)
    # off for a correct task-level reason: a mirror turns a left spin into a right one
    assert runner.algorithm.symmetry_cfg is None


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
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_rate_reward_follows_the_envelope_and_the_hints_fade_before_the_rest():
    """The objective and the two mechanism hints, evaluated under physics across the cycle.

    Three contracts are checked at once, because they are the same envelope read three ways:

    * at the hold phase the rate reward is maximal when the trunk is turning at the target rate and
      falls off on either side of it, and it is *not* maximal at rest;
    * at the standing rest the target is zero, so a stationary robot scores full marks there and a
      spinning one does not -- which is what makes the trick end rather than run on;
    * both mechanism hints are identically zero across the standing rest and non-zero during the
      braking ramp, so the shaping fades out with the rotation instead of being cut at the top of it.
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

        def set_phase(phase: float) -> None:
            command = unwrapped.command_manager.get_command("base_velocity")
            command[:, 0] = math.cos(2.0 * math.pi * phase)
            command[:, 1] = math.sin(2.0 * math.pi * phase)

        def spin_at(yaw_rate: float) -> None:
            velocity = torch.zeros((unwrapped.num_envs, 6), device=unwrapped.device)
            velocity[:, 5] = yaw_rate
            robot.write_root_link_velocity_to_sim_index(root_velocity=velocity)
            unwrapped.sim.forward()

        track_params = unwrapped.reward_manager.get_term_cfg("spin_rate_track").params
        hold_phase = 0.5 * (MICRODUCK_SPIN_ACCEL_END + MICRODUCK_SPIN_HOLD_END)
        rest_phase = 0.5 * (MICRODUCK_SPIN_BRAKE_END + 1.0)

        # 1. during the hold the target is the peak rate
        set_phase(hold_phase)
        spin_at(MICRODUCK_SPIN_RATE_MAX)
        on_target = mdp.spin_rate_track(unwrapped, **track_params)
        spin_at(0.0)
        at_rest_during_hold = mdp.spin_rate_track(unwrapped, **track_params)
        spin_at(2.0 * MICRODUCK_SPIN_RATE_MAX)
        overspun = mdp.spin_rate_track(unwrapped, **track_params)
        assert float(on_target.min()) > 0.95
        assert float(at_rest_during_hold.max()) < float(on_target.min())
        assert float(overspun.max()) < float(on_target.min())

        # 2. during the standing rest the target is zero, so the preference inverts
        set_phase(rest_phase)
        spin_at(0.0)
        still_at_rest = mdp.spin_rate_track(unwrapped, **track_params)
        spin_at(MICRODUCK_SPIN_RATE_MAX)
        spinning_at_rest = mdp.spin_rate_track(unwrapped, **track_params)
        assert float(still_at_rest.min()) > 0.95
        assert float(spinning_at_rest.max()) < float(still_at_rest.min())

        # 3. both hints are gated to zero across the rest and live during the braking ramp
        joint_pos = robot.data.default_joint_pos.torch.clone()
        joint_pos[:, list(robot.joint_names).index("left_hip_pitch")] += 0.3
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
        unwrapped.sim.forward()
        scissor_params = unwrapped.reward_manager.get_term_cfg("leg_antisymmetry").params
        grounded_params = unwrapped.reward_manager.get_term_cfg("spin_grounded").params

        set_phase(rest_phase)
        assert float(mdp.leg_antisymmetry(unwrapped, **scissor_params).abs().max()) == pytest.approx(0.0, abs=1e-6)
        assert float(mdp.spin_grounded(unwrapped, **grounded_params).abs().max()) == pytest.approx(0.0, abs=1e-6)

        braking_phase = 0.5 * (MICRODUCK_SPIN_HOLD_END + MICRODUCK_SPIN_BRAKE_END)
        set_phase(braking_phase)
        assert float(mdp.leg_antisymmetry(unwrapped, **scissor_params).abs().min()) > 0.0
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
