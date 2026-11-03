# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe-parity, defect-recording and smoke tests for the contributed MicroDuck crouch-glide task.

The parity tests need neither the asset nor the simulator: they read the assembled configuration and
compare it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference_tasks3.md`` section 8. The expected values are spelled out
rather than imported from the configuration under test, so that a drifting value fails rather than
agrees with itself.

Two of the tests here are unusual and deliberate:

* :func:`test_the_crouch_keyframe_is_recorded_as_violating_the_compiled_joint_limits` **records a
  known upstream defect** against the *compiled model* rather than against a transcribed constant.
  The crouch keyframe asks two joints for angles their stops do not allow (upstream issue draft 018),
  the port reproduces it verbatim for parity, and this test is what stops that being invisible: it
  asserts the violation is exactly the two joints and exactly the magnitude measured, so a future
  asset or pose change that removes it fails loudly and can be adopted deliberately.
* :func:`test_the_degenerate_wheel_event_survives_and_nothing_depends_on_it` is the **interlock**
  between upstream's two coupled defects (sections 13.1 and 13.3). Both halves are asserted: that
  the degenerate event is still registered, and that the per-environment friction storage it
  accidentally props up upstream exists here without it.

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
from isaaclab.actuators import BamActuatorCfg
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
from isaaclab_tasks.contrib.microduck.rollercrouch.agents.rsl_rl_ppo_cfg import MicroDuckRollerCrouchPPORunnerCfg
from isaaclab_tasks.contrib.microduck.rollercrouch.rollercrouch_env_cfg import (
    MICRODUCK_CROUCH_DESCENT_END,
    MICRODUCK_CROUCH_HOLD_END,
    MICRODUCK_CROUCH_LEAN_PITCH,
    MICRODUCK_CROUCH_PERIOD,
    MICRODUCK_CROUCH_POSE,
    MICRODUCK_CROUCH_POSE_STD,
    MICRODUCK_CROUCH_RISE_END,
    MICRODUCK_ENTRY_VELOCITY_X,
    MICRODUCK_STAND_POSE,
    MicroDuckRollerCrouchFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import MicroDuckVelocityRollersFlatEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_rollers_usd = pytest.mark.usefixtures("microduck_rollers_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-RollerCrouch-Flat-MicroDuck"

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, upstream's ``num_steps_per_env``."""

ACTOR_OBSERVATION_DIM = 61
CRITIC_OBSERVATION_DIM = 78
"""The shared deploy contract and the roller family's privileged group (sections 11.1 and 11.2)."""

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
EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
EXPECTED_TIRE_BODY_NAMES = ["tire", "tire_2", "tire_3", "tire_4"]
EXPECTED_WHEEL_JOINT_NAMES = ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]

_POSE_SEGMENTS = {"descent_end": 0.10, "hold_end": 0.50, "rise_end": 0.60}

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "upright": (2.0, {"std": math.sqrt(0.2)}),
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    # upstream declares -1.0 and its curriculum's stage 0 is -0.5, which is what is ever live
    "action_rate_l2": (-0.5, {}),
    "crouch_glide_pose": (
        6.0,
        {
            "command_name": "base_velocity",
            "crouch_pose": MICRODUCK_CROUCH_POSE,
            "stand_pose": MICRODUCK_STAND_POSE,
            "std": 0.4,
            **_POSE_SEGMENTS,
        },
    ),
    "crouch_glide_pose_l1": (
        2.0,
        {
            "command_name": "base_velocity",
            "crouch_pose": MICRODUCK_CROUCH_POSE,
            "stand_pose": MICRODUCK_STAND_POSE,
            **_POSE_SEGMENTS,
        },
    ),
    "forward_speed": (1.0, {"vel_ref": 0.2}),
    "crouch_forward_lean": (
        1.0,
        {"command_name": "base_velocity", "target_pitch": 0.08, "std": 0.1, **_POSE_SEGMENTS},
    ),
    "feet_flat": (-2.0, {"normal_axis": (0.0, 1.0, 0.0), "bodies_per_foot": 2}),
    "self_collisions": (-1.0, {}),
    "neck_action_rate_l2": (-0.5, {"action_name": "joint_pos"}),
    "joint_torques_l2": (-1e-3, {}),
}
"""Upstream's twelve-term reward recipe (section 8.4), keyed by term name, at its live weights."""

EXPECTED_DELETED_REWARDS = [
    # the whole skating stroke, which a crouch-glide is not
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    # the ``keep`` set also removes the regularizer every ``del``-list task silently inherits
    "dof_pos_limits",
    "neck_joint_pos_l2",
    "action_over_limit",
]
"""Terms of the skating recipe upstream's ``keep`` set removes (section 8.4)."""

EXPECTED_CURRICULUM_TERMS = {"action_rate_weight", "com_range", "head_com_range"}
"""Three schedules, the leanest in the family (section 8.9). No wheel-friction ramp -- see below."""

EXPECTED_WEIGHT_STAGES = {"action_rate_weight": ([-0.5, -0.8, -1.0], [0, 250, 500])}
EXPECTED_RANGE_STAGES = {
    "com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
    "head_com_range": ([0.003, 0.005, 0.01], [0, 500, 1000]),
}

EXPECTED_CROUCH_LIMIT_VIOLATIONS = {
    # joint: (target [rad], the stop it passes [rad], the overshoot [rad])
    "neck_pitch": (1.0937, 1.0472, 0.0465),
    "right_knee": (-1.5907, -1.5708, 0.0199),
}
"""The two crouch targets that lie outside the model's hard joint limits (section 13.6, draft 018).

The stops are quoted here only so the *reason* is legible; the test resolves them from the compiled
model and compares against the model, not against these numbers.
"""


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which are pinned by name in their own tests."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


##
# Recipe parity against upstream (section 8)
##


@pytest.mark.unit
def test_the_reward_recipe_matches_upstream_term_for_term():
    """Every reward slot upstream trains with is present, with its weight and its parameters."""
    rewards = MicroDuckRollerCrouchFlatEnvCfg().rewards

    # two-sided, so a skating term the ``keep`` set should have removed also fails
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
def test_the_cycle_segments_are_upstream_durations_on_a_five_second_clock():
    """The four segments are a 0.5 s fold, a 2 s glide, a 0.5 s rise and a 2 s standing rest."""
    cfg = MicroDuckRollerCrouchFlatEnvCfg()

    assert pytest.approx(5.0) == MICRODUCK_CROUCH_PERIOD
    assert (MICRODUCK_CROUCH_DESCENT_END, MICRODUCK_CROUCH_HOLD_END, MICRODUCK_CROUCH_RISE_END) == (0.10, 0.50, 0.60)
    durations = [
        MICRODUCK_CROUCH_DESCENT_END,
        MICRODUCK_CROUCH_HOLD_END - MICRODUCK_CROUCH_DESCENT_END,
        MICRODUCK_CROUCH_RISE_END - MICRODUCK_CROUCH_HOLD_END,
        1.0 - MICRODUCK_CROUCH_RISE_END,
    ]
    assert [d * MICRODUCK_CROUCH_PERIOD for d in durations] == pytest.approx([0.5, 2.0, 0.5, 2.0])
    # and a twenty-second episode is exactly four cycles
    assert cfg.episode_length_s / MICRODUCK_CROUCH_PERIOD == pytest.approx(4.0)


@pytest.mark.unit
def test_the_phase_command_starts_every_episode_standing():
    """``randomize_phase`` is False here and True on the ground-pick task, and the difference matters.

    The deployed runtime launches the cycle from a standing button press, so a randomized start would
    train "stay low" from spawns that are already low -- the exact failure upstream's comment warns
    about (section 8.6).
    """
    cfg = MicroDuckRollerCrouchFlatEnvCfg()

    assert set(vars(cfg.commands)) == {"base_velocity"}
    command = cfg.commands.base_velocity
    assert isinstance(command, mdp.GroundPickPhaseCommandCfg)
    assert command.period == pytest.approx(MICRODUCK_CROUCH_PERIOD)
    assert command.randomize_phase is False
    assert command.heading_command is False
    # the inherited velocity ranges are never sampled: the term writes the phase encoding directly
    assert command.ranges.lin_vel_x == command.ranges.lin_vel_y == command.ranges.ang_vel_z == (0.0, 0.0)


@pytest.mark.unit
def test_the_entry_momentum_is_injected_through_the_root_reset_and_not_through_a_push():
    """A reset-mode push adds to the *current* root velocity, which on a diverged env means NaN.

    Upstream fixed exactly this and locks the distinction with its own regression test; its spin task
    restates the warning verbatim (section 8.7). Reintroducing the push would reintroduce a NaN
    generator, so the check is that the entry speed is in the root reset and that no reset-mode push
    exists to carry it.
    """
    events = MicroDuckRollerCrouchFlatEnvCfg().events

    assert events.reset_base.params["velocity_range"] == {"x": MICRODUCK_ENTRY_VELOCITY_X}
    assert MICRODUCK_ENTRY_VELOCITY_X == (0.2, 0.5)
    assert events.push_robot.mode == "interval"
    reset_pushes = [
        name for name, term in vars(events).items() if getattr(term, "mode", None) == "reset" and "push" in name
    ]
    assert reset_pushes == []
    # and the spawn height is unchanged: the roller task's absolute (0.1335, 0.1435) band
    spawn_z = MicroDuckRollerCrouchFlatEnvCfg().scene.robot.init_state.pos[2]
    low, high = events.reset_base.params["pose_range"]["z"]
    assert (spawn_z + low, spawn_z + high) == pytest.approx((0.1335, 0.1435))


@pytest.mark.unit
def test_the_degenerate_wheel_event_survives_and_nothing_depends_on_it():
    """The interlock between upstream's two coupled defects (sections 13.1 and 13.3).

    Upstream ships a wheel-friction randomization whose range is ``(0.0, 0.0)`` and, unlike the
    skating task's, no curriculum that ramps it -- so it writes zero forever and the environment
    trains on frictionless bearings (issue draft 017). It is nonetheless **load-bearing upstream**:
    it is the sole declarer of ``dof_frictionloss``, which is what puts that model field into
    MuJoCo's per-world expansion set, and upstream omits the dedicated ``expand_bam_friction_fields``
    event that should have done that job. Deleting the "useless" event there breaks the BAM actuator
    at the first multi-environment step.

    Both halves are checked here:

    * the degenerate event is still registered, verbatim, so a cleanup pass that removed it would
      fail rather than silently change what the policy trained against;
    * nothing in this port depends on it, which is the other half of the same ruling. The
      per-environment friction storage is asserted in
      :func:`test_the_bam_friction_storage_is_per_environment_without_the_wheel_event`, under
      physics and with the event removed.
    """
    events = MicroDuckRollerCrouchFlatEnvCfg().events
    curriculum = MicroDuckRollerCrouchFlatEnvCfg().curriculum

    assert events.randomize_wheel_friction is not None
    assert events.randomize_wheel_friction.mode == "reset"
    assert events.randomize_wheel_friction.func is mdp.randomize_joint_dry_friction
    assert events.randomize_wheel_friction.params["friction_range"] == (0.0, 0.0)
    assert events.randomize_wheel_friction.params["asset_cfg"].joint_names == EXPECTED_WHEEL_JOINT_NAMES
    # the schedule the skating task drives it with is deliberately absent, which is what makes the
    # range permanent rather than a starting point
    assert "wheel_friction" not in vars(curriculum)
    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS
    # and there is no expansion event to add: it has no counterpart in this port on any task
    assert not hasattr(events, "expand_bam_friction_fields")


@pytest.mark.unit
def test_the_two_keyframes_are_upstream_captures_reproduced_verbatim():
    """Both poses were read off the physical robot, asymmetries and stale comments and all."""
    rewards = MicroDuckRollerCrouchFlatEnvCfg().rewards

    assert set(MICRODUCK_CROUCH_POSE) == set(MICRODUCK_STAND_POSE) == set(EXPECTED_SERVO_JOINT_NAMES)
    for term in (rewards.crouch_glide_pose, rewards.crouch_glide_pose_l1):
        assert term.params["crouch_pose"] == MICRODUCK_CROUCH_POSE
        assert term.params["stand_pose"] == MICRODUCK_STAND_POSE
        # both poses name all fourteen servos, so the mean is over the head as well as the legs --
        # unlike the sit/stand task, where the head is command-driven
        assert term.params["asset_cfg"].joint_names == EXPECTED_SERVO_JOINT_NAMES
        assert term.params["asset_cfg"].preserve_order
    assert rewards.crouch_glide_pose.params["std"] == pytest.approx(MICRODUCK_CROUCH_POSE_STD)
    # the capture is not mirrored, which is what a reading off a real robot looks like
    assert abs(MICRODUCK_CROUCH_POSE["left_knee"]) != pytest.approx(abs(MICRODUCK_CROUCH_POSE["right_knee"]))
    # the L1 companion negates itself, hence its positive weight
    assert rewards.crouch_glide_pose_l1.weight > 0.0
    assert rewards.crouch_forward_lean.params["target_pitch"] == pytest.approx(MICRODUCK_CROUCH_LEAN_PITCH)


@pytest.mark.unit
def test_the_curricula_reproduce_upstream_stage_tables():
    """The staged schedules carry upstream's payloads at upstream's iteration boundaries."""
    curriculum = MicroDuckRollerCrouchFlatEnvCfg().curriculum

    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS
    for name, (weights, iterations) in EXPECTED_WEIGHT_STAGES.items():
        stages = getattr(curriculum, name).params["weight_stages"]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage["weight"] for stage in stages] == pytest.approx(weights), name
    for name, (ranges, iterations) in EXPECTED_RANGE_STAGES.items():
        stages = getattr(curriculum, name).params["range_stages"]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage["range"] for stage in stages] == pytest.approx(ranges), name
    # stage 0 agrees with the declared weight, so nothing is silently dead
    assert MicroDuckRollerCrouchFlatEnvCfg().rewards.action_rate_l2.weight == pytest.approx(
        curriculum.action_rate_weight.params["weight_stages"][0]["weight"]
    )


@pytest.mark.unit
def test_the_scene_actions_observations_and_terminations_are_the_roller_task_untouched():
    """Upstream rebuilds these from the mjlab template and arrives at the roller recipe (section 8.7)."""
    crouch = MicroDuckRollerCrouchFlatEnvCfg()
    rollers = MicroDuckVelocityRollersFlatEnvCfg()

    assert crouch.observations.to_dict() == rollers.observations.to_dict()
    assert crouch.actions.to_dict() == rollers.actions.to_dict()
    assert crouch.terminations.to_dict() == rollers.terminations.to_dict()
    assert crouch.scene.to_dict() == rollers.scene.to_dict()
    assert crouch.episode_length_s == pytest.approx(rollers.episode_length_s)
    # a bad crouch ends the episode: unlike the stand-up family, the tilt termination stays live
    assert crouch.terminations.fell_over.params["limit_angle"] == pytest.approx(math.radians(70.0))
    # the events differ in exactly one parameter
    crouch_events, roller_events = crouch.events.to_dict(), rollers.events.to_dict()
    differing = [name for name in crouch_events if crouch_events[name] != roller_events[name]]
    assert differing == ["reset_base"]


@pytest.mark.unit
def test_the_contact_budget_is_measured_rather_than_inherited():
    """The fold is a contact set the skating profile does not cover, so it was profiled separately."""
    solver = MicroDuckRollerCrouchFlatEnvCfg().sim.physics.default.solver_cfg
    roller_solver = MicroDuckVelocityRollersFlatEnvCfg().sim.physics.default.solver_cfg

    # measured peaks: 90 constraints and 29 contacts per environment, against the roller task's 83/26
    assert solver.njmax >= 90
    assert solver.nconmax >= 29
    assert solver.njmax > roller_solver.njmax
    assert solver.nconmax > roller_solver.nconmax
    # and above the mjlab template's inherited 35, which upstream never revisits here
    assert solver.nconmax >= 35
    # the solver iteration counts are upstream's template values, untouched
    assert (solver.iterations, solver.ls_iterations) == (10, 20)


@pytest.mark.unit
def test_the_runner_keeps_the_family_hyper_parameters_under_its_own_log_tree():
    """Upstream's runner differs from the velocity one in two fields (section 8.10)."""
    runner = MicroDuckRollerCrouchPPORunnerCfg()

    assert runner.experiment_name == "microduck_rollercrouch"
    assert runner.max_iterations == 8000
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    # the family's exploration bonus, not the skating task's tripled one
    assert runner.algorithm.entropy_coef == pytest.approx(0.01)
    assert runner.algorithm.symmetry_cfg is None
    assert runner.obs_groups == {"actor": ["policy"], "critic": ["critic"]}


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
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_crouch_keyframe_is_recorded_as_violating_the_compiled_joint_limits():
    """Records upstream issue draft 018 against the model, so the defect is pinned rather than hidden.

    The crouch keyframe was read off the physical robot and never checked against the simulation
    model's stops. Two of its fourteen targets lie outside them (section 13.6), so both pose rewards
    charge a residual on those joints that no policy can zero and the "hold the crouch" reward can
    never saturate. The port reproduces the pose verbatim on parity grounds -- the deployed policy
    was trained against these targets -- and this test is the record.

    The limits are read from the **compiled model**, not transcribed, so the expectation tracks the
    asset. If a future asset or pose change removes the violation this test fails, which is the
    intent: the defect stops being true and that should be adopted deliberately rather than silently.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=2)
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        env.reset()
        robot = env.unwrapped.scene["robot"]
        limits = robot.data.joint_pos_limits.torch[0]

        violations = {}
        for name, target in MICRODUCK_CROUCH_POSE.items():
            index = list(robot.joint_names).index(name)
            lower, upper = float(limits[index, 0]), float(limits[index, 1])
            if target > upper:
                violations[name] = (target, upper, target - upper)
            elif target < lower:
                violations[name] = (target, lower, lower - target)

        assert set(violations) == set(EXPECTED_CROUCH_LIMIT_VIOLATIONS), (
            f"the recorded draft-018 violation changed: measured {violations}"
        )
        for name, (target, stop, overshoot) in EXPECTED_CROUCH_LIMIT_VIOLATIONS.items():
            measured_target, measured_stop, measured_overshoot = violations[name]
            assert measured_target == pytest.approx(target, abs=1e-4), name
            assert measured_stop == pytest.approx(stop, abs=1e-3), name
            assert measured_overshoot == pytest.approx(overshoot, abs=1e-3), name

        # the standing keyframe is inside the stops on every joint, so the defect is the crouch's
        for name, target in MICRODUCK_STAND_POSE.items():
            index = list(robot.joint_names).index(name)
            assert float(limits[index, 0]) <= target <= float(limits[index, 1]), name
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_bam_friction_storage_is_per_environment_without_the_wheel_event():
    """The other half of the section 13.1 / 13.3 interlock: nothing here props up the expansion.

    Upstream's BAM actuator writes per-environment friction into a *shared* MuJoCo model field that
    has to be expanded per world first, and in this environment the only thing that expands it is the
    degenerate wheel event of section 13.3 -- so removing that event breaks the actuator there. Isaac
    Lab's actuator owns per-environment storage unconditionally, which is why upstream's
    ``expand_bam_friction_fields`` has no counterpart in this port.

    This test builds the environment with the wheel event **removed** and asserts the storage is
    still per-environment and still writable, i.e. the interlock does not exist here. The companion
    unit test asserts the event is nonetheless kept.
    """
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAME, device="cuda", num_envs=4)
        env_cfg.events.randomize_wheel_friction = None
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        unwrapped = env.unwrapped
        env.reset()

        assert "randomize_wheel_friction" not in unwrapped.event_manager.active_terms["reset"]

        robot = unwrapped.scene["robot"]
        # the BAM friction budget is stored per environment by the actuator itself, on both
        # execution paths, rather than in a shared model field that an event has to expand
        group = next(name for name, cfg in robot.cfg.actuators.items() if isinstance(cfg, BamActuatorCfg))
        friction_scale = read_group_parameter(robot.actuators, group, "drive", "friction_scale")
        assert friction_scale.shape[0] == unwrapped.num_envs

        # and the joint-level dry friction the wheel event would have written is per-environment too:
        # write four different values and read four different values back
        wheel_ids = [list(robot.joint_names).index(name) for name in EXPECTED_WHEEL_JOINT_NAMES]
        written = torch.arange(unwrapped.num_envs, device=unwrapped.device, dtype=torch.float32).unsqueeze(1) * 1e-3
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=written.expand(-1, len(wheel_ids)).contiguous(),
            joint_ids=wheel_ids,
            env_ids=torch.arange(unwrapped.num_envs, device=unwrapped.device),
        )
        read_back = robot.data.joint_friction_coeff.torch[:, wheel_ids]
        assert torch.allclose(read_back, written.expand(-1, len(wheel_ids)), atol=1e-9)

        # and it steps, which is the failure upstream sees at the first multi-environment step
        action = torch.zeros((unwrapped.num_envs, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
        with torch.inference_mode():
            for _ in range(4):
                env.step(action)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_pose_reward_follows_the_phase_from_the_stand_keyframe_to_the_crouch_one():
    """The task's objective, evaluated under physics across the cycle it is defined on.

    The robot is held at each keyframe in turn and the reward is read at the two phases the cycle
    dwells on. The contract is a *crossing*: at phase 0 the standing keyframe scores highest, at the
    middle of the crouch dwell the crouched one does, and the same term produces both.

    It also records upstream issue draft 018 as a *reward* rather than as a joint angle: the crouch
    target is scored twice at the crouch phase, once at the exact keyframe and once at the keyframe
    clamped into the compiled model's joint stops. The second is the best a policy can physically
    reach, and it is strictly below the first -- so the "hold the crouch" reward can never saturate.
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
        term_cfg = unwrapped.reward_manager.get_term_cfg("crouch_glide_pose")

        def hold(pose: dict[str, float]) -> torch.Tensor:
            joint_pos = robot.data.default_joint_pos.torch.clone()
            for name, angle in pose.items():
                joint_pos[:, list(robot.joint_names).index(name)] = angle
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
            unwrapped.sim.forward()
            return mdp.crouch_glide_pose_gaussian(unwrapped, **term_cfg.params)

        def set_phase(phase: float) -> None:
            command = unwrapped.command_manager.get_command("base_velocity")
            command[:, 0] = math.cos(2.0 * math.pi * phase)
            command[:, 1] = math.sin(2.0 * math.pi * phase)

        # phase 0: the cycle is standing, so the standing keyframe is the target
        set_phase(0.0)
        standing_at_stand = hold(MICRODUCK_STAND_POSE)
        crouched_at_stand = hold(MICRODUCK_CROUCH_POSE)
        assert float(standing_at_stand.min()) > float(crouched_at_stand.max())

        # mid-dwell: the cycle is crouched, so the crouched keyframe is
        set_phase(0.5 * (MICRODUCK_CROUCH_DESCENT_END + MICRODUCK_CROUCH_HOLD_END))
        standing_at_crouch = hold(MICRODUCK_STAND_POSE)
        crouched_at_crouch = hold(MICRODUCK_CROUCH_POSE)
        assert float(crouched_at_crouch.min()) > float(standing_at_crouch.max())

        # both on-target scores saturate, because holding a keyframe exactly is what the term asks
        assert float(standing_at_stand.min()) == pytest.approx(1.0, abs=1e-3)
        assert float(crouched_at_crouch.min()) == pytest.approx(1.0, abs=1e-3)

        # draft 018, recorded as a reward: the crouch keyframe as a *reachable* pose -- clamped into
        # the model's stops -- scores strictly below the unreachable target it is clamped from, so
        # the term's maximum is not attainable and the residual is permanent
        limits = robot.data.joint_pos_limits.torch[0]
        reachable = {
            name: min(
                max(angle, float(limits[list(robot.joint_names).index(name), 0])),
                float(limits[list(robot.joint_names).index(name), 1]),
            )
            for name, angle in MICRODUCK_CROUCH_POSE.items()
        }
        assert reachable != MICRODUCK_CROUCH_POSE
        crouched_reachable = hold(reachable)
        assert float(crouched_reachable.max()) < float(crouched_at_crouch.min())
        # and the shortfall is small, which is why it is a transfer hazard rather than a blocker
        assert float(crouched_reachable.min()) > 0.99
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
