# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe-parity and smoke tests for the contributed MicroDuck swizzle environment.

The parity tests need neither the asset nor the simulator: they read the assembled configuration and
compare it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference_tasks3.md`` section 7. The expected values are spelled out
rather than imported from the configuration under test, so that a drifting value fails rather than
agrees with itself.

The task is a *delta* on the roller-skating recipe, so this file tests the delta and asserts that
everything else is untouched -- the scene, the sensors, the events, the terminations and the physics
preset are compared object-for-object against
``test_microduck_rollers_env.py``'s subject rather than re-transcribed. What is transcribed here is
the reward swap, the two re-opened command clamps and the four new curricula.

Simulator-backed checks retrieve the prebuilt USD through Isaac Lab's native asset cache.
Recipe checks do not need an asset; source fidelity comparisons use optional read-only fixtures."""

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
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckSwizzlePPORunnerCfg
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import MicroDuckVelocityRollersFlatEnvCfg
from isaaclab_tasks.contrib.microduck.velocity.swizzle_env_cfg import MicroDuckVelocitySwizzleEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_rollers_usd = pytest.mark.usefixtures("microduck_rollers_usd")
"""Skips the tests that spawn the robot. The parity tests do not need the asset."""

TASK_NAME = "IsaacContrib-Velocity-Swizzle-MicroDuck"

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, upstream's ``num_steps_per_env``."""

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
ACTOR_OBSERVATION_DIM = 61
"""The deployed MicroDuck observation layout (section 11.1).

Identical to the stride task's, which is the point of swapping the head slot's *content* rather than
inserting a term: the head command replaces a zero pad in place, so the width and the order both
survive.
"""

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
CRITIC_OBSERVATION_DIM = 78
"""The privileged critic layout (section 11.2), which is not a deploy contract."""

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
EXPECTED_LEFT_LEG_JOINT_NAMES = EXPECTED_LEG_JOINT_NAMES[:5]
EXPECTED_RIGHT_LEG_JOINT_NAMES = EXPECTED_LEG_JOINT_NAMES[5:]
EXPECTED_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
EXPECTED_TIRE_BODY_NAMES = ["tire", "tire_2", "tire_3", "tire_4"]

_STD_STANDING = {
    ".*hip_yaw.*": 0.05,
    ".*hip_roll.*": 0.05,
    ".*hip_pitch.*": 0.05,
    ".*knee.*": 0.05,
    ".*ankle.*": 0.05,
}
_STD_WALKING = {
    ".*hip_yaw.*": 0.3,
    ".*hip_roll.*": 0.6,
    ".*hip_pitch.*": 0.4,
    ".*knee.*": 0.4,
    ".*ankle.*": 0.25,
}
_STD_RUNNING = {
    ".*hip_yaw.*": 0.5,
    ".*hip_roll.*": 0.8,
    ".*hip_pitch.*": 0.8,
    ".*knee.*": 0.8,
    ".*ankle.*": 0.5,
}
"""The stride task's tolerance dictionaries with the neck, head and wheel entries filtered out.

Upstream filters the dictionaries and narrows the selector together (section 7.4), which is what
makes the filtered entries dead rather than merely unused.
"""

EXPECTED_ADDED_REWARDS = {
    # name: (weight, scalar params)
    "leg_symmetry": (2.0, {}),
    "grounded": (1.0, {"command_name": "base_velocity", "bodies_per_foot": 2}),
    "heading_tracking": (0.0, {"command_name": "base_velocity", "std": 0.5}),
    "head_pose_tracking": (0.0, {"command_name": "head_pose", "std": 0.5}),
}
"""The four terms the swizzle recipe adds to the stride one (section 7.5).

The last two are declared at weight zero and reach 3.0 and 4.0 through the curricula below, which is
where the tension between the two heading terms is managed.
"""

EXPECTED_REMOVED_REWARDS = [
    # the five anti-swizzle terms: this gait is what they exist to suppress
    "single_support",
    "glide",
    "skating_air_time",
    "gait_symmetry",
    "hip_roll_neutral",
    # a negative throttle means backwards here, not brake
    "braking",
    # it would fight the head command
    "neck_joint_pos_l2",
]
"""The seven stride terms upstream deletes (section 7.1, 7.2 and 7.4)."""

EXPECTED_REWARD_COUNT = 18
"""Upstream's swizzle reward dict has eighteen terms (section 7.5): 21 - 7 + 4."""

EXPECTED_CURRICULUM_TERMS = {
    # inherited from the stride recipe
    "action_rate_weight",
    "wheel_friction",
    "com_range",
    "head_com_range",
    # the swizzle's own
    "heading_hold_weight",
    "heading_tracking_weight",
    "head_pose_tracking_weight",
    "head_pose_range",
}

EXPECTED_WEIGHT_STAGES = {
    "heading_hold_weight": ([1.0, 1.0, 0.5, 0.0], [0, 1000, 1750, 2500]),
    "heading_tracking_weight": ([0.0, 0.0, 1.5, 3.0], [0, 1000, 1750, 2500]),
    "head_pose_tracking_weight": ([0.0, 0.0, 2.0, 4.0], [0, 1500, 2250, 3000]),
}
"""The three schedules that introduce the task's re-enabled capabilities (sections 7.3 and 7.4)."""

EXPECTED_HEAD_POSE_RANGE_STAGES = [
    (0, ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (1500, ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (2250, ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15))),
    (3000, ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))),
]
"""The head command envelope, held tiny until the swizzle is solid (section 7.4)."""


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which are pinned by name in their own tests."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


def _live_terms(manager_cfg) -> dict:
    """The manager's terms that are actually registered, i.e. the ones that are not ``None``."""
    return {name: term for name, term in vars(manager_cfg).items() if term is not None}


##
# Recipe parity against upstream (section 7)
##


@pytest.mark.unit
def test_the_reward_recipe_is_the_stride_one_minus_seven_terms_plus_four():
    """The swizzle is the stride recipe with its anti-swizzle half deleted and symmetry paid for."""
    swizzle = MicroDuckVelocitySwizzleEnvCfg().rewards
    stride = MicroDuckVelocityRollersFlatEnvCfg().rewards

    live = _live_terms(swizzle)
    assert len(live) == EXPECTED_REWARD_COUNT
    for name in EXPECTED_REMOVED_REWARDS:
        assert getattr(swizzle, name) is None, name
        # each of them is a real stride term, so none of these deletions is a no-op
        assert getattr(stride, name) is not None, name
    assert set(live) == (set(_live_terms(stride)) - set(EXPECTED_REMOVED_REWARDS)) | set(EXPECTED_ADDED_REWARDS)

    for name, (weight, params) in EXPECTED_ADDED_REWARDS.items():
        term = getattr(swizzle, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        assert set(actual) == set(params), name
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_surviving_stride_terms_keep_their_weights_and_parameters():
    """Only ``pose`` and ``wheel_speed`` are edited; the other nine are carried across untouched."""
    swizzle = MicroDuckVelocitySwizzleEnvCfg().rewards
    stride = MicroDuckVelocityRollersFlatEnvCfg().rewards

    edited = {"pose", "wheel_speed"}
    carried = (set(_live_terms(swizzle)) & set(_live_terms(stride))) - edited
    # 21 stride terms, minus the seven deleted and the two edited
    assert len(carried) == 12
    for name in carried:
        assert getattr(swizzle, name).weight == pytest.approx(getattr(stride, name).weight), name
        assert _scalar_params(getattr(swizzle, name)) == _scalar_params(getattr(stride, name)), name


@pytest.mark.unit
def test_the_posture_reward_is_narrowed_to_the_legs_which_changes_the_value_of_the_mean():
    """The kernel is a mean over the selection, so a ten-joint selection is a different reward.

    Upstream narrows it deliberately (section 7.4): the neck and head are command-driven here, and
    holding them at the stand pose is exactly what the head command asks them not to do. The wheels'
    999.0 tolerance entries -- the stride task's way of neutralizing them inside an eighteen-joint
    mean -- go with the wheels, and this test pins that they are gone rather than left dead.
    """
    swizzle = MicroDuckVelocitySwizzleEnvCfg().rewards.pose
    stride = MicroDuckVelocityRollersFlatEnvCfg().rewards.pose

    assert swizzle.weight == pytest.approx(2.0)
    assert swizzle.params["asset_cfg"].joint_names == EXPECTED_LEG_JOINT_NAMES
    assert len(stride.params["asset_cfg"].joint_names) == 18
    for key, expected in (
        ("std_standing", _STD_STANDING),
        ("std_walking", _STD_WALKING),
        ("std_running", _STD_RUNNING),
    ):
        assert swizzle.params[key] == expected, key
        # every surviving pattern carries the stride task's value: the change is scope, not tuning
        for pattern, value in expected.items():
            assert stride.params[key][pattern] == pytest.approx(value), f"{key}.{pattern}"
        assert ".*passive_.*" not in swizzle.params[key], key
        assert ".*neck.*" not in swizzle.params[key], key
    assert swizzle.params["walking_threshold"] == pytest.approx(0.01)
    assert swizzle.params["running_threshold"] == pytest.approx(0.5)


@pytest.mark.unit
def test_the_wheel_reward_becomes_bidirectional_and_keeps_its_known_stale_radius():
    """A negative throttle now means *skate backwards*, which is why ``braking`` is deleted with it."""
    swizzle = MicroDuckVelocitySwizzleEnvCfg().rewards
    stride = MicroDuckVelocityRollersFlatEnvCfg().rewards

    assert swizzle.wheel_speed.params["bidirectional"] is True
    assert stride.wheel_speed.params.get("bidirectional", False) is False
    assert swizzle.braking is None
    # parity over correctness, as on the stride task: the measured tire radius is 0.0150 m and the
    # deployed policies were trained against the reward's 0.0175 m default (section 7.6)
    assert swizzle.wheel_speed.params["wheel_radius"] == pytest.approx(0.0175)
    assert swizzle.wheel_speed.params["vel_scale"] == pytest.approx(0.3)
    assert swizzle.wheel_speed.weight == pytest.approx(10.0)


@pytest.mark.unit
def test_the_two_gait_terms_read_the_legs_and_the_tires_in_a_pinned_order():
    """The symmetry residual pairs joint against joint, so the two selections are a contract."""
    rewards = MicroDuckVelocitySwizzleEnvCfg().rewards

    left = rewards.leg_symmetry.params["left_joint_cfg"]
    right = rewards.leg_symmetry.params["right_joint_cfg"]
    assert left.joint_names == EXPECTED_LEFT_LEG_JOINT_NAMES
    assert right.joint_names == EXPECTED_RIGHT_LEG_JOINT_NAMES
    assert left.preserve_order and right.preserve_order
    # a joint appearing on both sides would make the residual compare a joint against itself
    assert not set(left.joint_names) & set(right.joint_names)

    sensor_cfg = rewards.grounded.params["sensor_cfg"]
    assert sensor_cfg.name == "contact_forces"
    assert sensor_cfg.body_names == EXPECTED_TIRE_BODY_NAMES
    assert sensor_cfg.preserve_order
    assert rewards.head_pose_tracking.params["asset_cfg"].joint_names == EXPECTED_HEAD_JOINT_NAMES


@pytest.mark.unit
def test_the_command_opens_both_clamps_the_stride_task_closed():
    """The throttle is symmetrized and the heading clamp is opened from zero to +/-0.5 rad."""
    cfg = MicroDuckVelocitySwizzleEnvCfg()
    command = cfg.commands.base_velocity

    assert isinstance(command, mdp.RelativeHeadingVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (-0.6, 0.6)
    assert command.ranges.ang_vel_z == (-0.5, 0.5)
    # a skate still cannot translate sideways, and everything else is the stride task's
    stride_command = MicroDuckVelocityRollersFlatEnvCfg().commands.base_velocity
    assert command.ranges.lin_vel_y == stride_command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.resampling_time_range == stride_command.resampling_time_range
    assert command.rel_forward_envs == stride_command.rel_forward_envs
    assert command.heading_command is False


@pytest.mark.unit
def test_the_head_command_replaces_a_zero_pad_in_place_on_both_groups():
    """Re-assigning an existing term keeps its slot, so the 61-wide deploy contract survives."""
    cfg = MicroDuckVelocitySwizzleEnvCfg()

    assert set(vars(cfg.commands)) == {"base_velocity", "head_pose"}
    head_pose = cfg.commands.head_pose
    assert isinstance(head_pose, mdp.UniformPoseDeltaCommandCfg)
    assert head_pose.resampling_time_range == (2.0, 5.0)
    assert head_pose.ranges == EXPECTED_HEAD_POSE_RANGE_STAGES[0][1]

    for group, expected in (
        (cfg.observations.policy, ACTOR_OBSERVATION_TERMS),
        (cfg.observations.critic, CRITIC_OBSERVATION_TERMS),
    ):
        terms = _observation_terms(group)
        assert list(terms) == [name for name, _ in expected]
        assert terms["head_pose_commands"].func is mdp.generated_commands
        assert terms["head_pose_commands"].params == {"command_name": "head_pose"}
        # the body-pose slot stays a zero pad: this task has no trunk-pose command
        assert terms["body_pose_commands"].func is mdp.zero_command_padding
        assert terms["body_pose_commands"].params == {"dim": 6}
    assert sum(width for _, width in ACTOR_OBSERVATION_TERMS) == ACTOR_OBSERVATION_DIM
    assert sum(width for _, width in CRITIC_OBSERVATION_TERMS) == CRITIC_OBSERVATION_DIM


@pytest.mark.unit
def test_the_curricula_hand_the_heading_over_and_open_the_head_late():
    """Upstream's stage tables, and the crossover the two heading terms fight across."""
    curriculum = MicroDuckVelocitySwizzleEnvCfg().curriculum

    # two-sided, so a schedule quietly appearing or disappearing also fails
    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS

    for name, (weights, iterations) in EXPECTED_WEIGHT_STAGES.items():
        stages = getattr(curriculum, name).params["weight_stages"]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage["weight"] for stage in stages] == pytest.approx(weights), name

    stages = curriculum.head_pose_range.params["range_stages"]
    assert [stage["step"] for stage in stages] == [
        it * STEPS_PER_ITERATION for it, _ in EXPECTED_HEAD_POSE_RANGE_STAGES
    ]
    assert [stage["ranges"] for stage in stages] == [ranges for _, ranges in EXPECTED_HEAD_POSE_RANGE_STAGES]

    # stage 0 of each schedule agrees with the weight the term ships, so nothing is silently dead
    rewards = MicroDuckVelocitySwizzleEnvCfg().rewards
    for name, reward_name in (
        ("heading_hold_weight", "heading_hold"),
        ("heading_tracking_weight", "heading_tracking"),
        ("head_pose_tracking_weight", "head_pose_tracking"),
    ):
        stage_zero = getattr(curriculum, name).params["weight_stages"][0]["weight"]
        assert getattr(rewards, reward_name).weight == pytest.approx(stage_zero), name


@pytest.mark.unit
def test_the_two_heading_terms_are_both_live_across_the_hand_over():
    """Reproduced deliberately: upstream trains the crossover, and nothing in its cfg says so.

    ``heading_hold`` pays for staying at the spawn heading and ``heading_tracking`` for reaching a
    resampled one, so between iterations 1750 and 2500 the policy is paid for both at once
    (section 13.14). The port keeps the overlap rather than squaring the schedules off, because the
    deployed policy was trained through it -- this test is the record that it is a choice.
    """
    curriculum = MicroDuckVelocitySwizzleEnvCfg().curriculum

    hold = {stage["step"]: stage["weight"] for stage in curriculum.heading_hold_weight.params["weight_stages"]}
    track = {stage["step"]: stage["weight"] for stage in curriculum.heading_tracking_weight.params["weight_stages"]}
    crossover = 1750 * STEPS_PER_ITERATION
    assert hold[crossover] > 0.0 and track[crossover] > 0.0
    # and the hand-over completes: exactly one of them is live at the end
    end = 2500 * STEPS_PER_ITERATION
    assert hold[end] == pytest.approx(0.0)
    assert track[end] > 0.0


@pytest.mark.unit
def test_the_scene_events_terminations_and_physics_are_the_stride_task_untouched():
    """This is the same robot on the same floor under the same disturbances, doing a different gait."""
    swizzle = MicroDuckVelocitySwizzleEnvCfg()
    stride = MicroDuckVelocityRollersFlatEnvCfg()

    assert swizzle.events.to_dict() == stride.events.to_dict()
    assert swizzle.terminations.to_dict() == stride.terminations.to_dict()
    assert swizzle.actions.to_dict() == stride.actions.to_dict()
    assert swizzle.scene.to_dict() == stride.scene.to_dict()
    assert swizzle.sim.to_dict() == stride.sim.to_dict()
    assert swizzle.episode_length_s == pytest.approx(stride.episode_length_s)
    assert swizzle.decimation == stride.decimation
    # the contact budget is the stride task's measured one, which bounds this gait's contact set
    assert swizzle.sim.physics.default.solver_cfg.nconmax == 32


@pytest.mark.unit
def test_the_runner_inherits_the_roller_hyper_parameters_under_a_new_log_tree():
    """Upstream replaces two string fields on the roller runner and changes nothing else."""
    runner = MicroDuckSwizzlePPORunnerCfg()

    assert runner.experiment_name == "microduck_velocity_swizzle"
    assert runner.num_steps_per_env == STEPS_PER_ITERATION
    assert runner.save_interval == 250
    # inherited from the roller runner, including its 50 000-iteration ceiling (section 12.4)
    assert runner.max_iterations == 50000
    assert runner.algorithm.entropy_coef == pytest.approx(0.03)
    # off, although this is the one task in the family whose premise is left-right symmetry
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
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_the_leg_symmetry_reward_scores_a_mirrored_pose_above_a_scissored_one():
    """The gait's defining term, evaluated under physics on the joints it resolves by name.

    The condition is ``q_left + q_right ~= 0`` rather than ``q_left ~= q_right``, because the model
    uses mirrored left/right sign conventions -- so this test drives the legs into both poses and
    checks the term prefers the one the swizzle is made of. It also pins that the reward is
    *maximal* at the stand pose, which is mirrored by construction.
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

        term_cfg = unwrapped.reward_manager.get_term_cfg("leg_symmetry")
        left_ids = term_cfg.params["left_joint_cfg"].joint_ids
        right_ids = term_cfg.params["right_joint_cfg"].joint_ids
        assert [robot.joint_names[int(i)] for i in left_ids] == EXPECTED_LEFT_LEG_JOINT_NAMES
        assert [robot.joint_names[int(i)] for i in right_ids] == EXPECTED_RIGHT_LEG_JOINT_NAMES

        def score() -> torch.Tensor:
            unwrapped.sim.forward()
            return mdp.leg_symmetry_reward(unwrapped, **term_cfg.params)

        joint_pos = robot.data.default_joint_pos.torch.clone()
        velocity = torch.zeros_like(joint_pos)
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=velocity)
        at_stand = score()

        # mirrored: the hips swing equal and opposite, which reads as a zero residual
        mirrored = joint_pos.clone()
        mirrored[:, left_ids[2]] += 0.3
        mirrored[:, right_ids[2]] -= 0.3
        robot.write_joint_position_to_sim_index(position=mirrored)
        robot.write_joint_velocity_to_sim_index(velocity=velocity)
        at_mirrored = score()

        # scissored: both hips swing the same way, which is the maximum residual for that excursion
        scissored = joint_pos.clone()
        scissored[:, left_ids[2]] += 0.3
        scissored[:, right_ids[2]] += 0.3
        robot.write_joint_position_to_sim_index(position=scissored)
        robot.write_joint_velocity_to_sim_index(velocity=velocity)
        at_scissored = score()

        # the stand pose is mirrored by construction, so it is at the term's maximum
        assert float(at_stand.max()) == pytest.approx(0.0, abs=1e-3)
        assert float(at_mirrored.min()) == pytest.approx(float(at_stand.min()), abs=1e-3)
        # and the scissor is charged: 0.6 rad of residual spread over five joint pairs
        assert float(at_scissored.max()) < float(at_mirrored.min()) - 0.1
        assert float(at_scissored.mean()) == pytest.approx(-0.6 / 5.0, abs=1e-2)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_rollers_usd
def test_environment_steps_with_random_actions():
    """The registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(TASK_NAME, device="cuda", num_envs=2, num_steps=10)
