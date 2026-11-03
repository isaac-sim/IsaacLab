# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke and recipe-parity tests for the contributed MicroDuck velocity-tracking environments.

The smoke tests spawn :data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_CFG`, whose
prebuilt USD is hosted on Nucleus.
The simulator fixtures resolve it through the native asset cache; source-only comparisons
remain optional read-only checks. Asset setup uses the prebuilt assets through Isaac Lab's native asset cache.

The parity test needs neither the asset nor the simulator: it reads the assembled configuration and
compares it against the upstream recipe, transcribed here from
``artifacts/microduck/upstream_reference.md``. The expected values are spelled out rather than
imported from the configuration under test, so that a drifting value fails rather than agrees with
itself.
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
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg
from isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import MicroDuckVelocityRoughEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

requires_microduck_usd = pytest.mark.usefixtures("microduck_walk_usd")
"""Skips the tests that spawn the robot. The parity test does not need the asset."""

TASK_NAMES = ["IsaacContrib-Velocity-Flat-MicroDuck-Pollen", "IsaacContrib-Velocity-Rough-MicroDuck-Pollen"]
"""The registered MicroDuck velocity tasks."""

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
"""The deployed MicroDuck observation layout: each block of the flat actor vector, in order.

Reference section 7. This is a **deploy contract**: the runtime on the robot rebuilds this vector
by hand from its own sensor reads, so a term that moves or changes width silently invalidates every
trained checkpoint. The two IMU blocks come first -- the design doc's ordering, which puts the
joint blocks first, is wrong and section 7 corrects it.
"""

ACTOR_OBSERVATION_DIM = 61
"""Actor observation width the deployed MicroDuck policy expects.

48 proprioception values plus the 13-wide command block ``[twist(3), head_pose(4), body_pose(6)]``.
"""

CRITIC_OBSERVATION_TERMS = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("velocity_commands", 3),
    ("foot_height", 2),
    ("foot_air_time", 2),
    ("foot_contact", 2),
    ("foot_contact_forces", 6),
    ("head_pose_commands", 4),
    ("body_pose_commands", 6),
]
"""The privileged critic layout (reference section 7), which is not a deploy contract.

Upstream's critic is the actor's terms with the corruption removed, plus the base linear velocity
it deletes from the actor and four sensor-derived foot terms. The sensor widths follow from two
feet, so the total is measured here rather than taken from upstream's own count.
"""

CRITIC_OBSERVATION_DIM = 76
"""Critic observation width, measured from the assembled group."""


def _term_slices(terms: list[tuple[str, int]]) -> dict[str, slice]:
    """Lay a term table out into the column ranges its blocks occupy once concatenated."""
    slices, start = {}, 0
    for name, width in terms:
        slices[name] = slice(start, start + width)
        start += width
    return slices


ACTOR_OBSERVATION_SLICES = _term_slices(ACTOR_OBSERVATION_TERMS)
"""Column range of each actor block in the flat 61-vector."""

_TWIST_COMMAND_CONSTANT = 0.25
_BODY_COMMAND_CONSTANT = 0.5
"""Distinct constants the command blocks are pinned to, so a block read at the wrong offset fails.

The head block is pinned to exactly zero, which is the third distinct value.
"""

_DISPLACED_JOINTS = [(7, "head_yaw", 0.5), (9, "right_hip_yaw", -0.3)]
"""``(column, joint, displacement [rad])`` the joint-block order is probed with.

Upstream's servo layout (reference section 7) is indices 0-4 left leg, 5-8 neck/head, 9-13 right
leg. ``head_yaw`` at 7 and ``right_hip_yaw`` at 9 straddle the neck/right-leg boundary, so any
regrouping of the blocks -- Isaac Lab resolves joints in USD order, which is not this one -- moves
at least one of them.
"""


@pytest.mark.integration
@requires_microduck_usd
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_the_observation_groups_are_the_widths_their_contracts_name(task_name):
    """The actor group is the deployed 61-vector, term for term, and the critic is privileged."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(task_name, device="cuda", num_envs=2)
        env = gym.make(task_name, cfg=env_cfg)
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
@requires_microduck_usd
def test_the_actor_observation_blocks_hold_the_terms_the_layout_names():
    """Every block of the flat actor vector carries the signal the deploy layout puts there."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(TASK_NAMES[0], device="cuda", num_envs=4)
        # pin each command block to its own constant. The curricula are switched off wholesale
        # because several of them write command ranges back at step 0.
        for name in vars(env_cfg.curriculum):
            setattr(env_cfg.curriculum, name, None)
        env_cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
        env_cfg.commands.body_pose.ranges = ((_BODY_COMMAND_CONSTANT, _BODY_COMMAND_CONSTANT),) * 6
        # every bucket off and no heading controller, so the twist is the sampled constant
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_forward_envs = 0.0
        env_cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            setattr(env_cfg.commands.base_velocity.ranges, axis, (_TWIST_COMMAND_CONSTANT,) * 2)

        env = gym.make(TASK_NAMES[0], cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        obs, _ = env.reset()
        policy = obs["policy"]

        # command blocks: no noise rides on them, so these are exact
        assert policy[:, ACTOR_OBSERVATION_SLICES["head_pose_commands"]].abs().max().item() == 0.0
        for name, constant in (
            ("velocity_commands", _TWIST_COMMAND_CONSTANT),
            ("body_pose_commands", _BODY_COMMAND_CONSTANT),
        ):
            block = policy[:, ACTOR_OBSERVATION_SLICES[name]]
            assert torch.equal(block, torch.full_like(block, constant)), name

        # joint block: displace two joints that straddle a block boundary and read the columns back
        robot = env.unwrapped.scene["robot"]
        joint_names = [name for _, name, _ in _DISPLACED_JOINTS]
        joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
        displacement = torch.tensor([[value for _, _, value in _DISPLACED_JOINTS]], device=env.unwrapped.device).repeat(
            env.unwrapped.num_envs, 1
        )
        robot.write_joint_position_to_sim_index(
            position=robot.data.default_joint_pos.torch[:, joint_ids] + displacement,
            joint_ids=joint_ids,
        )

        joint_block = env.unwrapped.observation_manager.compute()["policy"][:, ACTOR_OBSERVATION_SLICES["joint_pos"]]
        expected = torch.zeros_like(joint_block)
        for column, _, value in _DISPLACED_JOINTS:
            expected[:, column] = value
        # the encoder bias (+/-0.015 rad) and the observation noise (+/-0.001 rad) ride on top
        assert torch.allclose(joint_block, expected, atol=0.02)
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@requires_microduck_usd
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_environment_steps_with_random_actions(task_name):
    """Each registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(task_name, device="cuda", num_envs=2, num_steps=10)


##
# Recipe parity against upstream
##

_SQRT_0_1 = math.sqrt(0.1)
_SQRT_0_5 = math.sqrt(0.5)
_SQRT_0_05 = math.sqrt(0.05)

_SOLE_TARGET_HEIGHT = 0.02
"""Upstream's foot-height target [m], measured at the sole site."""

_MEASURED_SOLE_TO_ANKLE_OFFSET = 0.022496
"""Height [m] of the ankle body frame above the foot site at the STAND2 home pose.

Measured independently of the port, with MuJoCo forward kinematics on the pinned
``robot_walk.xml``: ``data.xpos[ankle_left][2] - data.site_xpos[left_foot][2]``.
"""

_SOLE_TO_ANKLE_OFFSET = 0.0225
"""The same offset [m], rounded to a tenth of a millimetre, which is the form the recipe carries."""

EXPECTED_REWARDS = {
    # name: (weight, scalar params)
    "track_lin_vel": (2.0, {"command_name": "base_velocity", "std": _SQRT_0_1}),
    "track_ang_vel": (2.0, {"command_name": "base_velocity", "std": _SQRT_0_5}),
    "upright": (2.0, {"std": _SQRT_0_05}),
    "pose": (1.0, {"command_name": "base_velocity", "walking_threshold": 0.01, "running_threshold": 1.5}),
    "body_ang_vel": (-0.05, {}),
    "angular_momentum": (-0.02, {}),
    "dof_pos_limits": (-1.0, {}),
    "action_rate_l2": (-0.1, {}),
    "air_time": (
        3.0,
        {
            "command_name": "base_velocity",
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_threshold": 0.01,
        },
    ),
    "foot_clearance": (
        -2.0,
        {
            "command_name": "base_velocity",
            "command_threshold": 0.01,
            "target_height": _SOLE_TARGET_HEIGHT + _SOLE_TO_ANKLE_OFFSET,
        },
    ),
    "foot_swing_height": (
        # the relative-error form is not invariant under moving the measurement frame, so the
        # weight carries the compensating factor -- see MICRODUCK_FOOT_TARGET_HEIGHT
        -0.25 * ((_SOLE_TARGET_HEIGHT + _SOLE_TO_ANKLE_OFFSET) / _SOLE_TARGET_HEIGHT) ** 2,
        {
            "command_name": "base_velocity",
            "command_threshold": 0.01,
            "target_height": _SOLE_TARGET_HEIGHT + _SOLE_TO_ANKLE_OFFSET,
        },
    ),
    "foot_slip": (-0.1, {"command_name": "base_velocity", "command_threshold": 0.01}),
    "self_collisions": (-1.0, {}),
    "head_pose_tracking": (2.0, {"command_name": "head_pose", "std": 0.5}),
    "head_pose_bias": (0.0, {"command_name": "head_pose", "tau_s": 1.0}),
    "body_pose_tracking": (
        0.0,
        {
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15.0),
        },
    ),
}
"""Upstream's reward recipe (reference sections 2.4 and 6), keyed by term name.

``soft_landing`` is absent because upstream removes it from its own recipe, and there is no energy
or torque penalty because the mjlab base template has none.
"""

EXPECTED_EVENTS = {
    # name: (mode, scalar params)
    "foot_friction": ("startup", {"static_friction_range": (0.7, 1.3), "restitution_range": (0.0, 0.0)}),
    "encoder_bias": ("startup", {"bias_range": (-0.015, 0.015)}),
    "mass_inertia": ("startup", {"mass_distribution_params": (0.95, 1.05), "operation": "scale"}),
    "reset_base": (
        "reset",
        {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    ),
    "reset_robot_joints": ("reset", {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}),
    "randomize_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_head_com": ("reset", {"com_range": {axis: (-0.003, 0.003) for axis in "xyz"}}),
    "randomize_joint_friction": ("reset", {"scale_range": (0.9, 1.1)}),
    "randomize_armature": ("reset", {"armature_distribution_params": (0.9, 1.1), "operation": "scale"}),
    "push_robot": ("interval", {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}),
}
"""Upstream's domain-randomization suite (reference section 2.6), keyed by term name.

Upstream's ``base_com`` is absent because it selects zero bodies upstream; its
``expand_bam_friction_fields`` is absent because it only registers MuJoCo's
``dof_frictionloss``/``dof_damping`` for per-world expansion, which the BAM actuator's
per-environment storage already is; and the three terms it ships disabled are absent because a term
that is never enabled is not part of the recipe.
"""

EXPECTED_TERMINATIONS = {
    # name: (time_out flag, scalar params)
    "time_out": (True, {}),
    "fell_over": (False, {"limit_angle": math.radians(70.0)}),
    "out_of_terrain_bounds": (True, {"distance_buffer": 20.3}),
    "nan_state": (False, {"sensor_names": ("contact_forces",)}),
}
"""Upstream's terminations (reference sections 2.5 and 6), keyed by term name.

``distance_buffer`` is the one value that is not upstream's literal. Upstream trips at
``|x| > num_rows * size_x / 2 - 0.3``; this term trips at ``|x| > map_width / 2 - buffer`` with
``map_width = num_rows * size_x + 2 * border_width``, so reproducing upstream's bound needs
``buffer = border_width + 0.3``.
"""

EXPECTED_CURRICULUM_TERMS = {
    "terrain_levels",
    "action_rate_weight",
    "head_pose_bias_weight",
    "standing_envs",
    "head_pose_range",
    "body_pose_range",
    "com_range",
    "head_com_range",
}
"""Upstream's curriculum term names (reference sections 2.8 and 6).

``command_vel`` is absent because upstream deletes it: MicroDuck's velocity ranges never widen.
"""

EXPECTED_STAGE_TABLES = {
    "action_rate_weight": (
        "weight_stages",
        "weight",
        [-0.1, -0.2, -0.4, -0.6, -0.8, -1.0],
        [0, 500, 750, 1000, 1250, 1500],
    ),
    "head_pose_bias_weight": ("weight_stages", "weight", [0.0, 1.0, 2.0, 3.0], [0, 600, 1000, 1500]),
    "standing_envs": (
        "standing_stages",
        "rel_standing_envs",
        [0.02, 0.05, 0.1, 0.15, 0.2, 0.25],
        [0, 500, 750, 1000, 1500, 2000],
    ),
    "com_range": ("range_stages", "range", [0.003, 0.005, 0.01, 0.015], [0, 500, 1000, 1500]),
    "head_com_range": ("range_stages", "range", [0.003, 0.005, 0.01], [0, 500, 1000]),
}
"""Upstream's scalar curriculum tables (reference section 6): payloads and iteration boundaries."""

EXPECTED_HEAD_POSE_STAGES = [
    (0, ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (500, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
    (1000, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
    (1500, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))),
    (2000, ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))),
]
"""Upstream's head-pose range ramp, per ``(neck_pitch, head_pitch, head_yaw, head_roll)``."""

STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, upstream's ``num_steps_per_env``."""


EXPECTED_ACTOR_NOISE = {
    "base_ang_vel": 0.03,
    "projected_gravity": 0.01,
    "joint_pos": 0.001,
    "joint_vel": 0.25,
}
"""Half-width of the uniform noise on each corrupted actor term (reference section 2.3).

These are MicroDuck's own overrides of the mjlab base template, which is an order of magnitude
noisier throughout.
"""

EXPECTED_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound on the IMU mounting-misalignment angle (reference section 6)."""

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


def _scalar_params(term_cfg) -> dict:
    """Drop the entity selections, which are compared separately and are not upstream's values."""
    return {key: value for key, value in term_cfg.params.items() if not isinstance(value, SceneEntityCfg)}


def _observation_terms(group) -> dict[str, ObservationTermCfg]:
    """The group's terms in declaration order, which is the order the manager concatenates them."""
    return {name: term for name, term in vars(group).items() if isinstance(term, ObservationTermCfg)}


@pytest.mark.unit
def test_the_actor_observation_layout_is_the_deploy_contract():
    """The actor group is upstream's term order at upstream's noise levels, with corruption on."""
    observations = MicroDuckVelocityRoughEnvCfg().observations
    terms = _observation_terms(observations.policy)

    assert list(terms) == [name for name, _ in ACTOR_OBSERVATION_TERMS]
    assert sum(width for _, width in ACTOR_OBSERVATION_TERMS) == ACTOR_OBSERVATION_DIM
    assert observations.policy.concatenate_terms
    # the deployed policy is trained against corrupted observations; the critic is not
    assert observations.policy.enable_corruption
    for name, term in terms.items():
        magnitude = EXPECTED_ACTOR_NOISE.get(name)
        if magnitude is None:
            assert term.noise is None, name
        else:
            assert (term.noise.n_min, term.noise.n_max) == (-magnitude, magnitude), name


@pytest.mark.unit
def test_the_actor_reads_its_sensors_through_the_upstream_imperfections():
    """The actor sees a biased encoder, a misaligned IMU and a bus latency, as the robot does."""
    cfg = MicroDuckVelocityRoughEnvCfg()
    terms = _observation_terms(cfg.observations.policy)

    for name, wrapped in (
        ("base_ang_vel", mdp.base_ang_vel_imu_misaligned),
        ("projected_gravity", mdp.projected_gravity_imu_misaligned),
    ):
        params = terms[name].params
        assert terms[name].func is mdp.delayed_observation, name
        assert params["term_func"] is wrapped, name
        assert params["term_params"]["max_angle_deg"] == EXPECTED_IMU_MISALIGNMENT_DEG, name
        assert (params["min_lag"], params["max_lag"], params["update_period"]) == (0, 1, 64), name
    # both terms read one shared rotation, so they have to agree on its bound or the accessor raises
    assert (
        terms["base_ang_vel"].params["term_params"]["max_angle_deg"]
        == terms["projected_gravity"].params["term_params"]["max_angle_deg"]
    )
    # distinct configuration objects: an alias would share one delay term's buffer between them
    assert terms["base_ang_vel"] is not terms["projected_gravity"]

    # the servo bus derives velocity from the previous position-sample window: a constant one-step lag
    joint_vel = terms["joint_vel"].params
    assert terms["joint_vel"].func is mdp.delayed_observation
    assert joint_vel["term_func"] is mdp.joint_vel_rel
    assert (joint_vel["min_lag"], joint_vel["max_lag"], joint_vel["update_period"]) == (1, 1, 0)

    # the encoder bias is a closed loop: the observation adds what the action term subtracts. The
    # biased observation without the biased action trains a permanent joint-space offset instead.
    assert terms["joint_pos"].func is mdp.joint_pos_rel_biased
    assert terms["joint_pos"].params["biased"] is True
    assert isinstance(cfg.actions.joint_pos, mdp.BiasedJointPositionActionCfg)


@pytest.mark.unit
def test_the_critic_group_is_privileged_and_reads_the_true_state():
    """The critic sees what the robot cannot: the true sensors, plus the foot and velocity terms."""
    cfg = MicroDuckVelocityRoughEnvCfg()
    terms = _observation_terms(cfg.observations.critic)

    assert list(terms) == [name for name, _ in CRITIC_OBSERVATION_TERMS]
    assert cfg.observations.critic.concatenate_terms
    assert not cfg.observations.critic.enable_corruption
    # ``enable_corruption`` strips the noise, but neither the delay nor the misalignment is gated by
    # it, so the critic has to be wired to the stock terms rather than to the actor's wrapped ones
    for name, term in terms.items():
        assert term.noise is None, name
        assert term.func is not mdp.delayed_observation, name
    assert terms["base_ang_vel"].func is mdp.base_ang_vel
    assert terms["projected_gravity"].func is mdp.projected_gravity
    assert terms["joint_vel"].func is mdp.joint_vel_rel
    assert terms["joint_pos"].params["biased"] is False
    # the privileged half: the base velocity upstream deletes from the actor, and the foot terms
    assert terms["base_lin_vel"].func is mdp.base_lin_vel
    assert terms["foot_height"].func is mdp.foot_height_safe
    assert terms["foot_air_time"].func is mdp.foot_air_time_safe
    assert terms["foot_contact"].func is mdp.foot_contact
    assert terms["foot_contact_forces"].func is mdp.foot_contact_forces_safe
    # a privileged group the runner never reads is dead weight
    assert MicroDuckPPORunnerCfg().obs_groups == {"actor": ["policy"], "critic": ["critic"]}


@pytest.mark.unit
def test_both_observation_groups_read_the_servos_in_the_deploy_order():
    """Isaac Lab resolves joints in USD order; the deployed vector is in MJCF actuator order."""
    observations = MicroDuckVelocityRoughEnvCfg().observations
    actor = _observation_terms(observations.policy)
    critic = _observation_terms(observations.critic)

    selections = {
        "actor.joint_pos": actor["joint_pos"].params["asset_cfg"],
        "actor.joint_vel": actor["joint_vel"].params["term_params"]["asset_cfg"],
        "critic.joint_pos": critic["joint_pos"].params["asset_cfg"],
        "critic.joint_vel": critic["joint_vel"].params["asset_cfg"],
    }
    for name, asset_cfg in selections.items():
        # spelling the servos out also reproduces upstream's ``^(?!passive_).*`` exclusion
        assert asset_cfg.joint_names == EXPECTED_SERVO_JOINT_NAMES, name
        assert asset_cfg.preserve_order, name
    # the columns the integration test probes are the ones the layout names
    for column, joint, _ in _DISPLACED_JOINTS:
        assert EXPECTED_SERVO_JOINT_NAMES[column] == joint


@pytest.mark.unit
def test_the_reward_recipe_matches_upstream_term_for_term():
    """Every reward slot upstream trains with is present, with its weight and its parameters."""
    rewards = MicroDuckVelocityRoughEnvCfg().rewards

    assert set(vars(rewards)) == set(EXPECTED_REWARDS)
    for name, (weight, params) in EXPECTED_REWARDS.items():
        term = getattr(rewards, name)
        assert term.weight == pytest.approx(weight), name
        actual = _scalar_params(term)
        for key, value in params.items():
            assert actual[key] == pytest.approx(value), f"{name}.{key}"


@pytest.mark.unit
def test_the_foot_height_targets_are_offset_to_the_ankle_frame_they_are_measured_in():
    """The sole-height target is re-based, not copied, and the swing weight compensates for it."""
    rewards = MicroDuckVelocityRoughEnvCfg().rewards

    # the rounded offset the recipe uses is the offset MuJoCo reports, to a tenth of a millimetre
    assert pytest.approx(_MEASURED_SOLE_TO_ANKLE_OFFSET, abs=5e-5) == _SOLE_TO_ANKLE_OFFSET
    # a target copied verbatim would ask for a foot 2 cm below the ground it stands on
    assert rewards.foot_clearance.params["target_height"] > _MEASURED_SOLE_TO_ANKLE_OFFSET
    # the swing-height term charges a relative error, so re-basing it also rescales its gradient;
    # weight * (peak / target - 1)^2 must equal upstream's -0.25 * (peak_sole / 0.02 - 1)^2
    target = rewards.foot_swing_height.params["target_height"]
    peak_sole = 0.031
    ported = rewards.foot_swing_height.weight * ((peak_sole + _SOLE_TO_ANKLE_OFFSET) / target - 1.0) ** 2
    upstream = -0.25 * (peak_sole / _SOLE_TARGET_HEIGHT - 1.0) ** 2
    assert ported == pytest.approx(upstream)


@pytest.mark.unit
def test_the_head_pose_terms_pin_the_joint_order_the_command_columns_assume():
    """The head rewards index command columns positionally, so their joint order is load-bearing."""
    rewards = MicroDuckVelocityRoughEnvCfg().rewards

    for name in ("head_pose_tracking", "head_pose_bias"):
        asset_cfg = getattr(rewards, name).params["asset_cfg"]
        assert asset_cfg.joint_names == ["neck_pitch", "head_pitch", "head_yaw", "head_roll"], name
        assert asset_cfg.preserve_order, name


@pytest.mark.unit
def test_the_randomization_suite_matches_upstream_term_for_term():
    """Every domain randomization upstream enables is present, in its mode, with its ranges."""
    events = MicroDuckVelocityRoughEnvCfg().events

    assert set(vars(events)) == set(EXPECTED_EVENTS)
    for name, (mode, params) in EXPECTED_EVENTS.items():
        term = getattr(events, name)
        assert term.mode == mode, name
        actual = _scalar_params(term)
        for key, value in params.items():
            assert actual[key] == value, f"{name}.{key}"
    assert events.push_robot.interval_range_s == (3.0, 6.0)


@pytest.mark.unit
def test_the_terminations_match_upstream_including_the_rebased_terrain_bound():
    """Every termination upstream ends an episode on is present, with its time-out classification."""
    terminations = MicroDuckVelocityRoughEnvCfg().terminations

    assert set(vars(terminations)) == set(EXPECTED_TERMINATIONS)
    for name, (time_out, params) in EXPECTED_TERMINATIONS.items():
        term = getattr(terminations, name)
        assert term.time_out is time_out, name
        assert _scalar_params(term) == pytest.approx(params), name


@pytest.mark.unit
def test_the_head_centre_of_mass_randomization_drops_the_upstream_hip_link():
    """``bearing_roll`` is the right hip-yaw link; upstream lists it among the head bodies in error."""
    body_names = MicroDuckVelocityRoughEnvCfg().events.randomize_head_com.params["asset_cfg"].body_names

    assert "bearing_roll" not in body_names
    assert len(body_names) == 4


@pytest.mark.unit
def test_the_scalar_curricula_reproduce_upstream_stage_tables():
    """The staged schedules carry upstream's payloads at upstream's iteration boundaries."""
    curriculum = MicroDuckVelocityRoughEnvCfg().curriculum

    # two-sided, so an unexpected extra schedule quietly rewriting a weight or a range also fails
    assert set(vars(curriculum)) == EXPECTED_CURRICULUM_TERMS

    for name, (params_key, payload_key, payloads, iterations) in EXPECTED_STAGE_TABLES.items():
        stages = getattr(curriculum, name).params[params_key]
        assert [stage["step"] for stage in stages] == [it * STEPS_PER_ITERATION for it in iterations], name
        assert [stage[payload_key] for stage in stages] == pytest.approx(payloads), name


@pytest.mark.unit
def test_the_head_pose_range_curriculum_reproduces_upstream_stage_table():
    """The head command opens through upstream's five stages, keeping its final caps verbatim."""
    stages = MicroDuckVelocityRoughEnvCfg().curriculum.head_pose_range.params["range_stages"]

    assert len(stages) == len(EXPECTED_HEAD_POSE_STAGES)
    for stage, (iteration, ranges) in zip(stages, EXPECTED_HEAD_POSE_STAGES):
        assert stage["step"] == iteration * STEPS_PER_ITERATION
        assert stage["ranges"] == ranges


@pytest.mark.unit
def test_the_pose_commands_are_registered_at_their_initial_ranges():
    """The head and body pose commands exist, are as wide as their reward terms index, and resample."""
    commands = MicroDuckVelocityRoughEnvCfg().commands

    assert commands.head_pose.ranges == EXPECTED_HEAD_POSE_STAGES[0][1]
    assert commands.head_pose.resampling_time_range == (2.0, 5.0)
    assert commands.body_pose.ranges == ((-0.005, 0.005),) * 3 + ((-0.05, 0.05),) * 3
    assert commands.body_pose.resampling_time_range == (2.0, 5.0)
    # the velocity command's own ranges are fixed; only the standing fraction is on a curriculum
    assert commands.base_velocity.ranges.lin_vel_x == (-0.4, 0.4)
    assert commands.base_velocity.ranges.lin_vel_y == (-0.3, 0.3)
    assert commands.base_velocity.ranges.ang_vel_z == (-1.0, 1.0)


@pytest.mark.unit
def test_the_velocity_command_buckets_and_sim_rates_match_upstream():
    """Pins the recipe scalars the parity tables above do not touch: bucket fractions and rates."""
    cfg = MicroDuckVelocityRoughEnvCfg()
    base_velocity = cfg.commands.base_velocity

    assert base_velocity.resampling_time_range == (3.0, 8.0)
    assert base_velocity.rel_standing_envs == pytest.approx(0.02)
    assert base_velocity.rel_forward_envs == pytest.approx(0.2)
    assert base_velocity.rel_turn_in_place_envs == pytest.approx(0.15)
    assert base_velocity.rel_heading_envs == pytest.approx(0.0)

    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.episode_length_s == pytest.approx(20.0)


@pytest.mark.unit
def test_the_servos_run_on_the_backend_native_path():
    """MicroDuck trains on the Newton-native BAM path, which needs an even decimation to capture.

    Upstream executes its servo model inside the MuJoCo step, republishing the live friction budget
    and viscous coefficient into ``dof_frictionloss``/``dof_damping`` every physics tick; the
    Isaac Lab-executed path instead clips the torque once per control step and never writes to the
    solver. The two are different plants, so which one the task selects is part of the recipe.

    The decimation is asserted here as well as with the rates it belongs to, because it is a
    *precondition* of this choice and not just a control rate: the BAM delay line is actuator state,
    and :class:`~isaaclab_newton.physics.NewtonManager` refuses to CUDA-graph-capture stateful
    Newton actuators at a decimation of one and warns that an odd decimation drops the last state
    update of every replay. Four is even, so the capture is balanced.
    """
    for cfg in (MicroDuckVelocityRoughEnvCfg(), MicroDuckVelocityFlatEnvCfg()):
        assert cfg.sim.use_newton_actuators is True
        assert cfg.decimation % 2 == 0


@pytest.mark.unit
def test_the_runner_iteration_size_is_tied_to_the_curriculum_step_constant():
    """A runner change would otherwise silently desync every curriculum boundary from 'iteration N'.

    ``num_steps_per_env`` exists in two places that must agree: the runner's own field and the
    constant ``velocity_env_cfg.MICRODUCK_STEPS_PER_ITERATION`` the curriculum stage tables convert
    upstream's iteration counts with. The runner imports that constant rather than restating it, so
    this assertion is really pinning that the import stayed wired -- see
    :data:`STEPS_PER_ITERATION` above for the independent, upstream-derived value both must equal.
    """
    assert MicroDuckPPORunnerCfg().num_steps_per_env == STEPS_PER_ITERATION


@pytest.mark.unit
def test_the_terrain_generator_carries_upstream_sub_terrain_mix():
    """The rough terrain is upstream's gentle mix, and the flat task keeps a plain plane."""
    rough = MicroDuckVelocityRoughEnvCfg()
    generator = rough.scene.terrain.terrain_generator

    assert rough.scene.terrain.terrain_type == "generator"
    assert {name: cfg.proportion for name, cfg in generator.sub_terrains.items()} == {
        "flat": 0.25,
        "pyramid_stairs": 0.25,
        "random_grid": 0.30,
        "pyramid_slope": 0.20,
    }
    # the robot lifts its feet 1-2 cm, so nothing it walks over may be taller than that
    assert generator.sub_terrains["pyramid_stairs"].step_height_range == (0.0, 0.015)
    assert generator.sub_terrains["random_grid"].grid_height_range == (0.0, 0.010)
    assert generator.sub_terrains["pyramid_slope"].slope_range == (0.03, 0.10)
    # upstream rings none of its sub-terrains with a flat border
    assert generator.sub_terrains["pyramid_slope"].border_width == 0.0

    flat = MicroDuckVelocityFlatEnvCfg()
    assert flat.scene.terrain.terrain_type == "plane"
    assert flat.scene.terrain.terrain_generator is None
    # the terrain-level curriculum reads the generator's configuration, so it cannot stay behind
    assert flat.curriculum.terrain_levels is None
    assert rough.curriculum.terrain_levels is not None


@pytest.mark.unit
def test_the_self_collision_sensor_reports_a_force_matrix():
    """``self_collision_cost`` reads a filtered force matrix, which needs a dedicated sensor."""
    scene = MicroDuckVelocityRoughEnvCfg().scene

    assert scene.self_collision.filter_prim_paths_expr
    # one sensing body against one filter body: the two soles are the only collidable geometries on
    # the robot, so this is the whole self-collision signal, counted once rather than from both ends
    assert "ankle_left" in scene.self_collision.prim_path
    assert "ankle_right" in scene.self_collision.filter_prim_paths_expr[0]
    assert not scene.contact_forces.filter_prim_paths_expr


@pytest.mark.integration
@requires_microduck_usd
def test_bam_friction_randomization_updates_only_selected_environments():
    """The live native drive shares one friction draw per robot and preserves other robots."""
    from isaaclab.actuators.newton.adapter import read_group_parameter

    task = "IsaacContrib-Velocity-Flat-MicroDuck-Pollen"
    cfg = parse_env_cfg(task, device="cuda", num_envs=4)
    env = gym.make(task, cfg=cfg)
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        before = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale").clone()
        assert torch.all((before >= 0.9) & (before <= 1.1))
        torch.testing.assert_close(before, before[:, :1].expand_as(before))
        selected = torch.tensor([1, 3], device=env.unwrapped.device)
        mdp.randomize_bam_friction(env.unwrapped, selected, scale_range=(2.0, 3.0))
        after = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale")
        assert after.shape == (4, 14)
        assert torch.all((after[selected] >= 2.0) & (after[selected] <= 3.0))
        torch.testing.assert_close(after[selected], after[selected, :1].expand(-1, 14))
        torch.testing.assert_close(after[[0, 2]], before[[0, 2]])
        assert after[selected, 0].std() > 0.0

        # Measure MuJoCo's resolved joint limits instead of testing a removed configuration flag.
        solver = env.unwrapped.sim.backend_manager.backend.solver
        solref = torch.as_tensor(solver.mj_model.jnt_solref)
        expected = torch.tensor([0.02, 1.0], dtype=solref.dtype).expand_as(solref)
        torch.testing.assert_close(solref, expected)
    finally:
        env.close()
        SimulationContext.clear_instance()
