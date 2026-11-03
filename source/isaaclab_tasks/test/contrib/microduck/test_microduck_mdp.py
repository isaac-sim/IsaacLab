# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the MicroDuck-specific MDP terms.

The command and observation terms only need ``num_envs``, ``device`` and the asset they read, so
they run against an environment double instead of a simulated scene. The bucket fractions are
driven to their extremes (0.0 or 1.0) so every assertion is exact rather than statistical; the
distributional claims -- that the velocity buckets are independent draws rather than a partition,
and that the IMU misalignment is zero-centred rather than a fixed tilt -- are pinned with enough
environments that a false negative is impossible in practice.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


HEAD_POSE_RANGES = ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
"""Initial head-pose command ranges [rad] for ``(neck_pitch, head_pitch, head_yaw, head_roll)``."""

BODY_POSE_RANGES = (
    (-0.005, 0.005),
    (-0.005, 0.005),
    (-0.005, 0.005),
    (-0.05, 0.05),
    (-0.05, 0.05),
    (-0.05, 0.05),
)
"""Initial body-pose command ranges for ``(x, y, z)`` [m] and ``(roll, pitch, yaw)`` [rad]."""

ANG_VEL_Z_RANGE = (-1.0, 1.0)
"""Yaw-rate command range [rad/s] MicroDuck trains with."""


JOINT_NAMES = ["hip", "knee", "ankle", "neck"]
"""Joint names of the articulation double, standing in for the 14 MicroDuck servos."""

BODY_NAMES = ["trunk", "ankle_left", "ankle_right"]
"""Body names of the articulation double, in the resolved order the real scene reports."""

ENCODER_BIAS_RANGE = (-0.015, 0.015)
"""Per-joint encoder-bias range [rad] MicroDuck randomizes over (reference section 2.6)."""

IMU_MISALIGNMENT_ANGLE_DEG = 6.0
"""Upper bound [deg] on the IMU mounting misalignment (reference section 6)."""


class _DummyTensorView:
    """Stands in for a ``ProxyArray``, which exposes its contents under ``.torch``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.torch = tensor

    def __len__(self) -> int:
        return len(self.torch)


class _DummyRobotData:
    """The articulation state the MicroDuck command, observation and action terms read."""

    def __init__(self, num_envs: int, device: str) -> None:
        self.root_lin_vel_b = _DummyTensorView(torch.zeros(num_envs, 3, device=device))
        self.root_ang_vel_b = _DummyTensorView(torch.zeros(num_envs, 3, device=device))
        self.heading_w = _DummyTensorView(torch.zeros(num_envs, device=device))
        gravity = torch.tensor([0.0, 0.0, -1.0], device=device).expand(num_envs, 3)
        self.projected_gravity_b = _DummyTensorView(gravity.clone())
        self.joint_pos = _DummyTensorView(torch.zeros(num_envs, len(JOINT_NAMES), device=device))
        self.default_joint_pos = _DummyTensorView(torch.zeros(num_envs, len(JOINT_NAMES), device=device))
        self.body_pos_w = _DummyTensorView(torch.zeros(num_envs, len(BODY_NAMES), 3, device=device))
        self.root_link_pos_w = _DummyTensorView(torch.zeros(num_envs, 3, device=device))
        # Isaac Lab orders quaternions (x, y, z, w), so identity is the last column
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(num_envs, 4)
        self.root_link_quat_w = _DummyTensorView(identity.clone())


class _DummyRobot:
    def __init__(self, num_envs: int, device: str) -> None:
        self.data = _DummyRobotData(num_envs, device)
        self.num_joints = len(JOINT_NAMES)
        self.device = device
        self.joint_position_target: torch.Tensor | None = None
        self.actuators = SimpleNamespace(target_command=SimpleNamespace(set_position_index=self._set_position))

    def find_joints(self, name_keys, preserve_order: bool = False, *, as_proxy: bool = False):
        names = list(name_keys)
        ids = [JOINT_NAMES.index(name) for name in names]
        indices = torch.tensor(ids, dtype=torch.long, device=self.device)
        return (_DummyTensorView(indices) if as_proxy else ids), names

    def _set_position(self, value: torch.Tensor, joint_ids) -> None:
        self.joint_position_target = value.clone()


class _DummyContactSensorData:
    def __init__(self, num_envs: int, device: str) -> None:
        self.net_forces_w = _DummyTensorView(torch.zeros(num_envs, len(BODY_NAMES), 3, device=device))
        self.current_air_time = _DummyTensorView(torch.zeros(num_envs, len(BODY_NAMES), device=device))


class _DummyContactSensor:
    def __init__(self, num_envs: int, device: str) -> None:
        self.data = _DummyContactSensorData(num_envs, device)


class _DummyScene(dict):
    """Scene double: terms index it like a mapping and read the sensors and origins off it."""

    def __init__(self, mapping: dict, num_envs: int, device: str) -> None:
        super().__init__(mapping)
        self.sensors = {"contact_forces": _DummyContactSensor(num_envs, device)}
        self.env_origins = torch.zeros(num_envs, 3, device=device)


class _DummyVisMarkerRegistry:
    """Absorbs the debug-visualization handshake ``CommandTerm.__init__`` performs."""

    def clear_debug_vis_callback(self, term) -> None:
        pass


class _DummySimulation:
    def __init__(self) -> None:
        self.vis_marker_registry = _DummyVisMarkerRegistry()


class _DummyEnv:
    """Minimal environment double for command, observation and action terms."""

    def __init__(self, num_envs: int = 8, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        self.extras: dict = {}
        self.common_step_counter = 0
        self.sim = _DummySimulation()
        self.scene = _DummyScene({"robot": _DummyRobot(num_envs, device)}, num_envs, device)


def _make_pose_command(ranges, resampling_time_range=(2.0, 5.0), num_envs=8) -> mdp.UniformPoseDeltaCommand:
    env = _DummyEnv(num_envs=num_envs)
    cfg = mdp.UniformPoseDeltaCommandCfg(resampling_time_range=resampling_time_range, ranges=ranges)
    return mdp.UniformPoseDeltaCommand(cfg, cast("ManagerBasedRLEnv", env))


def _make_velocity_command(
    num_envs: int = 8,
    rel_standing_envs: float = 0.0,
    rel_forward_envs: float = 0.0,
    rel_turn_in_place_envs: float = 0.0,
) -> mdp.MicroDuckVelocityCommand:
    env = _DummyEnv(num_envs=num_envs)
    cfg = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=rel_standing_envs,
        rel_heading_envs=0.0,
        rel_forward_envs=rel_forward_envs,
        rel_turn_in_place_envs=rel_turn_in_place_envs,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=ANG_VEL_Z_RANGE,
            heading=(-math.pi, math.pi),
        ),
    )
    return mdp.MicroDuckVelocityCommand(cfg, cast("ManagerBasedRLEnv", env))


def _all_env_ids(term) -> torch.Tensor:
    return torch.arange(term.num_envs, device=term.device)


##
# Uniform pose-delta command
##


@pytest.mark.parametrize(
    "ranges,expected_dim",
    [(HEAD_POSE_RANGES, 4), (BODY_POSE_RANGES, 6)],
    ids=["head", "body"],
)
def test_pose_delta_command_width_follows_the_range_tuple(ranges, expected_dim):
    """The command is as wide as the range tuple is long."""
    term = _make_pose_command(ranges)

    assert term.command.shape == (term.num_envs, expected_dim)


@pytest.mark.parametrize(
    "ranges",
    [HEAD_POSE_RANGES, BODY_POSE_RANGES],
    ids=["head", "body"],
)
def test_pose_delta_command_samples_each_dimension_inside_its_own_range(ranges):
    """Every dimension stays inside the range configured for it, not the widest one."""
    torch.manual_seed(0)
    term = _make_pose_command(ranges, num_envs=512)

    term._resample_command(_all_env_ids(term))

    for dim, (low, high) in enumerate(ranges):
        column = term.command[:, dim]
        assert torch.all(column >= low)
        assert torch.all(column <= high)
        # a per-dimension range that is merely inherited from a neighbour would fail this
        assert column.abs().max() > 0.5 * high


def test_pose_delta_command_holds_its_value_until_the_resampling_clock_expires():
    """The command is held between resamples and redrawn once the sampled interval elapses."""
    torch.manual_seed(0)
    term = _make_pose_command(HEAD_POSE_RANGES, resampling_time_range=(2.0, 5.0))
    term.reset(_all_env_ids(term))
    held = term.command.clone()

    assert torch.all(term.time_left >= 2.0)
    assert torch.all(term.time_left <= 5.0)

    # below the shortest sampled interval nothing may change
    for _ in range(19):
        term.compute(dt=0.1)
    torch.testing.assert_close(term.command, held)
    assert torch.all(term.command_counter == 1)

    # past the longest sampled interval every environment must have been redrawn
    for _ in range(31):
        term.compute(dt=0.1)
    assert torch.all(term.command_counter >= 2)
    assert not torch.allclose(term.command, held)


def test_pose_delta_command_accepts_a_widened_range_but_rejects_a_narrowed_one():
    """A curriculum may replace the ranges, but only with one range per commanded dimension."""
    torch.manual_seed(0)
    term = _make_pose_command(HEAD_POSE_RANGES, num_envs=64)

    # widening in place is the curriculum's whole job
    term.cfg.ranges = tuple((10.0 * low, 10.0 * high) for low, high in HEAD_POSE_RANGES)
    term._resample_command(_all_env_ids(term))
    assert term.command[:, 0].abs().max() > HEAD_POSE_RANGES[0][1]

    # dropping a dimension would otherwise leave the last one holding a stale value
    term.cfg.ranges = HEAD_POSE_RANGES[:3]
    with pytest.raises(AssertionError):
        term._resample_command(_all_env_ids(term))


##
# MicroDuck velocity command
##


def test_turn_in_place_envs_zero_the_linear_command_and_force_a_yaw_rate():
    """A turn-in-place environment spins on the spot at no less than 40% of the yaw-rate limit."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_turn_in_place_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    max_rate = max(abs(ANG_VEL_Z_RANGE[0]), abs(ANG_VEL_Z_RANGE[1]))
    torch.testing.assert_close(term.command[:, :2], torch.zeros_like(term.command[:, :2]))
    assert torch.all(term.command[:, 2].abs() >= 0.4 * max_rate)
    assert torch.all(term.command[:, 2].abs() <= max_rate)
    # the sign is drawn, not fixed
    assert torch.any(term.command[:, 2] > 0.0)
    assert torch.any(term.command[:, 2] < 0.0)


def test_standing_envs_hold_a_zero_command():
    """A standing environment is commanded to stand still."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_standing_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    torch.testing.assert_close(term.command, torch.zeros_like(term.command))


def test_forward_envs_walk_forward_at_a_minimum_speed():
    """A forward-only environment gets a positive surge command and no sway or yaw."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_forward_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    assert torch.all(term.command[:, 0] >= 0.3)
    assert torch.all(term.command[:, 0] <= 0.4)
    torch.testing.assert_close(term.command[:, 1:], torch.zeros_like(term.command[:, 1:]))


def test_standing_takes_precedence_over_forward_only():
    """Upstream zeroes standing environments every step, after the forward bucket is applied."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_standing_envs=1.0, rel_forward_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    torch.testing.assert_close(term.command, torch.zeros_like(term.command))


def test_turn_in_place_takes_precedence_over_standing_and_forward_only():
    """Turn-in-place is applied last and clears the standing flag, so it wins every overlap."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512, rel_standing_envs=1.0, rel_forward_envs=1.0, rel_turn_in_place_envs=1.0)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    max_rate = max(abs(ANG_VEL_Z_RANGE[0]), abs(ANG_VEL_Z_RANGE[1]))
    torch.testing.assert_close(term.command[:, :2], torch.zeros_like(term.command[:, :2]))
    assert torch.all(term.command[:, 2].abs() >= 0.4 * max_rate)
    assert not torch.any(term.is_standing_env)


def test_buckets_are_independent_draws_rather_than_a_partition():
    """Each bucket is its own Bernoulli draw, so an environment can land in several at once."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=4096, rel_standing_envs=0.5, rel_forward_envs=0.5)

    term._resample_command(_all_env_ids(term))

    assert torch.any(term.is_standing_env & term.is_forward_env)
    assert torch.any(term.is_standing_env & ~term.is_forward_env)
    assert torch.any(~term.is_standing_env & term.is_forward_env)
    assert torch.any(~term.is_standing_env & ~term.is_forward_env)


def test_disabled_buckets_leave_the_uniform_sample_alone():
    """With every fraction at zero the term reproduces the stock uniform velocity command."""
    torch.manual_seed(0)
    term = _make_velocity_command(num_envs=512)

    term._resample_command(_all_env_ids(term))
    term._update_command()

    assert not torch.any(term.is_forward_env)
    assert not torch.any(term.is_standing_env)
    # a uniform draw over (-0.3, 0.3) practically never lands on the forward-only signature
    assert torch.any(term.command[:, 1] != 0.0)
    assert torch.any(term.command[:, 0] < 0.0)


##
# Encoder bias
##


def _make_obs_env(num_envs: int = 8) -> _DummyEnv:
    return _DummyEnv(num_envs=num_envs)


def _joint_cfg(joint_ids: list[int]) -> SceneEntityCfg:
    """A resolved joint selection: the manager fills ``joint_ids`` in before a term ever runs."""
    return SceneEntityCfg("robot", joint_names=[JOINT_NAMES[i] for i in joint_ids], joint_ids=joint_ids)


def test_encoder_bias_defaults_to_zero_until_the_randomization_runs():
    """A task that never wires the startup event still gets the unbiased joint positions."""
    env = _make_obs_env()
    asset_cfg = _joint_cfg([0, 1, 2, 3])
    env.scene["robot"].data.joint_pos.torch[:] = 0.3

    biased = mdp.joint_pos_rel_biased(cast("ManagerBasedEnv", env), asset_cfg, biased=True)

    torch.testing.assert_close(biased, torch.full_like(biased, 0.3))


def test_encoder_bias_is_a_constant_per_environment_offset_inside_its_range():
    """The bias is sampled once and read back identically on every later call."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=512)
    asset_cfg = _joint_cfg([0, 1, 2, 3])
    mdp.randomize_encoder_bias(cast("ManagerBasedEnv", env), None, bias_range=ENCODER_BIAS_RANGE)

    first = mdp.joint_pos_rel_biased(cast("ManagerBasedEnv", env), asset_cfg, biased=True)
    second = mdp.joint_pos_rel_biased(cast("ManagerBasedEnv", env), asset_cfg, biased=True)

    torch.testing.assert_close(first, second)
    assert torch.all(first >= ENCODER_BIAS_RANGE[0])
    assert torch.all(first <= ENCODER_BIAS_RANGE[1])
    # a per-environment draw, not one shared offset
    assert first[:, 0].std() > 0.0
    # ... and a per-joint draw, not one offset repeated across the joint block
    assert torch.any(first[:, 0] != first[:, 1])


def test_the_critic_reads_the_true_joint_positions():
    """``biased=False`` reproduces the stock ``joint_pos_rel``, which is what the critic sees."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=64)
    asset_cfg = _joint_cfg([0, 1, 2, 3])
    data = env.scene["robot"].data
    data.joint_pos.torch[:] = torch.rand_like(data.joint_pos.torch)
    data.default_joint_pos.torch[:] = 0.1
    mdp.randomize_encoder_bias(cast("ManagerBasedEnv", env), None, bias_range=ENCODER_BIAS_RANGE)

    truth = mdp.joint_pos_rel_biased(cast("ManagerBasedEnv", env), asset_cfg, biased=False)
    biased = mdp.joint_pos_rel_biased(cast("ManagerBasedEnv", env), asset_cfg, biased=True)

    expected = data.joint_pos.torch - data.default_joint_pos.torch
    torch.testing.assert_close(truth, expected)
    torch.testing.assert_close(biased - truth, mdp.encoder_bias(cast("ManagerBasedEnv", env)))


def test_the_bias_follows_the_requested_joint_order():
    """A reordered joint selection reorders the bias with it, so the blocks stay aligned."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=16)
    mdp.randomize_encoder_bias(cast("ManagerBasedEnv", env), None, bias_range=ENCODER_BIAS_RANGE)
    bias = mdp.encoder_bias(cast("ManagerBasedEnv", env))

    reordered = mdp.joint_pos_rel_biased(cast("ManagerBasedEnv", env), _joint_cfg([3, 0]), biased=True)

    torch.testing.assert_close(reordered, bias[:, [3, 0]])


def test_the_biased_action_subtracts_the_bias_the_observation_adds():
    """The two halves of the encoder-bias loop cancel, so the commanded joint lands on target."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=16)
    robot = env.scene["robot"]
    robot.data.default_joint_pos.torch[:] = 0.2
    mdp.randomize_encoder_bias(cast("ManagerBasedEnv", env), None, bias_range=ENCODER_BIAS_RANGE)
    cfg = mdp.BiasedJointPositionActionCfg(asset_name="robot", joint_names=JOINT_NAMES, scale=1.0)
    term = mdp.BiasedJointPositionAction(cfg, cast("ManagerBasedEnv", env))

    actions = torch.rand(env.num_envs, len(JOINT_NAMES), device=env.device)
    term.process_actions(actions)
    term.apply_actions()

    bias = mdp.encoder_bias(cast("ManagerBasedEnv", env))
    torch.testing.assert_close(robot.joint_position_target, 0.2 + actions - bias)
    # the policy commanding ``a`` reads back ``a`` once the joint reaches its target
    robot.data.joint_pos.torch[:] = robot.joint_position_target
    read_back = mdp.joint_pos_rel_biased(cast("ManagerBasedEnv", env), _joint_cfg([0, 1, 2, 3]), biased=True)
    torch.testing.assert_close(read_back, actions)


##
# IMU misalignment
##


def _misalignment_matrix(env: _DummyEnv, max_angle_deg: float) -> torch.Tensor:
    """The rotation the misaligned terms apply, read one basis column at a time."""
    columns = []
    for axis in range(3):
        env.scene["robot"].data.root_ang_vel_b.torch[:] = 0.0
        env.scene["robot"].data.root_ang_vel_b.torch[:, axis] = 1.0
        columns.append(mdp.base_ang_vel_imu_misaligned(cast("ManagerBasedEnv", env), max_angle_deg))
    return torch.stack(columns, dim=-1)


def test_imu_misalignment_stays_within_the_configured_angle():
    """The rotation magnitude is uniform on ``[0, max_angle_deg]`` about a random axis."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=4096)

    rotation = _misalignment_matrix(env, IMU_MISALIGNMENT_ANGLE_DEG)

    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    angle = torch.acos(torch.clamp(0.5 * (trace - 1.0), min=-1.0, max=1.0))
    bound = math.radians(IMU_MISALIGNMENT_ANGLE_DEG)
    assert torch.all(angle <= bound + 1e-5)
    # the whole range is used, and uniformly: a wrongly scaled angle fails both
    assert angle.max() > 0.95 * bound
    assert abs(float(angle.mean()) - 0.5 * bound) < 0.05 * bound


def test_imu_misalignment_is_zero_centred_rather_than_a_fixed_tilt():
    """A random axis makes the misalignment a magnitude, not a systematic pitch offset."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=4096)

    gravity = mdp.projected_gravity_imu_misaligned(cast("ManagerBasedEnv", env), IMU_MISALIGNMENT_ANGLE_DEG)

    truth = env.scene["robot"].data.projected_gravity_b.torch
    # every environment is tilted ...
    assert torch.all((gravity - truth).norm(dim=-1) > 0.0)
    # ... but the fleet is not, to well inside the per-environment tilt
    assert torch.all((gravity.mean(dim=0) - truth[0]).abs() < 0.01)


def test_both_imu_terms_share_one_misalignment_that_never_changes():
    """One mounting error per robot: the same rotation for gravity and rate, for the whole run."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=256)
    data = env.scene["robot"].data
    data.root_ang_vel_b.torch[:] = data.projected_gravity_b.torch

    gravity = mdp.projected_gravity_imu_misaligned(cast("ManagerBasedEnv", env), IMU_MISALIGNMENT_ANGLE_DEG)
    ang_vel = mdp.base_ang_vel_imu_misaligned(cast("ManagerBasedEnv", env), IMU_MISALIGNMENT_ANGLE_DEG)

    torch.testing.assert_close(gravity, ang_vel)
    # a later step must not redraw it
    torch.testing.assert_close(
        gravity, mdp.projected_gravity_imu_misaligned(cast("ManagerBasedEnv", env), IMU_MISALIGNMENT_ANGLE_DEG)
    )


def test_imu_misalignment_is_the_identity_when_disabled():
    """``max_angle_deg = 0`` turns both terms back into their stock counterparts."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=64)
    data = env.scene["robot"].data
    data.root_ang_vel_b.torch[:] = torch.rand_like(data.root_ang_vel_b.torch)

    gravity = mdp.projected_gravity_imu_misaligned(cast("ManagerBasedEnv", env), 0.0)
    ang_vel = mdp.base_ang_vel_imu_misaligned(cast("ManagerBasedEnv", env), 0.0)

    torch.testing.assert_close(gravity, data.projected_gravity_b.torch)
    torch.testing.assert_close(ang_vel, data.root_ang_vel_b.torch)


##
# Observation delay
##


def _step_index(env: ManagerBasedEnv) -> torch.Tensor:
    """A term whose value is the control step it was computed on, so a lag is readable off it."""
    return torch.full((env.num_envs, 1), float(env.common_step_counter), device=env.device)


def _make_delayed_term(env: _DummyEnv, min_lag: int, max_lag: int, update_period: int, hold_prob: float = 0.0):
    params = {
        "term_func": _step_index,
        "min_lag": min_lag,
        "max_lag": max_lag,
        "update_period": update_period,
        "hold_prob": hold_prob,
    }
    cfg = ObservationTermCfg(func=mdp.delayed_observation, params=params)
    term = mdp.delayed_observation(cfg, cast("ManagerBasedEnv", env))
    return term, params


def _advance(env: _DummyEnv, term, params: dict) -> torch.Tensor:
    """One control step: bump the step counter the way the environment does, then compute."""
    env.common_step_counter += 1
    return term(cast("ManagerBasedEnv", env), **params)


def test_a_fixed_one_step_lag_returns_the_previous_control_step_value():
    """MicroDuck's ``joint_vel`` delay: a constant one-step lag once the buffer has filled."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=8)
    term, params = _make_delayed_term(env, min_lag=1, max_lag=1, update_period=0)

    # the first step has nothing older to return, so it returns itself (reference section 8, rule 6)
    torch.testing.assert_close(_advance(env, term, params), torch.ones(env.num_envs, 1))
    for step in range(2, 8):
        torch.testing.assert_close(_advance(env, term, params), torch.full((env.num_envs, 1), float(step - 1)))


def test_the_delay_advances_once_per_control_step_however_often_it_is_computed():
    """Recomputing an observation inside one step must not push the buffer twice, or recompute it."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=8)
    term, params = _make_delayed_term(env, min_lag=1, max_lag=1, update_period=0)
    calls = 0

    def _counted(env):
        nonlocal calls
        calls += 1
        return _step_index(env)

    params["term_func"] = _counted

    _advance(env, term, params)
    _advance(env, term, params)
    repeated = term(cast("ManagerBasedEnv", env), **params)

    torch.testing.assert_close(repeated, torch.ones(env.num_envs, 1))
    assert calls == 2
    torch.testing.assert_close(_advance(env, term, params), torch.full((env.num_envs, 1), 2.0))


def test_the_lag_is_redrawn_only_on_the_update_period_boundary():
    """MicroDuck's IMU delay: a 0-or-1 step lag re-drawn every 64 steps, staggered per env."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=256)
    update_period = 64
    term, params = _make_delayed_term(env, min_lag=0, max_lag=1, update_period=update_period)

    # one warm-up step: with a single frame buffered the lag is clamped to 0 whatever was drawn
    _advance(env, term, params)
    lags = []
    for _ in range(4 * update_period):
        value = _advance(env, term, params)
        lags.append(env.common_step_counter - value[:, 0])
    lags = torch.stack(lags)  # (steps, num_envs)

    assert torch.all((lags == 0) | (lags == 1))
    # every environment holds its lag for whole periods, and they do not all switch together
    change_steps = [torch.nonzero(lags[1:, i] != lags[:-1, i]).flatten() for i in range(env.num_envs)]
    for steps in change_steps:
        assert torch.all(torch.diff(steps) % update_period == 0)
    assert len({int(steps[0]) for steps in change_steps if len(steps) > 0}) > 1
    # both lags are actually drawn
    assert torch.any(lags == 0) and torch.any(lags == 1)


def test_a_reset_environment_gets_the_freshest_frame():
    """A reset clears only the reset environments' history and lag (reference section 8)."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=4)
    term, params = _make_delayed_term(env, min_lag=1, max_lag=1, update_period=0)
    for _ in range(4):
        _advance(env, term, params)

    term.reset(torch.tensor([0, 2], device=env.device))
    value = _advance(env, term, params)

    step = float(env.common_step_counter)
    torch.testing.assert_close(value[:, 0], torch.tensor([step, step - 1.0, step, step - 1.0]))


def test_a_reset_between_two_computes_of_one_step_does_not_cost_the_others_a_frame():
    """The compute-final-observation pattern: compute, reset part of the batch, compute again.

    The second compute must not advance the buffer a second time, or every environment that was
    not reset silently skips a frame -- and its lag-update cadence drifts with it.
    """
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=4)
    term, params = _make_delayed_term(env, min_lag=1, max_lag=1, update_period=0)
    for _ in range(4):
        _advance(env, term, params)
    step = float(env.common_step_counter)

    # the environment records a final observation, resets the terminating environments, and
    # recomputes the group -- all within the same control step
    term.reset(torch.tensor([0, 2], device=env.device))
    recomputed = term(cast("ManagerBasedEnv", env), **params)

    # the reset environments have no history left, the others keep the frame they were served
    torch.testing.assert_close(recomputed[:, 0], torch.tensor([step, step - 1.0, step, step - 1.0]))
    # ... and the frame from this step is still there to be served on the next one
    value = _advance(env, term, params)
    torch.testing.assert_close(value[:, 0], torch.tensor([step + 1.0, step, step + 1.0, step]))


def test_a_reset_redraws_the_lag_update_phase():
    """The phase is re-drawn per environment on reset, so resets do not synchronize the fleet."""
    torch.manual_seed(0)
    env = _make_obs_env(num_envs=256)
    update_period = 2
    term, params = _make_delayed_term(env, min_lag=0, max_lag=1, update_period=update_period)

    def _phase_of_lag_changes(steps: int) -> torch.Tensor:
        """Recover each environment's update phase from the steps its lag changes on."""
        # the first two steps are the buffer warm-up, where the lag is clamped to the frames held
        lags = []
        for _ in range(steps + 2):
            value = _advance(env, term, params)
            lags.append(env.common_step_counter - value[:, 0])
        lags = torch.stack(lags[2:])
        changed = lags[1:] != lags[:-1]
        # a change can only happen on an update step, so any change step reveals the phase
        step_index = torch.arange(1, len(lags), device=env.device).unsqueeze(-1) % update_period
        return torch.where(changed, step_index, -torch.ones_like(step_index)).max(dim=0).values

    before = _phase_of_lag_changes(64)
    term.reset()
    after = _phase_of_lag_changes(64)

    # every environment must have been observed changing lag at least once in both runs
    assert torch.all(before >= 0) and torch.all(after >= 0)
    # a re-drawn phase lands on a different one for half the environments; a kept one for none
    assert (before != after).float().mean() > 0.2


def test_a_delayed_term_rejects_a_lag_window_it_cannot_serve():
    """A configuration whose lags fall outside the buffer is a configuration error."""
    env = _make_obs_env()

    with pytest.raises(ValueError):
        _make_delayed_term(env, min_lag=2, max_lag=1, update_period=0)
    with pytest.raises(ValueError):
        _make_delayed_term(env, min_lag=0, max_lag=0, update_period=0)
    with pytest.raises(ValueError):
        _make_delayed_term(env, min_lag=0, max_lag=1, update_period=0, hold_prob=1.5)


##
# NaN-safe critic terms
##


def _sensor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("contact_forces", body_names=BODY_NAMES[1:], body_ids=[1, 2])


def test_the_safe_foot_terms_report_zero_where_the_sensor_reports_a_non_finite_value():
    """A single non-finite sensor read must not poison a whole training run."""
    env = _make_obs_env(num_envs=3)
    sensor_cfg = _sensor_cfg()
    sensor = env.scene.sensors["contact_forces"]
    forces = sensor.data.net_forces_w.torch
    forces[:, 1:] = torch.tensor([0.0, 0.0, 2.0])
    forces[0, 1, 0] = float("nan")
    forces[1, 2, 1] = float("inf")
    air_time = sensor.data.current_air_time.torch
    air_time[:] = 0.25
    air_time[2, 1] = float("nan")

    contact_forces = mdp.foot_contact_forces_safe(cast("ManagerBasedEnv", env), sensor_cfg)
    air = mdp.foot_air_time_safe(cast("ManagerBasedEnv", env), sensor_cfg)

    assert torch.all(torch.isfinite(contact_forces))
    assert torch.all(torch.isfinite(air))
    assert contact_forces[0, 0] == 0.0 and contact_forces[1, 4] == 0.0
    assert air[2, 0] == 0.0
    # the finite entries keep upstream's signed log compression
    torch.testing.assert_close(contact_forces[2, 2], torch.tensor(math.log1p(2.0)))
    torch.testing.assert_close(air[0], torch.full((2,), 0.25))


def test_foot_contact_flags_only_the_loaded_feet():
    """The contact flag follows the net force, and reports the bodies in the requested order."""
    env = _make_obs_env(num_envs=2)
    forces = env.scene.sensors["contact_forces"].data.net_forces_w.torch
    forces[0, 1] = torch.tensor([0.0, 0.0, 3.0])
    forces[1, 2] = torch.tensor([0.0, 0.0, -3.0])

    contact = mdp.foot_contact(cast("ManagerBasedEnv", env), _sensor_cfg())

    torch.testing.assert_close(contact, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_the_safe_foot_height_is_the_sole_clearance_above_the_environment_origin():
    """Foot height is measured from the terrain the environment sits on, not the world origin."""
    env = _make_obs_env(num_envs=2)
    asset_cfg = SceneEntityCfg("robot", body_names=BODY_NAMES[1:], body_ids=[1, 2])
    body_pos = env.scene["robot"].data.body_pos_w.torch
    body_pos[:, :, 2] = 0.4
    env.scene.env_origins[:, 2] = 0.1
    body_pos[1, 2, 2] = float("nan")

    height = mdp.foot_height_safe(cast("ManagerBasedEnv", env), asset_cfg)

    torch.testing.assert_close(height, torch.tensor([[0.3, 0.3], [0.3, 0.0]]))


##
# Staged curricula
##


ACTION_RATE_WEIGHT_STAGES = [
    {"step": 0 * 24, "weight": -0.1},
    {"step": 500 * 24, "weight": -0.2},
    {"step": 750 * 24, "weight": -0.4},
    {"step": 1000 * 24, "weight": -0.6},
    {"step": 1250 * 24, "weight": -0.8},
    {"step": 1500 * 24, "weight": -1.0},
]
"""Upstream's ``action_rate_weight`` ramp (reference section 6, curriculum stage tables)."""

STANDING_ENVS_STAGES = [
    {"step": 0 * 24, "rel_standing_envs": 0.02},
    {"step": 500 * 24, "rel_standing_envs": 0.05},
    {"step": 750 * 24, "rel_standing_envs": 0.1},
    {"step": 1000 * 24, "rel_standing_envs": 0.15},
    {"step": 1500 * 24, "rel_standing_envs": 0.2},
    {"step": 2000 * 24, "rel_standing_envs": 0.25},
]
"""Upstream's ``standing_envs`` ramp (reference section 6, curriculum stage tables)."""


class _DummyRewardManager:
    """Reward-manager double exposing the two accessors the weight curriculum uses."""

    def __init__(self, term_cfgs: dict) -> None:
        self._term_cfgs = term_cfgs

    def get_term_cfg(self, term_name: str):
        return self._term_cfgs[term_name]

    def set_term_cfg(self, term_name: str, cfg) -> None:
        self._term_cfgs[term_name] = cfg


class _DummyCommandManager:
    def __init__(self, terms: dict) -> None:
        self._terms = terms

    def get_term(self, name: str):
        return self._terms[name]


class _DummyEventManager:
    def __init__(self, term_cfgs: dict) -> None:
        self._term_cfgs = term_cfgs

    def get_term_cfg(self, term_name: str):
        return self._term_cfgs[term_name]


class _DummyTerminationManager:
    """Termination-manager double exposing the accessor the termination curriculum uses."""

    def __init__(self, term_cfgs: dict) -> None:
        self._term_cfgs = term_cfgs

    def get_term_cfg(self, term_name: str):
        return self._term_cfgs[term_name]


class _DummyTermCfg:
    """Stands in for a term configuration the curricula mutate in place."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def _make_curriculum_env() -> _DummyEnv:
    """An environment double carrying the reward, command and event terms the curricula mutate."""
    env = _DummyEnv(num_envs=4)
    env.reward_manager = _DummyRewardManager(
        {
            "action_rate_l2": _DummyTermCfg(weight=-0.1),
            "head_pose_bias": _DummyTermCfg(weight=0.0),
        }
    )
    env.command_manager = _DummyCommandManager(
        {
            "base_velocity": _DummyTermCfg(cfg=_DummyTermCfg(rel_standing_envs=0.02)),
            "head_pose": _DummyTermCfg(cfg=_DummyTermCfg(ranges=HEAD_POSE_RANGES)),
        }
    )
    env.event_manager = _DummyEventManager(
        {
            "randomize_com": _DummyTermCfg(params={"com_range": {"x": (-0.003, 0.003)}}),
        }
    )
    env.termination_manager = _DummyTerminationManager(
        {
            "fell_over": _DummyTermCfg(params={"limit_angle": math.radians(70.0)}),
        }
    )
    return env


def _apply(env: _DummyEnv, func, **params):
    return func(cast("ManagerBasedRLEnv", env), slice(None), **params)


@pytest.mark.parametrize(
    "step, expected",
    [
        (0, -0.1),
        (500 * 24, -0.1),
        (500 * 24 + 1, -0.2),
        (1000 * 24 + 1, -0.6),
        (1500 * 24 + 1, -1.0),
        (10_000 * 24, -1.0),
    ],
)
def test_the_reward_weight_ramp_steps_on_a_strictly_greater_step_boundary(step, expected):
    """The weight is the payload of the last stage the environment has strictly passed."""
    env = _make_curriculum_env()
    env.common_step_counter = step

    value = _apply(env, mdp.reward_weight_stages, reward_name="action_rate_l2", weight_stages=ACTION_RATE_WEIGHT_STAGES)

    assert value == pytest.approx(expected)
    assert env.reward_manager.get_term_cfg("action_rate_l2").weight == pytest.approx(expected)


def test_the_reward_weight_ramp_reaches_every_stage_it_lists():
    """Every listed weight is served, so no stage is unreachable."""
    env = _make_curriculum_env()
    served = []
    for stage in ACTION_RATE_WEIGHT_STAGES:
        env.common_step_counter = stage["step"] + 1
        served.append(
            _apply(env, mdp.reward_weight_stages, reward_name="action_rate_l2", weight_stages=ACTION_RATE_WEIGHT_STAGES)
        )

    assert served == pytest.approx([stage["weight"] for stage in ACTION_RATE_WEIGHT_STAGES])


@pytest.mark.parametrize(
    "step, expected",
    [(0, 0.02), (500 * 24, 0.02), (500 * 24 + 1, 0.05), (2000 * 24 + 1, 0.25)],
)
def test_the_standing_fraction_ramp_mutates_the_live_command_configuration(step, expected):
    """The curriculum writes the live command term's configuration, not the environment's copy."""
    env = _make_curriculum_env()
    env.common_step_counter = step

    value = _apply(env, mdp.standing_envs_stages, command_name="base_velocity", standing_stages=STANDING_ENVS_STAGES)

    assert value == pytest.approx(expected)
    assert env.command_manager.get_term("base_velocity").cfg.rel_standing_envs == pytest.approx(expected)


def test_the_command_range_ramp_steps_on_an_inclusive_step_boundary():
    """Upstream's pose-range curriculum triggers at the stage step, unlike the weight ramps."""
    stages = [
        {"step": 0, "ranges": HEAD_POSE_RANGES},
        {"step": 500 * 24, "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
    ]
    env = _make_curriculum_env()
    env.common_step_counter = 500 * 24

    value = _apply(env, mdp.command_range_stages, command_name="head_pose", range_stages=stages)

    assert env.command_manager.get_term("head_pose").cfg.ranges == stages[1]["ranges"]
    # the logged value is the widest half-range currently commanded
    assert value == pytest.approx(1.40)


def test_the_command_range_ramp_restates_the_first_stage_before_any_boundary():
    """A fresh run is pinned to the first stage rather than to whatever the configuration held."""
    stages = [{"step": 100, "ranges": HEAD_POSE_RANGES}]
    env = _make_curriculum_env()
    env.command_manager.get_term("head_pose").cfg.ranges = ((-9.0, 9.0),) * 4
    env.common_step_counter = 0

    _apply(env, mdp.command_range_stages, command_name="head_pose", range_stages=stages)

    assert env.command_manager.get_term("head_pose").cfg.ranges == HEAD_POSE_RANGES


@pytest.mark.parametrize(
    "step, expected",
    [(0, 0.003), (500 * 24, 0.003), (500 * 24 + 1, 0.005), (1500 * 24 + 1, 0.015)],
)
def test_the_event_range_ramp_widens_a_symmetric_range_on_every_axis(step, expected):
    """The centre-of-mass ramp rewrites the event's own parameters, symmetrically about zero."""
    stages = [
        {"step": 0 * 24, "range": 0.003},
        {"step": 500 * 24, "range": 0.005},
        {"step": 1000 * 24, "range": 0.01},
        {"step": 1500 * 24, "range": 0.015},
    ]
    env = _make_curriculum_env()
    env.common_step_counter = step

    value = _apply(env, mdp.event_range_stages, event_name="randomize_com", range_stages=stages)

    assert value == pytest.approx(expected)
    assert env.event_manager.get_term_cfg("randomize_com").params["com_range"] == {
        "x": (-expected, expected),
        "y": (-expected, expected),
        "z": (-expected, expected),
    }


@pytest.mark.parametrize(
    "stages",
    [
        [],
        [{"step": 500, "weight": -0.2}, {"step": 0, "weight": -0.1}],
        [{"step": 0, "weight": -0.1}, {"step": 0, "weight": -0.2}],
        [{"step": 0}],
    ],
)
def test_a_malformed_stage_table_is_rejected_rather_than_silently_misapplied(stages):
    """An empty, unsorted, duplicated or incomplete stage table cannot be applied by accident."""
    env = _make_curriculum_env()

    with pytest.raises(ValueError):
        _apply(env, mdp.reward_weight_stages, reward_name="action_rate_l2", weight_stages=stages)


##
# Ball state, for the ball-kick critic (addendum section 6.4)
##


class _DummyBallData:
    """The free-body state the two privileged ball observations read."""

    def __init__(self, position: torch.Tensor, velocity: torch.Tensor) -> None:
        self.root_link_pos_w = _DummyTensorView(position)
        self.root_link_lin_vel_w = _DummyTensorView(velocity)


class _DummyBall:
    def __init__(self, position: torch.Tensor, velocity: torch.Tensor) -> None:
        self.data = _DummyBallData(position, velocity)


def _ball_obs_env(
    ball_pos: list[list[float]],
    ball_vel: list[list[float]],
    robot_pos: list[list[float]] | None = None,
    yaw: float = 0.0,
) -> _DummyEnv:
    """An environment double holding a ball and a robot yawed by ``yaw`` [rad]."""
    env = _DummyEnv(num_envs=len(ball_pos))
    env.scene["ball"] = _DummyBall(torch.tensor(ball_pos), torch.tensor(ball_vel))
    if robot_pos is not None:
        env.scene["robot"].data.root_link_pos_w.torch[:] = torch.tensor(robot_pos)
    half = yaw * 0.5
    # (x, y, z, w) yaw rotation
    env.scene["robot"].data.root_link_quat_w.torch[:] = torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)])
    return env


def test_ball_pos_in_base_is_the_relative_position_rotated_into_the_base_frame():
    """A ball straight ahead of a robot yawed 90 degrees sits on the base frame's ``-y`` axis.

    Both halves matter and a term could get either one wrong on its own: subtracting the robot
    position without rotating gives ``(1, 0, 0)``, and rotating without subtracting gives
    ``(2, 1, 0)`` rotated.
    """
    env = _ball_obs_env(
        ball_pos=[[3.0, 1.0, 0.035]], ball_vel=[[0.0, 0.0, 0.0]], robot_pos=[[2.0, 1.0, 0.115]], yaw=math.pi / 2.0
    )

    position = mdp.ball_pos_in_base(cast("ManagerBasedEnv", env), asset_cfg=SceneEntityCfg("ball"))

    torch.testing.assert_close(position, torch.tensor([[0.0, -1.0, -0.08]]), atol=1e-6, rtol=0.0)


def test_ball_vel_in_base_rotates_the_world_velocity_without_subtracting_the_robots_own():
    """Upstream reads the ball's *world* velocity, so a robot walking at a still ball reads zero.

    That is reproduced rather than corrected: the critic is trained against upstream's signal, and a
    closing velocity would be a different observation with the same name.
    """
    env = _ball_obs_env(ball_pos=[[1.0, 0.0, 0.035]], ball_vel=[[0.6, 0.0, 0.0]], yaw=math.pi / 2.0)

    velocity = mdp.ball_vel_in_base(cast("ManagerBasedEnv", env), asset_cfg=SceneEntityCfg("ball"))

    torch.testing.assert_close(velocity, torch.tensor([[0.0, -0.6, 0.0]]), atol=1e-6, rtol=0.0)


def test_the_ball_observations_survive_a_ball_the_solver_has_lost():
    """Nothing in the ball-kick task NaN-checks the ball, so these two terms guard themselves."""
    env = _ball_obs_env(ball_pos=[[float("nan"), 0.0, 0.0]], ball_vel=[[0.0, float("inf"), 0.0]])

    position = mdp.ball_pos_in_base(cast("ManagerBasedEnv", env), asset_cfg=SceneEntityCfg("ball"))
    velocity = mdp.ball_vel_in_base(cast("ManagerBasedEnv", env), asset_cfg=SceneEntityCfg("ball"))

    assert torch.isfinite(position).all()
    assert torch.isfinite(velocity).all()


##
# Ground-state reset: bucket parameters
##


@pytest.mark.parametrize(
    "probabilities, missing",
    [
        ({"face_down_prob": 1.0}, "prone_z_range"),
        ({"face_up_prob": 1.0}, "prone_z_range"),
        ({"sitting_prob": 1.0}, "sitting_z_range/sitting_joint_pos"),
        ({"standing_prob": 1.0}, "standing_z_range"),
    ],
)
def test_the_ground_state_reset_refuses_a_live_bucket_it_cannot_spawn(probabilities, missing):
    """A bucket that can be drawn must have the band and keyframe it spawns from.

    The parameters are optional so that a standing-only task -- the ball kick -- does not have to
    invent values for three buckets it never draws. That only stays safe while a *live* bucket
    without them is an error rather than a silent fallback to some default band.
    """
    env = _DummyEnv(num_envs=2)
    buckets = {
        "face_down_prob": 0.0,
        "face_up_prob": 0.0,
        "sitting_prob": 0.0,
        "standing_prob": 0.0,
        **probabilities,
    }

    with pytest.raises(ValueError, match=missing.replace("/", ".")):
        mdp.reset_ground_state(cast("ManagerBasedEnv", env), env_ids=None, **buckets)


FELL_OVER_STAGES = [
    {"step": 0, "params": {"limit_angle": math.radians(70.0)}},
    {"step": 500 * 24, "params": {"limit_angle": math.pi}},
]
"""Upstream's ``fell_over_disable`` schedule (addendum section 3.6): the tilt termination is widened
to half a turn once the walk is established, so a fall becomes recovery training instead of an
episode end."""


@pytest.mark.parametrize(
    "step, expected",
    [
        (0, math.radians(70.0)),
        (500 * 24 - 1, math.radians(70.0)),
        (500 * 24, math.pi),
        (10_000 * 24, math.pi),
    ],
)
def test_the_termination_parameter_ramp_steps_on_an_inclusive_step_boundary(step, expected):
    """Upstream compares the step counter inclusively here, unlike its reward-weight ramps."""
    env = _make_curriculum_env()
    env.common_step_counter = step

    index = _apply(env, mdp.termination_param_stages, term_name="fell_over", param_stages=FELL_OVER_STAGES)

    assert env.termination_manager.get_term_cfg("fell_over").params["limit_angle"] == pytest.approx(expected)
    assert index == pytest.approx(0.0 if expected < math.pi else 1.0)


def test_the_termination_parameter_ramp_merges_shallowly_rather_than_replacing_the_parameters():
    """A stage lists only what it changes; anything the term also carries has to survive it."""
    env = _make_curriculum_env()
    env.termination_manager.get_term_cfg("fell_over").params["asset_cfg"] = "robot"
    env.common_step_counter = 500 * 24

    _apply(env, mdp.termination_param_stages, term_name="fell_over", param_stages=FELL_OVER_STAGES)

    params = env.termination_manager.get_term_cfg("fell_over").params
    assert params == {"limit_angle": pytest.approx(math.pi), "asset_cfg": "robot"}


##
# Ground-state reset: the crouching bucket
##


CROUCH_ANCHOR = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}
"""Upstream's deep-crouch anchor (addendum section 3.4, ``_CROUCH_ANCHOR_BY_NAME``)."""


def test_the_ground_state_reset_refuses_a_live_crouch_bucket_it_cannot_spawn():
    """The crouching bucket needs both the height band it interpolates and the pose it folds toward."""
    env = _DummyEnv(num_envs=2)
    buckets = {
        "face_down_prob": 0.0,
        "face_up_prob": 0.0,
        "sitting_prob": 0.0,
        "standing_prob": 0.0,
        "crouch_prob": 1.0,
    }

    with pytest.raises(ValueError, match="crouch_z_range.crouch_joint_pos"):
        mdp.reset_ground_state(cast("ManagerBasedEnv", env), env_ids=None, **buckets)


##
# Rolling entry, for the roller-slope task (addendum section 4.3)
##


WHEEL_RADIUS = 0.0175
"""Rolling radius [m] upstream's roller terms default to (addendum section 9.3: the model measures
0.0150, and the stale value is ported verbatim for parity)."""


class _RollingEntryRobot:
    """Articulation double that records the two writes the rolling entry makes."""

    def __init__(self, num_envs: int, device: str) -> None:
        self.data = _DummyRobotData(num_envs, device)
        self.root_velocity_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.joint_velocity_writes: list[tuple[torch.Tensor, list[int], torch.Tensor]] = []

    def write_root_link_velocity_to_sim_index(self, *, root_velocity, env_ids=None, skip_forward=False) -> None:
        del skip_forward
        self.root_velocity_writes.append((root_velocity.clone(), env_ids))

    def write_joint_velocity_to_sim_index(self, *, velocity, joint_ids=None, env_ids=None, skip_forward=False) -> None:
        del skip_forward
        self.joint_velocity_writes.append((velocity.clone(), joint_ids, env_ids))


def _rolling_entry_env(num_envs: int = 6) -> _DummyEnv:
    env = _DummyEnv(num_envs=num_envs)
    env.scene["robot"] = _RollingEntryRobot(num_envs, env.device)
    return env


def _wheel_entity(joint_ids: list[int]) -> SceneEntityCfg:
    cfg = SceneEntityCfg("robot")
    cfg.joint_ids = joint_ids
    cfg.joint_names = [f"passive_{name}_wheel" for name in ("LF", "LR", "RF", "RR")]
    return cfg


def test_the_rolling_entry_pushes_the_base_forward_only():
    """A world-frame ``+x`` shove and nothing else: the ramp descends along ``+x``."""
    env = _rolling_entry_env()
    robot = env.scene["robot"]

    mdp.reset_rolling_entry(
        cast("ManagerBasedEnv", env), env_ids=None, asset_cfg=_wheel_entity([1, 3, 5, 7]), speed_range=(0.25, 0.45)
    )

    velocity, env_ids = robot.root_velocity_writes[0]
    assert velocity.shape == (env.num_envs, 6)
    assert torch.all((velocity[:, 0] >= 0.25) & (velocity[:, 0] <= 0.45))
    # every other component -- the lateral and vertical push and the whole angular velocity -- is
    # written as zero rather than left alone, so an inherited base velocity cannot survive the reset
    torch.testing.assert_close(velocity[:, 1:], torch.zeros(env.num_envs, 5))
    torch.testing.assert_close(env_ids, torch.arange(env.num_envs))


def test_the_rolling_entry_spins_every_wheel_to_match_its_own_environments_base_speed():
    """``omega * r == v`` per environment: the entry rolls rather than skids.

    Upstream added this after a base-only shove -- moving base, motionless wheels -- produced a
    contact spike on the first step that diverged into NaN.
    """
    env = _rolling_entry_env()
    robot = env.scene["robot"]

    mdp.reset_rolling_entry(
        cast("ManagerBasedEnv", env),
        env_ids=None,
        asset_cfg=_wheel_entity([1, 3, 5, 7]),
        speed_range=(0.25, 0.45),
        wheel_radius=WHEEL_RADIUS,
    )

    speed = robot.root_velocity_writes[0][0][:, 0]
    wheel_velocity, joint_ids, _ = robot.joint_velocity_writes[0]
    assert wheel_velocity.shape == (env.num_envs, 4)
    assert joint_ids == [1, 3, 5, 7]
    torch.testing.assert_close(wheel_velocity * WHEEL_RADIUS, speed.unsqueeze(1).expand(-1, 4).contiguous())


def test_the_rolling_entry_draws_one_speed_per_environment():
    """One draw per environment, not one shared draw: the entry speed is a randomization."""
    env = _rolling_entry_env(num_envs=64)

    mdp.reset_rolling_entry(
        cast("ManagerBasedEnv", env), env_ids=None, asset_cfg=_wheel_entity([1, 3, 5, 7]), speed_range=(0.25, 0.45)
    )

    speed = env.scene["robot"].root_velocity_writes[0][0][:, 0]
    assert torch.unique(speed).numel() > 1


def test_the_rolling_entry_touches_only_the_environments_it_was_given():
    """Reset events are handed the restarting environments, and the rest are mid-episode."""
    env = _rolling_entry_env()
    env_ids = torch.tensor([1, 4])

    mdp.reset_rolling_entry(
        cast("ManagerBasedEnv", env), env_ids=env_ids, asset_cfg=_wheel_entity([1, 3, 5, 7]), speed_range=(0.3, 0.3)
    )

    velocity, written_ids = env.scene["robot"].root_velocity_writes[0]
    assert velocity.shape == (2, 6)
    torch.testing.assert_close(written_ids, env_ids)
    torch.testing.assert_close(env.scene["robot"].joint_velocity_writes[0][0].shape[0], 2)


def test_pinning_the_entry_speed_range_makes_the_entry_deterministic():
    """What the accuracy-gate harness relies on to compare a rollout against upstream's."""
    env = _rolling_entry_env()

    mdp.reset_rolling_entry(
        cast("ManagerBasedEnv", env),
        env_ids=None,
        asset_cfg=_wheel_entity([1, 3, 5, 7]),
        speed_range=(0.35, 0.35),
        wheel_radius=WHEEL_RADIUS,
    )

    velocity, _ = env.scene["robot"].root_velocity_writes[0]
    wheel_velocity = env.scene["robot"].joint_velocity_writes[0][0]
    torch.testing.assert_close(velocity[:, 0], torch.full((env.num_envs,), 0.35))
    torch.testing.assert_close(wheel_velocity, torch.full((env.num_envs, 4), 0.35 / WHEEL_RADIUS))


##
# Slope curriculum (addendum section 4.4)
##


SLOPE_TILE_SIZE_X = 15.0
"""Length [m] of the roller-slope tile, which both curriculum thresholds are fractions of."""

SLOPE_MOVE_CASES = [
    # (x displacement [m], promoted, demoted) at the live 15 m tile: promote above 6.0, demote below 3.0
    (-7.0, False, True),
    (0.0, False, True),
    (2.999, False, True),
    (3.0, False, False),
    (4.5, False, False),
    (6.0, False, False),
    (6.001, True, False),
    (12.0, True, False),
]
"""The promotion and demotion bands, evaluated by hand from ``d > 0.4 * size_x`` and
``d < 0.2 * size_x``. Both comparisons are strict, so the two boundary values themselves hold."""


@pytest.mark.parametrize("distance, promoted, demoted", SLOPE_MOVE_CASES)
def test_the_slope_move_masks_promote_past_forty_percent_and_demote_below_twenty(distance, promoted, demoted):
    move_up, move_down = mdp.slope_move_masks(torch.tensor([distance]), SLOPE_TILE_SIZE_X)

    assert bool(move_up[0]) is promoted
    assert bool(move_down[0]) is demoted


class _DummyTerrainGeneratorCfg:
    def __init__(self, size: tuple[float, float]) -> None:
        self.size = size


class _DummyTerrainCfg:
    def __init__(self, size: tuple[float, float]) -> None:
        self.terrain_generator = _DummyTerrainGeneratorCfg(size)


class _DummyTerrain:
    """Terrain-importer double recording the promotion decision it is handed."""

    def __init__(self, num_envs: int, device: str, size: tuple[float, float]) -> None:
        self.cfg = _DummyTerrainCfg(size)
        self.terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.updates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def update_env_origins(self, env_ids, move_up, move_down) -> None:
        # the real importer also wraps an environment that clears the last level to a random one;
        # this double keeps the floor, which is the half these tests exercise
        self.updates.append((env_ids, move_up.clone(), move_down.clone()))
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids] + move_up.long() - move_down.long(), 0)


def _slope_curriculum_env(root_x: list[float], root_y: list[float] | None = None) -> _DummyEnv:
    env = _DummyEnv(num_envs=len(root_x))
    env.scene.terrain = _DummyTerrain(env.num_envs, env.device, (SLOPE_TILE_SIZE_X, 4.0))
    positions = env.scene["robot"].data.root_link_pos_w.torch
    positions[:, 0] = torch.tensor(root_x)
    if root_y is not None:
        positions[:, 1] = torch.tensor(root_y)
    return env


def test_the_slope_curriculum_measures_the_signed_distance_down_the_slope():
    """Signed ``x``, not a planar norm: sliding backwards must demote rather than promote.

    A norm would read a 7 m slide back up the ramp as 7 m of progress, and a robot that fell over
    and skidded uphill would be handed a steeper slope.
    """
    env = _slope_curriculum_env(root_x=[7.0, -7.0, 4.0], root_y=[0.0, 0.0, 9.0])

    mdp.terrain_levels_slope(cast("ManagerBasedRLEnv", env), env_ids=torch.arange(3))

    _, move_up, move_down = env.scene.terrain.updates[0]
    torch.testing.assert_close(move_up, torch.tensor([True, False, False]))
    torch.testing.assert_close(move_down, torch.tensor([False, True, False]))


def test_the_slope_curriculum_measures_the_distance_from_each_environments_own_origin():
    """The origins sit on the ramp of each environment's own tile, at whatever world x that is."""
    env = _slope_curriculum_env(root_x=[107.0, 7.0])
    env.scene.env_origins[0, 0] = 100.0

    mdp.terrain_levels_slope(cast("ManagerBasedRLEnv", env), env_ids=torch.arange(2))

    _, move_up, _ = env.scene.terrain.updates[0]
    torch.testing.assert_close(move_up, torch.tensor([True, True]))


def test_the_slope_curriculum_reports_the_mean_level_over_every_environment():
    """Matches the stock ``terrain_levels_vel``, which the training log reads as one scalar."""
    env = _slope_curriculum_env(root_x=[7.0, 7.0, 0.0, 0.0])

    mean_level = mdp.terrain_levels_slope(cast("ManagerBasedRLEnv", env), env_ids=torch.arange(4))

    # two promoted to level 1, two held at level 0 by the floor inside ``update_env_origins``
    torch.testing.assert_close(mean_level, torch.tensor(0.5))
    torch.testing.assert_close(env.scene.terrain.terrain_levels, torch.tensor([1, 1, 0, 0]))


@pytest.mark.parametrize("kind", ["pose", "velocity"])
def test_command_reset_accepts_environment_slices(kind):
    """Manager resets pass slices; selected robots resample and others keep their commands."""
    torch.manual_seed(0)
    term = _make_pose_command(HEAD_POSE_RANGES) if kind == "pose" else _make_velocity_command()
    term.reset()
    before = term.command.clone()
    term.reset(slice(1, None, 2))
    torch.testing.assert_close(term.command[::2], before[::2])
    assert not torch.equal(term.command[1::2], before[1::2])
    assert torch.isfinite(term.command).all()
