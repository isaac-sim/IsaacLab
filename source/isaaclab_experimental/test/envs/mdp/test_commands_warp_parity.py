# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for the warp-first command terms.

The command manager runs captured, so each command term is recorded once, has its inputs overwritten
in place, and is replayed; the result is compared against the stable term evaluated on the new data.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

# Skip entire module if no CUDA device available
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

from isaaclab_experimental.envs.mdp.commands import NullCommand, UniformPoseCommand, UniformVelocityCommand
from isaaclab_experimental.utils.warp import resolve_1d_mask
from parity_helpers import (
    DEVICE,
    NUM_BODIES,
    NUM_ENVS,
    MockArticulation,
    MockArticulationData,
    assert_close,
    assert_equal,
    copy_np_to_wp,
    make_pose_command_term,
    mutate_body_data,
    quat_rotate_inv_np,
)

import isaaclab.envs.mdp.commands.velocity_command as stable_velocity_command
from isaaclab.envs.mdp.commands.commands_cfg import NullCommandCfg, UniformPoseCommandCfg, UniformVelocityCommandCfg
from isaaclab.utils.seed import WarpRng

_DT = 0.02


@pytest.fixture(autouse=True)
def _rng_state():
    WarpRng.state = wp.array(np.arange(NUM_ENVS, dtype=np.uint32) + 7, device=DEVICE)
    yield
    WarpRng.state = None


def _command_env(robot) -> SimpleNamespace:
    """The environment surface a command term reads."""
    all_mask = wp.ones(NUM_ENVS, dtype=wp.bool, device=DEVICE)
    scratch_mask = wp.zeros(NUM_ENVS, dtype=wp.bool, device=DEVICE)
    return SimpleNamespace(
        scene={"robot": robot},
        num_envs=NUM_ENVS,
        device=DEVICE,
        resolve_env_mask=lambda env_ids=None, env_mask=None: resolve_1d_mask(
            ids=env_ids, mask=env_mask, all_mask=all_mask, scratch_mask=scratch_mask, device=DEVICE
        ),
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None)),
    )


def _make_term(term_class, cfg, robot):
    """Construct a command term the way :class:`CommandManager` does."""
    term = term_class(cfg, _command_env(robot))
    term._prepare_reset_extras()
    return term


def _capture_compute(term) -> wp.Graph:
    """Record one :meth:`compute` without resampling; the resampling timer is pushed past the test."""
    term.time_left_wp.fill_(1.0e6)
    term.compute(_DT)  # warm-up outside the capture
    with wp.ScopedCapture() as capture:
        term.compute(_DT)
    return capture.graph


def test_pose_command_matches_stable_after_capture_and_mutation():
    """World-frame goal, tracking errors and the sticky success flag follow the stable term on replay."""
    art_data = MockArticulationData(num_bodies=NUM_BODIES)
    robot = MockArticulation(art_data, num_bodies=NUM_BODIES)
    stable = make_pose_command_term(robot)
    stable.metrics = {}
    cfg = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="body",
        resampling_time_range=(1.0, 1.0),
        position_success_threshold=stable.cfg.position_success_threshold,
        orientation_success_threshold=stable.cfg.orientation_success_threshold,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )
    term = _make_term(UniformPoseCommand, cfg, robot)
    graph = _capture_compute(term)

    # both the articulation state and the command move, so a stale view of either shows up on replay
    mutate_body_data(art_data)
    fresh = make_pose_command_term(robot, seed=505)
    stable.pose_command_b[:] = fresh.pose_command_b
    term.pose_command_b[:] = fresh.pose_command_b
    # cleared so the replay has to set the sticky flags again
    stable._succeeded.zero_()
    term._succeeded.zero_()
    wp.synchronize()
    wp.capture_launch(graph)
    wp.synchronize()
    stable._update_metrics()

    assert_close(term.pose_command_w, stable.pose_command_w)
    assert_close(wp.to_torch(term.metrics["position_error"]), stable.metrics["position_error"])
    assert_close(wp.to_torch(term.metrics["orientation_error"]), stable.metrics["orientation_error"], atol=1e-4)
    assert_equal(term._succeeded, stable._succeeded)
    assert stable._succeeded.any() and not stable._succeeded.all(), "thresholds must split the environments"

    # the episode success rate is the mean sticky flag over the reset envs, which then start clean
    selected = torch.arange(NUM_ENVS, device=DEVICE) % 2 == 0
    success_rate = term.reset(env_mask=wp.from_torch(selected))["success_rate"]
    wp.synchronize()
    assert success_rate.item() == pytest.approx(stable._succeeded[selected].float().mean().item())
    assert not term._succeeded[selected].any()
    assert_equal(term._succeeded[~selected], stable._succeeded[~selected])


def test_velocity_command_matches_stable_after_capture_and_mutation():
    """Heading control, standing envs and tracking-error sums follow the stable term on replay."""
    art_data = MockArticulationData()
    robot = MockArticulation(art_data)
    cfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0, 1.0),
        heading_command=True,
        heading_control_stiffness=0.7,
        rel_standing_envs=0.3,
        rel_heading_envs=0.6,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.8, 0.8), heading=(-math.pi, math.pi)
        ),
    )
    term = _make_term(UniformVelocityCommand, cfg, robot)
    term.reset()  # samples commands, heading targets and roles
    graph = _capture_compute(term)
    stable_error_sums = (term._error_xy_sum.numpy().copy(), term._error_yaw_sum.numpy().copy())

    # yaw-only root rotations, so the stable term's heading is the yaw by construction
    rng = np.random.RandomState(606)
    yaw = rng.uniform(-math.pi, math.pi, NUM_ENVS).astype(np.float32)
    quat = np.zeros((NUM_ENVS, 4), dtype=np.float32)
    quat[:, 2] = np.sin(yaw / 2.0)
    quat[:, 3] = np.cos(yaw / 2.0)
    lin_vel_w = rng.randn(NUM_ENVS, 3).astype(np.float32)
    ang_vel_w = rng.randn(NUM_ENVS, 3).astype(np.float32)
    pose = np.concatenate([art_data.root_pos_w.warp.numpy(), quat], axis=1)
    copy_np_to_wp(art_data.root_link_pose_w, pose)
    copy_np_to_wp(art_data.root_quat_w, quat)
    copy_np_to_wp(art_data.root_com_vel_w, np.concatenate([lin_vel_w, ang_vel_w], axis=1))
    copy_np_to_wp(art_data.root_lin_vel_b, quat_rotate_inv_np(quat, lin_vel_w))
    copy_np_to_wp(art_data.root_ang_vel_b, quat_rotate_inv_np(quat, ang_vel_w))
    art_data.heading_w = SimpleNamespace(torch=torch.tensor(yaw, device=DEVICE))

    stable = object.__new__(stable_velocity_command.UniformVelocityCommand)
    stable.cfg = cfg
    stable.robot = robot
    stable.vel_command_b = term.vel_command_b.clone()
    stable.heading_target = wp.to_torch(term._heading_target_wp).clone()
    stable.is_heading_env = wp.to_torch(term._is_heading_env_wp).clone()
    stable.is_standing_env = wp.to_torch(term._is_standing_env_wp).clone()
    stable._error_xy_sum = torch.tensor(stable_error_sums[0], device=DEVICE)
    stable._error_yaw_sum = torch.tensor(stable_error_sums[1], device=DEVICE)
    stable._step_count = torch.zeros(NUM_ENVS, device=DEVICE)
    assert stable.is_heading_env.any() and stable.is_standing_env.any() and not stable.is_standing_env.all()

    wp.capture_launch(graph)
    wp.synchronize()
    stable._update_metrics()
    stable._update_command()

    assert_close(term.vel_command_b, stable.vel_command_b)
    assert_close(wp.to_torch(term._error_xy_sum), stable._error_xy_sum)
    assert_close(wp.to_torch(term._error_yaw_sum), stable._error_yaw_sum)


def test_null_command_leaves_resampling_time_finite():
    """The null command never draws from its infinite resampling range."""
    term = _make_term(NullCommand, NullCommandCfg(), robot=None)

    assert term.reset() == {}
    term.compute(_DT)

    assert np.isfinite(term.time_left_wp.numpy()).all()
