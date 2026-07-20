# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp-native registered command terms."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import warp as wp
from isaaclab_experimental.envs import mdp as experimental_mdp
from isaaclab_experimental.envs.mdp.commands import (
    UniformPoseCommand,
    UniformPoseCommandCfg,
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from isaaclab_experimental.managers import CommandManager, CommandTerm

from isaaclab.envs.mdp.commands import UniformPoseCommandCfg as StableUniformPoseCommandCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg as StableUniformVelocityCommandCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique


class _ArrayProxy:
    """Minimal backend-neutral data proxy used by command terms."""

    def __init__(self, values: np.ndarray, dtype):
        self.warp = wp.array(values, dtype=dtype, device="cpu")
        self.torch = wp.to_torch(self.warp)


class _Robot:
    """Minimal articulation data needed by velocity and pose commands."""

    def __init__(self, num_envs: int):
        identity_quat = np.zeros((num_envs, 4), dtype=np.float32)
        identity_quat[:, 3] = 1.0
        identity_pose = np.zeros((num_envs, 7), dtype=np.float32)
        identity_pose[:, 6] = 1.0
        self.data = SimpleNamespace(
            root_pose_w=_ArrayProxy(identity_pose, wp.transformf),
            root_vel_w=_ArrayProxy(np.zeros((num_envs, 6), dtype=np.float32), wp.spatial_vectorf),
            root_pos_w=_ArrayProxy(np.zeros((num_envs, 3), dtype=np.float32), wp.vec3f),
            root_quat_w=_ArrayProxy(identity_quat, wp.quatf),
            root_lin_vel_b=_ArrayProxy(np.zeros((num_envs, 3), dtype=np.float32), wp.vec3f),
            root_ang_vel_b=_ArrayProxy(np.zeros((num_envs, 3), dtype=np.float32), wp.vec3f),
            heading_w=_ArrayProxy(np.zeros(num_envs, dtype=np.float32), wp.float32),
            body_pos_w=_ArrayProxy(np.zeros((num_envs, 1, 3), dtype=np.float32), wp.vec3f),
            body_quat_w=_ArrayProxy(identity_quat[:, None, :], wp.quatf),
        )
        self.is_initialized = True

    def find_bodies(self, body_name: str) -> tuple[list[int], list[str]]:
        return [0], [body_name]


class _Env:
    """Minimal manager-based environment used by command terms."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = {"robot": _Robot(num_envs)}
        self.rng_state_wp = wp.array(np.arange(num_envs, dtype=np.uint32) + 41, device=self.device)
        self.extras = {}
        self.sim = SimpleNamespace(
            vis_marker_registry=SimpleNamespace(
                add_debug_vis_callback=lambda term: None,
                clear_debug_vis_callback=lambda term: None,
            )
        )


def _velocity_cfg() -> UniformVelocityCommandCfg:
    return UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 2.0),
        heading_command=True,
        heading_control_stiffness=2.0,
        rel_heading_envs=1.0,
        rel_standing_envs=0.0,
        vel_xy_success_threshold=0.5,
        vel_yaw_success_threshold=0.4,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(2.0, 2.0),
            ang_vel_z=(0.3, 0.3),
            heading=(0.7, 0.7),
        ),
        debug_vis=False,
    )


def _pose_cfg() -> UniformPoseCommandCfg:
    return UniformPoseCommandCfg(
        asset_name="robot",
        body_name="tool",
        resampling_time_range=(3.0, 3.0),
        make_quat_unique=True,
        position_success_threshold=0.25,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(1.0, 1.0),
            pos_y=(2.0, 2.0),
            pos_z=(3.0, 3.0),
            roll=(0.2, 0.2),
            pitch=(-0.3, -0.3),
            yaw=(4.0, 4.0),
        ),
        debug_vis=False,
    )


def test_configs_exports_and_command_accessors_preserve_shared_contracts():
    """Configs should inherit stable fields and terms should expose debug UI and persistent storage."""
    assert issubclass(UniformVelocityCommandCfg, StableUniformVelocityCommandCfg)
    assert issubclass(UniformPoseCommandCfg, StableUniformPoseCommandCfg)
    assert experimental_mdp.UniformVelocityCommandCfg is UniformVelocityCommandCfg
    assert experimental_mdp.UniformPoseCommandCfg is UniformPoseCommandCfg
    assert str(_velocity_cfg().class_type).endswith(".commands.velocity_command:UniformVelocityCommand")
    assert str(_pose_cfg().class_type).endswith(".commands.pose_command:UniformPoseCommand")
    assert UniformVelocityCommand._set_debug_vis_impl.__module__.endswith(".commands._debug_vis")
    assert UniformPoseCommand._set_debug_vis_impl.__module__.endswith(".commands._debug_vis")

    env = _Env()
    velocity_term = UniformVelocityCommand(_velocity_cfg(), env)
    pose_term = UniformPoseCommand(_pose_cfg(), env)
    manager = CommandManager.__new__(CommandManager)
    manager._terms = {"velocity": velocity_term, "pose": pose_term}

    assert velocity_term.command_wp is velocity_term.command_wp
    assert pose_term.command_wp is pose_term.command_wp
    assert manager.get_command_wp("velocity") is velocity_term.command_wp
    assert manager.get_command_wp("pose") is pose_term.command_wp
    assert manager.get_command("velocity") is velocity_term.command
    assert manager.get_command("pose") is pose_term.command
    torch.testing.assert_close(pose_term.command[:, :6], torch.zeros(4, 6))
    torch.testing.assert_close(pose_term.command[:, 6], torch.ones(4))


def test_debug_visualization_uses_current_simulation_registry():
    """Command terms should register visualization through the current simulation API."""
    registry = SimpleNamespace(add_debug_vis_callback=Mock(return_value="handle"), clear_debug_vis_callback=Mock())
    term = SimpleNamespace(
        has_debug_vis_implementation=True,
        _set_debug_vis_impl=Mock(),
        _debug_vis_handle=None,
        _env=SimpleNamespace(sim=SimpleNamespace(vis_marker_registry=registry)),
    )

    assert CommandTerm.set_debug_vis(term, True)
    assert term._debug_vis_handle == "handle"
    registry.add_debug_vis_callback.assert_called_once_with(term)

    assert CommandTerm.set_debug_vis(term, False)
    registry.clear_debug_vis_callback.assert_called_once_with(term)

    CommandTerm.__del__(term)
    assert registry.clear_debug_vis_callback.call_count == 2


def test_uniform_velocity_command_updates_and_resets_selected_rows():
    """Velocity commands should update in Warp and preserve unselected rows during sparse reset."""
    env = _Env()
    term = UniformVelocityCommand(_velocity_cfg(), env)
    term._prepare_reset_extras()
    command_pointer = term.command.data_ptr()
    command_wp = term.command_wp

    term.command[:] = torch.tensor([[1.0, 2.0, 0.3], [2.0, -1.0, -0.2], [0.0, 0.5, 0.1], [-1.0, -2.0, 0.0]])
    env.scene["robot"].data.root_vel_w.torch[:, :3] = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [-2.0, -2.0, 0.0]]
    )
    env.scene["robot"].data.root_vel_w.torch[:, 5] = torch.tensor([0.1, -0.1, 0.4, 0.0])
    term._update_metrics()
    wp.synchronize()
    torch.testing.assert_close(
        wp.to_torch(term._error_xy_sum_wp),
        torch.tensor([np.sqrt(5.0), 1.0, 0.5, 1.0], dtype=torch.float32),
    )
    torch.testing.assert_close(wp.to_torch(term._error_yaw_sum_wp), torch.tensor([0.2, 0.1, 0.3, 0.0]))

    term.command.fill_(-9.0)
    wp.to_torch(term._error_xy_sum_wp)[:] = torch.tensor([0.2, 9.0, 2.0, 8.0])
    wp.to_torch(term._error_yaw_sum_wp)[:] = torch.tensor([0.1, 9.0, 2.0, 8.0])
    wp.to_torch(term._step_count_wp)[:] = torch.tensor([1.0, 1.0, 2.0, 1.0])
    wp.to_torch(term.time_left_wp)[:] = torch.tensor([-1.0, 6.0, -1.0, 8.0])
    wp.to_torch(term.command_counter_wp)[:] = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    env_mask = wp.array([True, False, True, False], dtype=wp.bool, device="cpu")

    extras = term.reset(env_mask=env_mask)
    wp.synchronize()

    assert term.command_wp is command_wp
    assert term.command.data_ptr() == command_pointer
    torch.testing.assert_close(term.command[[0, 2]], torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]))
    torch.testing.assert_close(term.command[[1, 3]], torch.full((2, 3), -9.0))
    torch.testing.assert_close(wp.to_torch(term.time_left_wp), torch.tensor([2.0, 6.0, 2.0, 8.0]))
    torch.testing.assert_close(wp.to_torch(term.command_counter_wp), torch.tensor([1, 6, 1, 8], dtype=torch.int32))
    torch.testing.assert_close(extras["error_vel_xy"], torch.tensor(0.6))
    torch.testing.assert_close(extras["error_vel_yaw"], torch.tensor(0.55))
    torch.testing.assert_close(extras["success_rate"], torch.tensor(0.5))
    torch.testing.assert_close(wp.to_torch(term._error_xy_sum_wp), torch.tensor([0.0, 9.0, 0.0, 8.0]))
    torch.testing.assert_close(wp.to_torch(term._step_count_wp), torch.tensor([0.0, 1.0, 0.0, 1.0]))

    term.command[:] = torch.tensor([[1.0, 2.0, 0.4]]).repeat(4, 1)
    term.heading_target[:] = torch.tensor([np.pi / 2, 0.0, np.pi / 2, 0.0])
    term.is_heading_env[:] = torch.tensor([True, False, True, False])
    term.is_standing_env[:] = torch.tensor([False, False, True, True])
    term._update_command()
    wp.synchronize()
    torch.testing.assert_close(
        term.command,
        torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_uniform_velocity_command_replay_reads_current_root_state():
    """Captured command kernels should read current Tier-1 root state on every replay."""
    env = _Env()
    cfg = _velocity_cfg()
    cfg.ranges.ang_vel_z = (-2.0, 2.0)
    term = UniformVelocityCommand(cfg, env)
    robot_data = env.scene["robot"].data

    term.command.zero_()
    term.command[:, 0] = 1.0
    term.heading_target.fill_(np.pi / 2)
    term.is_heading_env.fill_(True)
    term.is_standing_env.fill_(False)

    with wp.ScopedCapture(device=env.device) as metrics_capture:
        term._update_metrics()
    with wp.ScopedCapture(device=env.device) as command_capture:
        term._update_command()

    # Identity pose with matching world-frame velocity has zero body-frame XY error.
    robot_data.root_vel_w.torch[:, 0] = 1.0
    wp.capture_launch(metrics_capture.graph)
    wp.synchronize()
    torch.testing.assert_close(wp.to_torch(term._error_xy_sum_wp), torch.zeros(env.num_envs))

    # A 90-degree root yaw exactly matches the heading target, so replay should command zero yaw rate.
    half_sqrt = np.sqrt(0.5)
    robot_data.root_pose_w.torch[:, 5] = half_sqrt
    robot_data.root_pose_w.torch[:, 6] = half_sqrt
    wp.capture_launch(command_capture.graph)
    wp.synchronize()
    torch.testing.assert_close(term.command[:, 2], torch.zeros(env.num_envs), atol=1.0e-6, rtol=1.0e-6)


def test_uniform_pose_command_matches_stable_math_and_tracks_sticky_success():
    """Pose commands should use xyzw math, update metrics, and reset sticky success by mask."""
    env = _Env()
    term = UniformPoseCommand(_pose_cfg(), env)
    term._prepare_reset_extras()
    command_pointer = term.command.data_ptr()
    command_wp = term.command_wp
    term.command.fill_(-9.0)
    wp.to_torch(term.time_left_wp)[:] = torch.tensor([-1.0, 6.0, -1.0, 8.0])
    wp.to_torch(term.command_counter_wp)[:] = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    env_mask = wp.array([True, False, True, False], dtype=wp.bool, device="cpu")

    term.reset(env_mask=env_mask)
    wp.synchronize()

    expected_quat = quat_unique(quat_from_euler_xyz(torch.tensor([0.2]), torch.tensor([-0.3]), torch.tensor([4.0])))[0]
    assert term.command_wp is command_wp
    assert term.command.data_ptr() == command_pointer
    torch.testing.assert_close(term.command[[0, 2], :3], torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    torch.testing.assert_close(term.command[[0, 2], 3:], expected_quat.repeat(2, 1), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(term.command[[1, 3]], torch.full((2, 7), -9.0))
    torch.testing.assert_close(wp.to_torch(term.time_left_wp), torch.tensor([3.0, 6.0, 3.0, 8.0]))
    torch.testing.assert_close(wp.to_torch(term.command_counter_wp), torch.tensor([1, 6, 1, 8], dtype=torch.int32))

    term.command[[1, 3]] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    roll = torch.tensor([0.0, 0.1, -0.2, 0.3])
    pitch = torch.tensor([0.0, -0.1, 0.2, -0.3])
    yaw = torch.tensor([0.5, -0.4, 0.3, -0.2])
    root_quat = quat_from_euler_xyz(roll, pitch, yaw)
    env.scene["robot"].data.root_pos_w.torch[:] = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [-5.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
    )
    env.scene["robot"].data.root_quat_w.torch[:] = root_quat
    expected_position_w, expected_orientation_w = combine_frame_transforms(
        env.scene["robot"].data.root_pos_w.torch,
        root_quat,
        term.command[:, :3],
        term.command[:, 3:],
    )
    offsets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.5, 0.0, 0.0]])
    env.scene["robot"].data.body_pos_w.torch[:, 0] = expected_position_w + offsets
    identity_quat = torch.zeros(4, 4)
    identity_quat[:, 3] = 1.0
    env.scene["robot"].data.body_quat_w.torch[:, 0] = identity_quat

    term._update_metrics()
    wp.synchronize()

    expected_position_delta, expected_orientation_delta = compute_pose_error(
        expected_position_w,
        expected_orientation_w,
        env.scene["robot"].data.body_pos_w.torch[:, 0],
        env.scene["robot"].data.body_quat_w.torch[:, 0],
    )
    torch.testing.assert_close(term.pose_command_w[:, :3], expected_position_w, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(term.pose_command_w[:, 3:], expected_orientation_w, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(
        wp.to_torch(term.metrics["position_error"]), torch.linalg.norm(expected_position_delta, dim=-1)
    )
    torch.testing.assert_close(
        wp.to_torch(term.metrics["orientation_error"]),
        torch.linalg.norm(expected_orientation_delta, dim=-1),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(wp.to_torch(term._success_rate_wp), torch.tensor([1.0, 0.0, 1.0, 0.0]))

    env.scene["robot"].data.body_pos_w.torch[0, 0] += 2.0
    term._update_metrics()
    wp.synchronize()
    torch.testing.assert_close(wp.to_torch(term._success_rate_wp), torch.tensor([1.0, 0.0, 1.0, 0.0]))

    reset_mask = wp.array([True, False, False, True], dtype=wp.bool, device="cpu")
    extras = term.reset(env_mask=reset_mask)
    wp.synchronize()
    torch.testing.assert_close(extras["success_rate"], torch.tensor(0.5))
    torch.testing.assert_close(wp.to_torch(term._success_rate_wp), torch.tensor([0.0, 0.0, 1.0, 0.0]))
