# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the unified dexterous Lift and Reorient tasks."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CommandTerm

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.mdp.commands.pose_commands import (
    CableUniformPoseCommand,
    DeformableUniformPoseCommand,
    ObjectUniformPoseCommand,
)


class _MarkerSpy:
    def __init__(self, _cfg=None):
        self.calls: list[tuple[tuple, dict]] = []

    def set_visibility(self, _visible: bool) -> None:
        pass

    def visualize(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class _FakeScene(dict):
    def __init__(self, environment_ids: torch.Tensor, **assets):
        super().__init__(assets)
        self._ALL_INDICES = environment_ids
        self.env_origins = torch.zeros((len(environment_ids), 3))


def test_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must not depend on per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0])
    depth = torch.tensor([0.0, 2.0])

    assert torch.allclose(mdp.vision_camera._rgb_norm(None, rgb), torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.allclose(mdp.vision_camera._depth_norm(None, depth), torch.tanh(depth / 2) - 0.5)


def test_lift_pose_markers_forward_environment_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every lift pose, goal, and success marker should retain its environment ownership."""
    num_envs = 3
    environment_ids = torch.arange(num_envs)
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 0] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    root_pose_w = torch.cat((root_pos_w, identity_quat), dim=-1)

    robot = SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        ),
    )
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
            root_link_pose_w=SimpleNamespace(torch=root_pose_w),
        )
    )
    success_asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root_pos_w)))
    scene = _FakeScene(environment_ids, robot=robot, object=object_asset, table=success_asset)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene)
    cfg = SimpleNamespace(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="table",
        success_visualizer_cfg=object(),
        goal_pose_visualizer_cfg=object(),
        curr_pose_visualizer_cfg=object(),
        position_only=True,
        cmd_kind=None,
        element_names=None,
    )

    def _initialize_command_term(command, command_cfg, command_env) -> None:
        command.cfg = command_cfg
        command._env = command_env
        command.metrics = {}

    monkeypatch.setattr(CommandTerm, "__init__", _initialize_command_term)
    monkeypatch.setattr("isaaclab.markers.VisualizationMarkers", _MarkerSpy)

    command = ObjectUniformPoseCommand(cfg, env)
    command._set_debug_vis_impl(True)
    command._debug_vis_callback(None)
    command.cfg.position_only = False
    command._debug_vis_callback(None)
    command._update_metrics()
    DeformableUniformPoseCommand._update_metrics(command)
    command._segment_position_w = lambda: root_pos_w
    CableUniformPoseCommand._update_metrics(command)
    CableUniformPoseCommand._debug_vis_callback(command, None)

    expected_call_counts = {
        command.success_visualizer: 4,
        command.goal_visualizer: 3,
        command.curr_visualizer: 3,
    }
    for visualizer, expected_count in expected_call_counts.items():
        assert len(visualizer.calls) == expected_count
        for _, kwargs in visualizer.calls:
            assert torch.equal(kwargs["environment_ids"], environment_ids)


def test_lift_point_cloud_markers_repeat_environment_ids_per_point() -> None:
    """Flattened point-cloud markers should retain env-major ownership."""
    num_envs = 3
    num_points = 4
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 0] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    points_local = torch.arange(num_envs * num_points * 3, dtype=torch.float32).view(num_envs, num_points, 3)

    term = object.__new__(mdp.object_point_cloud_b)
    term.object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.ref_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.points_local = points_local
    term.points_w = torch.zeros_like(points_local)
    term.visualizer = _MarkerSpy()
    env = SimpleNamespace(num_envs=num_envs)

    term(env, num_points=num_points, visualize=True)

    assert len(term.visualizer.calls) == 1
    _, kwargs = term.visualizer.calls[0]
    assert torch.equal(kwargs["translations"], term.points_w.view(-1, 3))
    assert torch.equal(kwargs["environment_ids"], torch.arange(num_envs).repeat_interleave(num_points))
