# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for perspective views tied to an environment or an asset."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.visualizers import PerspectiveCameraCfg, SceneCameraCfg, VisualizerCfg
from isaaclab.visualizers.camera_follower import CameraFollower
from isaaclab.visualizers.visualizer_cfg import with_free_view


def _quat_z(yaw: float) -> list[float]:
    """Quaternion (x, y, z, w) of a rotation of *yaw* about +Z."""
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


class _Scene:
    """Two environments, 10 m apart, with a robot at the origin of the second."""

    num_envs = 2
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    def __init__(self, ready: bool = True):
        data = SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]])),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])),
        )
        self.robot = SimpleNamespace(is_initialized=ready, data=data)

    def __getitem__(self, name):
        return self.robot


def _eye(follower: CameraFollower, time: float = 0.1) -> torch.Tensor:
    return torch.tensor(follower.pose(time)[0])


def test_a_view_with_an_environment_stays_fixed_relative_to_its_origin():
    cfg = PerspectiveCameraCfg(env_index=1, eye=(1.0, 2.0, 3.0), lookat=(0.0, 0.0, 1.0))
    eye, target = CameraFollower(cfg, _Scene()).pose(0.1)
    assert eye == (11.0, 2.0, 3.0)
    assert target == (10.0, 0.0, 1.0)


def test_a_tracked_view_follows_the_asset_and_rotates_with_its_heading():
    scene = _Scene()
    cfg = PerspectiveCameraCfg(env_index=1, track_path="robot", eye=(-3.0, 0.0, 1.0), follow_heading=True)
    follower = CameraFollower(cfg, scene)
    torch.testing.assert_close(_eye(follower), torch.tensor([7.0, 0.0, 1.5]))
    # turned a quarter turn, the robot's back is at -y
    scene.robot.data.root_quat_w.torch[1] = torch.tensor(_quat_z(math.pi / 2))
    torch.testing.assert_close(_eye(follower), torch.tensor([10.0, -3.0, 1.5]), atol=1e-5, rtol=0)
    # without heading the offsets stay aligned with the world axes
    cfg.follow_heading = False
    torch.testing.assert_close(_eye(follower), torch.tensor([7.0, 0.0, 1.5]))


def test_smoothing_lags_a_turn_but_not_the_first_sample_or_a_reset():
    def run(jump: float) -> torch.Tensor:
        scene = _Scene()
        cfg = PerspectiveCameraCfg(track_path="robot", eye=(-3.0, 0.0, 1.0), follow_heading=True)
        cfg.heading_smoothing_time_constant = 0.5
        follower = CameraFollower(cfg, scene)
        follower.pose(0.0)
        scene.robot.data.root_quat_w.torch[0] = torch.tensor(_quat_z(math.pi / 2))
        scene.robot.data.root_pos_w.torch[0, 0] += jump  # an environment reset moves the asset
        return _eye(follower)

    smooth, reset = run(0.0), run(5.0)
    assert smooth[1] > -2.0  # still mostly behind the old heading
    torch.testing.assert_close(reset[1] - 0.0, torch.tensor(-3.0), atol=1e-4, rtol=0)  # the new heading at once


def test_a_view_waits_for_its_asset_and_rejects_bad_targets():
    assert CameraFollower(PerspectiveCameraCfg(track_path="robot"), _Scene(ready=False)).pose(0.1) is None
    with pytest.raises(ValueError, match="outside"):
        CameraFollower(PerspectiveCameraCfg(env_index=2), _Scene())
    with pytest.raises(ValueError, match="non-negative"):
        PerspectiveCameraCfg(env_index=-1)
    with pytest.raises(ValueError, match="non-negative"):
        PerspectiveCameraCfg(heading_smoothing_time_constant=-1.0)


def test_task_views_get_a_free_view_beside_them():
    free, task = PerspectiveCameraCfg(), PerspectiveCameraCfg(env_index=0)
    args = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 30.0)
    assert with_free_view([task], *args)[0].eye == (1.0, 2.0, 3.0)
    assert len(with_free_view([task], *args)) == 2
    assert with_free_view([free, task], *args) == [free, task]
    assert with_free_view([SceneCameraCfg(prim_path="/cam")], *args) == [SceneCameraCfg(prim_path="/cam")]
    # a task view first does not replace the visualizer's own pose or its streaming choice
    cfg = VisualizerCfg(eye=(7.0, 0.0, 0.0), streaming_view=True, cameras=[task])
    assert cfg.eye == (7.0, 0.0, 0.0) and cfg.streaming_view
    assert not VisualizerCfg(streaming_view=True, cameras=[free]).streaming_view
