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

from isaaclab.visualizers import PerspectiveCameraCfg, SceneCameraCfg, TrackingCameraCfg, VisualizerCfg
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
    cfg = TrackingCameraCfg(env_index=1, eye=(1.0, 2.0, 3.0), lookat=(0.0, 0.0, 1.0))
    eye, target = CameraFollower(cfg, _Scene()).pose(0.1)
    assert eye == (11.0, 2.0, 3.0)
    assert target == (10.0, 0.0, 1.0)


def test_a_tracked_view_follows_the_asset_and_rotates_with_its_heading():
    scene = _Scene()
    cfg = TrackingCameraCfg(env_index=1, track_path="robot", eye=(-3.0, 0.0, 1.0), follow_heading=True)
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
        cfg = TrackingCameraCfg(track_path="robot", eye=(-3.0, 0.0, 1.0), follow_heading=True)
        cfg.heading_smoothing_time_constant = 0.5
        follower = CameraFollower(cfg, scene)
        follower.pose(0.0)
        scene.robot.data.root_quat_w.torch[0] = torch.tensor(_quat_z(0.2))  # 2 rad/s
        scene.robot.data.root_pos_w.torch[0, 0] += jump  # an environment reset also moves the asset
        return _eye(follower)

    smooth, reset = run(0.0), run(5.0)
    assert smooth[1] > -0.3  # filtered toward the new heading
    torch.testing.assert_close(reset[1], torch.tensor(-3.0 * math.sin(0.2)), atol=1e-4, rtol=0)  # the new one at once


def test_a_reset_in_place_with_a_new_heading_is_not_smoothed_but_a_real_turn_is():
    def run(yaw: float) -> torch.Tensor:
        scene = _Scene()
        cfg = TrackingCameraCfg(track_path="robot", eye=(-3.0, 0.0, 1.0), follow_heading=True)
        cfg.heading_smoothing_time_constant = 0.5
        follower = CameraFollower(cfg, scene)
        follower.pose(0.0)
        scene.robot.data.root_quat_w.torch[0] = torch.tensor(_quat_z(yaw))  # the position does not change
        return torch.tensor(follower.pose(0.02)[0])

    torch.testing.assert_close(run(math.pi / 2)[1], torch.tensor(-3.0), atol=1e-4, rtol=0)  # 78 rad/s: a reset
    assert run(0.05)[1] > -0.2  # 2.5 rad/s: a turn, filtered toward the new heading


def test_a_view_waits_for_its_asset_and_rejects_bad_targets():
    assert CameraFollower(TrackingCameraCfg(track_path="robot"), _Scene(ready=False)).pose(0.1) is None
    with pytest.raises(ValueError, match="outside"):
        CameraFollower(TrackingCameraCfg(env_index=2), _Scene())
    with pytest.raises(ValueError, match="non-negative"):
        TrackingCameraCfg(env_index=-1)
    with pytest.raises(ValueError, match="non-negative"):
        TrackingCameraCfg(heading_smoothing_time_constant=-1.0)


def test_tracking_cameras_get_a_free_view_beside_them():
    free, task = PerspectiveCameraCfg(), TrackingCameraCfg(env_index=0)
    args = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 30.0)
    assert with_free_view([task], *args)[0].eye == (1.0, 2.0, 3.0)
    assert len(with_free_view([task], *args)) == 2
    assert with_free_view([free, task], *args) == [free, task]
    # scene cameras declared ahead of the tracking camera stay first
    sensor = SceneCameraCfg(prim_path="/cam")
    views = with_free_view([sensor, task], *args)
    assert views[0] is sensor and views[1].eye == (1.0, 2.0, 3.0) and views[2] is task
    assert with_free_view([SceneCameraCfg(prim_path="/cam")], *args) == [SceneCameraCfg(prim_path="/cam")]
    # a tracking camera first does not replace the visualizer's own pose or its streaming choice
    cfg = VisualizerCfg(eye=(7.0, 0.0, 0.0), streaming_view=True, cameras=[task])
    assert cfg.eye == (7.0, 0.0, 0.0) and cfg.streaming_view
    assert not VisualizerCfg(streaming_view=True, cameras=[free]).streaming_view


class _NamedScene(_Scene):
    """A scene that knows its assets by name, as an interactive scene does."""

    def __init__(self):
        super().__init__()
        self.robot.body_names = ["base", "LF_FOOT"]
        self.robot.find_bodies = lambda name: (
            [self.robot.body_names.index(name)] if name in self.robot.body_names else [],
            [],
        )
        self.articulations, self.rigid_objects = {"robot": self.robot}, {"cube": SimpleNamespace(is_initialized=True)}

    def __getitem__(self, name):
        try:
            return {**self.articulations, **self.rigid_objects}[name]
        except KeyError:
            raise KeyError(name) from None


def test_the_followed_asset_can_be_switched_and_cleared_while_running():
    cfg = TrackingCameraCfg(env_index=1, eye=(1.0, 0.0, 0.0), lookat=(0.0, 0.0, 0.0))
    follower = CameraFollower(cfg, _NamedScene())
    assert _eye(follower).tolist() == [11.0, 0.0, 0.0]  # Fixed relative to the environment origin.

    follower.set_track_path("robot")
    assert cfg.track_path == "robot"
    assert _eye(follower).tolist() == [11.0, 0.0, 0.5]  # Offset from the robot at (10, 0, 0.5).

    follower.set_track_path(None)
    assert cfg.track_path is None
    assert _eye(follower).tolist() == [11.0, 0.0, 0.0]


def test_an_unknown_asset_or_ambiguous_body_is_rejected_and_keeps_the_current_target():
    cfg = TrackingCameraCfg(env_index=0, track_path="robot")
    follower = CameraFollower(cfg, _NamedScene())

    with pytest.raises(ValueError, match="unknown scene asset"):
        follower.set_track_path("ghost")
    with pytest.raises(ValueError, match="exactly one body"):
        follower.set_track_path("robot/no_such_body")

    assert cfg.track_path == "robot"
    follower.set_track_path("robot/LF_FOOT")
    assert cfg.track_path == "robot/LF_FOOT"


def test_the_followable_assets_and_bodies_are_listed():
    follower = CameraFollower(TrackingCameraCfg(env_index=0), _NamedScene())

    assert follower.asset_names() == ["robot", "cube"]
    assert follower.body_names("robot") == ["base", "LF_FOOT"]
    assert follower.body_names("cube") == []
    assert follower.body_names("missing") == []


def test_clearing_the_asset_keeps_the_view_tied_to_its_environment():
    cfg = TrackingCameraCfg(track_path="robot")
    follower = CameraFollower(cfg, _NamedScene())
    assert isinstance(cfg, TrackingCameraCfg)
    follower.set_track_path(None)
    assert cfg.track_path is None and isinstance(cfg, TrackingCameraCfg)


def test_two_updates_a_millisecond_apart_are_not_mistaken_for_an_environment_reset():
    cfg = TrackingCameraCfg(env_index=0, track_path="robot", follow_heading=True, heading_smoothing_time_constant=1.0)
    follower = CameraFollower(cfg, _NamedScene())
    follower.pose(0.0)
    follower._yaw = 0.0
    follower._last_position = [0.0, 0.0, 0.5]
    assert follower._filtered_yaw([0.0, 0.0, 0.0099995, 0.99995], [0.0, 0.0, 0.5], 0.001) < 0.02


def test_without_an_env_index_the_middle_visible_environment_is_followed():
    class _Many(_Scene):
        num_envs = 256
        env_origins = torch.zeros(256, 3)

    assert CameraFollower(TrackingCameraCfg(), _Many()).env == 127
    assert CameraFollower(TrackingCameraCfg(), _Many(), env_ids=[3, 10, 40]).env == 10
    assert CameraFollower(TrackingCameraCfg(env_index=5), _Many()).env == 5
