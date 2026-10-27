# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton viewer perspective views that follow an asset or frame an environment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("newton")
from isaaclab_visualizers.newton.newton_visualizer import NewtonVisualizerBase
from pyglet.math import Vec3

from isaaclab.visualizers import PerspectiveCameraCfg
from isaaclab.visualizers.camera_follower import CameraFollower


class _Camera:
    """The viewer camera: a position, a look-at target stored as angles, and a field of view."""

    def __init__(self):
        self.pos, self.pitch, self.yaw, self.fov = Vec3(1.0, 1.0, 1.0), 0.1, 0.2, 40.0
        self.target = None

    def look_at(self, target):
        self.target = tuple(target)


class _Scene:
    num_envs = 2
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    def __getitem__(self, name):
        return SimpleNamespace(is_initialized=False)


def _visualizer(views) -> NewtonVisualizerBase:
    viz = NewtonVisualizerBase.__new__(NewtonVisualizerBase)
    viz.cfg = SimpleNamespace(focal_length=12.0, eye=(4.0, -4.0, 3.0), lookat=(0.0, 0.0, 0.0))
    viz._followers, viz._last_render_step = {}, 5
    viz._tracking_defaults, viz._tracking_error = {}, ""
    camera = _Camera()

    def set_camera(pos, pitch, yaw):
        camera.pos, camera.pitch, camera.yaw = pos, pitch, yaw

    viz._viewer = SimpleNamespace(camera=camera, set_camera=set_camera)
    viz._sim = SimpleNamespace(
        get_scene_data_provider=lambda: SimpleNamespace(get_interactive_scene=lambda: _Scene()),
        get_physics_step_count=lambda: 10,
        get_physics_dt=lambda: 0.1,
    )
    viz._image_views = list(views)
    viz._followers = {view: CameraFollower(view.cfg.source, _Scene()) for view in views if view.cfg.source.is_task_view}
    viz.current_image_view = viz._image_views[0]
    return viz


class _View:
    """A perspective image view: hashable like the real one, with no scene camera."""

    camera, render = None, None

    def __init__(self, cfg: PerspectiveCameraCfg):
        self.cfg = SimpleNamespace(source=cfg)


def _view(cfg: PerspectiveCameraCfg) -> _View:
    return _View(cfg)


def test_a_task_view_points_the_viewer_without_changing_the_configured_pose():
    task = _view(PerspectiveCameraCfg(env_index=1, eye=(1.0, 2.0, 3.0), lookat=(0.0, 0.0, 1.0), focal_length=24.0))
    viz = _visualizer([task])
    assert viz._apply_view_pose(task)
    camera = viz._viewer.camera
    assert (camera.pos.x, camera.pos.y, camera.pos.z) == (11.0, 2.0, 3.0)
    assert camera.target == (10.0, 0.0, 1.0)
    assert camera.fov == pytest.approx(viz._focal_length_to_vertical_fov_degrees(24.0))
    assert viz.cfg.eye == (4.0, -4.0, 3.0) and viz.cfg.focal_length == 12.0


def test_capturing_another_view_hands_the_viewer_camera_back():
    free, task = _view(PerspectiveCameraCfg()), _view(PerspectiveCameraCfg(env_index=1, eye=(1.0, 2.0, 3.0)))
    viz = _visualizer([free, task])
    seen = {}

    def capture(output):
        # the window's own view is borrowed for the capture, and the camera moves while it renders
        seen["view"] = viz.current_image_view
        viz._viewer.camera.pos, viz._viewer.camera.fov = Vec3(11.0, 2.0, 3.0), 20.0
        return "frame"

    viz._capture_frame = capture
    assert viz._capture_view(task, None) == "frame"
    assert seen["view"] is task
    camera = viz._viewer.camera
    assert (camera.pos.x, camera.pos.y, camera.pos.z, camera.fov) == (1.0, 1.0, 1.0, 40.0)
    assert viz.current_image_view is free and viz._last_render_step == -1


def test_the_task_view_recording_picks_the_first_tied_view():
    free = _view(PerspectiveCameraCfg())
    first, second = _view(PerspectiveCameraCfg(track_path="robot")), _view(PerspectiveCameraCfg(env_index=1))
    viz = _visualizer([free, first, second])
    viz._sim.get_physics_step_count = lambda: 3
    first.read_rgb, second.read_rgb = (lambda step: ("first", step)), (lambda step: ("second", step))
    assert viz.render_task_view_rgb_array() == ("first", 3)


def test_the_task_view_recording_reports_a_visualizer_without_one():
    with pytest.raises(RuntimeError, match="tied to an environment or an asset"):
        _visualizer([_view(PerspectiveCameraCfg())]).render_task_view_rgb_array()


def test_a_task_view_without_a_follower_has_no_pose_instead_of_failing():
    task = _view(PerspectiveCameraCfg(env_index=1))
    viz = _visualizer([task])
    viz._followers.clear()  # no interactive scene was registered when the viewer initialized
    assert not viz._apply_view_pose(task)


def test_the_selector_names_the_tracking_camera_and_the_free_perspective():
    sensor = SimpleNamespace(camera=object(), cfg=SimpleNamespace(source="/World/envs/env_0/cam"))
    viz = _visualizer([_view(PerspectiveCameraCfg()), _view(PerspectiveCameraCfg(env_index=1))])
    viz._image_views.append(sensor)
    assert viz._view_labels() == ["Perspective", "Tracking Camera", "/World/envs/env_0/cam"]

    viz = _visualizer([_view(PerspectiveCameraCfg(env_index=0)), _view(PerspectiveCameraCfg(track_path="robot"))])
    assert viz._view_labels() == ["Tracking Camera 1", "Tracking Camera 2"]


def test_resetting_the_tracking_camera_restores_what_the_task_declared():
    task = _view(PerspectiveCameraCfg(env_index=0, eye=(1.0, 0.0, 0.0), track_path="robot", follow_heading=True))
    viz = _visualizer([task])
    cfg, follower = task.cfg.source, viz._followers[task]
    viz._tracking_defaults = {
        task: {
            "eye": cfg.eye, "lookat": cfg.lookat, "focal_length": cfg.focal_length, "track_path": "robot",
            "follow_heading": True, "heading_smoothing_time_constant": 0.0, "env": follower.env,
        }
    }  # fmt: skip
    cfg.eye, cfg.follow_heading = (5.0, 5.0, 5.0), False
    follower.set_env(1)
    follower.set_track_path(None)

    assert viz._reset_tracking_camera(follower)
    assert (cfg.eye, cfg.track_path, cfg.follow_heading, follower.env) == ((1.0, 0.0, 0.0), "robot", True, 0)


def test_an_unknown_asset_is_reported_and_the_camera_keeps_following_its_target():
    task = _view(PerspectiveCameraCfg(env_index=0, track_path="robot"))
    viz = _visualizer([task])
    viz._tracking_error = ""
    follower = viz._followers[task]

    class _Scene2(_Scene):
        def __getitem__(self, name):
            if name != "robot":
                raise KeyError(name)
            return super().__getitem__(name)

    follower._scene = _Scene2()
    assert not viz._set_tracking_path(follower, "ghost")
    assert "unknown scene asset" in viz._tracking_error and task.cfg.source.track_path == "robot"
    assert viz._set_tracking_path(follower, "robot") and viz._tracking_error == ""
