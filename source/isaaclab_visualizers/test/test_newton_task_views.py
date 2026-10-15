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

    viz._capture_perspective = capture
    assert viz._capture_view(task, None) == "frame"
    assert seen["view"] is task
    camera = viz._viewer.camera
    assert (camera.pos.x, camera.pos.y, camera.pos.z, camera.fov) == (1.0, 1.0, 1.0, 40.0)
    assert viz.current_image_view is free and viz._last_render_step == -1


def test_the_task_view_recording_picks_the_first_tied_view_and_reports_none():
    free, task = _view(PerspectiveCameraCfg()), _view(PerspectiveCameraCfg(track_path="robot"))
    viz = _visualizer([free, task])
    viz._sim.get_physics_step_count = lambda: 3
    task.read_rgb = lambda step: ("task", step)
    assert viz.render_task_view_rgb_array() == ("task", 3)
    with pytest.raises(RuntimeError, match="tied to an environment or an asset"):
        _visualizer([free]).render_task_view_rgb_array()


def test_the_selector_names_what_a_task_view_follows_or_frames():
    label = NewtonVisualizerBase._view_label
    assert label(0, _view(PerspectiveCameraCfg())) == "Perspective 1"
    assert label(1, _view(PerspectiveCameraCfg(env_index=2))) == "Environment 2"
    assert label(1, _view(PerspectiveCameraCfg(track_path="robot/base"))) == "Following robot/base (env 0)"
    assert label(2, SimpleNamespace(camera=object(), cfg=SimpleNamespace(source="/World/envs/env_0/cam"))) == (
        "/World/envs/env_0/cam"
    )
