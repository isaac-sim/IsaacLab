# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tracking Camera views of the Newton GL window."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("newton")
from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonVisualizerBase
from pyglet.math import Vec3

from isaaclab.visualizers import PerspectiveCameraCfg, TrackingCameraCfg
from isaaclab.visualizers.camera_follower import CameraFollower


class _Camera:
    """The viewer camera: a position, a look-at target, angles, and a field of view."""

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


class _View:
    """An image view; a class, not a namespace, so that it can key the follower dictionary."""

    def __init__(self, source, camera=None):
        self.cfg, self.camera = SimpleNamespace(source=source), camera


def _view(source):
    return _View(source)


def _visualizer(views) -> NewtonVisualizerBase:
    viz = NewtonVisualizerBase.__new__(NewtonVisualizerBase)
    viz.cfg = SimpleNamespace(focal_length=12.0)
    camera = _Camera()

    def set_camera(pos, pitch, yaw):
        camera.pos, camera.pitch, camera.yaw = pos, pitch, yaw

    viz._viewer = SimpleNamespace(camera=camera, set_camera=set_camera)
    viz._sim = SimpleNamespace(get_physics_step_count=lambda: 10, get_physics_dt=lambda: 0.1)
    viz._image_views = list(views)
    viz._followers = {view: CameraFollower(view.cfg.source, _Scene()) for view in views if viz._is_task_view(view)}
    viz.rtx_rendering, viz._rtx_pose_view, viz._perspective_poses = False, None, {}
    return viz


def test_a_lone_free_view_is_called_perspective_and_a_task_view_the_tracking_camera():
    free, task = _view(PerspectiveCameraCfg()), _view(TrackingCameraCfg(env_index=0, eye=(2.0, 0.0, 1.0)))
    viz = _visualizer([free, task])

    assert viz._view_labels([0, 1]) == ["Perspective", "Tracking Camera"]
    assert [viz._is_task_view(view) for view in viz._image_views] == [False, True]


def test_views_are_numbered_only_when_several_share_a_name():
    views = [_view(PerspectiveCameraCfg()), _view(PerspectiveCameraCfg()), _view(TrackingCameraCfg(env_index=0))]
    views += [_view(TrackingCameraCfg(env_index=1))]
    viz = _visualizer(views)

    assert viz._view_labels([0, 1, 2, 3]) == [
        "Perspective 1",
        "Perspective 2",
        "Tracking Camera 1",
        "Tracking Camera 2",
    ]


def test_scene_cameras_are_named_by_their_prim_without_the_environment_pattern():
    camera = SimpleNamespace(cfg=SimpleNamespace(data_types=["rgba"]))
    sensor = _View("/World/envs/env_[^/]+/Robot/wrist_camera", camera=camera)
    viz = _visualizer([_view(PerspectiveCameraCfg()), sensor])

    assert viz._view_labels([0, 1]) == ["Perspective", "wrist_camera (RGB sensor)"]


def test_the_camera_list_puts_perspective_first_then_tracking_cameras_then_sensors():
    sensor = _View("/World/envs/env_[^/]+/Camera", camera=object())
    tracking, free = _view(TrackingCameraCfg(env_index=0)), _view(PerspectiveCameraCfg())
    viz = _visualizer([sensor, tracking, free])

    order = sorted(range(3), key=lambda i: viz._source_rank(viz._image_views[i]))

    assert order == [2, 1, 0]


def test_a_task_view_moves_the_viewer_camera_to_the_followed_environment():
    view = _view(TrackingCameraCfg(env_index=1, eye=(2.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.5), focal_length=24.0))
    viz = _visualizer([view])

    assert viz._apply_view_pose(view)

    camera = viz._viewer.camera
    assert (camera.pos.x, camera.pos.y, camera.pos.z) == pytest.approx((12.0, 0.0, 1.0))  # Environment 1 origin.
    assert camera.target == pytest.approx((10.0, 0.0, 0.5))


def test_switching_the_followed_environment_re_aims_the_camera():
    view = _view(TrackingCameraCfg(env_index=0, eye=(2.0, 0.0, 1.0)))
    viz = _visualizer([view])

    viz._followers[view].set_env(1)
    viz._apply_view_pose(view)

    assert viz._viewer.camera.pos.x == pytest.approx(12.0)


def test_resetting_the_tracking_camera_restores_what_the_task_declared():
    view = _view(TrackingCameraCfg(env_index=0, eye=(2.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.5), focal_length=24.0))
    viz = _visualizer([view])
    cfg = view.cfg.source
    viz._tracking_defaults = {
        view: {
            "eye": cfg.eye,
            "lookat": cfg.lookat,
            "focal_length": cfg.focal_length,
            "track_path": cfg.track_path,
            "follow_heading": cfg.follow_heading,
            "heading_smoothing_time_constant": cfg.heading_smoothing_time_constant,
            "env": 0,
        }
    }
    viz.current_image_view = view
    viz._tracking_error = ""
    follower = viz._followers[view]

    cfg.eye, cfg.focal_length = (9.0, 9.0, 9.0), 60.0
    follower.set_env(1)
    assert viz._reset_tracking_camera(follower)

    assert (cfg.eye, cfg.focal_length, follower.env) == ((2.0, 0.0, 1.0), 24.0, 0)


def test_an_invalid_followed_asset_is_reported_not_raised():
    view = _view(TrackingCameraCfg(env_index=0))
    viz = _visualizer([view])
    viz._tracking_error = ""

    class Scene(_Scene):
        def __getitem__(self, name):
            raise KeyError(name)

    viz._followers[view]._scene = Scene()

    assert not viz._set_tracking_path(viz._followers[view], "ghost")
    assert "unknown scene asset" in viz._tracking_error


def test_the_free_perspective_pose_is_kept_while_another_source_is_shown():
    free, task = _view(PerspectiveCameraCfg()), _view(TrackingCameraCfg(env_index=0))
    viz = _visualizer([free, task])
    viz._focal_length_to_vertical_fov_degrees = lambda focal=None: 40.0
    viz.current_image_view = free
    viz._viewer.camera.fov = 0.0
    camera = viz._viewer.camera
    camera.pos, camera.pitch, camera.yaw = (1.0, 2.0, 3.0), -10.0, 45.0

    viz._remember_perspective_pose()
    camera.pos, camera.pitch, camera.yaw = (9.0, 9.0, 9.0), 0.0, 0.0  # A Tracking Camera or sensor moved it.
    viz.cfg.focal_length = 50.0

    assert viz._restore_perspective_pose(free)
    assert (tuple(camera.pos), camera.pitch, camera.yaw, viz.cfg.focal_length) == ((1.0, 2.0, 3.0), -10.0, 45.0, 12.0)
    assert not viz._restore_perspective_pose(task)  # A view that was never left has nothing to restore.


def test_resetting_the_tracking_camera_works_while_the_rtx_view_is_current():
    view = _view(TrackingCameraCfg(env_index=0, eye=(2.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.5), focal_length=24.0))
    rtx = _View("/World/viewer_camera_0_rtx", camera=object())
    viz = _visualizer([view])
    cfg = view.cfg.source
    viz._tracking_defaults = {
        view: {
            "eye": cfg.eye,
            "lookat": cfg.lookat,
            "focal_length": cfg.focal_length,
            "track_path": cfg.track_path,
            "follow_heading": cfg.follow_heading,
            "heading_smoothing_time_constant": cfg.heading_smoothing_time_constant,
            "env": 0,
        }
    }
    viz.current_image_view, viz.rtx_rendering, viz._rtx_pose_view = (
        rtx,
        True,
        view,
    )  # RTX is showing the Tracking Camera.
    viz._tracking_error = ""

    cfg.eye = (9.0, 9.0, 9.0)
    assert viz._reset_tracking_camera(viz._followers[view])

    assert cfg.eye == (2.0, 0.0, 1.0)


def _rtx_view():
    camera = SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/viewer_camera_0", renderer_cfg="ovrtx"))
    return _View("/World/viewer_camera_0", camera=camera)


def _switchable(views):
    """A visualizer whose camera selection only records which view became current."""
    base = _visualizer(views)
    viz = NewtonGLVisualizer.__new__(NewtonGLVisualizer)
    viz.__dict__.update(base.__dict__)
    viz._overlay_publisher = SimpleNamespace(hide_all=lambda renderers: viz.hidden.append(renderers))
    viz._sim.get_or_create_backend = lambda cfg: cfg
    viz.hidden, viz.current_image_view = [], views[0]

    def select(index, keep_pose=False):
        viz.current_image_view = viz._image_views[index]

    viz._select_camera = select
    return viz


def test_switching_to_rtx_and_back_returns_to_the_perspective_view():
    free, rtx = _view(PerspectiveCameraCfg()), _rtx_view()
    viz = _switchable([free, rtx])

    viz.set_rtx_rendering(True)
    assert viz.current_image_view is rtx and viz.rtx_rendering

    viz.set_rtx_rendering(False)
    assert viz.current_image_view is free and not viz.rtx_rendering
    assert viz.hidden == [("ovrtx",)]  # The overlay meshes are hidden when OVRTX stops drawing.


def test_switching_to_rtx_while_a_tracking_camera_is_shown_returns_to_it():
    free, task, rtx = _view(PerspectiveCameraCfg()), _view(TrackingCameraCfg(env_index=0)), _rtx_view()
    viz = _switchable([free, task, rtx])
    viz.current_image_view = task

    viz.set_rtx_rendering(True)
    viz.set_rtx_rendering(False)

    assert viz.current_image_view is task


def test_rtx_rendering_needs_the_viewer_camera_declared_at_launch():
    viz = _switchable([_view(PerspectiveCameraCfg())])

    with pytest.raises(RuntimeError, match="no OVRTX viewer camera"):
        viz.set_rtx_rendering(True)
