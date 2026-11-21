# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the camera sources of the Newton GL window."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("newton")
from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonVisualizerBase
from pyglet.math import Vec3

from isaaclab.visualizers import PerspectiveCameraCfg

pytestmark = pytest.mark.kitless


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
    """An image view; a class, not a namespace, so that it can key the pose dictionary."""

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
    viz.rtx_rendering, viz._perspective_pose = False, None
    return viz


def test_a_lone_free_view_is_called_perspective():
    viz = _visualizer([_view(PerspectiveCameraCfg())])

    assert viz._view_labels([0]) == ["Perspective"]


def test_scene_cameras_are_named_by_their_prim_without_the_environment_pattern():
    camera = SimpleNamespace(cfg=SimpleNamespace(data_types=["rgba"]))
    sensor = _View("/World/envs/env_[^/]+/Robot/wrist_camera", camera=camera)
    viz = _visualizer([_view(PerspectiveCameraCfg()), sensor])

    assert viz._view_labels([0, 1]) == ["Perspective", "wrist_camera (RGB sensor)"]


def test_the_camera_list_puts_perspective_first_then_sensors():
    sensor = _View("/World/envs/env_[^/]+/Camera", camera=object())
    free = _view(PerspectiveCameraCfg())
    viz = _visualizer([sensor, free])

    order = sorted(range(2), key=lambda i: viz._source_rank(viz._image_views[i]))

    assert order == [1, 0]


def test_the_free_perspective_pose_is_kept_while_another_source_is_shown():
    free = _view(PerspectiveCameraCfg())
    viz = _visualizer([free])
    viz._focal_length_to_vertical_fov_degrees = lambda focal=None: 40.0
    viz.current_image_view = free
    viz._viewer.camera.fov = 0.0
    camera = viz._viewer.camera
    camera.pos, camera.pitch, camera.yaw = (1.0, 2.0, 3.0), -10.0, 45.0

    viz._remember_perspective_pose()
    camera.pos, camera.pitch, camera.yaw = (9.0, 9.0, 9.0), 0.0, 0.0  # A sensor moved it.
    viz.cfg.focal_length = 50.0

    assert viz._restore_perspective_pose()
    assert (tuple(camera.pos), camera.pitch, camera.yaw, viz.cfg.focal_length) == ((1.0, 2.0, 3.0), -10.0, 45.0, 12.0)


def test_a_perspective_that_was_never_left_has_no_pose_to_restore():
    viz = _visualizer([_view(PerspectiveCameraCfg())])

    assert not viz._restore_perspective_pose()


def _rtx_view():
    camera = SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/viewer_camera_0", renderer_cfg="ovrtx"))
    return _View("/World/viewer_camera_0", camera=camera)


def _switchable(views):
    """A visualizer whose camera selection only records which view became current."""
    base = _visualizer(views)
    viz = NewtonGLVisualizer.__new__(NewtonGLVisualizer)
    viz.__dict__.update(base.__dict__)
    viz.cfg = SimpleNamespace(backend=SimpleNamespace(rtx_camera_path="/World/viewer_camera_0"))
    viz._overlay_publisher = SimpleNamespace(hide_all=lambda renderers: viz.hidden.append(renderers))
    viz._sim.get_or_create_backend = lambda cfg: cfg
    viz.hidden, viz.current_image_view = [], views[0]

    viz.keep_pose = []

    def select(index, keep_pose=False):
        viz.keep_pose.append(keep_pose)
        viz.current_image_view = viz._image_views[index]

    viz._select_camera = select
    viz._apply_rtx_visible_envs = lambda: None
    return viz


def test_switching_to_rtx_and_back_returns_to_the_perspective_view():
    free, rtx = _view(PerspectiveCameraCfg()), _rtx_view()
    viz = _switchable([free, rtx])

    viz.set_rtx_rendering(True)
    assert viz.current_image_view is rtx and viz.rtx_rendering

    viz.set_rtx_rendering(False)
    assert viz.current_image_view is free and not viz.rtx_rendering


def test_choosing_the_perspective_while_rtx_is_shown_keeps_the_rtx_view():
    free, rtx = _view(PerspectiveCameraCfg()), _rtx_view()
    viz = _switchable([free, rtx])
    viz.set_rtx_rendering(True)

    viz._choose_camera_source(0)  # the OVRTX camera already follows the Perspective

    assert viz.current_image_view is rtx and viz.rtx_rendering


def test_the_rtx_switch_waits_a_frame_so_the_message_is_shown_first():
    viz = _switchable([_view(PerspectiveCameraCfg()), _rtx_view()])
    viz._rtx_switch = "requested"
    switched = []
    viz.set_rtx_rendering = lambda enabled: switched.append((enabled, viz._rtx_switch))
    viz._rtx_appearance = None
    viz._viewer.begin_frame = lambda t: (_ for _ in ()).throw(RuntimeError("stop"))
    viz._sim_time = 0.0

    with pytest.raises(RuntimeError, match="stop"):
        NewtonGLVisualizer._render_frame(viz)

    assert viz._rtx_switch is None  # a failed switch does not leave the message up
    assert switched == [(True, "loading")]  # the HUD already showed the message for the previous frame


def test_a_request_made_while_drawing_the_hud_survives_to_the_next_frame():
    free = _view(PerspectiveCameraCfg())
    free.invalidate = lambda: None
    viz = _switchable([free, _rtx_view()])
    viz._rtx_appearance, viz._sim_time = None, 0.0
    viz._pending_mesh_submissions = {}
    viz._last_render_step = 0
    viz._sim.get_physics_step_count = lambda: 0

    def end_frame():  # the HUD is drawn here, and the RTX button was clicked
        viz._rtx_switch = "requested"

    viz._viewer = SimpleNamespace(
        begin_frame=lambda t: None,
        end_frame=end_frame,
        is_paused=lambda: True,
        is_running=lambda: True,
    )

    NewtonGLVisualizer._render_frame(viz)

    assert viz._rtx_switch == "requested"


def test_leaving_rtx_hides_the_overlays_and_suspends_the_viewer_lights():
    viz = _switchable([_view(PerspectiveCameraCfg()), _rtx_view()])
    suspended = []
    viz._rtx_appearance = SimpleNamespace(available=True, suspend=lambda: suspended.append(True))

    NewtonGLVisualizer._release_rtx_scene(viz)

    assert viz.hidden == [("ovrtx",)] and suspended == [True]


def test_leaving_rtx_without_a_free_perspective_view_shows_the_first_other_source():
    sensor = _View("sensor", camera=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/envs/env_0/Camera")))
    rtx = _rtx_view()
    viz = _switchable([sensor, rtx])
    viz.current_image_view = rtx

    viz.set_rtx_rendering(False)

    assert viz.current_image_view is sensor
    assert viz.keep_pose == [False]  # A scene sensor is not moved to the viewer's pose.


def test_meshes_reach_the_ovrtx_scene_camera_a_standalone_window_displays():
    ovrtx = SimpleNamespace(renderer_type="ovrtx")
    sensor = _View("sensor", camera=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/Camera", renderer_cfg=ovrtx)))
    other = _View("other", camera=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/Other", renderer_cfg=ovrtx)))
    viz = _switchable([sensor, other])
    viz._sim.get_or_create_backend = lambda cfg: cfg

    assert viz.rtx_view_index is None
    assert viz._rtx_renderers == (ovrtx,)  # only the displayed sensor, not every scene camera


def test_rtx_rendering_needs_the_viewer_camera_declared_at_launch():
    viz = _switchable([_view(PerspectiveCameraCfg())])

    with pytest.raises(RuntimeError, match="no OVRTX viewer camera"):
        viz.set_rtx_rendering(True)


def test_the_shared_renderer_hides_environments_only_while_the_rtx_view_is_shown():
    calls = []
    renderer = SimpleNamespace(
        _env_paths=["a", "b", "c"], set_environment_visibility=lambda v: calls.append(v.tolist())
    )
    rtx = _rtx_view()
    rtx.invalidate = lambda: None
    viz = _switchable([_view(PerspectiveCameraCfg()), rtx])
    viz._apply_rtx_visible_envs = NewtonGLVisualizer._apply_rtx_visible_envs.__get__(viz)
    viz._sim.get_or_create_backend = lambda cfg: renderer
    viz._env_ids = [0]

    viz._apply_rtx_visible_envs()  # OpenGL is shown, so the renderer the sensors share is left alone
    assert calls == []
    viz.current_image_view = rtx
    viz._apply_rtx_visible_envs()
    viz.current_image_view = viz._image_views[0]
    viz._apply_rtx_visible_envs()  # leaving RTX lifts the limit once

    assert calls == [[True, False, False], [True, True, True]]


def test_the_standalone_rtx_window_has_no_opengl_view_state():
    viz = NewtonVisualizerBase.__new__(NewtonVisualizerBase)

    assert viz.rtx_view_index is None and not viz.rtx_rendering and not viz.sensor_view_selected


def test_a_free_view_records_the_rtx_window_instead_of_an_opengl_render():
    free, rtx = _view(PerspectiveCameraCfg()), _rtx_view()
    viz = _switchable([free, rtx])
    viz.current_image_view = rtx
    viz._capture_frame = lambda output: "window frame"

    assert viz._capture_view(free, None) == "window frame"
    assert viz.current_image_view is rtx
