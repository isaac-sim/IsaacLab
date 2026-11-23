# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tracking Camera views of the Newton GL window."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("newton")
from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer, NewtonVisualizerBase
from pyglet.math import Vec3

from isaaclab.visualizers import PerspectiveCameraCfg, TrackingCameraCfg
from isaaclab.visualizers.camera_follower import CameraFollower

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
    viz._followers = {view: CameraFollower(view.cfg.source, _Scene()) for view in views if viz._is_tracking_view(view)}
    viz.rtx_rendering, viz._rtx_pose_view, viz._perspective_pose = False, None, None
    return viz


def test_a_lone_free_view_is_called_perspective_and_a_task_view_the_tracking_camera():
    free, task = _view(PerspectiveCameraCfg()), _view(TrackingCameraCfg(env_index=0, eye=(2.0, 0.0, 1.0)))
    viz = _visualizer([free, task])

    assert viz._view_labels([0, 1]) == ["Perspective (interactive)", "Tracking Camera (fixed to env)"]
    assert [viz._is_tracking_view(view) for view in viz._image_views] == [False, True]


def test_tracking_cameras_are_numbered_only_when_several_share_a_name():
    views = [
        _view(PerspectiveCameraCfg()),
        _view(TrackingCameraCfg(env_index=0)),
        _view(TrackingCameraCfg(env_index=1)),
    ]
    viz = _visualizer(views)

    assert viz._view_labels([0, 1, 2]) == [
        "Perspective (interactive)",
        "Tracking Camera 1 (fixed to env)",
        "Tracking Camera 2 (fixed to env)",
    ]


def test_scene_cameras_are_named_by_their_prim_without_the_environment_pattern():
    camera = SimpleNamespace(cfg=SimpleNamespace(data_types=["rgba"]))
    sensor = _View("/World/envs/env_[^/]+/Robot/wrist_camera", camera=camera)
    viz = _visualizer([_view(PerspectiveCameraCfg()), sensor])

    assert viz._view_labels([0, 1]) == ["Perspective (interactive)", "wrist_camera (RGB sensor)"]


def test_the_camera_list_puts_perspective_first_then_tracking_cameras_then_sensors():
    sensor = _View("/World/envs/env_[^/]+/Camera", camera=object())
    tracking, free = _view(TrackingCameraCfg(env_index=0)), _view(PerspectiveCameraCfg())
    viz = _visualizer([sensor, tracking, free])

    order = sorted(range(3), key=lambda i: viz._source_rank(viz._image_views[i]))

    assert order == [2, 1, 0]


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

    assert viz._restore_perspective_pose()
    assert (tuple(camera.pos), camera.pitch, camera.yaw, viz.cfg.focal_length) == ((1.0, 2.0, 3.0), -10.0, 45.0, 12.0)


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


def test_a_perspective_that_was_never_left_has_no_pose_to_restore():
    viz = _visualizer([_view(PerspectiveCameraCfg())])

    assert not viz._restore_perspective_pose()


def _rtx_view():
    camera = SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/viewer_camera_0", renderer_cfg="ovrtx"))
    view = _View("/World/viewer_camera_0", camera=camera)
    view.invalidate = lambda: None
    return view


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


def test_picking_a_sensor_while_rtx_is_on_returns_the_rtx_camera_to_rendering_on_demand():
    rtx = _rtx_view()
    sensor = _View("/World/sensor", camera=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/sensor")))
    sensor.invalidate = lambda: None
    viz = _switchable([_view(PerspectiveCameraCfg()), rtx, sensor])
    viz._viewer.picking = None
    viz._viewer.camera.get_view_matrix = lambda: np.eye(4)
    viz._release_rtx_scene = lambda: None
    viz._place_scene_camera_at_viewer = lambda camera: None
    viz._pose_started = False
    viz.current_image_view = rtx
    rtx.camera.cfg.render_on_demand = False  # the shared render while RTX is shown

    NewtonGLVisualizer._select_camera(viz, 2, keep_pose=True)

    assert rtx.camera.cfg.render_on_demand is True
    assert viz.current_image_view is sensor


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


def test_the_camera_speed_slider_sets_the_free_cameras_keyboard_speed():
    free, tracking = _view(PerspectiveCameraCfg()), _view(TrackingCameraCfg(env_index=0))
    viz = _switchable([free, tracking])
    viz._viewer = SimpleNamespace(_camera_speed=5.0, camera_speed=5.0, camera_speed_boost=2.0)
    viz._pose_source_view = lambda: viz.current_image_view
    imgui = SimpleNamespace(
        slider_float=lambda *args: (True, 12.0),
        SliderFlags_=SimpleNamespace(logarithmic=0),
        is_item_hovered=lambda *args: False,
        HoveredFlags_=SimpleNamespace(allow_when_disabled=0),
    )
    viz.current_image_view = tracking
    viz._draw_camera_speed(imgui)
    assert viz._viewer.camera_speed == 5.0  # a tracking camera does not move with the keys

    viz.current_image_view = free
    viz._draw_camera_speed(imgui)
    assert viz._viewer.camera_speed == 12.0 and viz._viewer.camera_speed_boost == 12.0


def test_a_window_with_only_the_rtx_camera_draws_its_camera_controls():
    from unittest.mock import Mock

    viz = _switchable([_rtx_view()])
    viz.current_image_view = viz._image_views[0]
    viz._borrowing_view = False
    viz._draw_appearance = viz._draw_navigation_controls = lambda imgui: None
    viz._draw_camera_info = viz._draw_window_options = lambda imgui: None
    imgui = Mock()
    imgui.is_item_hovered.return_value = False
    imgui.radio_button.return_value = False  # disabled radio buttons are never clicked

    viz._draw_streaming_view_controls(imgui)  # no other source to list: must not raise

    imgui.combo.assert_not_called()
    imgui.begin_disabled.assert_called_once()


def test_camera_view_controls_are_grouped_under_headers_in_order():
    from unittest.mock import Mock

    viz = _switchable([_view(PerspectiveCameraCfg()), _rtx_view()])
    viz._borrowing_view = False
    order = []
    viz._draw_appearance = lambda imgui: order.append("appearance")
    viz._draw_focal_length = lambda imgui: order.append("focal length")
    viz._draw_camera_speed = lambda imgui: order.append("camera speed")
    imgui = Mock()
    imgui.is_item_hovered.return_value = False
    imgui.combo.side_effect = lambda *args: order.append("camera source") or (False, 0)
    imgui.radio_button.side_effect = lambda *args: order.append("renderer") or False
    imgui.separator_text.side_effect = lambda text: order.append(text)
    imgui.checkbox.side_effect = lambda label, value: (False, value)
    viz._viewer.vsync = False
    camera = viz._viewer.camera
    camera.pos, camera.pitch, camera.yaw, camera.fov = (1.0, 2.0, 3.0), 0.0, 0.0, 40.0

    viz._draw_streaming_view_controls(imgui)

    assert order == [
        "Camera Source",
        "camera source",
        "Rendering",
        "renderer",
        "renderer",
        "appearance",
        "Camera Settings",
        "focal length",
        "camera speed",
        "Camera Data",
        "Window",
    ]
    imgui.text_disabled.assert_not_called()  # the camera data is plain text, not greyed out
    assert [call.args[0].split(":")[0] for call in imgui.text.call_args_list] == [
        "Renderer",
        "Position",
        "Rotation",
        "FOV",
    ]


def test_the_vsync_checkbox_sets_the_window_vsync():
    from unittest.mock import Mock

    viz = _switchable([_view(PerspectiveCameraCfg()), _rtx_view()])
    viz._viewer.vsync = False
    imgui = Mock()
    imgui.checkbox.return_value = (True, True)

    viz._draw_window_options(imgui)

    assert viz._viewer.vsync is True


def test_the_docs_link_does_not_block_the_render_thread(monkeypatch):
    import threading

    from isaaclab_visualizers.newton import newton_visualizer

    release, started = threading.Event(), threading.Event()
    monkeypatch.setattr(newton_visualizer.webbrowser, "open", lambda url: (started.set(), release.wait(5)))

    newton_visualizer._open_in_browser("https://example.com")  # returns while the browser call is still blocked

    assert started.wait(5)
    release.set()
