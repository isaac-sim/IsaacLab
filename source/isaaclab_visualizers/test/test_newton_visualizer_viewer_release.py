# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Presentation teardown leaves the shared GL context alone and closes the viewer after a frame."""

from unittest.mock import Mock

import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import pytest
from isaaclab_visualizers.newton import (
    NewtonGLBackendCfg,
    NewtonGLVisualizer,
)

from isaaclab.visualizers import VisualizerCfg

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize(("platform", "window_open"), [("linux", True), ("win32", True), ("win32", False)])
def test_close_does_not_destroy_shared_gl_context(
    monkeypatch: pytest.MonkeyPatch, platform: str, window_open: bool
) -> None:
    """Closing a live Windows window releases focus before dropping the viewer."""
    monkeypatch.setattr(newton_visualizer.sys, "platform", platform)
    visualizer = NewtonGLVisualizer(VisualizerCfg(backend=NewtonGLBackendCfg()))
    viewer = visualizer._viewer = Mock()
    window = viewer.renderer.window
    if not window_open:
        window.context = None

    visualizer.close()

    if platform == "win32" and window_open:
        window.set_visible.assert_called_once_with(False)
    else:
        window.set_visible.assert_not_called()
    viewer.close.assert_not_called()
    assert visualizer._viewer is None


@pytest.mark.parametrize("requested", [False, True])
def test_gl_close_request_closes_after_frame(monkeypatch: pytest.MonkeyPatch, requested: bool) -> None:
    """A close request must not destroy the GL context inside the UI render callback."""
    events: list[str] = []
    monkeypatch.setattr(newton_visualizer.ViewerGL, "end_frame", lambda self: events.append("frame"))
    monkeypatch.setattr(newton_visualizer.ViewerGL, "close", lambda self: events.append("close"))
    viewer = object.__new__(newton_visualizer.NewtonViewerGL)
    viewer._close_requested = False
    viewer.keyboard_frame_hook = lambda: events.append("keyboard")
    if requested:
        viewer.request_close()
    viewer.renderer = Mock()
    viewer.renderer.window.switch_to.side_effect = lambda: events.append("context")

    viewer.end_frame()

    assert events == (["frame", "keyboard", "context", "close"] if requested else ["frame", "keyboard"])


def test_close_lifts_the_visible_environment_limit_of_the_shared_renderer():
    """Closing while RTX limits the environments restores them for the scene cameras and still finishes."""
    visualizer = NewtonGLVisualizer(VisualizerCfg(backend=NewtonGLBackendCfg()))
    visualizer._viewer = Mock()
    visualizer._rtx_visibility_limited = True
    visualizer._env_ids = [0]
    seen = []
    visualizer._apply_rtx_visible_envs = lambda: seen.append(visualizer._env_ids)

    visualizer.close()

    assert seen == [None] and visualizer._is_closed


def test_close_unbinds_the_perspective_views_it_rendered():
    """Perspective views bound as per-view captures are released on close, and sensor views are left alone."""
    from functools import partial

    visualizer = NewtonGLVisualizer(VisualizerCfg(backend=NewtonGLBackendCfg()))
    visualizer._viewer = Mock()
    perspective = Mock(camera=None)
    perspective.render = partial(visualizer._capture_view, perspective)
    sensor = Mock(render=None)
    visualizer._image_views = [perspective, sensor]

    visualizer.close()

    assert perspective.render is None
    perspective.invalidate.assert_called_once()
    sensor.invalidate.assert_not_called()


def test_window_controls_do_not_need_a_capture_to_have_run():
    """A fresh visualizer is not borrowing a view, so its window controls draw before the first capture."""
    assert NewtonGLVisualizer(VisualizerCfg(backend=NewtonGLBackendCfg()))._borrowing_view is False


def test_window_without_the_rtx_camera_shows_a_disabled_renderer_toggle():
    """Without the OVRTX viewer camera the RTX choice is greyed out, with why it is opt-in and how to enable it."""
    visualizer = NewtonGLVisualizer(VisualizerCfg(backend=NewtonGLBackendCfg()))
    imgui = Mock()
    imgui.is_item_hovered.return_value = True

    visualizer._draw_rtx_unavailable(imgui)

    imgui.begin_disabled.assert_called_once()
    imgui.end_disabled.assert_called_once()
    assert [call.args[0] for call in imgui.radio_button.call_args_list] == ["OpenGL", "RTX"]
    assert "opt-in" in imgui.set_tooltip.call_args.args[0]
    assert any("--viz newton_rtx" in str(call.args) for call in imgui.text_wrapped.call_args_list)
