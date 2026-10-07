# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Presentation teardown releases owned products and preserves shared renderer and sensor lifetimes."""

from unittest.mock import Mock

import isaaclab_visualizers.newton.newton_viewer as newton_viewer
import pytest
from isaaclab_visualizers.newton import (
    NewtonGLVisualizer,
    NewtonGLVisualizerCfg,
    NewtonRTXVisualizer,
    NewtonRTXVisualizerCfg,
)

pytestmark = [pytest.mark.unit]


def test_release_viewer_does_not_close_gl_viewer() -> None:
    """GL teardown must not prevent another viewer from starting in the same process."""
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())
    viewer = visualizer._viewer = Mock()
    visualizer._release_viewer()
    viewer.close.assert_not_called()
    assert visualizer._viewer is None


@pytest.mark.parametrize("requested", [False, True])
def test_gl_close_request_closes_after_frame(monkeypatch: pytest.MonkeyPatch, requested: bool) -> None:
    """A close request must not destroy the GL context inside the UI render callback."""
    events: list[str] = []
    monkeypatch.setattr(newton_viewer.ViewerGL, "end_frame", lambda self: events.append("frame"))
    viewer = object.__new__(newton_viewer.NewtonViewerGL)
    viewer._close_requested = False
    if requested:
        viewer.request_close()
    viewer.renderer = type("Renderer", (), {"close": lambda self: events.append("close")})()

    viewer.end_frame()

    assert events == (["frame", "close"] if requested else ["frame"])


@pytest.mark.parametrize("failure", [None, "product", "window"])
def test_close_releases_owned_resources_even_on_failure(failure):
    """Teardown is idempotent and leaves shared sensors and renderer usable, even when one release fails."""
    renderer = Mock()
    visualizer = NewtonRTXVisualizer(NewtonRTXVisualizerCfg(), renderer=renderer)
    viewer = visualizer._viewer = Mock()
    product = visualizer._render_data = object()
    camera = visualizer._camera_sensor = Mock()
    visualizer._camera_choices = [camera]
    visualizer._scene_stage = object()
    if failure == "product":
        renderer.cleanup.side_effect = RuntimeError("release failed")
    elif failure == "window":
        viewer.close.side_effect = RuntimeError("release failed")
    if failure is None:
        visualizer.close()
    else:
        with pytest.raises(RuntimeError, match="release failed"):
            visualizer.close()
    visualizer.close()

    renderer.cleanup.assert_called_once_with(product)
    viewer.close.assert_called_once()
    renderer.close.assert_not_called()
    camera.close.assert_not_called()
    assert visualizer._viewer is visualizer._render_data is visualizer._renderer is None
    assert visualizer._scene_stage is visualizer._camera_sensor is None
    assert not visualizer._camera_choices
    assert visualizer._is_closed
