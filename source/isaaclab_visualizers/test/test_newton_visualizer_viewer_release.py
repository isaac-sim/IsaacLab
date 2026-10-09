# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Presentation teardown closes the native viewer and preserves borrowed sensor lifetimes."""

from unittest.mock import Mock

import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import pytest
from isaaclab_visualizers.newton import (
    NewtonGLVisualizer,
    NewtonGLVisualizerCfg,
    NewtonRTXVisualizer,
    NewtonRTXVisualizerCfg,
)

pytestmark = [pytest.mark.unit]


def test_close_does_not_destroy_shared_gl_context() -> None:
    """GL teardown must not prevent another viewer from starting in the same process."""
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())
    viewer = visualizer._viewer = Mock()
    visualizer.close()
    viewer.close.assert_not_called()
    assert visualizer._viewer is None


@pytest.mark.parametrize("requested", [False, True])
def test_gl_close_request_closes_after_frame(monkeypatch: pytest.MonkeyPatch, requested: bool) -> None:
    """A close request must not destroy the GL context inside the UI render callback."""
    events: list[str] = []
    monkeypatch.setattr(newton_visualizer.ViewerGL, "end_frame", lambda self: events.append("frame"))
    viewer = object.__new__(newton_visualizer.NewtonViewerGL)
    viewer._close_requested = False
    if requested:
        viewer.request_close()
    viewer.renderer = type("Renderer", (), {"close": lambda self: events.append("close")})()

    viewer.end_frame()

    assert events == (["frame", "close"] if requested else ["frame"])


@pytest.mark.parametrize("failure", [False, True])
def test_close_releases_owned_resources_even_on_failure(failure):
    """Teardown is idempotent and leaves borrowed sensor references untouched, even when one release fails."""
    visualizer = NewtonRTXVisualizer(NewtonRTXVisualizerCfg())
    viewer = visualizer._viewer = Mock()
    camera = Mock()
    visualizer.image_view = Mock(camera=camera)
    visualizer._image_views = [visualizer.image_view]
    visualizer._sim = Mock()
    if failure:
        viewer.close.side_effect = RuntimeError("release failed")
    if not failure:
        visualizer.close()
    else:
        with pytest.raises(RuntimeError, match="release failed"):
            visualizer.close()
    visualizer.close()

    viewer.close.assert_called_once()
    camera.close.assert_not_called()
    assert visualizer._viewer is visualizer.backend is None
    assert visualizer._sim is visualizer.image_view is None
    assert not visualizer._image_views
    assert visualizer._is_closed
