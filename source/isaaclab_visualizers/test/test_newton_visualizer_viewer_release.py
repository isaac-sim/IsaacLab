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


def test_close_does_not_destroy_shared_gl_context() -> None:
    """GL teardown must not prevent another viewer from starting in the same process."""
    visualizer = NewtonGLVisualizer(VisualizerCfg(backend=NewtonGLBackendCfg()))
    viewer = visualizer._viewer = Mock()
    visualizer.close()
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
