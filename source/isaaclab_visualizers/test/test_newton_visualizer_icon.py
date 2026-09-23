# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :func:`isaaclab_visualizers.newton.newton_visualizer._apply_newton_icon`.

``_apply_newton_icon`` is a temporary Isaac Lab-side workaround for ``ViewerRTX`` never setting
its own window icon (see the function's docstring). It narrows exception suppression so that a
failure loading Newton's bundled icon (for example if Newton's internal ``_src`` path layout
changes) is logged rather than silently swallowed, while a failure from ``window.set_icon``
itself (which can legitimately happen on headless/EGL windows) stays silent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import pyglet.util
import pytest

pytestmark = [pytest.mark.unit]


def test_apply_newton_icon_swallows_set_icon_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``window.set_icon`` failure (e.g. on a headless/EGL window) must stay silent."""
    fake_images = [object(), object(), object()]
    monkeypatch.setattr(newton_visualizer, "_load_newton_icon_images", lambda: fake_images)
    window = MagicMock()
    window.set_icon.side_effect = RuntimeError("no display")

    newton_visualizer._apply_newton_icon(window)  # must not raise

    window.set_icon.assert_called_once_with(*fake_images)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("Newton's bundled icon directory could not be located."),
        OSError("truncated file"),
        pyglet.util.DecodeException("No decoders were able to decode the image"),
    ],
    ids=["missing-icon-dir", "os-error", "pyglet-decode-error"],
)
def test_apply_newton_icon_logs_and_does_not_call_set_icon_when_loading_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    """A failure loading Newton's bundled icon must be logged, not silently swallowed.

    Covers pyglet.util.DecodeException specifically: pyglet.image.load() raises it (not an
    OSError) when none of its codecs can decode a corrupt or unsupported icon file, and it must
    not escape uncaught out of NewtonViewerRTX._init_window() for what is meant to be a purely
    cosmetic, best-effort icon fix.
    """

    def _raise() -> list:
        raise error

    monkeypatch.setattr(newton_visualizer, "_load_newton_icon_images", _raise)
    window = MagicMock()

    with caplog.at_level("DEBUG", logger=newton_visualizer.logger.name):
        newton_visualizer._apply_newton_icon(window)  # must not raise

    window.set_icon.assert_not_called()
    assert any("Could not load Newton's bundled icon" in record.message for record in caplog.records)
