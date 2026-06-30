# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for settings published by :class:`isaaclab.app.AppLauncher`."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

import isaaclab.app.app_launcher as app_launcher_module
from isaaclab.app.app_launcher import AppLauncher


@pytest.mark.parametrize(
    ("headless", "livestream", "xr", "expected_has_gui"),
    [
        pytest.param(False, 0, False, True, id="local-window"),
        pytest.param(True, 0, False, False, id="headless"),
        pytest.param(True, 1, False, True, id="livestream"),
        pytest.param(True, 0, True, True, id="xr"),
    ],
)
def test_load_extensions_publishes_has_gui_setting(monkeypatch, headless, livestream, xr, expected_has_gui):
    """Publish the GUI state consumed by SimulationContext and RTX rendering."""
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._apply_rtx_determinism = False
    launcher._python_logging_level = logging.ERROR
    launcher._headless = headless
    launcher._livestream = livestream
    launcher._enable_cameras = False
    launcher._offscreen_render = False
    launcher._render_viewport = False
    launcher._xr = xr
    launcher._video_enabled = False

    settings = MagicMock()
    monkeypatch.setattr(app_launcher_module, "initialize_carb_settings", MagicMock())
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", MagicMock(return_value=settings))

    with patch.object(AppLauncher, "_apply_python_logging_level"):
        launcher._load_extensions()

    settings.set_bool.assert_any_call("/isaaclab/has_gui", expected_has_gui)
