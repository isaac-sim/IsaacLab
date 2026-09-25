# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Kit extension utilities."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from isaaclab.sim.utils.extensions import disable_extension, enable_extension, get_extension_path


def _install_kit_app_module(monkeypatch, extension_manager):
    """Install a minimal ``omni.kit.app`` module hierarchy."""
    app = MagicMock()
    app.get_extension_manager.return_value = extension_manager
    app_module = types.ModuleType("omni.kit.app")
    app_module.get_app = MagicMock(return_value=app)
    kit_module = types.ModuleType("omni.kit")
    kit_module.app = app_module
    omni_module = types.ModuleType("omni")
    omni_module.kit = kit_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.kit", kit_module)
    monkeypatch.setitem(sys.modules, "omni.kit.app", app_module)
    return app_module


def test_enable_extension_enables_only_when_needed(monkeypatch):
    """Enable a disabled extension and leave an enabled extension unchanged."""
    extension_manager = MagicMock()
    extension_manager.is_extension_enabled.side_effect = [False, True]
    _install_kit_app_module(monkeypatch, extension_manager)

    enable_extension("example.extension")
    enable_extension("example.extension")

    extension_manager.set_extension_enabled_immediate.assert_called_once_with("example.extension", True)


def test_disable_extension_disables_only_when_needed(monkeypatch):
    """Disable an enabled extension and leave a disabled extension unchanged."""
    extension_manager = MagicMock()
    extension_manager.is_extension_enabled.side_effect = [True, False]
    _install_kit_app_module(monkeypatch, extension_manager)

    disable_extension("example.extension")
    disable_extension("example.extension")

    extension_manager.set_extension_enabled_immediate.assert_called_once_with("example.extension", False)


def test_extension_helpers_require_running_kit_app(monkeypatch):
    """Report a clear error when Kit modules exist without a running application."""
    app_module = _install_kit_app_module(monkeypatch, MagicMock())
    app_module.get_app.return_value = None

    with pytest.raises(RuntimeError, match="no Kit application is running"):
        enable_extension("example.extension")

    app_module.get_app.side_effect = RuntimeError("Failed to acquire interface: omni::kit::IApp")
    with pytest.raises(RuntimeError, match="no Kit application is running"):
        get_extension_path("example.extension")


def test_get_extension_path_resolves_enabled_extension_id(monkeypatch):
    """Resolve the path through the enabled extension identifier."""
    extension_manager = MagicMock()
    extension_manager.get_enabled_extension_id.return_value = "example.extension-1.0.0"
    extension_manager.get_extension_path.return_value = "/extensions/example.extension"
    _install_kit_app_module(monkeypatch, extension_manager)

    result = get_extension_path("example.extension")

    assert result == "/extensions/example.extension"
    extension_manager.get_extension_path.assert_called_once_with("example.extension-1.0.0")
