# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Kit viewport helpers."""

import sys
import types
from unittest.mock import MagicMock

from isaaclab_physx.renderers.kit_viewport_utils import _set_kit_camera_view


def test_set_kit_camera_view_uses_viewport_utility(monkeypatch):
    """Set the camera pose without importing Isaac Sim's rendering manager."""
    viewport_api = object()
    camera_state = MagicMock()
    camera_state_type = MagicMock(return_value=camera_state)
    utility_module = types.ModuleType("omni.kit.viewport.utility")
    utility_module.get_active_viewport = MagicMock(return_value=viewport_api)
    camera_state_module = types.ModuleType("omni.kit.viewport.utility.camera_state")
    camera_state_module.ViewportCameraState = camera_state_type
    gf_module = MagicMock()
    gf_module.Vec3d.side_effect = lambda *values: values
    pxr_module = types.ModuleType("pxr")
    pxr_module.Gf = gf_module

    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", utility_module)
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility.camera_state", camera_state_module)
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    _set_kit_camera_view((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), "/Camera")

    camera_state_type.assert_called_once_with("/Camera", viewport_api)
    camera_state.set_position_world.assert_called_once_with((1.0, 2.0, 3.0), True)
    camera_state.set_target_world.assert_called_once_with((4.0, 5.0, 6.0), True)


def test_set_kit_camera_view_requires_active_viewport(monkeypatch):
    """Raise a clear error when Kit has no active viewport."""
    utility_module = types.ModuleType("omni.kit.viewport.utility")
    utility_module.get_active_viewport = MagicMock(return_value=None)
    camera_state_module = types.ModuleType("omni.kit.viewport.utility.camera_state")
    camera_state_module.ViewportCameraState = MagicMock()
    pxr_module = types.ModuleType("pxr")
    pxr_module.Gf = MagicMock()

    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", utility_module)
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility.camera_state", camera_state_module)
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    try:
        _set_kit_camera_view((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), "/Camera")
    except RuntimeError as exc:
        assert str(exc) == "No active Kit viewport is available."
    else:
        raise AssertionError("Expected _set_kit_camera_view to require an active viewport.")
