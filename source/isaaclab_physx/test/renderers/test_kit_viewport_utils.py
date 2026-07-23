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
    camera_state.set_position_world.assert_called_once_with((1.0, 2.0, 3.0), False)
    camera_state.set_target_world.assert_called_once_with((4.0, 5.0, 6.0), True)


def test_set_kit_camera_view_does_not_require_authored_center_of_interest(monkeypatch):
    """Set the camera position without rotating around an unauthored center of interest."""

    class _CameraState:
        def __init__(self, camera_path, viewport_api):
            self.position_calls = []
            self.target_calls = []
            camera_state_holder["state"] = self

        def set_position_world(self, world_position, rotate):
            if rotate:
                raise TypeError("Matrix4d.Transform(Matrix4d, NoneType)")
            self.position_calls.append((world_position, rotate))

        def set_target_world(self, world_target, rotate):
            self.target_calls.append((world_target, rotate))

    camera_state_holder = {}
    utility_module = types.ModuleType("omni.kit.viewport.utility")
    utility_module.get_active_viewport = MagicMock(return_value=object())
    camera_state_module = types.ModuleType("omni.kit.viewport.utility.camera_state")
    camera_state_module.ViewportCameraState = _CameraState
    gf_module = MagicMock()
    gf_module.Vec3d.side_effect = lambda *values: values
    pxr_module = types.ModuleType("pxr")
    pxr_module.Gf = gf_module

    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", utility_module)
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility.camera_state", camera_state_module)
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    _set_kit_camera_view((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), "/OmniverseKit_Persp")

    state = camera_state_holder["state"]
    assert state.position_calls == [((1.0, 2.0, 3.0), False)]
    assert state.target_calls == [((4.0, 5.0, 6.0), True)]


def test_set_kit_camera_view_accepts_vertical_eye_target(monkeypatch):
    """Accept cameras directly above or below their target."""
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

    _set_kit_camera_view((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), "/Camera")
    _set_kit_camera_view((0.0, 0.0, -5.0), (0.0, 0.0, 0.0), "/Camera")

    assert camera_state.set_position_world.call_args_list == [
        (((0.0, 0.0, 5.0), False),),
        (((0.0, 0.0, -5.0), False),),
    ]
    assert camera_state.set_target_world.call_args_list == [
        (((0.0, 0.0, 0.0), True),),
        (((0.0, 0.0, 0.0), True),),
    ]


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
