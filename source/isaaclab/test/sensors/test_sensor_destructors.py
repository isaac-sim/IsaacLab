# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from isaaclab.sensors.camera.camera import Camera
from isaaclab.sensors.sensor_base import SensorBase

pytestmark = pytest.mark.unit


class _DummySensor(SensorBase):
    """Minimal concrete sensor used only to invoke ``SensorBase.__del__``."""

    @property
    def data(self):
        return None

    def _initialize_impl(self):
        return None

    def _update_buffers_impl(self, env_ids):
        return None


def test_sensor_destructor_clears_callbacks_before_shutdown(monkeypatch):
    """Sensor destructors should still unregister callbacks during normal runtime."""

    cleared = False

    def clear_callbacks(_self):
        nonlocal cleared
        cleared = True

    sensor = object.__new__(_DummySensor)
    monkeypatch.setattr(SensorBase, "_clear_callbacks", clear_callbacks)

    sensor.__del__()

    assert cleared


def test_sensor_destructor_skips_clear_after_import_shutdown(monkeypatch):
    """Sensor destructors should not run cleanup after import machinery is torn down."""

    def clear_callbacks(_self):
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    sensor = object.__new__(_DummySensor)
    monkeypatch.setattr(SensorBase, "_clear_callbacks", clear_callbacks)
    monkeypatch.setattr("sys.meta_path", None)

    sensor.__del__()


def test_camera_destructor_cleans_renderer_before_shutdown(monkeypatch):
    """Camera destructors should still release renderer resources during normal runtime."""

    class _View:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Renderer:
        def __init__(self):
            self.cleaned = False

        def cleanup(self, _render_data):
            self.cleaned = True

    view = _View()
    renderer = _Renderer()
    camera = object.__new__(Camera)
    camera._view = view
    camera._renderer = renderer
    camera._render_data = None
    monkeypatch.setattr(SensorBase, "_clear_callbacks", lambda _self: None)

    camera.__del__()

    assert view.closed
    assert renderer.cleaned
    assert camera._view is None


def test_camera_destructor_skips_cleanup_after_import_shutdown(monkeypatch):
    """Camera destructors should not run cleanup after import machinery is torn down."""

    class _Boom:
        def close(self):
            raise ImportError("sys.meta_path is None, Python is likely shutting down")

        def cleanup(self, _render_data):
            raise ImportError("sys.meta_path is None, Python is likely shutting down")

    camera = object.__new__(Camera)
    camera._view = _Boom()
    camera._renderer = _Boom()
    camera._render_data = None
    monkeypatch.setattr("sys.meta_path", None)

    camera.__del__()
