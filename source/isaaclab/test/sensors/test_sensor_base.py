# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors import SensorBase


class _TestSensor(SensorBase):
    @property
    def data(self):
        return None

    def _initialize_impl(self):
        pass

    def _update_buffers_impl(self, env_mask):
        pass


def test_sensor_base_destructor_clears_before_shutdown(monkeypatch):
    """Sensor destructors should still clean up during normal runtime."""

    cleared = False

    def clear_callbacks(_self):
        nonlocal cleared
        cleared = True

    sensor = object.__new__(_TestSensor)
    monkeypatch.setattr(_TestSensor, "_clear_callbacks", clear_callbacks)

    sensor.__del__()

    assert cleared


def test_sensor_base_destructor_skips_clear_after_import_shutdown(monkeypatch):
    """Sensor destructors should not import modules after import machinery is torn down."""

    def clear_callbacks(_self):
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    sensor = object.__new__(_TestSensor)
    monkeypatch.setattr(_TestSensor, "_clear_callbacks", clear_callbacks)
    monkeypatch.setattr("sys.meta_path", None)

    sensor.__del__()
