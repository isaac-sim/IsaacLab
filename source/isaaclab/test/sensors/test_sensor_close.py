# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the sensor teardown contract used at simulation shutdown.

Instances are built with ``object.__new__`` and only the attributes ``close`` touches are set,
so these run without a simulation app.
"""

from __future__ import annotations

from typing import Any

import pytest

from isaaclab.scene import InteractiveScene
from isaaclab.sensors.camera.camera import Camera
from isaaclab.sensors.sensor_base import SensorBase

pytestmark = pytest.mark.unit


class _FakeRenderer:
    """Records the render data it was asked to release."""

    def __init__(self, cleanup_log: list[Any] | None = None, raises: bool = False) -> None:
        self.cleanup_log = cleanup_log if cleanup_log is not None else []
        self.raises = raises

    def cleanup(self, render_data: Any) -> None:
        self.cleanup_log.append(render_data)
        if self.raises:
            raise RuntimeError("backend cleanup failed")


def _camera(renderer: _FakeRenderer, render_data: Any) -> Camera:
    camera = object.__new__(Camera)
    camera._renderer = renderer
    camera._render_data = render_data
    # Handles that SensorBase.__del__ clears when the instance is collected.
    camera._initialize_handle = None
    camera._invalidate_initialize_handle = None
    camera._prim_deletion_handle = None
    camera._debug_vis_handle = None
    return camera


def _scene(sensors: dict[str, Any]) -> InteractiveScene:
    scene = object.__new__(InteractiveScene)
    scene._sensors = sensors
    return scene


def test_camera_close_releases_render_data():
    """close hands the render data back to the renderer that created it."""
    renderer = _FakeRenderer()
    data = object()

    _camera(renderer, data).close()

    assert renderer.cleanup_log == [data]


def test_camera_close_releases_while_a_reference_survives():
    """close releases even though the caller still holds the camera.

    Regression test for the shutdown crash: a camera kept in a task attribute (in addition to
    the scene's own entry) never reached a zero reference count, so its finalizer never ran and
    its renderer resources were still registered when the app tore down.
    """
    renderer = _FakeRenderer()
    data = object()
    camera = _camera(renderer, data)
    surviving_reference = camera

    camera.close()

    assert renderer.cleanup_log == [data]
    assert surviving_reference is camera


def test_camera_close_is_idempotent():
    """A second close does not release the same render data twice."""
    renderer = _FakeRenderer()
    camera = _camera(renderer, object())

    camera.close()
    camera.close()

    assert len(renderer.cleanup_log) == 1


def test_camera_close_without_render_data_is_a_no_op():
    """A camera that never initialized has nothing to release."""
    renderer = _FakeRenderer()

    _camera(renderer, None).close()

    assert renderer.cleanup_log == []


def test_scene_close_closes_every_sensor():
    """close reaches all registered sensors, not just the first."""
    log: list[Any] = []
    first, second = _camera(_FakeRenderer(log), "a"), _camera(_FakeRenderer(log), "b")

    _scene({"first": first, "second": second}).close()

    assert sorted(log) == ["a", "b"]


def test_scene_close_continues_after_a_sensor_raises():
    """One sensor failing to close does not strand the others."""
    log: list[Any] = []
    failing = _camera(_FakeRenderer(raises=True), "boom")
    healthy = _camera(_FakeRenderer(log), "ok")

    _scene({"failing": failing, "healthy": healthy}).close()

    assert log == ["ok"]


def test_sensor_base_close_is_a_no_op_by_default():
    """Sensors without simulator-side resources need not override close."""
    SensorBase.close(object())
