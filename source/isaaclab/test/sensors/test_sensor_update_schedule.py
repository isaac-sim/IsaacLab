# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for when :class:`~isaaclab.sensors.SensorBase` refreshes its buffers.

The per-env sensor clocks are float32, so the runs are long enough for their rounding error to exceed
the ``1e-6`` due tolerance.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest
import warp as wp

from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit


def test_native_sensors_preserve_shared_environment_count():
    """Native bindings validate coverage; they must not resize SensorBase's environment rows and clocks."""
    for directory in (ISAACLAB_ROOT / "source").glob("isaaclab_*/isaaclab_*/sensors"):
        for path in directory.rglob("*.py"):
            if path.name.endswith("_data.py"):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                    assert node.attr != "_num_envs", f"{path}:{node.lineno} overrides SensorBase environment ownership"


class _RefreshLogSensor(SensorBase):
    """Sensor that records the step index of every buffer refresh, per environment."""

    def __init__(self, update_period: float, device: str, num_envs: int = 2, requires_physics_step: bool = False):
        # SensorBase.__init__ needs a USD stage; only the timing state is under test here.
        self.cfg = SensorBaseCfg(prim_path="/World/envs/env_.*/sensor", update_period=update_period)
        self._device = device
        self._num_envs = num_envs
        self._requires_physics_step = requires_physics_step
        self._is_initialized = True
        self._is_visualizing = False
        self._create_timing_buffers()
        self._data_dirty = True
        self.step = 0
        self.refreshes = [[] for _ in range(num_envs)]

    def __del__(self):
        pass

    @property
    def data(self):
        self._update_outdated_buffers()

    def _initialize_impl(self):
        pass

    def _update_buffers_impl(self, env_mask: wp.array):
        timestamps = self._timestamp.numpy() if self._requires_physics_step else None
        for env in np.flatnonzero(env_mask.numpy()):
            if self._requires_physics_step and timestamps[env] == 0.0:
                continue
            self.refreshes[env].append(self.step)


def _run(sensor: _RefreshLogSensor, dts: list[float], read_every: int = 1) -> list[list[int]]:
    """Calls ``update(dt)`` for each step and reads the data every ``read_every`` steps."""
    for step, dt in enumerate(dts, start=1):
        sensor.update(dt)
        if step % read_every == 0:
            sensor.step = step
            _ = sensor.data
    return sensor.refreshes


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_refresh_cadence_holds_as_the_clock_ages(device):
    """A 4-step period refreshes every 4 steps for 20 s (height scanner of the locomotion tasks)."""
    sensor = _RefreshLogSensor(update_period=0.02, device=device)
    refreshes = _run(sensor, [0.005] * 4000)
    assert refreshes == [list(range(1, 4001, 4))] * 2


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_unread_sensor_stays_due_until_read(device):
    """The period restarts at the read that refreshed the sensor, not on a fixed time grid."""
    # reads every 3 steps leave a 4-step sensor due and unread for 2 steps, so it refreshes every 6 steps
    sensor = _RefreshLogSensor(update_period=0.02, device=device)
    refreshes = _run(sensor, [0.005] * 4000, read_every=3)
    assert refreshes == [list(range(3, 4001, 6))] * 2


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_refresh_follows_the_time_passed_to_update(device):
    """The period is measured in the ``dt`` passed to each update, not in a number of updates."""
    sensor = _RefreshLogSensor(update_period=0.02, device=device)
    refreshes = _run(sensor, [0.005, 0.015] * 2000)
    assert refreshes == [list(range(1, 4001, 2))] * 2


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("requires_physics_step", [False, True], ids=["capture-on-reset", "capture-after-step"])
def test_reset_read_only_restarts_period_after_capture(device, requires_physics_step):
    """A skipped reset read leaves the sensor due; a completed capture starts its normal period."""
    sensor = _RefreshLogSensor(update_period=0.02, device=device, requires_physics_step=requires_physics_step)
    _run(sensor, [0.005])
    sensor.reset(env_mask=wp.array([True, False], dtype=wp.bool, device=device))
    _ = sensor.data
    for step in range(2, 11):
        sensor.step = step
        sensor.update(0.005)
        _ = sensor.data
    assert sensor.refreshes[0] == ([1, 2, 6, 10] if requires_physics_step else [1, 1, 5, 9])
    assert sensor.refreshes[1] == [1, 5, 9]
