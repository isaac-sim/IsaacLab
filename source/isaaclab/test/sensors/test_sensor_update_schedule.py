# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for when :class:`~isaaclab.sensors.SensorBase` refreshes its buffers.

The per-env sensor clocks are float32, so the runs are long enough for their rounding error to exceed
the ``1e-6`` due tolerance.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import warp as wp

from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit


class _RefreshLogSensor(SensorBase):
    """Sensor that records the step index of every buffer refresh, per environment."""

    def __init__(self, update_period: float, device: str, num_envs: int = 2):
        # SensorBase.__init__ needs a USD stage; only the timing state is under test here.
        self.cfg = SensorBaseCfg(prim_path="/World/envs/env_.*/sensor", update_period=update_period)
        self._device = device
        self._num_envs = num_envs
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
        for env in np.flatnonzero(env_mask.numpy()):
            self.refreshes[env].append(self.step)


class _SkippingLogSensor(_RefreshLogSensor):
    """Sensor that skips environments not stepped since their last reset.

    The contact, IMU, PVA and joint wrench sensors do the same. It also records the time since the last refresh
    seen by each refresh, which contact sensors add to their air and contact times.
    """

    _skips_unstepped_envs: ClassVar[bool] = True

    def __init__(self, update_period: float, device: str, num_envs: int = 2):
        super().__init__(update_period, device, num_envs)
        self.since_last_refresh = [[] for _ in range(num_envs)]

    def _update_buffers_impl(self, env_mask: wp.array):
        stepped = self._timestamp.numpy() > 0.0
        elapsed = self._elapsed_since_update.numpy()
        for env in np.flatnonzero(env_mask.numpy() & stepped):
            self.refreshes[env].append(self.step)
            self.since_last_refresh[env].append(round(float(elapsed[env]), 6))


def _run(sensor: _RefreshLogSensor, dts: list[float], read_every: int = 1, start: int = 1) -> list[list[int]]:
    """Calls ``update(dt)`` for each step, numbered from ``start``, and reads the data every ``read_every`` steps."""
    for step, dt in enumerate(dts, start=start):
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


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_skipped_read_after_reset_keeps_sensor_due(device):
    """A read between a reset and the next step that captured nothing leaves the sensor due.

    Sensors that skip environments not stepped since their last reset capture nothing at that read, so they
    refresh at the first read after the next step instead of holding their reset values for another period.
    Their period still counts from the reset, so the later refreshes match those of sensors that capture at the
    reset. Another reset clears this, so the period then counts from the first refresh after that reset.
    """
    capturing = _RefreshLogSensor(update_period=0.02, device=device, num_envs=3)
    skipping = _SkippingLogSensor(update_period=0.02, device=device, num_envs=3)
    for sensor in (capturing, skipping):
        _run(sensor, [0.005] * 6)
        sensor.reset(env_mask=wp.array([True, False, True], dtype=wp.bool, device=device))
        _ = sensor.data  # the observations read right after the reset, still at step 6
        sensor.reset(env_mask=wp.array([False, False, True], dtype=wp.bool, device=device))  # not read
        _run(sensor, [0.005] * 8, start=7)
    assert capturing.refreshes == [[1, 5, 6, 10, 14], [1, 5, 9, 13], [1, 5, 6, 7, 11]]
    assert skipping.refreshes == [[1, 5, 7, 10, 14], [1, 5, 9, 13], [1, 5, 7, 11]]
    # the refresh at step 7 does not restart the period, but the next one only sees the time since step 7
    assert skipping.since_last_refresh[0] == [0.005, 0.02, 0.005, 0.015, 0.02]
    assert skipping.since_last_refresh[2] == [0.005, 0.02, 0.005, 0.02]
