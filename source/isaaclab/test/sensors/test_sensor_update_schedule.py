# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for when :class:`~isaaclab.sensors.SensorBase` refreshes its buffers.

The per-env sensor clocks are float32, so the runs are long enough for their rounding error to exceed
the ``1e-6`` due tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.sensors.kernels import reset_envs_kernel, update_outdated_envs_kernel, update_timestamp_kernel
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


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


class _LegacyTimingSensor(_RefreshLogSensor):
    """Custom sensor that allocates only the timing buffers available before this fix."""

    def _create_timing_buffers(self):
        self._is_outdated = wp.ones(self._num_envs, dtype=wp.bool, device=self._device)
        self._timestamp = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        self._timestamp_last_update = wp.zeros_like(self._timestamp)


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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_custom_sensor_adopts_legacy_timing_buffers(device):
    """Custom allocation, reset-before-update, lazy reads and replacement preserve refresh timing."""
    sensor = _LegacyTimingSensor(update_period=0.02, device=device)
    mask = wp.ones(2, dtype=wp.bool, device=device)
    sensor.reset(env_mask=mask)
    assert _run(sensor, [0.005] * 5) == [[1, 5]] * 2

    sensor.update(0.005)
    sensor.reset(env_mask=wp.array([True, False], dtype=wp.bool, device=device))
    sensor.refreshes = [[], []]
    assert _run(sensor, [0.005] * 3) == [[1], [3]]

    # Replace buffers at the same size, retaining time accumulated since a previous refresh.
    sensor._timestamp = wp.full(2, 0.01, dtype=wp.float32, device=device)
    sensor._timestamp_last_update = wp.zeros_like(sensor._timestamp)
    sensor._is_outdated = wp.zeros(2, dtype=wp.bool, device=device)
    sensor.refreshes = [[], []]
    assert _run(sensor, [0.005] * 2) == [[2]] * 2

    # Resize without calling the new SensorBase timing allocator, then read before stepping.
    sensor._num_envs = 3
    sensor._create_timing_buffers()
    sensor.refreshes = [[], [], []]
    sensor.step = 0
    sensor._data_dirty = True
    _ = sensor.data
    assert _run(sensor, [0.005] * 4) == [[0, 4]] * 3


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_timestamp_kernel_signatures_remain_compatible(device):
    """Direct callers can still advance, refresh and partially reset with float32 timestamp arrays."""
    outdated = wp.zeros(2, dtype=wp.bool, device=device)
    timestamp = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
    last_update = wp.clone(timestamp)
    for _ in range(2):
        wp.launch(update_timestamp_kernel, dim=2, inputs=[outdated, timestamp, last_update, 0.005, 0.01], device=device)
    np.testing.assert_array_equal(outdated.numpy(), [True, True])
    np.testing.assert_allclose(timestamp.numpy(), [1.01, 2.01], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(last_update.numpy(), [1.0, 2.0])

    wp.launch(update_outdated_envs_kernel, dim=2, inputs=[outdated, timestamp, last_update], device=device)
    np.testing.assert_array_equal(outdated.numpy(), [False, False])
    np.testing.assert_allclose(last_update.numpy(), [1.01, 2.01], atol=1e-6, rtol=0)

    mask = wp.array([True, False], dtype=wp.bool, device=device)
    wp.launch(reset_envs_kernel, dim=2, inputs=[mask, outdated, timestamp, last_update], device=device)
    np.testing.assert_array_equal(outdated.numpy(), [True, False])
    np.testing.assert_allclose(timestamp.numpy(), [0.0, 2.01], atol=1e-6, rtol=0)
    np.testing.assert_allclose(last_update.numpy(), [0.0, 2.01], atol=1e-6, rtol=0)
