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
from isaaclab.utils.warp.capture import GraphedCallable

pytestmark = pytest.mark.unit


def test_sensor_lifecycle_has_single_owners():
    """Backends retain shared scheduling, lazy state reads, and one Fabric initialization owner."""
    for directory in (ISAACLAB_ROOT / "source").glob("isaaclab*/isaaclab*/sensors"):
        for path in directory.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Name):
                    assert node.id not in {"_capture_on_reset", "_requires_physics_step"}, path
                if isinstance(node, ast.Attribute):
                    assert node.attr not in {"_capture_on_reset", "_requires_physics_step"}, path
                if directory.parent.name == "isaaclab" or path.name.endswith("_data.py"):
                    continue
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                    assert node.attr != "_num_envs", f"{path}:{node.lineno} overrides SensorBase environment ownership"
                if isinstance(node, ast.FunctionDef):
                    assert all(arg.arg != "timestamp" for arg in node.args.args), (
                        f"{path}:{node.lineno} duplicates SensorBase capture readiness"
                    )
                    if directory.parts[-2] == "isaaclab_newton" and node.name == "_update_buffers_impl":
                        assert not any(
                            isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Attribute)
                            and call.func.attr == "forward"
                            for call in ast.walk(node)
                        ), f"{path}:{node.lineno} bypasses the lazy state read boundary"
    for name in (
        "isaaclab/isaaclab/sim/views/fabric_xform_selection.py",
        "isaaclab_physx/isaaclab_physx/sim/views/fabric_frame_view.py",
        "isaaclab_newton/isaaclab_newton/sim/views/newton_site_frame_view.py",
    ):
        path = ISAACLAB_ROOT / "source" / name
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                assert node.name not in {"_sync_fabric_from_usd_initial", "parent_world_rw_ifa"}, path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"SetLocalXformFromUsd", "SetWorldXformFromUsd", "forward"}, path


class _RefreshLogSensor(SensorBase):
    """Sensor that records the step index of every buffer refresh, per environment."""

    def __init__(self, update_period: float, device: str, num_envs: int = 2):
        # SensorBase.__init__ needs a USD stage; only the timing state is under test here.
        self.cfg = SensorBaseCfg(prim_path="/World/envs/env_.*/sensor", update_period=update_period)
        self._device = device
        self._num_envs = num_envs
        self._ALL_ENV_MASK = wp.ones(num_envs, dtype=wp.bool, device=device)
        self._reset_mask = wp.zeros_like(self._ALL_ENV_MASK)
        self._is_initialized = True
        self._is_visualizing = False
        self._create_timing_buffers()
        self._data_dirty = False
        self.backend_calls = 0
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
        self.backend_calls += 1
        for env in np.flatnonzero(env_mask.numpy()):
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
@pytest.mark.parametrize("invalidate", [False, True], ids=["cleared-output", "derived-query"])
def test_reset_read_only_restarts_period_after_capture(device, invalidate):
    """Cleared output stays cached; explicitly invalidated queries refresh and restart their period."""
    sensor = _RefreshLogSensor(update_period=0.02, device=device)
    _run(sensor, [0.005])
    reset_mask = wp.array([True, False], dtype=wp.bool, device=device)
    sensor.reset(env_mask=reset_mask)
    if invalidate:
        sensor._invalidate_buffers(reset_mask)
    _ = sensor.data
    for step in range(2, 11):
        sensor.step = step
        sensor.update(0.005)
        _ = sensor.data
    assert sensor.refreshes[0] == ([1, 1, 5, 9] if invalidate else [1, 2, 6, 10])
    assert sensor.refreshes[1] == [1, 5, 9]


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.parametrize("period", [0.0, 0.02])
def test_reset_cache_refreshes_on_positive_update_or_explicit_invalidation(device, period):
    """Zero-dt reads retain reset outputs; the first positive update captures before a long period expires."""
    sensor = _RefreshLogSensor(update_period=period, device=device)
    for iteration in range(2):
        sensor.update(0.0, force_recompute=True)
        _ = sensor.data
        assert sensor.refreshes == [[], []]
        assert sensor.backend_calls == 2 * iteration
        _run(sensor, [0.005])
        _ = sensor.data
        assert sensor.refreshes == [[1], [1]]
        sensor.update(0.0, force_recompute=True)
        _ = sensor.data
        assert sensor.refreshes == [[1], [1]]
        assert sensor.backend_calls == 2 * iteration + 1
        sensor._invalidate_buffers(wp.array([True, False], dtype=wp.bool, device=device))
        sensor.update(0.0, force_recompute=True)
        assert sensor.refreshes == [[1, 1], [1]]
        sensor.reset()
        sensor.refreshes = [[], []]

    sensor._invalidate_buffers()
    sensor.update(0.0, force_recompute=True)
    assert sensor.refreshes == [[1], [1]]


@wp.kernel
def _count_refresh(env_mask: wp.array(dtype=wp.bool), refreshes: wp.array(dtype=wp.int32)):
    env = wp.tid()
    if env_mask[env]:
        refreshes[env] += 1


class _DeviceRefreshSensor(_RefreshLogSensor):
    """Sensor that counts the buffer refreshes of each environment on the device, as a CUDA graph replays them."""

    def __init__(self, device: str):
        super().__init__(update_period=0.0, device=device)
        self.device_refreshes = wp.zeros(self._num_envs, dtype=wp.int32, device=device)

    def _update_buffers_impl(self, env_mask: wp.array):
        wp.launch(_count_refresh, dim=self._num_envs, inputs=[env_mask, self.device_refreshes], device=self._device)


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_recorded_read_refreshes_on_every_replay(device, monkeypatch):
    """A read recorded while the sensor is not dirty, as after a reset, still refreshes the sensor on each replay."""
    monkeypatch.setattr(GraphedCallable, "enabled", True)
    sensor = _DeviceRefreshSensor(device)
    sensor.update(0.005)
    _ = sensor.data  # the sensor is clean when the read is recorded
    read = GraphedCallable(lambda: sensor.data, device)
    read()  # records the refresh of the outdated environments, none yet
    for _ in range(3):
        sensor.update(0.005)
        read()
    # one eager refresh, then one per replay
    assert sensor.device_refreshes.numpy().tolist() == [4, 4]


class _Reader:
    """Two recorded parts that read the same sensor, as the post-physics and observation parts of a step do."""

    def __init__(self, sensor: _DeviceRefreshSensor):
        self.sensor = sensor
        self.first = GraphedCallable(self._read, sensor._device)
        self.second = GraphedCallable(self._read, sensor._device)

    def _read(self):
        _ = self.sensor.data


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_recorded_parts_reading_a_sensor_refresh_it_each_step(device, monkeypatch):
    """The data of a sensor that recorded parts read refreshes once per step, also when a step skips the part that
    refreshed it first: each recording holds the refresh of the environments still outdated."""
    monkeypatch.setattr(GraphedCallable, "enabled", True)
    sensor = _DeviceRefreshSensor(device)
    reader = _Reader(sensor)
    for _ in range(4):  # the first step records both parts, the others replay them
        sensor.update(0.005)
        reader.first()
        reader.second()
    sensor.update(0.005)
    reader.second()
    assert sensor.device_refreshes.numpy().tolist() == [5, 5]
