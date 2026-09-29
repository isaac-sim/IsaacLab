# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for when :class:`~isaaclab.sensors.SensorBase` refreshes its buffers.

The real :meth:`SensorBase.update`, :attr:`SensorBase.data` and :meth:`SensorBase.reset` code paths are
driven with a stand-in simulation context, and every refresh is compared against an exact-arithmetic
model of the scheduling rule: a sensor becomes due once the time since its last refresh (or reset)
plus ``1e-6`` reaches ``update_period``, it stays due until its data is read, and that read refreshes
it and restarts the elapsed time.

The per-env sensor clocks are float32 and grow with the time since the last reset, so the runs are long
enough for the clocks' rounding error to exceed that tolerance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import warp as wp

from isaaclab import sim as sim_utils
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit

_VARIABLE_DTS = [0.005, 0.005, 0.010, 0.002, 0.003, 0.004, 0.001, 0.0075]


@wp.kernel
def _record_refresh_kernel(
    env_mask: wp.array(dtype=wp.bool),
    step: wp.int32,
    refresh_count: wp.array(dtype=wp.int32),
    refresh_steps: wp.array2d(dtype=wp.int32),
):
    env = wp.tid()
    if env_mask[env]:
        count = refresh_count[env]
        if count < refresh_steps.shape[1]:
            refresh_steps[env, count] = step
        refresh_count[env] = count + 1


class _RefreshLogSensor(SensorBase):
    """Sensor that records the step index of every buffer refresh, per environment."""

    def __init__(self, update_period: float, num_envs: int, device: str, max_refreshes: int):
        # SensorBase.__init__ needs a USD stage; only the timing state is under test here.
        self.cfg = SensorBaseCfg(prim_path="/World/envs/env_.*/sensor", update_period=update_period)
        self._initialize_handle = None
        self._invalidate_initialize_handle = None
        self._prim_deletion_handle = None
        self._debug_vis_handle = None
        self._is_initialized = False
        self._is_visualizing = False
        self._max_refreshes = max_refreshes
        self.step = 0
        sim = SimpleNamespace(
            device=device,
            backend="stub",
            get_physics_dt=lambda: 0.0,
            get_clone_plan=lambda: SimpleNamespace(topology=SimpleNamespace(world_prototype_layout=[0] * num_envs)),
        )
        with patch.object(sim_utils.SimulationContext, "instance", staticmethod(lambda: sim)):
            self._initialize_impl()
        self._is_initialized = True

    def _initialize_impl(self):
        super()._initialize_impl()
        self._refresh_count = wp.zeros(self._num_envs, dtype=wp.int32, device=self._device)
        self._refresh_steps = wp.zeros((self._num_envs, self._max_refreshes), dtype=wp.int32, device=self._device)

    @property
    def data(self):
        self._update_outdated_buffers()
        return self._refresh_count

    def _update_buffers_impl(self, env_mask: wp.array):
        wp.launch(
            _record_refresh_kernel,
            dim=self._num_envs,
            inputs=[env_mask, self.step, self._refresh_count, self._refresh_steps],
            device=self._device,
        )

    def refresh_steps(self) -> list[list[int]]:
        counts = self._refresh_count.numpy()
        steps = self._refresh_steps.numpy()
        assert counts.max() <= self._max_refreshes, "refresh log overflow"
        return [steps[env, : counts[env]].tolist() for env in range(self._num_envs)]


def _run(
    sensor: _RefreshLogSensor,
    dts: Sequence[float],
    read_every: int = 1,
    resets: Callable[[int], Sequence[int]] | None = None,
) -> list[list[int]]:
    """Advance the sensor by ``dts``, reading its data every ``read_every`` updates.

    Per step: ``update(dt)``, then the read (if any), then the resets returned by ``resets(step)``.
    This is the order of :meth:`ManagerBasedRLEnv.step` (physics and sensor updates, observations and
    rewards, then resets).
    """
    for step, dt in enumerate(dts, start=1):
        sensor.update(dt)
        if step % read_every == 0:
            sensor.step = step
            _ = sensor.data
        if resets is not None:
            env_ids = resets(step)
            if env_ids:
                sensor.reset(env_ids=list(env_ids))
    return sensor.refresh_steps()


def _expected_refreshes(
    dts: Sequence[float],
    update_period: float,
    read_every: int = 1,
    reset_steps: set[int] | frozenset[int] = frozenset(),
) -> list[int]:
    """Steps at which one environment is refreshed, in exact arithmetic."""
    period = Fraction(update_period)
    tolerance = Fraction(1e-6)
    elapsed = Fraction(0)
    # sensors start outdated so that the first read fills the buffers
    is_due = True
    refreshes = []
    for step, dt in enumerate(dts, start=1):
        elapsed += Fraction(dt)
        if elapsed + tolerance >= period:
            is_due = True
        if step % read_every == 0 and is_due:
            refreshes.append(step)
            elapsed = Fraction(0)
            is_due = False
        if step in reset_steps:
            elapsed = Fraction(0)
            is_due = True
    return refreshes


def _assert_schedule(actual: list[int], expected: list[int]):
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a == e, f"refresh #{i}: expected step {e}, got step {a}"
    assert len(actual) == len(expected), f"expected {len(expected)} refreshes, got {len(actual)}"


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize(
    "dt, period_steps, duration",
    [
        (0.005, 4, 20.0),  # locomotion height scanner
        (0.02, 1, 70.0),
        (0.002, 2, 40.0),
        (0.001, 20, 5.0),
        (0.005, 10, 20.0),  # ten float64 additions of dt fall short of 10 * dt
        (0.002, 1000, 6.0),
    ],
)
def test_integer_period_keeps_its_cadence_as_the_clock_ages(device, dt, period_steps, duration):
    """A period of k steps refreshes exactly every k steps, however long the sensor has been running."""
    dts = [dt] * round(duration / dt)
    update_period = period_steps * dt
    sensor = _RefreshLogSensor(update_period, num_envs=2, device=device, max_refreshes=len(dts))
    actual = _run(sensor, dts)
    expected = _expected_refreshes(dts, update_period)
    assert expected == list(range(1, len(dts) + 1, period_steps))
    for env_refreshes in actual:
        _assert_schedule(env_refreshes, expected)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("dt, period_steps, read_every", [(0.005, 2.5, 1), (0.001, 3.7, 4)])
def test_non_integer_period_waits_for_the_next_step_after_the_refresh(device, dt, period_steps, read_every):
    """A period between two step multiples is counted from the last refresh, not from a fixed time grid."""
    dts = [dt] * round(20.0 / dt)
    update_period = period_steps * dt
    sensor = _RefreshLogSensor(update_period, num_envs=2, device=device, max_refreshes=len(dts))
    actual = _run(sensor, dts, read_every=read_every)
    expected = _expected_refreshes(dts, update_period, read_every=read_every)
    for env_refreshes in actual:
        _assert_schedule(env_refreshes, expected)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("update_period, read_every", [(0.02, 1), (0.1, 4)])
def test_variable_dt(device, update_period, read_every):
    """``update(dt)`` may receive a different ``dt`` on every call."""
    dts = _VARIABLE_DTS * round(40.0 / sum(_VARIABLE_DTS))
    sensor = _RefreshLogSensor(update_period, num_envs=2, device=device, max_refreshes=len(dts))
    actual = _run(sensor, dts, read_every=read_every)
    expected = _expected_refreshes(dts, update_period, read_every=read_every)
    for env_refreshes in actual:
        _assert_schedule(env_refreshes, expected)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_lazy_reads_shift_the_refresh_phase(device):
    """An unread sensor stays due, and the next period is counted from the read that refreshed it."""
    # reads every 3 steps leave a 4-step sensor due and unread for 2 steps, so it refreshes every 6 steps
    dt, update_period, read_every = 0.005, 0.02, 3
    dts = [dt] * round(20.0 / dt)
    sensor = _RefreshLogSensor(update_period, num_envs=2, device=device, max_refreshes=len(dts))
    actual = _run(sensor, dts, read_every=read_every)
    expected = _expected_refreshes(dts, update_period, read_every=read_every)
    for env_refreshes in actual:
        _assert_schedule(env_refreshes, expected)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_due_sensor_is_refreshed_once_when_read(device):
    """Updates while a sensor is already due do not queue extra refreshes or move its next refresh."""
    dt, update_period = 0.005, 0.02
    sensor = _RefreshLogSensor(update_period, num_envs=2, device=device, max_refreshes=1000)
    # 18 s into the episode, where the float32 clock is coarsest relative to dt
    _run(sensor, [dt] * 3600)
    for _ in range(50):
        sensor.update(dt)
    sensor.step = 3650
    _ = sensor.data
    _ = sensor.data
    # the read at step 3650 restarted the period: due again after exactly 4 more updates
    for step in range(3651, 3659):
        sensor.update(dt)
        sensor.step = step
        _ = sensor.data
    for env_refreshes in sensor.refresh_steps():
        assert env_refreshes[-4:] == [3597, 3650, 3654, 3658]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_partial_resets_restart_each_env_independently(device):
    """Resetting some environments restarts only their schedule; the others keep their phase."""
    dt, update_period, read_every = 0.005, 0.02, 4
    dts = [dt] * round(20.0 / dt)
    # env k is reset every episode_steps[k] steps; env 0 is never reset
    episode_steps = [None, 1, 997, 2531, 3203, 3999]
    reset_steps = [frozenset() if n is None else frozenset(range(n, len(dts) + 1, n)) for n in episode_steps]
    sensor = _RefreshLogSensor(update_period, num_envs=len(episode_steps), device=device, max_refreshes=len(dts))
    actual = _run(
        sensor,
        dts,
        read_every=read_every,
        resets=lambda step: [env for env, steps in enumerate(reset_steps) if step in steps],
    )
    for env, env_refreshes in enumerate(actual):
        expected = _expected_refreshes(dts, update_period, read_every=read_every, reset_steps=reset_steps[env])
        _assert_schedule(env_refreshes, expected)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_zero_period_refreshes_on_every_read(device):
    """``update_period == 0`` refreshes whenever the data is read after an update."""
    pattern = [0.005, 0.010, 0.002, 0.0075]
    dts = pattern * 1000
    sensor = _RefreshLogSensor(0.0, num_envs=2, device=device, max_refreshes=len(dts))
    actual = _run(sensor, dts, read_every=3)
    expected = list(range(3, len(dts) + 1, 3))
    assert _expected_refreshes(dts, 0.0, read_every=3) == expected
    for env_refreshes in actual:
        _assert_schedule(env_refreshes, expected)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
@pytest.mark.parametrize(
    "dts",
    [[0.005] * 4000, _VARIABLE_DTS * round(40.0 / sum(_VARIABLE_DTS))],
    ids=["constant_dt", "variable_dt"],
)
def test_cpu_refresh_schedule(device, dts):
    """The CPU build of the scheduling kernels follows the same schedule as the long CUDA runs above."""
    update_period = 0.02
    sensor = _RefreshLogSensor(update_period, num_envs=2, device=device, max_refreshes=len(dts))
    actual = _run(sensor, dts)
    expected = _expected_refreshes(dts, update_period)
    for env_refreshes in actual:
        _assert_schedule(env_refreshes, expected)
