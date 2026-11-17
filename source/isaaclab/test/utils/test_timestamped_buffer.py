# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""When a :class:`TimestampedBuffer` owner updates its data, eagerly and inside graph captures."""

from __future__ import annotations

import contextlib

import pytest
import warp as wp

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.buffers import TimestampedBuffer
from isaaclab.utils.warp.capture import GraphedCallable

pytestmark = pytest.mark.unit


@wp.kernel
def _double(source: wp.array(dtype=wp.float32), value: wp.array(dtype=wp.float32), updates: wp.array(dtype=wp.int32)):
    value[0] = 2.0 * source[0]
    updates[0] += 1


class _LazyValue:
    """Twice a source value, updated lazily; the updates are counted on the device."""

    def __init__(self, device: str, update_in_graph: bool = True):
        self.device = device
        self.timestamp = 0.0
        self.source = wp.ones(1, dtype=wp.float32, device=device)
        self.value = wp.zeros(1, dtype=wp.float32, device=device)
        self.updates = wp.zeros(1, dtype=wp.int32, device=device)
        self._value = TimestampedBuffer(self.value, update_in_graph=update_in_graph)

    def set_source(self, source: float) -> None:
        self.source.fill_(source)
        self.timestamp += 0.01

    def read(self) -> wp.array:
        if self._value.timestamp < self.timestamp:
            wp.launch(_double, dim=1, inputs=[self.source, self.value, self.updates], device=self.device)
            self._value.timestamp = self.timestamp
        return self.value

    @property
    def num_updates(self) -> int:
        return int(self.updates.numpy()[0])


def _capture(device: str, fn) -> wp.Graph:
    with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
        fn()
    return capture.graph


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_a_capture_holds_the_update_of_every_read(device):
    """A read captured while the data is current still updates it on each replay."""
    value = _LazyValue(device)
    value.read()  # the data is current when the read is captured
    graph = _capture(device, value.read)
    for source in (2.0, 3.0):
        value.set_source(source)
        wp.capture_launch(graph)
    assert (value.value.numpy()[0], value.num_updates) == (6.0, 3)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_a_capture_commits_no_freshness(device):
    """An update captured, not run, leaves the data stale for the next eager read, whether the graph is launched or
    abandoned."""
    value = _LazyValue(device)
    value.set_source(2.0)
    _capture(device, value.read)  # never launched
    assert value.read().numpy()[0] == 4.0
    value.set_source(3.0)
    with contextlib.suppress(RuntimeError), wp.ScopedDevice(device), wp.ScopedCapture():
        value.read()
        raise RuntimeError("abandon the capture")
    assert value.read().numpy()[0] == 6.0


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_an_abandoned_recording_leaves_the_next_plain_read_correct(device):
    """A :class:`GraphedCallable` that reads the data and raises while recording launches nothing, so the next plain
    read updates the data."""
    value = _LazyValue(device)
    value.read()

    def read_then_raise():
        value.read()
        raise RuntimeError("abandon the recording")

    value.set_source(5.0)
    with pytest.raises(RuntimeError, match="abandon the recording"):
        GraphedCallable(read_then_raise, device)()
    assert value.read().numpy()[0] == 10.0


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_an_opted_out_buffer_keeps_the_timestamp_rule_inside_a_capture(device):
    """A buffer opting out holds its update in a capture only when it is stale by timestamp, as outside one."""
    value = _LazyValue(device, update_in_graph=False)
    value.read()
    graph = _capture(device, value.read)
    value.timestamp += 0.01
    wp.capture_launch(graph)
    assert value.num_updates == 1
