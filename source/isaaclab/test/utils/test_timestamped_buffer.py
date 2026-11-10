# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""When a :class:`TimestampedBuffer` owner updates its data, eagerly and while a CUDA graph records."""

from __future__ import annotations

import pytest
import warp as wp

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.buffers import TimestampedBuffer, reset_timestamps

pytestmark = pytest.mark.unit


@wp.kernel
def _count_update(updates: wp.array(dtype=wp.int32)):
    updates[0] += 1


class _LazyValue:
    """A lazily updated value counting its updates on the device."""

    def __init__(self, device: str, update_in_graph: bool = True):
        self.device = device
        self.timestamp = 0.0
        self.updates = wp.zeros(1, dtype=wp.int32, device=device)
        self._value = TimestampedBuffer(self.updates, update_in_graph=update_in_graph)

    def read(self) -> None:
        if self._value.timestamp < self.timestamp:
            wp.launch(_count_update, dim=1, inputs=[self.updates], device=self.device)
            self._value.timestamp = self.timestamp

    @property
    def num_updates(self) -> int:
        return int(self.updates.numpy()[0])


def test_eager_reads_update_once_per_timestamp():
    value = _LazyValue("cpu")
    for _ in range(2):
        value.timestamp += 0.01
        value.read()
        value.read()
    reset_timestamps([value._value])
    value.read()
    assert value.num_updates == 3


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_recorded_read_updates_on_every_replay(device):
    """A read recorded while the data is current still updates it on each replay, once per recording."""
    value = _LazyValue(device)
    value.read()  # the data is current when the read is recorded
    with wp.ScopedCapture(device=device) as capture:
        value.read()
        value.read()
    for _ in range(3):
        value.timestamp += 0.01
        wp.capture_launch(capture.graph)
    # one eager update, then one per replay
    assert value.num_updates == 4


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_each_recording_holds_its_own_update(device):
    """An invalidation inside a recording records the update again, and a later recording records its own."""
    value = _LazyValue(device)
    value.read()
    with wp.ScopedCapture(device=device) as invalidated:
        value.read()
        reset_timestamps([value._value])
        value.read()
    with wp.ScopedCapture(device=device) as later:
        value.read()
    value.read()  # an eager read after the recordings keeps the eager rule
    assert value.num_updates == 1
    wp.capture_launch(invalidated.graph)
    wp.capture_launch(later.graph)
    assert value.num_updates == 4


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_an_opted_out_buffer_keeps_the_timestamp_rule_while_recording(device):
    """A buffer opting out records its update only when it is stale by timestamp, as outside a recording."""
    value = _LazyValue(device, update_in_graph=False)
    value.read()
    with wp.ScopedCapture(device=device) as capture:
        value.read()
    assert value._value.timestamp == value.timestamp
    for _ in range(2):
        wp.capture_launch(capture.graph)
    assert value.num_updates == 1
