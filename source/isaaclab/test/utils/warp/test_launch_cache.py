# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import warp as wp

import isaaclab.utils.warp.launch_cache as launch_cache
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache


def test_warp_launch_cache(monkeypatch):
    class Command:
        launches = 0

        def launch(self):
            self.launches += 1

    class Device:
        is_cuda = True
        is_capturing = False

    device = Device()
    command = Command()
    result = [command]
    calls = []

    def launch(*args, **kwargs):
        calls.append(kwargs)
        return result[0]

    monkeypatch.setattr(launch_cache.wp, "get_device", lambda _: device)
    monkeypatch.setattr(launch_cache.wp, "launch", launch)
    cache = _WarpLaunchCache("cuda:0")
    kernel = object()

    cache.launch("read", launch, kernel, dim=1, inputs=[], outputs=[])
    cache.launch("read", launch, kernel, dim=1, inputs=[], outputs=[])
    assert len(calls) == 1
    assert calls[0]["record_cmd"] is True
    assert command.launches == 2

    cache.launch("other", launch, kernel, dim=1, inputs=[], outputs=[])
    cache.clear("read")
    cache.launch("read", launch, kernel, dim=1, inputs=[], outputs=[])
    cache.launch("other", launch, kernel, dim=1, inputs=[], outputs=[])
    assert len(calls) == 3

    cache.clear()
    cache.launch("read", launch, kernel, dim=1, inputs=[], outputs=[])
    assert len(calls) == 4

    device.is_capturing = True
    cache.launch("capture", launch, kernel, dim=1, inputs=[], outputs=[])
    device.is_capturing = False
    device.is_cuda = False
    cache.launch("cpu", launch, kernel, dim=1, inputs=[], outputs=[])
    assert all("record_cmd" not in call for call in calls[-2:])

    device.is_cuda = True
    cache.clear()
    result[0] = None
    cache.launch("empty", launch, kernel, dim=0, inputs=[], outputs=[])
    cache.launch("empty", launch, kernel, dim=0, inputs=[], outputs=[])
    assert len(calls) == 8


def test_another_operation_records_again_when_an_array_or_scalar_argument_changes(monkeypatch):
    """An operation other than ``wp.launch`` cannot take an argument in place, so the call site records it again."""

    class Device:
        is_cuda = True
        is_capturing = False

    calls = []

    def operation(values, scale, device, record_cmd=False):
        calls.append((values, scale))
        return SimpleNamespace(launch=lambda: None)

    monkeypatch.setattr(launch_cache.wp, "get_device", lambda _: Device())
    cache = _WarpLaunchCache("cuda:0")
    first, second = wp.zeros(2, dtype=wp.float32, device="cpu"), wp.zeros(2, dtype=wp.float32, device="cpu")
    for values, scale in ((first, 1.0), (first, 1.0), (second, 1.0), (second, 2.0), (second, 2.0)):
        cache.launch("site", operation, values, scale)
    assert calls == [(first, 1.0), (second, 1.0), (second, 2.0)]
    assert len(cache._commands) == 1


@wp.kernel
def _scale_into(values: wp.array(dtype=wp.float32), scale: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = values[i] * scale


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_a_cached_launch_takes_changed_arguments_in_place(device, monkeypatch):
    """A call site keeps one recorded launch: a reallocated array, a changed scalar and a changed dimension are set on
    it in place, without recording a new launch."""
    recorded = []
    launch = wp.launch

    def counting_launch(*args, **kwargs):
        if kwargs.get("record_cmd"):
            recorded.append(True)
        return launch(*args, **kwargs)

    monkeypatch.setattr(launch_cache.wp, "launch", counting_launch)
    cache = _WarpLaunchCache(device)
    out = wp.zeros(4, dtype=wp.float32, device=device)

    def scale(values: wp.array, factor: float, dim: int = 4):
        cache.launch("scale", counting_launch, _scale_into, dim=dim, inputs=[values, factor], outputs=[out])
        return out.numpy().tolist()

    assert scale(wp.full(4, 1.0, dtype=wp.float32, device=device), 2.0) == [2.0] * 4
    reallocated = wp.full(4, 3.0, dtype=wp.float32, device=device)
    assert scale(reallocated, 5.0) == [15.0] * 4
    assert scale(reallocated, 1.0, dim=2) == [3.0, 3.0, 15.0, 15.0]
    assert (len(recorded), len(cache._commands)) == (1, 1)
    cache.clear()
    assert scale(reallocated, 2.0) == [6.0] * 4
    assert len(recorded) == 2
