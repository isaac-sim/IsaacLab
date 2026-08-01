# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.utils.warp.launch_cache as launch_cache
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

    cache.launch("read", kernel, dim=1, inputs=[], outputs=[])
    cache.launch("read", kernel, dim=1, inputs=[], outputs=[])
    assert len(calls) == 1
    assert calls[0]["record_cmd"] is True
    assert command.launches == 2

    cache.clear()
    cache.launch("read", kernel, dim=1, inputs=[], outputs=[])
    assert len(calls) == 2

    device.is_capturing = True
    cache.launch("capture", kernel, dim=1, inputs=[], outputs=[])
    device.is_capturing = False
    device.is_cuda = False
    cache.launch("cpu", kernel, dim=1, inputs=[], outputs=[])
    assert all("record_cmd" not in call for call in calls[-2:])

    device.is_cuda = True
    cache.clear()
    result[0] = None
    cache.launch("empty", kernel, dim=0, inputs=[], outputs=[])
    cache.launch("empty", kernel, dim=0, inputs=[], outputs=[])
    assert len(calls) == 6
