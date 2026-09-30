# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check all physical state on its device with one small host readback."""

import time

import numpy as np
import warp as wp


@wp.kernel
def check_finite(values: wp.array[float], flags: wp.array[int], slot: int):
    if not wp.isfinite(values[wp.tid()]):
        wp.atomic_max(flags, slot, 1)


class StateCheck:
    def __init__(self, errors, arrays, capture=False, stream_readback=False, profile=False):
        self.errors = errors
        self.names = list(arrays)
        self.values = [a.view(float).flatten() for a in arrays.values()]
        self.flags = wp.zeros(3 + len(arrays), dtype=int, device=errors.device)
        self.profile = profile
        if profile:
            self.begin = wp.Event(device=errors.device, enable_timing=True)
            self.end = wp.Event(device=errors.device, enable_timing=True)
        self.last_profile = {}
        self.host_flags = None
        if stream_readback:
            if not errors.device.is_cuda:
                raise ValueError("Stream readback requires CUDA")
            self.host_flags = wp.empty_like(self.flags, device="cpu", pinned=True)
        self.graph = None
        if capture:
            if not errors.device.is_cuda:
                raise ValueError("State-check graph capture requires CUDA")
            with wp.ScopedCapture(device=errors.device) as captured:
                self._launch()
            self.graph = captured.graph

    def _launch(self):
        self.flags.zero_()
        wp.copy(self.flags, self.errors, count=3)
        for slot, values in enumerate(self.values, 3):
            wp.launch(check_finite, dim=len(values), inputs=[values, self.flags, slot], device=values.device)

    def check(self):
        started = time.perf_counter() if self.profile else 0.0
        if self.profile:
            wp.record_event(self.begin)
        if self.graph is None:
            self._launch()
        else:
            wp.capture_launch(self.graph)
        if self.profile:
            wp.record_event(self.end)
        launched = time.perf_counter() if self.profile else 0.0
        if self.host_flags is None:
            flags = self.flags.numpy()
        else:
            stream = wp.get_stream(self.flags.device)
            wp.copy(self.host_flags, self.flags, stream=stream)
            wp.synchronize_stream(stream)
            flags = self.host_flags.numpy()
        if self.profile:
            self.last_profile = dict(
                check_submit_ms=(launched - started) * 1000,
                check_readback_wait_ms=(time.perf_counter() - launched) * 1000,
                check_gpu_ms=wp.get_event_elapsed_time(self.begin, self.end, synchronize=False),
            )
        if np.any(flags[:3]):
            raise RuntimeError(f"Explicit MPM errors [inversion, domain exit, active capacity]: {flags[:3]}")
        bad = np.flatnonzero(flags[3:])
        if len(bad):
            raise RuntimeError(f"Nonfinite {self.names[bad[0]]}")
