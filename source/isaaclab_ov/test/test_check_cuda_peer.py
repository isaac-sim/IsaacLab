# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The split-GPU preflight checks the renderer's publication copy and reports failures."""

import sys

import numpy as np
import pytest
from isaaclab_ov.renderers import check_cuda_peer

from isaaclab.test.utils import DeviceScope, test_devices


def _run(monkeypatch, *devices):
    monkeypatch.setattr(sys, "argv", ["check_cuda_peer", "--devices", *devices])
    return check_cuda_peer.main()


def _publication_buffers():
    """Return the renderer's publication copy, or skip without the OV rendering packages."""
    return pytest.importorskip("isaaclab_ov.renderers.publication_buffers")._GeometryPublicationBuffers


def _cuda_devices(count: int) -> list[str]:
    """Return ``count`` distinct CUDA devices selected for this run, or skip."""
    devices = test_devices(DeviceScope.CUDA)
    if len(devices) < count:
        pytest.skip(f"requires {count} selected CUDA device(s)")
    return [devices[0], devices[-1]][:count]


@pytest.mark.parametrize("other", ["same-gpu", "cpu"])
def test_preflight_rejects_non_split_selections(monkeypatch, capsys, other):
    _publication_buffers()
    (device,) = _cuda_devices(1)
    assert _run(monkeypatch, device if other == "same-gpu" else "cpu", device) == 1
    assert "two distinct CUDA devices" in capsys.readouterr().err


def test_preflight_passes_and_exercises_the_publication_copy(monkeypatch, capsys):
    publication_buffers = _publication_buffers()
    first, second = _cuda_devices(2)
    calls = []
    copy = publication_buffers.copy_completed

    def record(self, key, source):
        calls.append((str(source.device), str(self._device)))
        return copy(self, key, source)

    monkeypatch.setattr(publication_buffers, "copy_completed", record)
    assert _run(monkeypatch, first, second) == 0
    assert {call for call in calls if call[0] != call[1]} == {(first, second), (second, first)}
    assert "PASS" in capsys.readouterr().out


def test_preflight_fails_on_a_corrupted_publication_copy(monkeypatch, capsys):
    import warp as wp

    monkeypatch.setattr(
        _publication_buffers(),
        "copy_completed",
        lambda self, _key, source: wp.zeros(source.shape, dtype=source.dtype, device=self._device),
    )
    assert _run(monkeypatch, *_cuda_devices(2)) == 1
    captured = capsys.readouterr()
    assert "4096/4096 incorrect values" in captured.out
    assert "Do not use this split-GPU path" in captured.err


def test_preflight_fails_a_skipped_copy_into_recycled_target_memory(monkeypatch, capsys):
    """Target memory that already holds the expected values must not hide a copy that never ran."""
    import warp as wp

    _publication_buffers()
    first, second = _cuda_devices(2)
    empty_like, copy = wp.empty_like, wp.copy
    skipped = []

    def recycled_allocation(src, device=None, **kwargs):
        array = empty_like(src, device=device, **kwargs)
        if array.device.is_cuda:
            # Pool memory can still hold the first pattern from an earlier, freed allocation.
            array.assign(np.arange(1, array.size + 1, dtype=np.float32))
        return array

    def skip_first_cross_gpu_copy(dest, src, *args, **kwargs):
        if not skipped and dest.device.is_cuda and src.device.is_cuda and dest.device != src.device:
            skipped.append(src)
            return
        copy(dest, src, *args, **kwargs)

    monkeypatch.setattr(wp, "empty_like", recycled_allocation)
    monkeypatch.setattr(wp, "copy", skip_first_cross_gpu_copy)
    assert _run(monkeypatch, first, second) == 1
    assert skipped
    assert f"publication path {first} -> {second}, sign=+1: 4096/4096 incorrect values" in capsys.readouterr().out
