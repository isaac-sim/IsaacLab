# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The split-GPU preflight checks the renderer's publication copy and reports failures."""

import sys

import pytest

pytest.importorskip("ovstage")
torch = pytest.importorskip("torch")

from isaaclab_teleop import check_cuda_peer  # noqa: E402


def _run(monkeypatch, *devices):
    monkeypatch.setattr(sys, "argv", ["check_cuda_peer", "--devices", *devices])
    return check_cuda_peer.main()


@pytest.mark.parametrize("devices", [("cuda:0", "cuda:0"), ("cpu", "cuda:0")])
def test_preflight_rejects_non_split_selections(monkeypatch, capsys, devices):
    assert _run(monkeypatch, *devices) == 1
    assert "two distinct CUDA devices" in capsys.readouterr().err


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_preflight_passes_and_exercises_the_publication_copy(monkeypatch, capsys):
    from isaaclab_ov.renderers.publication_buffers import _GeometryPublicationBuffers

    calls = []
    copy = _GeometryPublicationBuffers.copy_completed

    def record(self, key, source):
        calls.append((str(source.device), str(self._device)))
        return copy(self, key, source)

    monkeypatch.setattr(_GeometryPublicationBuffers, "copy_completed", record)
    assert _run(monkeypatch, "cuda:0", "cuda:1") == 0
    assert sorted(set(calls)) == [("cuda:0", "cuda:1"), ("cuda:1", "cuda:0")]
    assert "PASS" in capsys.readouterr().out


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_preflight_fails_on_a_corrupted_publication_copy(monkeypatch, capsys):
    import warp as wp
    from isaaclab_ov.renderers.publication_buffers import _GeometryPublicationBuffers

    monkeypatch.setattr(
        _GeometryPublicationBuffers,
        "copy_completed",
        lambda self, _key, source: wp.zeros(source.shape, dtype=source.dtype, device=self._device),
    )
    assert _run(monkeypatch, "cuda:0", "cuda:1") == 1
    captured = capsys.readouterr()
    assert "4096/4096 incorrect values" in captured.out
    assert "Do not use this split-GPU path" in captured.err
