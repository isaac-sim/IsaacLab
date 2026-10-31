# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The split-GPU preflight checks the renderer's publication copy and reports failures."""

import sys

import pytest

from isaaclab_teleop import check_cuda_peer

torch = pytest.importorskip("torch")


def _run(monkeypatch, *devices):
    monkeypatch.setattr(sys, "argv", ["check_cuda_peer", "--devices", *devices])
    return check_cuda_peer.main()


def _publication_buffers():
    """Return the renderer's publication copy, or skip without the OV rendering packages."""
    return pytest.importorskip("isaaclab_ov.renderers.publication_buffers")._GeometryPublicationBuffers


def test_preflight_explains_a_selection_without_the_ov_rendering_packages(monkeypatch, capsys):
    # A None entry fails the import as in a Kit or teleop-only environment.
    monkeypatch.setitem(sys.modules, "isaaclab_ov.renderers.publication_buffers", None)
    assert _run(monkeypatch, "cuda:0", "cuda:1") == 1
    error = capsys.readouterr().err
    assert "uv run --extra ovxr python -m isaaclab_teleop.check_cuda_peer --devices cuda:0 cuda:1" in error
    assert "PCIe" not in error


@pytest.mark.parametrize("devices", [("cuda:0", "cuda:0"), ("cpu", "cuda:0")])
def test_preflight_rejects_non_split_selections(monkeypatch, capsys, devices):
    _publication_buffers()
    assert _run(monkeypatch, *devices) == 1
    assert "two distinct CUDA devices" in capsys.readouterr().err


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_preflight_passes_and_exercises_the_publication_copy(monkeypatch, capsys):
    publication_buffers = _publication_buffers()
    calls = []
    copy = publication_buffers.copy_completed

    def record(self, key, source):
        calls.append((str(source.device), str(self._device)))
        return copy(self, key, source)

    monkeypatch.setattr(publication_buffers, "copy_completed", record)
    assert _run(monkeypatch, "cuda:0", "cuda:1") == 0
    assert sorted(set(calls)) == [("cuda:0", "cuda:1"), ("cuda:1", "cuda:0")]
    assert "PASS" in capsys.readouterr().out


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_preflight_fails_on_a_corrupted_publication_copy(monkeypatch, capsys):
    import warp as wp

    monkeypatch.setattr(
        _publication_buffers(),
        "copy_completed",
        lambda self, _key, source: wp.zeros(source.shape, dtype=source.dtype, device=self._device),
    )
    assert _run(monkeypatch, "cuda:0", "cuda:1") == 1
    captured = capsys.readouterr()
    assert "4096/4096 incorrect values" in captured.out
    assert "Do not use this split-GPU path" in captured.err
