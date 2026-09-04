# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resolving the weights a component loads at runtime."""

import os

import pytest

from isaaclab.utils import checkpoints
from isaaclab.utils.checkpoints import Checkpoint


def test_run_artifact_resolves_to_the_newest_file_the_run_wrote(tmp_path):
    """A local play loads the latest checkpoint a training run wrote."""
    older, newer = tmp_path / "cnn_100_0.5.pth", tmp_path / "cnn_200_0.1.pth"
    older.touch()
    newer.touch()
    os.utime(older, (1_000_000, 1_000_000))  # same-second touches tie on mtime

    assert Checkpoint(name="fe", run_glob="cnn_*.pth").resolve(str(tmp_path)) == str(newer)


def test_published_copy_wins_over_the_native_name(tmp_path):
    """A pretrained play loads the published file even beside native ones."""
    (tmp_path / "cnn_200_0.1.pth").touch()
    (tmp_path / "Isaac-Task_physx_rtx_rsl_rl_fe.pth").touch()

    assert Checkpoint(name="fe", run_glob="cnn_*.pth").resolve(str(tmp_path)).endswith("_fe.pth")


def test_missing_run_artifact_names_the_directory(tmp_path):
    """The error says where it looked, so the user can tell an unbuilt tree from a wrong path."""
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        Checkpoint(name="fe", run_glob="cnn_*.pth").resolve(str(tmp_path))


def test_url_weights_are_fetched_into_the_cache(monkeypatch):
    """Pre-existing weights go through the shared download, into the requested cache directory."""
    calls = []
    monkeypatch.setattr(checkpoints, "retrieve_file_path", lambda url, d: calls.append((url, d)) or "/cache/vae.pt")

    path = Checkpoint(name="vae", url="omniverse://IsaacLab/Contrib/vae.pt").resolve(cache_dir="/cache")

    assert path == "/cache/vae.pt"
    assert calls == [("omniverse://IsaacLab/Contrib/vae.pt", "/cache")]


def test_extension_follows_the_declared_source():
    """The published extension is whatever the component actually writes or fetches."""
    assert Checkpoint(name="a", run_glob="enc_*.safetensors").extension == ".safetensors"
    assert Checkpoint(name="b", url="omniverse://x/vae.pt").extension == ".pt"
