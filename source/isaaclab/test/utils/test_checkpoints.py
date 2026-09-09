# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resolving the weights a component loads at runtime."""

import os

import pytest

from isaaclab.utils.checkpoints import Checkpoint


def test_run_artifact_resolves_to_the_newest_file_the_run_wrote(tmp_path):
    """A local play loads the latest checkpoint a training run wrote."""
    older, newer = tmp_path / "cnn_100_0.5.pth", tmp_path / "cnn_200_0.1.pth"
    older.touch()
    newer.touch()
    os.utime(older, (1_000_000, 1_000_000))  # same-second touches tie on mtime

    assert Checkpoint(name="fe", run_glob="cnn_*.pth").resolve(str(tmp_path)) == str(newer)


def test_fetched_copy_wins_over_the_files_the_run_wrote(tmp_path):
    """A pretrained play loads the copy the fetch recorded, wherever the download landed."""
    (tmp_path / "cnn_200_0.1.pth").touch()
    fetched = "/cache/omniverse/host/Isaac-Task_physx_rtx_rsl_rl_fe.pth"

    assert Checkpoint(name="fe", run_glob="cnn_*.pth", local_path=fetched).resolve(str(tmp_path)) == fetched


def test_missing_run_artifact_names_the_directory(tmp_path):
    """The error says where it looked, so the user can tell an unbuilt tree from a wrong path."""
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        Checkpoint(name="fe", run_glob="cnn_*.pth").resolve(str(tmp_path))
