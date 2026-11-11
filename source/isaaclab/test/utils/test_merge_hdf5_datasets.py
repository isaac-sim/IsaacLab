# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _create_dataset(path: Path, format_version: int | None) -> None:
    with h5py.File(path, "w") as file:
        if format_version is not None:
            file.attrs["format_version"] = format_version
        data = file.create_group("data")
        data.attrs["env_args"] = '{"env_name": "test"}'
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((1, 1), dtype=np.float32))


def _tool_path(name: str) -> Path:
    return Path(__file__).resolve().parents[4] / "scripts" / "tools" / name


def test_merge_preserves_dataset_format_version(tmp_path):
    """Merged current-format datasets must not be reclassified as legacy datasets."""
    input_path = tmp_path / "input.hdf5"
    output_path = tmp_path / "merged.hdf5"
    _create_dataset(input_path, format_version=1)

    subprocess.run(
        [
            sys.executable,
            str(_tool_path("merge_hdf5_datasets.py")),
            "--input_files",
            str(input_path),
            "--output_file",
            str(output_path),
        ],
        check=True,
    )

    with h5py.File(output_path, "r") as file:
        assert file.attrs["format_version"] == 1


def test_augmented_dataset_preserves_format_and_merges_with_source(tmp_path):
    """The documented original-plus-augmented merge path must keep one consistent format version."""
    source_path = tmp_path / "source.hdf5"
    augmented_path = tmp_path / "augmented.hdf5"
    merged_path = tmp_path / "merged.hdf5"
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    _create_dataset(source_path, format_version=1)

    subprocess.run(
        [
            sys.executable,
            str(_tool_path("mp4_to_hdf5.py")),
            "--input_file",
            str(source_path),
            "--videos_dir",
            str(videos_dir),
            "--output_file",
            str(augmented_path),
        ],
        check=True,
    )

    with h5py.File(augmented_path, "r") as file:
        assert file.attrs["format_version"] == 1

    subprocess.run(
        [
            sys.executable,
            str(_tool_path("merge_hdf5_datasets.py")),
            "--input_files",
            str(source_path),
            str(augmented_path),
            "--output_file",
            str(merged_path),
        ],
        check=True,
    )

    with h5py.File(merged_path, "r") as file:
        assert file.attrs["format_version"] == 1
        assert len(file["data"]) == 2
