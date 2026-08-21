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


def _merge_script_path() -> Path:
    return Path(__file__).resolve().parents[4] / "scripts" / "tools" / "merge_hdf5_datasets.py"


def test_merge_preserves_dataset_format_version(tmp_path):
    """Merged current-format datasets must not be reclassified as legacy datasets."""
    input_path = tmp_path / "input.hdf5"
    output_path = tmp_path / "merged.hdf5"
    _create_dataset(input_path, format_version=1)

    subprocess.run(
        [sys.executable, str(_merge_script_path()), "--input_files", str(input_path), "--output_file", str(output_path)],
        check=True,
    )

    with h5py.File(output_path, "r") as file:
        assert file.attrs["format_version"] == 1


def test_merge_rejects_mixed_dataset_format_versions(tmp_path):
    """A merged file cannot safely represent episodes using different quaternion format versions."""
    current_path = tmp_path / "current.hdf5"
    legacy_path = tmp_path / "legacy.hdf5"
    output_path = tmp_path / "merged.hdf5"
    _create_dataset(current_path, format_version=1)
    _create_dataset(legacy_path, format_version=None)

    result = subprocess.run(
        [
            sys.executable,
            str(_merge_script_path()),
            "--input_files",
            str(current_path),
            str(legacy_path),
            "--output_file",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "different format_version values" in result.stderr
    assert not output_path.exists()
