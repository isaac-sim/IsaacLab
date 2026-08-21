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


def test_merge_preserves_dataset_format_version(tmp_path):
    """Merged current-format datasets must not be reclassified as legacy datasets."""
    input_path = tmp_path / "input.hdf5"
    output_path = tmp_path / "merged.hdf5"

    with h5py.File(input_path, "w") as file:
        file.attrs["format_version"] = 1
        data = file.create_group("data")
        data.attrs["env_args"] = '{"env_name": "test"}'
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((1, 1), dtype=np.float32))

    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "scripts" / "tools" / "merge_hdf5_datasets.py"
    subprocess.run(
        [sys.executable, str(script_path), "--input_files", str(input_path), "--output_file", str(output_path)],
        check=True,
    )

    with h5py.File(output_path, "r") as file:
        assert file.attrs["format_version"] == 1
