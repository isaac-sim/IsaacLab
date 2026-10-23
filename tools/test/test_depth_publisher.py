# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware-free checks of the standalone G1 depth publisher and its wire format."""

import importlib.util
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_copied_publisher_depth_processing_and_wire_format(tmp_path):
    """The copied script needs no repository imports and preserves valid depth and image orientation."""
    path = tmp_path / "depth_publisher.py"
    shutil.copyfile(Path(__file__).parents[2] / "scripts/deploy/depth_publisher.py", path)
    subprocess.run([sys.executable, "-I", str(path), "--help"], cwd=tmp_path, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("depth_publisher", path)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)

    # These intrinsics give a wider horizontal and narrower vertical field of view than training.
    intrinsics = SimpleNamespace(width=848, height=480, fx=428.0, fy=427.0)
    assert publisher.crop_to_fov(intrinsics) == (18, 0, 812, 480)

    frame = np.full((76, 128), 4.0, dtype=np.float32)
    frame[:2, :2] = [[1.0, 3.0], [0.0, np.nan]]
    frame[:2, 2:4] = [[0.0, np.inf], [-1.0, np.nan]]
    frame[-2:, -2:] = 1.234
    small = publisher.downsample_masked(frame)
    expected = np.full((38, 64), 4.0, dtype=np.float32)
    expected[0, :2] = [2.0, 0.0]
    expected[-1, -1] = 1.234
    np.testing.assert_array_equal(small, expected)

    packet = publisher.pack_frame(2**32 + 7, small, 123456789)
    assert len(packet) == 4884
    assert struct.unpack_from("<IIQHH", packet) == (0x47314450, 7, 123456789, 38, 64)
    millimetres = np.frombuffer(packet, dtype="<u2", offset=20).reshape(38, 64)
    expected_mm = np.full((38, 64), 4000, dtype=np.uint16)
    expected_mm[0, :2] = [2000, 0]
    expected_mm[-1, -1] = 1234
    np.testing.assert_array_equal(millimetres, expected_mm)
