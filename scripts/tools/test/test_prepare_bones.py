# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent unit, frame-rate and physical-support contracts for BONES preparation."""

import io
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from isaaclab_tasks.contrib.g1_tracking.kinematics import G1_JOINT_NAMES

from scripts.tools import prepare_bones


@pytest.fixture
def elevated_clip():
    columns = [
        "Frame",
        *[f"root_translate{x}" for x in "XYZ"],
        *[f"root_rotate{x}" for x in "XYZ"],
        *[f"{name}_dof" for name in G1_JOINT_NAMES],
    ]
    values = np.zeros((121, len(columns)))
    values[:, 0], values[:, 3], values[:, 6], values[:, 7] = np.arange(121), 300, 90, 45
    order = np.arange(len(columns))[::-1]
    stream = io.StringIO()
    np.savetxt(stream, values[:, order], delimiter=",", header=",".join(columns[i] for i in order), comments="")
    return stream.getvalue().encode()


def test_conversion_uses_header_order_and_named_units(elevated_clip):
    poses = prepare_bones.convert_csv(elevated_clip)
    np.testing.assert_allclose(poses[:, :3], np.tile([0, 0, 3], (121, 1)))
    np.testing.assert_allclose(poses[:, 7], np.pi / 4)
    np.testing.assert_allclose(Rotation.from_quat(poses[0, 3:7]).apply([1, 0, 0]), [0, 1, 0], atol=1e-6)
    with pytest.raises(ValueError, match="fields"):
        prepare_bones.convert_csv(elevated_clip.replace(b"root_translateX", b"root_translateY"))


def test_filter_resamples_and_rejects_missing_support(elevated_clip, tmp_path):
    data_dir = os.environ.get("ISAACLAB_G1_TRACKING_DATA")
    if not data_dir:
        pytest.skip("Set ISAACLAB_G1_TRACKING_DATA to the prepared G1 tracking directory.")
    mjcf = Path(data_dir) / "g1_description/g1_29dof.xml"
    prepare_bones.initialize_worker(str(mjcf), str(tmp_path), 120)
    result = prepare_bones.process_motion("session/elevated.csv", elevated_clip)
    assert result["reason"] == "elevated_support" and result["frames"] == 51
    assert not list(tmp_path.rglob("*.npy"))
