# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import pytest

from isaaclab.utils.sensors import convert_camera_intrinsics_to_usd

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("c_x", "c_y"), [(40.0, 40.0), (50.0, 30.0)])
def test_camera_intrinsics_warn_for_negative_aperture_offsets(caplog, c_x, c_y):
    """Principal-point offsets on either side of image center should emit the unsupported-offset warning."""
    intrinsic_matrix = [100.0, 0.0, c_x, 0.0, 100.0, c_y, 0.0, 0.0, 1.0]

    with caplog.at_level(logging.WARNING, logger="isaaclab.utils.sensors"):
        convert_camera_intrinsics_to_usd(intrinsic_matrix, width=100, height=80)

    assert "Camera aperture offsets are not supported by Omniverse" in caplog.text


def test_camera_intrinsics_centered_principal_point_does_not_warn(caplog):
    """A centered principal point should not emit an aperture-offset warning."""
    intrinsic_matrix = [100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0]

    with caplog.at_level(logging.WARNING, logger="isaaclab.utils.sensors"):
        convert_camera_intrinsics_to_usd(intrinsic_matrix, width=100, height=80)

    assert "Camera aperture offsets are not supported by Omniverse" not in caplog.text
