# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils.images import normalize_camera_output_for_display

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("data_type", ["depth", "distance_to_camera", "distance_to_image_plane"])
def test_depth_display_normalization_ignores_nonfinite_values(data_type):
    src = torch.tensor([0.0, 2.0, float("inf"), 4.0, float("nan")])

    out = normalize_camera_output_for_display(src, data_type)

    expected = torch.tensor([0.0, 0.5, 0.0, 1.0, 0.0])
    torch.testing.assert_close(out, expected)
    assert torch.isfinite(out).all()


def test_depth_display_normalization_handles_all_nonfinite_values():
    src = torch.tensor([float("inf"), float("nan")])

    out = normalize_camera_output_for_display(src, "depth")

    torch.testing.assert_close(out, torch.zeros_like(src))
