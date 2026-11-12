# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the locomanipulation SDG path helpers."""

import pytest
import torch

from isaaclab_mimic.locomanipulation_sdg.path_utils import ParameterizedPath


@pytest.mark.parametrize(
    ("query", "expected_point", "expected_distance"),
    [
        # past the end of the first segment, next to the corner
        ([1.5, -0.1], [1.0, 0.0], 1.0),
        # behind the start of the path
        ([-0.5, 0.0], [0.0, 0.0], 0.0),
        # beside the interior of the first segment
        ([0.5, 0.2], [0.5, 0.0], 0.5),
    ],
)
def test_find_nearest_distance_matches_nearest_point(query, expected_point, expected_distance):
    """The distance along the path is the arc length of the returned nearest point, also at the endpoints."""
    path = ParameterizedPath(torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))

    point, distance, _, _ = path.find_nearest(torch.tensor(query))

    torch.testing.assert_close(point, torch.tensor(expected_point), atol=1e-5, rtol=0.0)
    assert float(distance) == pytest.approx(expected_distance, abs=1e-5)
