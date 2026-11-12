# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the locomanipulation SDG occupancy map helpers."""

import numpy as np
import pytest

from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import OccupancyMap


def _single_obstacle_map(size: int = 15, resolution: float = 0.05) -> OccupancyMap:
    occupied = np.zeros((size, size), dtype=bool)
    occupied[size // 2, size // 2] = True
    return OccupancyMap.from_masks(~occupied, occupied, resolution, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("radius", [0, 1, 3])
def test_buffered_grows_obstacles_by_radius_on_every_side(radius):
    """Dilation by ``radius`` pixels reaches exactly ``radius`` pixels in each direction and is symmetric."""
    omap = _single_obstacle_map()
    center = omap.height_pixels() // 2

    occupied = omap.buffered(radius).occupied_mask()

    rows, cols = np.nonzero(occupied)
    assert (rows.min(), rows.max()) == (center - radius, center + radius)
    assert (cols.min(), cols.max()) == (center - radius, center + radius)
    np.testing.assert_array_equal(occupied, occupied[::-1, :])
    np.testing.assert_array_equal(occupied, occupied[:, ::-1])


def test_buffered_meters_below_one_pixel_keeps_map():
    """A buffer distance shorter than one pixel leaves the occupied region unchanged."""
    omap = _single_obstacle_map(resolution=0.05)

    np.testing.assert_array_equal(omap.buffered_meters(0.02).occupied_mask(), omap.occupied_mask())
