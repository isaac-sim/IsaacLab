# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the locomanipulation SDG occupancy map helpers."""

import numpy as np
import pytest

from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import OccupancyMap

pytestmark = pytest.mark.kitless

_MAP_SIZE = 15


@pytest.mark.parametrize("radius", [0, 1, 3])
def test_buffered_grows_obstacles_by_radius_on_every_side(radius: int) -> None:
    omap = _single_obstacle_map()
    center = _MAP_SIZE // 2

    occupied = omap.buffered(radius).occupied_mask()

    rows, cols = np.nonzero(occupied)
    assert (rows.min(), rows.max()) == (center - radius, center + radius)
    assert (cols.min(), cols.max()) == (center - radius, center + radius)
    np.testing.assert_array_equal(occupied, occupied[::-1, :])
    np.testing.assert_array_equal(occupied, occupied[:, ::-1])


def test_buffered_meters_below_one_pixel_keeps_map() -> None:
    omap = _single_obstacle_map()

    np.testing.assert_array_equal(omap.buffered_meters(0.02).occupied_mask(), omap.occupied_mask())


def _single_obstacle_map() -> OccupancyMap:
    occupied = np.zeros((_MAP_SIZE, _MAP_SIZE), dtype=bool)
    occupied[_MAP_SIZE // 2, _MAP_SIZE // 2] = True
    return OccupancyMap.from_masks(~occupied, occupied, 0.05, (0.0, 0.0, 0.0))
