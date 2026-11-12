# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for world point bounds checks on the locomanipulation SDG occupancy map."""

import numpy as np
import pytest
import torch

from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import OccupancyMap, Point2d
from isaaclab_mimic.locomanipulation_sdg.path_utils import plan_path
from isaaclab_mimic.locomanipulation_sdg.scene_utils import HasPose2d

pytestmark = pytest.mark.kitless

# 10 x 10 pixels at 0.1 m, so the map covers [0, 1) m along x and y.
_MAP_SIZE = 10
_RESOLUTION = 0.1

_OUTSIDE_POINTS = [(-0.05, 0.5), (1.05, 0.5), (0.5, -0.05), (0.5, 1.05)]
_INSIDE_POINTS = [(0.05, 0.05), (0.05, 0.95), (0.95, 0.05), (0.5, 0.5)]


@pytest.mark.parametrize("xy", _OUTSIDE_POINTS)
def test_world_point_just_outside_edge_is_out_of_bounds(xy: tuple[float, float]) -> None:
    omap = _free_map()
    point = Point2d(x=xy[0], y=xy[1])

    assert not omap.check_world_point_in_bounds(point)
    assert not omap.check_world_point_in_freespace(point)


@pytest.mark.parametrize("xy", _INSIDE_POINTS)
def test_world_point_inside_map_is_in_bounds(xy: tuple[float, float]) -> None:
    omap = _free_map()
    point = Point2d(x=xy[0], y=xy[1])

    assert omap.check_world_point_in_bounds(point)
    assert omap.check_world_point_in_freespace(point)


@pytest.mark.parametrize("xy", _OUTSIDE_POINTS)
def test_plan_path_rejects_start_just_outside_edge(xy: tuple[float, float]) -> None:
    omap = _free_map()

    assert plan_path(_Pose2d(*xy), _Pose2d(0.5, 0.5), omap) is None


def _free_map() -> OccupancyMap:
    occupied = np.zeros((_MAP_SIZE, _MAP_SIZE), dtype=bool)
    return OccupancyMap.from_masks(~occupied, occupied, _RESOLUTION, (0.0, 0.0, 0.0))


class _Pose2d(HasPose2d):
    def __init__(self, x: float, y: float) -> None:
        self._pose = torch.tensor([[x, y, 0.0]])

    def get_pose_2d(self) -> torch.Tensor:
        return self._pose
