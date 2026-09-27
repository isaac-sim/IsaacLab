# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from isaaclab.utils.configclass import configclass

from .trail_cfg import TrailBaseCfg


@configclass
class TerrainParameterInterface:
    """This class exposes terrain parameters shared between trail terrain generators and MDP components.

    It also documents the parametrization scheme used to construct a terrain
    patch, illustrated below::

                | -B- |==P0==|========TRAIL==========|==P1==| -B- |

                +------------------------------------------------+----> x
                0                                          size[0]

            where:
            B: border of width ``border_width``.
            P0: start platform of length ``platform_length``.
            P1: end platform of length ``platform_length``.
            TRAIL: trail path connecting the initial and final platforms.
            size: (length, width) of the terrain patch.
    """

    def __init__(self, source_cfg: TrailBaseCfg = TrailBaseCfg()):
        """Constructor.

        Args:
            source_cfg: The source configuration from which parameters are copied.
        """
        self.border_width = source_cfg.border_width
        self.platform_length = source_cfg.platform_length
        self.distance_start_to_trail = source_cfg.distance_start_to_trail

    def get_trail_length(self, size: tuple[float, float]) -> float:
        """Absolute distance between P0 and P1 along the x axis in world frame [m]."""
        return size[0] - 2.0 * (self.platform_length + self.border_width)

    def get_distance_init_to_center(self, size: tuple[float, float]) -> float:
        """Distance from the spawn location to the terrain patch center along x [m]."""
        return self.distance_start_to_trail + 0.5 * self.get_trail_length(size=size)

    def get_center_to_terrain_border(self, size: tuple[float, float]) -> tuple[float, float]:
        """Return distances (dx, dy) from the terrain center to its borders.

        Returns distances along x and y from the patch center to the inner border (i.e., excluding the outer border
        width).
        """
        dim_x = 0.5 * size[0] - self.border_width
        dim_y = 0.5 * size[1] - self.border_width
        return (dim_x, dim_y)
