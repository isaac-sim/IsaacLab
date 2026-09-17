# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab.utils import REQUIRED


@dataclass
class ObstaclesSceneCfg:
    """Configuration for a terrain with floating obstacles."""

    min_num_obstacles: int = 1
    max_num_obstacles: int = 40
    ground_offset: float = 3.0

    env_size: tuple[float, float, float] = REQUIRED

    @dataclass
    class BoxCfg:
        """Configuration for a box-shaped obstacle or wall.

        Defines the size and placement constraints for rectangular obstacles within
        the environment. The center position is specified as ratios of the environment
        size, allowing for flexible scaling.

        Attributes:
            size: Tuple of (length, width, height) in meters.
            center_ratio_min: Minimum position as ratio of env_size (0.0 to 1.0) for
                each axis. Used for random placement bounds.
            center_ratio_max: Maximum position as ratio of env_size (0.0 to 1.0) for
                each axis. For fixed positions, set equal to center_ratio_min.
        """

        size: tuple[float, float, float] = REQUIRED
        center_ratio_min: tuple[float, float, float] = REQUIRED
        center_ratio_max: tuple[float, float, float] = REQUIRED

    # Obstacle configurations
    panel_obs_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(0.1, 1.2, 3.0), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.95, 0.95)
        )
    )

    small_wall_obs_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(0.1, 0.5, 0.5), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.9, 0.9)
        )
    )

    big_wall_obs_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(0.1, 1.0, 1.0), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.9, 0.9)
        )
    )

    small_cube_obs_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(0.4, 0.4, 0.4), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.9, 0.9)
        )
    )

    rod_obs_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(0.1, 0.1, 2.0), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.9, 0.9)
        )
    )

    # Wall configurations
    left_wall_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(12.0, 0.2, 6.0), center_ratio_min=(0.5, 1.0, 0.5), center_ratio_max=(0.5, 1.0, 0.5)
        )
    )

    right_wall_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(12.0, 0.2, 6.0), center_ratio_min=(0.5, 0.0, 0.5), center_ratio_max=(0.5, 0.0, 0.5)
        )
    )

    back_wall_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(0.2, 8.0, 6.0), center_ratio_min=(0.0, 0.5, 0.5), center_ratio_max=(0.0, 0.5, 0.5)
        )
    )

    front_wall_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(0.2, 8.0, 6.0), center_ratio_min=(1.0, 0.5, 0.5), center_ratio_max=(1.0, 0.5, 0.5)
        )
    )

    top_wall_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(12.0, 8.0, 0.2), center_ratio_min=(0.5, 0.5, 1.0), center_ratio_max=(0.5, 0.5, 1.0)
        )
    )

    bottom_wall_cfg: Any = field(
        default_factory=lambda: ObstaclesSceneCfg.BoxCfg(
            size=(12.0, 8.0, 0.2), center_ratio_min=(0.5, 0.5, 0.0), center_ratio_max=(0.5, 0.5, 0.0)
        )
    )

    wall_cfgs: Any = field(
        default_factory=lambda: {
            "left_wall": ObstaclesSceneCfg.BoxCfg(
                size=(12.0, 0.2, 6.0), center_ratio_min=(0.5, 1.0, 0.5), center_ratio_max=(0.5, 1.0, 0.5)
            ),
            "right_wall": ObstaclesSceneCfg.BoxCfg(
                size=(12.0, 0.2, 6.0), center_ratio_min=(0.5, 0.0, 0.5), center_ratio_max=(0.5, 0.0, 0.5)
            ),
            "back_wall": ObstaclesSceneCfg.BoxCfg(
                size=(0.2, 8.0, 6.0), center_ratio_min=(0.0, 0.5, 0.5), center_ratio_max=(0.0, 0.5, 0.5)
            ),
            "front_wall": ObstaclesSceneCfg.BoxCfg(
                size=(0.2, 8.0, 6.0), center_ratio_min=(1.0, 0.5, 0.5), center_ratio_max=(1.0, 0.5, 0.5)
            ),
            "bottom_wall": ObstaclesSceneCfg.BoxCfg(
                size=(12.0, 8.0, 0.2), center_ratio_min=(0.5, 0.5, 0.0), center_ratio_max=(0.5, 0.5, 0.0)
            ),
            "top_wall": ObstaclesSceneCfg.BoxCfg(
                size=(12.0, 8.0, 0.2), center_ratio_min=(0.5, 0.5, 1.0), center_ratio_max=(0.5, 0.5, 1.0)
            ),
        }
    )

    obstacle_cfgs: Any = field(
        default_factory=lambda: {
            "panel": ObstaclesSceneCfg.BoxCfg(
                size=(0.1, 1.2, 3.0), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.95, 0.95)
            ),
            "small_wall": ObstaclesSceneCfg.BoxCfg(
                size=(0.1, 0.5, 0.5), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.9, 0.9)
            ),
            "big_wall": ObstaclesSceneCfg.BoxCfg(
                size=(0.1, 1.0, 1.0), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.9, 0.9)
            ),
            "small_cube": ObstaclesSceneCfg.BoxCfg(
                size=(0.4, 0.4, 0.4), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.9, 0.9)
            ),
            "rod": ObstaclesSceneCfg.BoxCfg(
                size=(0.1, 0.1, 2.0), center_ratio_min=(0.3, 0.05, 0.05), center_ratio_max=(0.85, 0.9, 0.9)
            ),
        }
    )
