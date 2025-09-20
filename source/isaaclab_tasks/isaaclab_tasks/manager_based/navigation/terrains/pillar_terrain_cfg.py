# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Code adapted from https://github.com/leggedrobotics/nav-suite

# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.utils import configclass

from .pillar_terrain import pillar_terrain


@configclass
class MeshPillarTerrainCfg(SubTerrainBaseCfg):
    @configclass
    class CylinderCfg:
        """Configuration for repeated cylinder."""

        radius: tuple[float, float] = MISSING
        """The radius of the pyramids (in m). First value start of curriculum, second value end."""

        max_yx_angle: tuple[float, float] = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0. First value start of curriculum, second value end."""

        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

        num_objects: tuple[int, int] = MISSING
        """The number of objects to add to the terrain. First value start of curriculum, second value end."""

        height: tuple[float, float] = MISSING
        """The height (along z) of the object (in m). First value start of curriculum, second value end."""

        object_type: str = "cylinder"
        """The type of object to generate.

        The type can be a string or a callable. If it is a string, the function will look for a function called
        ``make_{object_type}`` in the current module scope. If it is a callable, the function will
        use the callable to generate the object.
        """

    @configclass
    class BoxCfg:
        """Configuration for repeated boxes."""

        width: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m). First value start of curriculum, second value end."""

        length: tuple[float, float] = MISSING
        """The length (along y) of the box (in m). First value start of curriculum, second value end."""

        max_yx_angle: tuple[float, float] = (0.0, 0.0)
        """The maximum angle along the y and x axis. Defaults to (0.0, 0.0). First value start of curriculum, second value end."""

        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

        num_objects: tuple[int, int] = MISSING
        """The number of objects to add to the terrain. First value start of curriculum, second value end."""

        height: tuple[float, float] = MISSING
        """The height (along z) of the object (in m). First value start of curriculum, second value end."""

        object_type: str = "box"
        """The type of object to generate.

        The type can be a string or a callable. If it is a string, the function will look for a function called
        ``make_{object_type}`` in the current module scope. If it is a callable, the function will
        use the callable to generate the object.
        """

    function = pillar_terrain

    rough_terrain: HfRandomUniformTerrainCfg = None
    """The configuration for the rough terrain. If None, the terrain will be flat. Defaults to None."""

    box_objects: BoxCfg = MISSING
    """add boxes to the terrain"""

    cylinder_cfg: CylinderCfg = MISSING
    """add cylinders to the terrain"""

    max_height_noise: float = 0.0
    """The maximum amount of noise to add to the height of the objects (in m). Defaults to 0.0."""

    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""
