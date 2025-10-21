# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .sub_terrain_cfg import SubTerrainBaseCfg
from .terrain_generator import TerrainGenerator


@configclass
class TerrainGeneratorCfg:
    """Configuration for the terrain generator."""

    class_type: type = TerrainGenerator
    """The class to use for the terrain generator.

    Defaults to :class:`isaaclab.terrains.terrain_generator.TerrainGenerator`.
    """

    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed from the
    current NumPy's random state is used.

    When the seed is set, the random number generator is initialized with the given seed. This ensures
    that the generated terrains are deterministic across different runs. If the seed is not set, the
    seed from the current NumPy's random state is used. This assumes that the seed is set elsewhere in
    the code.
    """

    curriculum: bool = False
    """Whether to use the curriculum mode. Defaults to False.

    If True, the terrains are generated based on their difficulty parameter. Otherwise,
    they are randomly generated.
    """

    size: tuple[float, float] = MISSING
    """The width (along x) and length (along y) of each sub-terrain (in m).

    Note:
      This value is passed on to all the sub-terrain configurations.
    """

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0."""

    border_height: float = 1.0
    """The height of the border around the terrain (in m). Defaults to 1.0.

    .. note::
      The default border extends below the ground. If you want to make the border above the ground, choose a negative value.
    """

    num_rows: int = 1
    """Number of rows of sub-terrains to generate. Defaults to 1."""

    num_cols: int = 1
    """Number of columns of sub-terrains to generate. Defaults to 1."""

    color_scheme: Literal["height", "random", "none"] = "none"
    """Color scheme to use for the terrain. Defaults to "none".

    The available color schemes are:

    - "height": Color based on the height of the terrain.
    - "random": Random color scheme.
    - "none": No color scheme.
    """

    horizontal_scale: float = 0.1
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1.

    This value is passed on to all the height field sub-terrain configurations.
    """

    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005.

    This value is passed on to all the height field sub-terrain configurations.
    """

    slope_threshold: float | None = 0.75
    """The slope threshold above which surfaces are made vertical. Defaults to 0.75.

    If None no correction is applied.

    This value is passed on to all the height field sub-terrain configurations.
    """

    sub_terrains: dict[str, SubTerrainBaseCfg] = MISSING
    """Dictionary of sub-terrain configurations.

    The keys correspond to the name of the sub-terrain configuration and the values are the corresponding
    configurations.
    """

    difficulty_range: tuple[float, float] = (0.0, 1.0)
    """The range of difficulty values for the sub-terrains. Defaults to (0.0, 1.0).

    If curriculum is enabled, the terrains will be generated based on this range in ascending order
    of difficulty. Otherwise, the terrains will be generated based on this range in a random order.
    """

    use_cache: bool = False
    """Whether to load the sub-terrain from cache if it exists. Defaults to False.

    If enabled, the generated terrains are stored in the cache directory. When generating terrains, the cache
    is checked to see if the terrain already exists. If it does, the terrain is loaded from the cache. Otherwise,
    the terrain is generated and stored in the cache. Caching can be used to speed up terrain generation.
    """

    cache_dir: str = "/tmp/isaaclab/terrains"
    """The directory where the terrain cache is stored. Defaults to "/tmp/isaaclab/terrains"."""
