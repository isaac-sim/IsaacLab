# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``omni.isaac.orbit.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import MISSING
from typing import Callable
from typing_extensions import Literal

from omni.isaac.orbit.utils import configclass


@configclass
class SubTerrainBaseCfg:
    """Base class for terrain configurations.

    All the sub-terrain configurations must inherit from this class.

    The :attr:`size` attribute is the size of the generated sub-terrain. Based on this, the terrain must
    extend from :math:`(0, 0)` to :math:`(size[0], size[1])`.
    """

    function: Callable[[float, SubTerrainBaseCfg], tuple[list[trimesh.Trimesh], np.ndarray]] = MISSING
    """Function to generate the terrain.

    This function must take as input the terrain difficulty and the configuration parameters and
    return a tuple with a list of ``trimesh`` mesh objects and the terrain origin.
    """

    proportion: float = 1.0
    """Proportion of the terrain to generate. Defaults to 1.0.

    This is used to generate a mix of terrains. The proportion corresponds to the probability of sampling
    the particular terrain. For example, if there are two terrains, A and B, with proportions 0.3 and 0.7,
    respectively, then the probability of sampling terrain A is 0.3 and the probability of sampling terrain B
    is 0.7.
    """

    size: tuple[float, float] = MISSING
    """The width (along x) and length (along y) of the terrain (in m)."""


@configclass
class TerrainGeneratorCfg:
    """Configuration for the terrain generator."""

    seed: int | None = None
    """The seed for the random number generator. Defaults to None,
    in which case the seed is not set."""

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
    """List of sub-terrain configurations."""

    difficulty_choices: list[float] = [0.5, 0.75, 0.9]
    """List of difficulty choices. Defaults to [0.5, 0.75, 0.9].

    The difficulty choices are used to sample the difficulty of the generated terrain. The specified
    choices are randomly sampled with equal probability.

    Note:
      This is used only when curriculum-based generation is disabled.
    """

    use_cache: bool = False
    """Whether to load the terrain from cache if it exists. Defaults to True."""

    cache_dir: str = "/tmp/orbit/terrains"
    """The directory where the terrain cache is stored. Defaults to "/tmp/orbit/terrains"."""
