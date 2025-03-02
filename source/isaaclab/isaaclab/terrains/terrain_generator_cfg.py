# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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

import numpy as np
import trimesh
from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class FlatPatchSamplingCfg:
    """Configuration for sampling flat patches on the sub-terrain.

    For a given sub-terrain, this configuration specifies how to sample flat patches on the terrain.
    The sampled flat patches can be used for spawning robots, targets, etc.

    Please check the function :meth:`~isaaclab.terrains.utils.find_flat_patches` for more details.
    """

    num_patches: int = MISSING
    """Number of patches to sample."""

    patch_radius: float | list[float] = MISSING
    """Radius of the patches.

    A list of radii can be provided to check for patches of different sizes. This is useful to deal with
    cases where the terrain may have holes or obstacles in some areas.
    """

    x_range: tuple[float, float] = (-1e6, 1e6)
    """The range of x-coordinates to sample from. Defaults to (-1e6, 1e6).

    This range is internally clamped to the size of the terrain mesh.
    """

    y_range: tuple[float, float] = (-1e6, 1e6)
    """The range of y-coordinates to sample from. Defaults to (-1e6, 1e6).

    This range is internally clamped to the size of the terrain mesh.
    """

    z_range: tuple[float, float] = (-1e6, 1e6)
    """Allowed range of z-coordinates for the sampled patch. Defaults to (-1e6, 1e6)."""

    max_height_diff: float = MISSING
    """Maximum allowed height difference between the highest and lowest points on the patch."""


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

    size: tuple[float, float] = (10.0, 10.0)
    """The width (along x) and length (along y) of the terrain (in m). Defaults to (10.0, 10.0).

    In case the :class:`~isaaclab.terrains.TerrainImporterCfg` is used, this parameter gets overridden by
    :attr:`isaaclab.scene.TerrainImporterCfg.size` attribute.
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    """Dictionary of configurations for sampling flat patches on the sub-terrain. Defaults to None,
    in which case no flat patch sampling is performed.

    The keys correspond to the name of the flat patch sampling configuration and the values are the
    corresponding configurations.
    """


@configclass
class TerrainGeneratorCfg:
    """Configuration for the terrain generator."""

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
    """The height of the border around the terrain (in m). Defaults to 1.0."""

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
    """Whether to load the sub-terrain from cache if it exists. Defaults to True.

    If enabled, the generated terrains are stored in the cache directory. When generating terrains, the cache
    is checked to see if the terrain already exists. If it does, the terrain is loaded from the cache. Otherwise,
    the terrain is generated and stored in the cache. Caching can be used to speed up terrain generation.
    """

    cache_dir: str = "/tmp/isaaclab/terrains"
    """The directory where the terrain cache is stored. Defaults to "/tmp/isaaclab/terrains"."""
