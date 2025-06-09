# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from ..sub_terrain_cfg import SubTerrainBaseCfg
from . import hf_terrains


@configclass
class HfTerrainBaseCfg(SubTerrainBaseCfg):
    """The base configuration for height field terrains."""

    border_width: float = 0.0
    """The width of the border/padding around the terrain (in m). Defaults to 0.0.

    The border width is subtracted from the :obj:`size` of the terrain. If non-zero, it must be
    greater than or equal to the :obj:`horizontal scale`.
    """
    horizontal_scale: float = 0.1
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""
    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""
    slope_threshold: float | None = None
    """The slope threshold above which surfaces are made vertical. Defaults to None,
    in which case no correction is applied."""


"""
Different height field terrain configurations.
"""


@configclass
class HfRandomUniformTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = hf_terrains.random_uniform_terrain

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


@configclass
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid sloped height field terrain."""

    function = hf_terrains.pyramid_sloped_terrain

    slope_range: tuple[float, float] = MISSING
    """The slope of the terrain (in radians)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the slopes are upwards.
    """


@configclass
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Configuration for an inverted pyramid sloped height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidSlopedTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """

    inverted: bool = True


@configclass
class HfPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid stairs height field terrain."""

    function = hf_terrains.pyramid_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float = MISSING
    """The width of the steps (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    inverted: bool = False
    """Whether the pyramid stairs is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the stairs are upwards.
    """


@configclass
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stairs height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidStairsTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """

    inverted: bool = True


@configclass
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function = hf_terrains.discrete_obstacles_terrain

    obstacle_height_mode: str = "choice"
    """The mode to use for the obstacle height. Defaults to "choice".

    The following modes are supported: "choice", "fixed".
    """
    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""
    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the obstacles (in m)."""
    num_obstacles: int = MISSING
    """The number of obstacles to generate."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a wave height field terrain."""

    function = hf_terrains.wave_terrain

    amplitude_range: tuple[float, float] = MISSING
    """The minimum and maximum amplitude of the wave (in m)."""
    num_waves: int = 1.0
    """The number of waves to generate. Defaults to 1.0."""


@configclass
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""

    function = hf_terrains.stepping_stones_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""
    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""
    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""
    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
