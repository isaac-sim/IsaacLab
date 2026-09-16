# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, dataclass

from isaaclab.utils import config_field

from ..sub_terrain_cfg import SubTerrainBaseCfg


@dataclass
class HfTerrainBaseCfg(SubTerrainBaseCfg):
    """The base configuration for height field terrains."""

    convert_to_heightfield: bool = config_field(True)
    """Whether the sub-terrain should be converted to a heightfield. Defaults to True.

    Height field terrains are generated from a height field, so the conversion reproduces them exactly.
    """

    border_width: float = config_field(0.0)
    """The width of the border/padding around the terrain (in m). Defaults to 0.0.

    The border width is subtracted from the :obj:`size` of the terrain. If non-zero, it must be
    greater than or equal to the :obj:`horizontal scale`.
    """

    horizontal_scale: float = config_field(0.1)
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""

    vertical_scale: float = config_field(0.005)
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""

    slope_threshold: float | None = config_field(None)
    """The slope threshold above which surfaces are made vertical. Defaults to None,
    in which case no correction is applied."""


"""
Different height field terrain configurations.
"""


@dataclass
class HfRandomUniformTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function: str = config_field("{DIR}.hf_terrains:random_uniform_terrain")

    noise_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""

    noise_step: float = config_field(MISSING)
    """The minimum height (in m) change between two points."""

    downsampled_scale: float | None = config_field(None)
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


@dataclass
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid sloped height field terrain."""

    function: str = config_field("{DIR}.hf_terrains:pyramid_sloped_terrain")

    slope_range: tuple[float, float] = config_field(MISSING)
    """The slope of the terrain (in radians)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = config_field(False)
    """Whether the pyramid is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the slopes are upwards.
    """


@dataclass
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Configuration for an inverted pyramid sloped height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidSlopedTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """

    inverted: bool = config_field(True)


@dataclass
class HfPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid stairs height field terrain."""

    function: str = config_field("{DIR}.hf_terrains:pyramid_stairs_terrain")

    step_height_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = config_field(MISSING)
    """The width of the steps (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = config_field(False)
    """Whether the pyramid stairs is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the stairs are upwards.
    """


@dataclass
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stairs height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidStairsTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """

    inverted: bool = config_field(True)


@dataclass
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function: str = config_field("{DIR}.hf_terrains:discrete_obstacles_terrain")

    obstacle_height_mode: str = config_field("choice")
    """The mode to use for the obstacle height. Defaults to "choice".

    The following modes are supported: "choice", "fixed".
    """

    obstacle_width_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum width of the obstacles (in m)."""

    obstacle_height_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the obstacles (in m)."""

    num_obstacles: int = config_field(MISSING)
    """The number of obstacles to generate."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@dataclass
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a wave height field terrain."""

    function: str = config_field("{DIR}.hf_terrains:wave_terrain")

    amplitude_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum amplitude of the wave (in m)."""

    num_waves: int = config_field(1)
    """The number of waves to generate. Defaults to 1."""


@dataclass
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""

    function: str = config_field("{DIR}.hf_terrains:stepping_stones_terrain")

    stone_height_max: float = config_field(MISSING)
    """The maximum height of the stones (in m)."""

    stone_width_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum distance between stones (in m)."""

    holes_depth: float = config_field(-10.0)
    """The depth of the holes (negative obstacles). Defaults to -10.0."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
