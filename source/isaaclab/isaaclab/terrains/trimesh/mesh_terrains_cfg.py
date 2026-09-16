# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import MISSING, dataclass
from typing import Literal

from isaaclab.utils import config_field

from ..sub_terrain_cfg import SubTerrainBaseCfg

"""
Different trimesh terrain configurations.
"""


@dataclass
class MeshPlaneTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function: str = config_field("{DIR}.mesh_terrains:flat_terrain")


@dataclass
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function: str = config_field("{DIR}.mesh_terrains:pyramid_stairs_terrain")

    border_width: float = config_field(0.0)
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """

    step_height_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = config_field(MISSING)
    """The width of the steps (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    holes: bool = config_field(False)
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """


@dataclass
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function: str = config_field("{DIR}.mesh_terrains:inverted_pyramid_stairs_terrain")


@dataclass
class MeshRandomGridTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a random grid mesh terrain."""

    function: str = config_field("{DIR}.mesh_terrains:random_grid_terrain")

    grid_width: float = config_field(MISSING)
    """The width of the grid cells (in m)."""

    grid_height_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the grid cells (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    holes: bool = config_field(False)
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have randomized grid cells only along the plane extending
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.
    """


@dataclass
class MeshRailsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with box rails as extrusions."""

    function: str = config_field("{DIR}.mesh_terrains:rails_terrain")

    rail_thickness_range: tuple[float, float] = config_field(MISSING)
    """The thickness of the inner and outer rails (in m)."""

    rail_height_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the rails (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@dataclass
class MeshPitTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a pit that leads out of the pit."""

    function: str = config_field("{DIR}.mesh_terrains:pit_terrain")

    pit_depth_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the pit (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    double_pit: bool = config_field(False)
    """If True, the pit contains two levels of stairs. Defaults to False."""


@dataclass
class MeshBoxTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""

    function: str = config_field("{DIR}.mesh_terrains:box_terrain")

    box_height_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the box (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    double_box: bool = config_field(False)
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""


@dataclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the platform."""

    function: str = config_field("{DIR}.mesh_terrains:gap_terrain")

    gap_width_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum width of the gap (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@dataclass
class MeshFloatingRingTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a floating ring around the center."""

    function: str = config_field("{DIR}.mesh_terrains:floating_ring_terrain")

    ring_width_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum width of the ring (in m)."""

    ring_height_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the ring (in m)."""

    ring_thickness: float = config_field(MISSING)
    """The thickness (along z) of the ring (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@dataclass
class MeshStarTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a star pattern."""

    function: str = config_field("{DIR}.mesh_terrains:star_terrain")

    num_bars: int = config_field(MISSING)
    """The number of bars per-side the star. Must be greater than 2."""

    bar_width_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum width of the bars in the star (in m)."""

    bar_height_range: tuple[float, float] = config_field(MISSING)
    """The minimum and maximum height of the bars in the star (in m)."""

    platform_width: float = config_field(1.0)
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""


@dataclass
class MeshRepeatedObjectsTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for a terrain with repeated objects."""

    @dataclass
    class ObjectCfg:
        """Configuration of repeated objects."""

        num_objects: int = config_field(MISSING)
        """The number of objects to add to the terrain."""
        height: float = config_field(MISSING)
        """The height (along z) of the object (in m)."""

    function: str = config_field("{DIR}.mesh_terrains:repeated_objects_terrain")

    object_type: Literal["cylinder", "box", "cone"] | callable = config_field(MISSING)
    """The type of object to generate.

    The type can be a string or a callable. If it is a string, the function will look for a function called
    ``make_{object_type}`` in the current module scope. If it is a callable, the function will
    use the callable to generate the object.
    """

    object_params_start: ObjectCfg = config_field(MISSING)
    """The object curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = config_field(MISSING)
    """The object curriculum parameters at the end of the curriculum."""

    max_height_noise: float | None = config_field(None)
    """"This parameter is deprecated, but stated here to support backward compatibility"""

    abs_height_noise: tuple[float, float] = config_field((0.0, 0.0))
    """The minimum and maximum amount of additive noise for the height of the objects. Default is set to 0.0,
    which is no noise.
    """

    rel_height_noise: tuple[float, float] = config_field((1.0, 1.0))
    """The minimum and maximum amount of multiplicative noise for the height of the objects. Default is set to 1.0,
    which is no noise.
    """

    platform_width: float = config_field(1.0)
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""

    platform_height: float = config_field(-1.0)
    """The height of the platform. Defaults to -1.0.

    If the value is negative, the height is the same as the object height.
    """

    def __post_init__(self):
        if self.max_height_noise is not None:
            warnings.warn(
                "MeshRepeatedObjectsTerrainCfg: max_height_noise:float is deprecated and support will be removed in the"
                " future. Use abs_height_noise:list[float] instead."
            )
            self.abs_height_noise = (-self.max_height_noise, self.max_height_noise)


@dataclass
class MeshRepeatedPyramidsTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated pyramids."""

    @dataclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for a curriculum of repeated pyramids."""

        radius: float = config_field(MISSING)
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = config_field(0.0)
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = config_field(True)
        """Whether the angle is in degrees. Defaults to True."""

    object_type: str = config_field("{DIR}.utils:make_cone")

    object_params_start: ObjectCfg = config_field(MISSING)
    """The object curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = config_field(MISSING)
    """The object curriculum parameters at the end of the curriculum."""


@dataclass
class MeshRepeatedBoxesTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated boxes."""

    @dataclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated boxes."""

        size: tuple[float, float] = config_field(MISSING)
        """The width (along x) and length (along y) of the box (in m)."""
        max_yx_angle: float = config_field(0.0)
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = config_field(True)
        """Whether the angle is in degrees. Defaults to True."""

    object_type: str = config_field("{DIR}.utils:make_box")

    object_params_start: ObjectCfg = config_field(MISSING)
    """The box curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = config_field(MISSING)
    """The box curriculum parameters at the end of the curriculum."""


@dataclass
class MeshRepeatedCylindersTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated cylinders."""

    @dataclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated cylinder."""

        radius: float = config_field(MISSING)
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = config_field(0.0)
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = config_field(True)
        """Whether the angle is in degrees. Defaults to True."""

    object_type: str = config_field("{DIR}.utils:make_cylinder")

    object_params_start: ObjectCfg = config_field(MISSING)
    """The box curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = config_field(MISSING)
    """The box curriculum parameters at the end of the curriculum."""
