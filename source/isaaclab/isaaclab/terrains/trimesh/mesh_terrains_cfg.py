# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from ..sub_terrain_cfg import SubTerrainBaseCfg

"""
Different trimesh terrain configurations.
"""


@configclass
class MeshPlaneTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.flat_terrain


@configclass
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = mesh_terrains.pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float = MISSING
    """The width of the steps (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    platform_height: float = -1.0
    """The height of the platform.  Defaults to -1.0.

    If the value is negative, the height is the same as the object height."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """


@configclass
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function = mesh_terrains.inverted_pyramid_stairs_terrain


@configclass
class MeshRandomGridTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a random grid mesh terrain."""

    function = mesh_terrains.random_grid_terrain

    grid_width: float = MISSING
    """The width of the grid cells (in m)."""
    grid_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the grid cells (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have randomized grid cells only along the plane extending
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.
    """


@configclass
class MeshRailsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with box rails as extrusions."""

    function = mesh_terrains.rails_terrain

    rail_thickness_range: tuple[float, float] = MISSING
    """The thickness of the inner and outer rails (in m)."""
    rail_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the rails (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshPitTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a pit that leads out of the pit."""

    function = mesh_terrains.pit_terrain

    pit_depth_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the pit (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    double_pit: bool = False
    """If True, the pit contains two levels of stairs. Defaults to False."""


@configclass
class MeshBoxTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""

    function = mesh_terrains.box_terrain

    box_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the box (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    double_box: bool = False
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""


@configclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the platform."""

    function = mesh_terrains.gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gap (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshFloatingRingTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a floating ring around the center."""

    function = mesh_terrains.floating_ring_terrain

    ring_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the ring (in m)."""
    ring_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the ring (in m)."""
    ring_thickness: float = MISSING
    """The thickness (along z) of the ring (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshStarTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a star pattern."""

    function = mesh_terrains.star_terrain

    num_bars: int = MISSING
    """The number of bars per-side the star. Must be greater than 2."""
    bar_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the bars in the star (in m)."""
    bar_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the bars in the star (in m)."""
    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshRepeatedObjectsTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for a terrain with repeated objects."""

    @configclass
    class ObjectCfg:
        """Configuration of repeated objects."""

        num_objects: int = MISSING
        """The number of objects to add to the terrain."""
        height: float = MISSING
        """The height (along z) of the object (in m)."""

    function = mesh_terrains.repeated_objects_terrain

    object_type: Literal["cylinder", "box", "cone"] | callable = MISSING
    """The type of object to generate.

    The type can be a string or a callable. If it is a string, the function will look for a function called
    ``make_{object_type}`` in the current module scope. If it is a callable, the function will
    use the callable to generate the object.
    """
    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""

    max_height_noise: float | None = None
    """"This parameter is deprecated, but stated here to support backward compatibility"""

    abs_height_noise: tuple[float, float] = (0.0, 0.0)
    """The minimum and maximum amount of additive noise for the height of the objects. Default is set to 0.0, which is no noise."""

    rel_height_noise: tuple[float, float] = (1.0, 1.0)
    """The minimum and maximum amount of multiplicative noise for the height of the objects. Default is set to 1.0, which is no noise."""

    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""

    def __post_init__(self):
        if self.max_height_noise is not None:
            warnings.warn(
                "MeshRepeatedObjectsTerrainCfg: max_height_noise:float is deprecated and support will be removed in the"
                " future. Use abs_height_noise:list[float] instead."
            )
            self.abs_height_noise = (-self.max_height_noise, self.max_height_noise)


@configclass
class MeshRepeatedPyramidsTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated pyramids."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for a curriculum of repeated pyramids."""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_cone

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""


@configclass
class MeshRepeatedBoxesTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated boxes."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated boxes."""

        size: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_box

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""


@configclass
class MeshRepeatedCylindersTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated cylinders."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated cylinder."""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_cylinder

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""
