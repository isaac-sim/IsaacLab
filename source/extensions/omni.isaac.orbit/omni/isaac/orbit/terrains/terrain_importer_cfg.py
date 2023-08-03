# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING
from typing_extensions import Literal

from omni.isaac.orbit.utils import configclass

from .terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from .terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class TerrainImporterCfg:
    """Configuration for the terrain manager."""

    cls_name: type = TerrainImporter
    """The class name of the terrain importer."""

    prim_path: str = MISSING
    """The absolute path of the USD terrain prim.

    All sub-terrains are imported relative to this prim path.
    """

    num_envs: int = MISSING
    """The number of environment origins to consider."""

    terrain_type: Literal["generator", "plane", "usd"] = "generator"
    """The type of terrain to generate. Defaults to "generator".

    Available options are "plane", "usd", and "generator".
    """

    terrain_generator: TerrainGeneratorCfg | None = None
    """The terrain generator configuration.

    Only used if ``terrain_type`` is set to "generator".
    """

    usd_path: str | None = None
    """The path to the USD file containing the terrain.

    Only used if ``terrain_type`` is set to "usd".
    """

    env_spacing: float | None = None
    """The spacing between environment origins when defined in a grid. Defaults to None.

    Note:
      This parameter is used only when the ``terrain_type`` is ``"plane"`` or ``"usd"``.
    """

    color: tuple[float, float, float] | None = (0.065, 0.0725, 0.080)
    """The color of the terrain. Defaults to (0.065, 0.0725, 0.080).

    If :obj:`None`, no color is applied to the prim.
    """

    static_friction: float = 1.0
    """The static friction coefficient of the terrain. Defaults to 1.0."""

    dynamic_friction: float = 1.0
    """The dynamic friction coefficient of the terrain. Defaults to 1.0."""

    restitution: float = 0.0
    """The restitution coefficient of the terrain. Defaults to 0.0."""

    improve_patch_friction: bool = False
    """Whether to enable patch friction. Defaults to False."""

    combine_mode: str = "average"
    """Determines the way physics materials will be combined during collisions. Defaults to `average`.

    Available options are `average`, `min`, `multiply`, `multiply`, and `max`.
    """

    max_init_terrain_level: int | None = None
    """The maximum initial terrain level for defining environment origins. Defaults to None.

    The terrain levels are specified by the number of rows in the grid arrangement of
    sub-terrains. If :obj:`None`, then the initial terrain level is set to the maximum
    terrain level available (``num_rows - 1``).

    Note:
      This parameter is used only when sub-terrain origins are defined.
    """
