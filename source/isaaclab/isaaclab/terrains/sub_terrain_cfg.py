# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    import numpy as np
    import trimesh


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
