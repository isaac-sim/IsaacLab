# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.utils import configclass


@configclass
class MjcfConverterCfg(AssetConverterBaseCfg):
    """The configuration class for MjcfConverter.

    .. note::
        From Isaac Sim 5.0 onwards, the MJCF importer was rewritten to use the ``mujoco-usd-converter``
        library. Several settings from the old importer (``fix_base``, ``link_density``,
        ``import_inertia_tensor``, ``import_sites``) are no longer available as they are handled
        automatically by the converter based on the MJCF file content.

    .. note::
        The :attr:`~AssetConverterBaseCfg.make_instanceable` setting from the base class is not
        supported by the new MJCF importer and will be ignored.
    """

    merge_mesh: bool = False
    """Merge meshes where possible to optimize the model. Defaults to False."""

    collision_from_visuals: bool = False
    """Generate collision geometry from visual geometries. Defaults to False."""

    collision_type: str = "default"
    """Type of collision geometry to use. Defaults to ``"default"``.

    Supported values are ``"default"``, ``"Convex Hull"``, and ``"Convex Decomposition"``.
    """

    self_collision: bool = False
    """Activate self-collisions between links of the articulation. Defaults to False."""
