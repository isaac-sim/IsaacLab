# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils import configclass

from . import wrappers


@configclass
class MultiAssetSpawnerCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for loading multiple assets from their individual configurations.

    Specifying values for any properties at the configuration level will override the settings of
    individual assets' configuration. For instance if the attribute
    :attr:`MultiAssetSpawnerCfg.mass_props` is specified, its value will overwrite the values of the
    mass properties in each configuration inside :attr:`assets_cfg` (wherever applicable).
    This is done to simplify configuring similar properties globally. By default, all properties are set to None.

    The following is an exception to the above:

    * :attr:`visible`: This parameter is ignored. Its value for the individual assets is used.
    * :attr:`semantic_tags`: If specified, it will be appended to each individual asset's semantic tags.

    """

    func = wrappers.spawn_multi_asset

    assets_cfg: list[SpawnerCfg] = MISSING
    """List of asset configurations to spawn."""

    random_choice: bool = True
    """Whether to randomly select an asset configuration. Default is True.

    If False, the asset configurations are spawned in the order they are provided in the list.
    If True, a random asset configuration is selected for each spawn.
    """


@configclass
class MultiUsdFileCfg(UsdFileCfg):
    """Configuration parameters for loading multiple USD files.

    Specifying values for any properties at the configuration level is applied to all the assets
    imported from their USD files.

    .. tip::
        It is recommended that all the USD based assets follow a similar prim-hierarchy.

    """

    func = wrappers.spawn_multi_usd_file

    usd_path: str | list[str] = MISSING
    """Path or a list of paths to the USD files to spawn asset from."""

    random_choice: bool = True
    """Whether to randomly select an asset configuration. Default is True.

    If False, the asset configurations are spawned in the order they are provided in the list.
    If True, a random asset configuration is selected for each spawn.
    """
