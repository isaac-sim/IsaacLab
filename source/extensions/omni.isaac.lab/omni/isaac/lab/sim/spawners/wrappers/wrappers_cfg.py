# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim.spawners.spawner_cfg import SpawnerCfg

from . import wrappers


@configclass
class MultiAssetSpawnerCfg:
    """Configuration parameters for loading multiple assets randomly."""

    func = wrappers.spawn_multi_asset
    """Function to spawn the asset."""

    assets_cfg: list[SpawnerCfg] = MISSING
    """List of asset configurations to spawn."""

    random_choice: bool = True
    """Whether to randomly select an asset configuration. Default is True.

    If False, the asset configurations are spawned in the order they are provided in the list.
    If True, a random asset configuration is selected for each spawn.
    """
