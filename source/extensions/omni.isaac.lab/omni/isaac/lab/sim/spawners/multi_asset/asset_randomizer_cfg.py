# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for loading multiple assets randomly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from .asset_randomizer import randomize_and_spawn_asset
from .randomizations.randomizations_cfg import RandomizationCfg


@configclass
class AssetRandomizerCfg(sim_utils.SpawnerCfg):
    """Configuration for loading multiple assets randomly."""

    func: sim_utils.SpawnerCfg.func = randomize_and_spawn_asset

    num_random_assets: int = 1
    """The number of random assets to spawn"""

    base_seed: int = 0
    """The seed to use for randomization for replicability"""

    child_spawner_cfg: sim_utils.SpawnerCfg = MISSING
    """The configuration to use to spawn each asset"""

    randomization_cfg: RandomizationCfg | Iterable[RandomizationCfg] = MISSING
    """Optional randomization configuration that is applied at the time of spawning prims."""
