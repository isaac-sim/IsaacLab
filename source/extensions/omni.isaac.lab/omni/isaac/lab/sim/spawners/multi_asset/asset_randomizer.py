# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pxr import Usd

if TYPE_CHECKING:
    from .asset_randomizer_cfg import AssetRandomizerCfg


def randomize_and_spawn_asset(
    prim_path: str,
    cfg: AssetRandomizerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Randomizes and spawns an asset.
    Args:
        prim_path: The path to the asset to spawn
        cfg: The configuration for the asset randomization
        translation:  The translation to apply to the prim w.r.t. its parent prim in meters. Defaults to None, in which case this is set to the origin. Gets passed to the child spawner defined in the configuration
        orientation:  The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None, in which case this is set to identity (1, 0, 0, 0). Gets passed to the child spawner defined in the configuration
    """
    prim = cfg.child_spawner_cfg.func(prim_path, cfg.child_spawner_cfg, translation, orientation)
    randomizations = cfg.randomization_cfg

    if not isinstance(randomizations, Iterable):
        randomizations = [randomizations]

    for randomization_cfg in randomizations:
        prim = randomization_cfg.func(prim, randomization_cfg)

    return prim
