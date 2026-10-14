# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source locations for the berry task assets: Isaac Lab's Nucleus asset root or a local override.

:func:`isaaclab.utils.assets.retrieve_file_path` resolves and locally caches whichever
of these is returned, refreshing the cache when the server reports a newer revision;
no separate download step is required.
"""

import os

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def _asset_root() -> str:
    """The assets' root: ``ISAACLAB_BERRY_ASSET_ROOT`` if set, else :data:`~isaaclab.utils.assets.ISAACLAB_NUCLEUS_DIR`.

    The override (another server, or a local directory for testing) mirrors the Nucleus layout.
    """
    override = os.environ.get("ISAACLAB_BERRY_ASSET_ROOT")
    return override.rstrip("/") if override else ISAACLAB_NUCLEUS_DIR


def raspberry_asset_path() -> str:
    """Source of the raspberry USDZ package: Gaussians, tissue particles, tetrahedral mesh, simulation metadata and
    shader."""
    return f"{_asset_root()}/Objects/Berries/raspberry.usdz"


def punnet_asset_path(name: str) -> str:
    """Source of a punnet's visual USDZ package, ``name``, next to the raspberry's."""
    return f"{_asset_root()}/Objects/Berries/{name}"


def room_scan_path() -> str:
    """Source of the aligned room scan package."""
    return f"{_asset_root()}/Environments/NuRec/hand_hold-endeavor-livingroom_aligned.usdz"
