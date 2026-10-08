# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source locations for the berry task assets: a Nucleus URL or a local override.

:func:`isaaclab.utils.assets.retrieve_file_path` resolves and locally caches whichever
of these is returned, refreshing the cache when the server reports a newer revision;
no separate download step is required.
"""

import os

# Temporary: a personal Nucleus folder, shared with other demos. To be replaced by the public asset repository
# (see "Known limitations" in the README).
_SHARED_BUNDLE_ROOT = "omniverse://content.ov.nvidia.com/Users/nicolasm@nvidia.com/isaaclab_assets/tasks"


def raspberry_asset_path() -> str:
    """Source of the raspberry USDZ package: Gaussians, tissue particles, simulation metadata and shader.

    ``ISAACLAB_BERRY_ASSET_ROOT`` overrides the root with a berry-only one (a future clean republish, or a local
    directory for testing), holding ``raspberry/raspberry_v3.usdz``. The default is the shared bundle's ``berry/``
    subtree.
    """
    override = os.environ.get("ISAACLAB_BERRY_ASSET_ROOT")
    root = override.rstrip("/") if override else f"{_SHARED_BUNDLE_ROOT}/berry"
    return f"{root}/raspberry/raspberry_v3.usdz"


def room_scan_root() -> str:
    """Source root of the room scan: ``<root>/aligned.usda`` + ``point_cloud.usd``.

    Reuses the ``ISAACLAB_BERRY_ASSET_ROOT`` override, expected to contain a
    ``background/ebc`` subtree there. The default falls back to the shared bundle's
    ``gaussian_twin/background/ebc`` subtree, where the room scan currently lives.
    """
    override = os.environ.get("ISAACLAB_BERRY_ASSET_ROOT")
    if override:
        return f"{override.rstrip('/')}/background/ebc"
    return f"{_SHARED_BUNDLE_ROOT}/gaussian_twin/background/ebc"
