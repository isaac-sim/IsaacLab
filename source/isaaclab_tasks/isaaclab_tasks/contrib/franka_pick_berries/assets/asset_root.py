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

# The published bundle is currently shared with the tomato and gaussian_twin tasks.
_SHARED_BUNDLE_ROOT = "omniverse://content.ov.nvidia.com/Users/nicolasm@nvidia.com/isaaclab_assets/tasks"


def berry_root() -> str:
    """Source root for the berry USDZ packages: ``<root>/<berry>/<berry>[_<version>].usdz``.

    ``ISAACLAB_BERRY_ASSET_ROOT`` overrides this with a berry-only root (a future clean
    republish, or a local directory for testing). The default falls back to the shared
    bundle's ``berry/`` subtree.
    """
    override = os.environ.get("ISAACLAB_BERRY_ASSET_ROOT")
    return override.rstrip("/") if override else f"{_SHARED_BUNDLE_ROOT}/berry"


def background_root() -> str:
    """Source root for the EBC background layer: ``<root>/aligned.usda`` + ``point_cloud.usd``.

    Reuses the ``ISAACLAB_BERRY_ASSET_ROOT`` override, expected to contain a
    ``background/ebc`` subtree there. The default falls back to the shared bundle's
    ``gaussian_twin/background/ebc`` subtree, where the EBC background currently lives.
    """
    override = os.environ.get("ISAACLAB_BERRY_ASSET_ROOT")
    if override:
        return f"{override.rstrip('/')}/background/ebc"
    return f"{_SHARED_BUNDLE_ROOT}/gaussian_twin/background/ebc"
