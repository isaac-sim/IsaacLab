# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset identifiers and contact conventions shared across task modules."""

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Contrib/Shoelace"

# Runtime assets; texture paths are resolved relative to each USD.
SHOELACE_ASSET = f"{ASSET_DIR}/shoelace.usda"  # Base asset
SHOELACE_ASSETS = (
    SHOELACE_ASSET,
    *(
        f"{ASSET_DIR}/{name}"
        for name in (
            "canvas_sand.usda",
            "canvas_rust.usda",
            "suede_sage.usda",
            "suede_burgundy.usda",
            "knit_ocean.usda",
            "knit_lavender.usda",
        )
    ),
)

# Keep the scene default and cable builder contact gaps consistent [m].
CONTACT_GAP = 1.0e-4
# No-candidate sensor value, also used for observation scaling and grasp validity [m].
CONTACT_DISTANCE_CAP = 2.0e-3
