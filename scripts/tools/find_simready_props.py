# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the SimReady props of the Kuka-Allegro ``simready`` object preset.

Searches the SimReady service for single-body PhysX props by size and mass and writes their paths, relative to
the SimReady asset folder, with the search that produced them. The task reads the file, so it never searches.

.. code-block:: bash

    uv run python scripts/tools/find_simready_props.py
"""

import json
import pathlib

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, search_simready_usd_paths

OUTPUT = pathlib.Path("source/isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/simready_objects.json")
SEARCH = {
    "size_range": [0.02, 0.15],  # [m] along every axis: what the hand can grasp
    "mass_range": [0.005, 3.0],  # [kg]: what the hand can hold
    "features": ["FET003_BASE_PHYSX"],  # PhysX rigid body ...
    "exclude_features": ["FET004_BASE_PHYSX"],  # ... with a single body
    "limit": 100,
}

paths = search_simready_usd_paths(**SEARCH)
relative = [path.removeprefix(f"{ISAAC_NUCLEUS_DIR}/SimReady/") for path in paths]
OUTPUT.write_text(json.dumps({"search": SEARCH, "usd_paths": relative}, indent=2) + "\n")
print(f"Wrote {len(relative)} props to {OUTPUT}")
