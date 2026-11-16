# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Search the SimReady service for USD assets and record their paths with the search that found them.

The options are the arguments of :func:`isaaclab.utils.assets.search_simready_usd_paths`; ``--param`` sends any other
field of the service's ``/search_hybrid`` request as given. The output holds the paths, relative to the SimReady asset
folder, and the search as keyword arguments, so ``search_simready_usd_paths(**search)`` repeats it.

The props of the Kuka-Allegro ``simready`` object preset were recorded with:

.. code-block:: bash

    uv run python scripts/tools/search_simready.py --size_range 0.02 0.15 --mass_range 0.005 3.0 \\
        --features FET003_BASE_PHYSX --exclude_features FET004_BASE_PHYSX \\
        --output source/isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/simready_objects.json
"""

import argparse
import json
import pathlib

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, search_simready_usd_paths

parser = argparse.ArgumentParser(description="Search SimReady assets and record their paths with the search.")
parser.add_argument("--query", help="Short description of the assets.")
parser.add_argument(
    "--size_range", nargs=2, type=float, metavar=("MIN", "MAX"), help="Bounding box extent [m] along every axis."
)
parser.add_argument("--mass_range", nargs=2, type=float, metavar=("MIN", "MAX"), help="Authored mass [kg].")
parser.add_argument("--features", nargs="+", default=[], help="SimReady features every asset must have passed.")
parser.add_argument("--exclude_features", nargs="+", default=[], help="SimReady features no asset may have passed.")
parser.add_argument("--limit", type=int, default=100, help="Maximum number of assets.")
parser.add_argument(
    "--param",
    nargs=2,
    action="append",
    default=[],
    metavar=("FIELD", "VALUE"),
    help="Any other /search_hybrid request field, e.g. --param deduplicate_by_hash true. VALUE is read as JSON, else"
    " as a string.",
)
parser.add_argument("--output", type=pathlib.Path, required=True, help="JSON file to write.")
args = parser.parse_args()

# 1. the search as keyword arguments: the options given, then the request fields passed through
search = {
    name: value for name, value in vars(args).items() if name not in ("param", "output") and value not in (None, [])
}
for field, value in args.param:
    try:
        search[field] = json.loads(value)
    except json.JSONDecodeError:
        search[field] = value
# 2. search, and record the paths relative to the SimReady asset folder
paths = search_simready_usd_paths(**search)
relative = [path.removeprefix(f"{ISAAC_NUCLEUS_DIR}/SimReady/") for path in paths]
args.output.write_text(json.dumps({"search": search, "usd_paths": relative}, indent=2) + "\n")
print(f"Wrote {len(relative)} assets to {args.output}")
