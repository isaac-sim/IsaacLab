# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawner configurations that resolve USD assets from SimReady search queries.

Instead of hardcoding USD file paths, these configurations query the SimReady USD-Search service
with a natural-language phrase (e.g. ``"food box"``) and spawn the top-ranked assets. The search
itself is performed by :func:`isaaclab.utils.assets.search_simready_usd_paths`. The service is
queried once, when the configuration is instantiated; no network calls are made during
simulation.

Using this sub-module requires the optional ``simready-search`` package. Install it with
``./isaaclab.sh -i simready`` or ``pip install simready-search``. The service requires HTTP Basic
authentication: set the ``USD_SEARCH_USERNAME`` and ``USD_SEARCH_PASSWORD`` environment variables
before the first search call.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
