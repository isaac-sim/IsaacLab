# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments.

The package is structured as follows:

- ``core``: Core task families maintained as part of Isaac Lab.
- ``contrib``: Contributed task families. These may depend on ``core`` tasks, but
  ``core`` tasks never depend on ``contrib`` tasks.
- ``utils``: These include utility functions for the tasks.

"""

import importlib.metadata
import os
import tomllib

ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

_ext_toml = os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml")
if os.path.exists(_ext_toml):
    with open(_ext_toml, "rb") as _f:
        ISAACLAB_TASKS_METADATA = tomllib.load(_f)
else:
    ISAACLAB_TASKS_METADATA = {}
"""Extension metadata dictionary parsed from the extension.toml file."""

try:
    __version__ = importlib.metadata.version("isaaclab_tasks")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

##
# Register Gym environments.
##

import builtins

from .utils import import_packages

# Guard: AppLauncher._create_app() temporarily removes all "lab" modules from
# sys.modules while creating SimulationApp.  If Kit re-imports this package
# during that window, __init__ runs again and re-registers every gym env.
# We stash a flag on builtins because it is never evicted from sys.modules.
if not getattr(builtins, "_isaaclab_tasks_registered", False):
    _BLACKLIST_PKGS = ["utils", ".mdp", "contrib.humanoid_amp.motions"]
    import_packages(__name__, _BLACKLIST_PKGS)
    builtins._isaaclab_tasks_registered = True
