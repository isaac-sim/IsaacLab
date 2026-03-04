# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments.

The package is structured as follows:

- ``direct``: These include single-file implementations of tasks.
- ``manager_based``: These include task implementations that use the manager-based API.
- ``utils``: These include utility functions for the tasks.

"""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

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
    _BLACKLIST_PKGS = ["utils", ".mdp", "direct.humanoid_amp.motions"]
    import_packages(__name__, _BLACKLIST_PKGS)
    builtins._isaaclab_tasks_registered = True
