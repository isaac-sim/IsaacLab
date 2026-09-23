# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the built-in Isaac Lab tasks with Gymnasium on import.

Import this module before listing or creating a built-in environment by name.
Importing :mod:`isaaclab_tasks` or one of its task modules does not register
unrelated environments.
"""

import builtins

from .utils import import_packages

# AppLauncher._create_app() may evict this module from sys.modules while
# creating SimulationApp. Keep the guard outside the module cache so Kit's
# re-import does not register every environment a second time.
if not getattr(builtins, "_isaaclab_tasks_registered", False):
    _BLACKLIST_PKGS = ["utils", ".mdp", "contrib.humanoid_amp.motions"]
    import_packages("isaaclab_tasks", _BLACKLIST_PKGS)
    builtins._isaaclab_tasks_registered = True
