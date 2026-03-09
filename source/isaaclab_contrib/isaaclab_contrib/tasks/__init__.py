# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Config-based workflow environments.
"""

##
# Register Gym environments.
##

import builtins

from isaaclab_tasks.utils import import_packages

# Guard: AppLauncher._create_app() temporarily removes all "lab" modules from
# sys.modules while creating SimulationApp.  If Kit re-imports this package
# during that window, __init__ runs again and re-registers every gym env.
# We stash a flag on builtins because it is never evicted from sys.modules.
if not getattr(builtins, "_isaaclab_contrib_tasks_registered", False):
    _BLACKLIST_PKGS = [".mdp"]
    import_packages(__name__, _BLACKLIST_PKGS)
    builtins._isaaclab_contrib_tasks_registered = True
