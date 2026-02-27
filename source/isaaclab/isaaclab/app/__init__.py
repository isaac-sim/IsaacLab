# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing app-specific functionalities.

These include:

* Ability to launch the simulation app with different configurations
* Run tests with the simulation app

"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .app_launcher import AppLauncher
    from .settings_manager import SettingsManager, get_settings_manager, initialize_carb_settings

from isaaclab.utils.module import lazy_export

lazy_export(
    ("app_launcher", "AppLauncher"),
    ("settings_manager", ["SettingsManager", "get_settings_manager", "initialize_carb_settings"]),
)
