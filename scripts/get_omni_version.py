# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.kit.app

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

app = omni.kit.app.get_app()
kit_version = app.get_kit_version()
print(kit_version)
