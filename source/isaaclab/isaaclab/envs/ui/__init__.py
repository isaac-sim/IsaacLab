# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module providing UI window implementation for environments.

The UI elements are used to control the environment and visualize the state of the environment.
This includes functionalities such as tracking a robot in the simulation,
toggling different debug visualization tools, and other user-defined functionalities.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "base_env_window": ["BaseEnvWindow"],
        "empty_window": ["EmptyWindow"],
        "manager_based_rl_env_window": ["ManagerBasedRLEnvWindow"],
        "viewport_camera_controller": ["ViewportCameraController"],
    },
)
