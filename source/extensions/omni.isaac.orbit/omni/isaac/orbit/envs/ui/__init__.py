# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-package contains implementations of UI elements for the environments.

The UI elements are used to control the environment and visualize the state of the environment.
"""

from __future__ import annotations

# enable the extension for UI elements
# this only needs to be done once
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.ui")

# import all UI elements here
from .base_env_window import BaseEnvWindow
from .rl_env_window import RLEnvWindow

__all__ = ["BaseEnvWindow", "RLEnvWindow"]
