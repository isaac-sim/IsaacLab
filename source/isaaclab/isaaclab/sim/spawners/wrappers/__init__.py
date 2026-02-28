# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for wrapping spawner configurations.

Unlike the other spawner modules, this module provides a way to wrap multiple spawner configurations
into a single configuration. This is useful when the user wants to spawn multiple assets based on
different configurations.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
