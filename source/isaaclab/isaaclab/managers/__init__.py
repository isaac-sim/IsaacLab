# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for environment managers.

The managers are used to handle various aspects of the environment such as randomization events, curriculum,
and observations. Each manager implements a specific functionality for the environment. The managers are
designed to be modular and can be easily extended to support new functionality.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
