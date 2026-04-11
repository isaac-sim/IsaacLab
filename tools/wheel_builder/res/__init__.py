# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from importlib.metadata import version

__version__ = version("isaaclab")

# Extend the package search path so subpackages (app/, envs/, etc.) in the
# nested source tree are importable as isaaclab.app, isaaclab.envs, etc.
__path__.append(os.path.join(os.path.dirname(__file__), "source", "isaaclab", "isaaclab"))

if __name__ == "__main__":
    raise NotImplementedError()
