# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments.

The package is structured as follows:

- ``core``: Core task families maintained as part of Isaac Lab.
- ``contrib``: Contributed task families. These may depend on ``core`` tasks, but
  ``core`` tasks never depend on ``contrib`` tasks.
- ``benchmark``: Benchmark-only task families used to measure simulation and rendering
  throughput. These may depend on ``core`` and ``contrib`` tasks, but neither depends
  on ``benchmark`` tasks.
- ``utils``: These include utility functions for the tasks.

"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("isaaclab_tasks")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
