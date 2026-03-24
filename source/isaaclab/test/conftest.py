# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test-level conftest that avoids SimulationApp shutdown hangs.

Some Omniverse processes, especially those using cameras or render products,
hang during ``SimulationApp.close()``. By the time ``atexit`` handlers run,
pytest has already written its JUnit XML report and torn down fixtures, so
forcing an immediate exit avoids blocking CI runners or interactive shells.
"""

import atexit
import os


def pytest_sessionfinish(session, exitstatus):
    """Register a forced process exit during interpreter shutdown.

    ``atexit`` handlers run in LIFO order after pytest completes. Registering
    ``os._exit`` here ensures it runs before SimulationApp's cleanup handler,
    which can hang.
    """
    atexit.register(os._exit, int(exitstatus))
