# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# NOTE: While we don't actually use the simulation app in this test, we still need to launch it
#       because warp is only available in the context of a running simulation
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import time

from isaaclab.utils.timer import Timer

# number of decimal places to check
PRECISION_PLACES = 2


def test_timer_as_object():
    """Test using a `Timer` as a regular object."""
    timer = Timer()
    timer.start()
    assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    time.sleep(1)
    assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
    timer.stop()
    assert abs(1 - timer.total_run_time) < 10 ** (-PRECISION_PLACES)


def test_timer_as_context_manager():
    """Test using a `Timer` as a context manager."""
    with Timer() as timer:
        assert abs(0 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
        time.sleep(1)
        assert abs(1 - timer.time_elapsed) < 10 ** (-PRECISION_PLACES)
