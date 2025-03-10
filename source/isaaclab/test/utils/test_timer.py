# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# NOTE: While we don't actually use the simulation app in this test, we still need to launch it
#       because warp is only available in the context of a running simulation
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import unittest

from isaaclab.utils.timer import Timer


class TestTimer(unittest.TestCase):
    """Test fixture for the Timer class."""

    def setUp(self):
        # number of decimal places to check
        self.precision_places = 2

    def test_timer_as_object(self):
        """Test using a `Timer` as a regular object."""
        timer = Timer()
        timer.start()
        self.assertAlmostEqual(0, timer.time_elapsed, self.precision_places)
        time.sleep(1)
        self.assertAlmostEqual(1, timer.time_elapsed, self.precision_places)
        timer.stop()
        self.assertAlmostEqual(1, timer.total_run_time, self.precision_places)

    def test_timer_as_context_manager(self):
        """Test using a `Timer` as a context manager."""
        with Timer() as timer:
            self.assertAlmostEqual(0, timer.time_elapsed, self.precision_places)
            time.sleep(1)
            self.assertAlmostEqual(1, timer.time_elapsed, self.precision_places)


if __name__ == "__main__":
    run_tests()
