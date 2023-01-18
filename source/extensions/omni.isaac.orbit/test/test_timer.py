# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
import unittest

from omni.isaac.orbit.utils.timer import Timer


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

    def test_tiemr_as_context_manager(self):
        """Test using a `Timer` as a context manager."""
        with Timer() as timer:
            self.assertAlmostEqual(0, timer.time_elapsed, self.precision_places)
            time.sleep(1)
            self.assertAlmostEqual(1, timer.time_elapsed, self.precision_places)


if __name__ == "__main__":
    unittest.main()
