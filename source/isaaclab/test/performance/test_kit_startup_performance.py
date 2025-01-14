# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import time
import unittest

from isaaclab.app import run_tests


class TestKitStartUpPerformance(unittest.TestCase):
    """Test kit startup performance."""

    def test_kit_start_up_time(self):
        """Test kit start-up time."""
        from isaaclab.app import AppLauncher

        start_time = time.time()
        self.app_launcher = AppLauncher(headless=True).app
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertLessEqual(elapsed_time, 10.0)


if __name__ == "__main__":
    run_tests()
