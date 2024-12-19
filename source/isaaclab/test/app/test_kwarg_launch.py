# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

from isaaclab.app import AppLauncher, run_tests


class TestAppLauncher(unittest.TestCase):
    """Test launching of the simulation app using AppLauncher."""

    def test_livestream_launch_with_kwarg(self):
        """Test launching with headless and livestreaming arguments."""
        # everything defaults to None
        app = AppLauncher(headless=True, livestream=1).app

        # import settings
        import carb

        # acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # check settings
        # -- no-gui mode
        self.assertEqual(carb_settings_iface.get("/app/window/enabled"), False)
        # -- livestream
        self.assertEqual(carb_settings_iface.get("/app/livestream/enabled"), True)

        # close the app on exit
        app.close()


if __name__ == "__main__":
    run_tests()
