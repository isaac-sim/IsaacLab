# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import unittest

import isaaclab.utils.assets as assets_utils


class TestAssetsUtils(unittest.TestCase):
    """Test cases for the assets utility functions."""

    def test_nucleus_connection(self):
        """Test checking the Nucleus connection."""
        # check nucleus connection
        self.assertIsNotNone(assets_utils.NUCLEUS_ASSET_ROOT_DIR)

    def test_check_file_path_nucleus(self):
        """Test checking a file path on the Nucleus server."""
        # robot file path
        usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
        # check file path
        self.assertEqual(assets_utils.check_file_path(usd_path), 2)

    def test_check_file_path_invalid(self):
        """Test checking an invalid file path."""
        # robot file path
        usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_xyz.usd"
        # check file path
        self.assertEqual(assets_utils.check_file_path(usd_path), 0)


if __name__ == "__main__":
    run_tests()
