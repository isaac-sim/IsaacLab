# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app

"""Rest everything follows."""

import numpy as np
import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.orbit.sim as sim_utils


class TestUtilities(unittest.TestCase):
    """Test fixture for the sim utility functions."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        stage_utils.update_stage()

    def tearDown(self) -> None:
        """Clear stage after each test."""
        stage_utils.clear_stage()

    def test_get_all_matching_child_prims(self):
        """Test get_all_matching_child_prims() function."""
        # create scene
        prim_utils.create_prim("/World/Floor")
        prim_utils.create_prim(
            "/World/Floor/thefloor", "Cube", position=np.array([75, 75, -150.1]), attributes={"size": 300}
        )
        prim_utils.create_prim("/World/Room", "Sphere", attributes={"radius": 1e3})

        # test
        isaac_sim_result = prim_utils.get_all_matching_child_prims("/World")
        orbit_result = sim_utils.get_all_matching_child_prims("/World")
        self.assertListEqual(isaac_sim_result, orbit_result)

        # test valid path
        with self.assertRaises(ValueError):
            sim_utils.get_all_matching_child_prims("World/Room")

    def test_find_matching_prim_paths(self):
        """Test find_matching_prim_paths() function."""
        # create scene
        for index in range(2048):
            random_pos = np.random.uniform(-100, 100, size=3)
            prim_utils.create_prim(f"/World/Floor_{index}", "Cube", position=random_pos, attributes={"size": 2.0})
            prim_utils.create_prim(f"/World/Floor_{index}/Sphere", "Sphere", attributes={"radius": 10})
            prim_utils.create_prim(f"/World/Floor_{index}/Sphere/childSphere", "Sphere", attributes={"radius": 1})
            prim_utils.create_prim(f"/World/Floor_{index}/Sphere/childSphere2", "Sphere", attributes={"radius": 1})

        # test leaf paths
        isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere")
        orbit_result = sim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere")
        self.assertListEqual(isaac_sim_result, orbit_result)

        # test non-leaf paths
        isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*")
        orbit_result = sim_utils.find_matching_prim_paths("/World/Floor_.*")
        self.assertListEqual(isaac_sim_result, orbit_result)

        # test child-leaf paths
        isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere/childSphere.*")
        orbit_result = sim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere/childSphere.*")
        self.assertListEqual(isaac_sim_result, orbit_result)

        # test valid path
        with self.assertRaises(ValueError):
            sim_utils.get_all_matching_child_prims("World/Floor_.*")


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
