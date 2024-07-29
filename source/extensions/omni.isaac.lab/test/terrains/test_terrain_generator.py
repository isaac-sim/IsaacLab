# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import unittest
import os
import shutil
import numpy as np

from omni.isaac.lab.terrains import TerrainGenerator, TerrainGeneratorCfg
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG


class TestTerrainGenerator(unittest.TestCase):
    """Test the procedural terrain generator."""

    def setUp(self) -> None:
        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(test_dir, "output", "generator")
        # Clean up the directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_generation(self) -> None:
        """Generates assorted terrains and tests that the resulting mesh has the expected size."""
        # create terrain generator
        cfg = ROUGH_TERRAINS_CFG.copy()
        terrain_generator = TerrainGenerator(cfg=cfg)

        # get size from mesh bounds
        bounds = terrain_generator.terrain_mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])
        # compute the expected size
        expectedSizeX = cfg.size[0] * cfg.num_rows + 2 * cfg.border_width
        expectedSizeY = cfg.size[1] * cfg.num_cols + 2 * cfg.border_width

        # check if the size is as expected
        self.assertAlmostEqual(actualSize[0], expectedSizeX)
        self.assertAlmostEqual(actualSize[1], expectedSizeY)

    def test_generation_with_curriculum_and_cache(self) -> None:
        """Generate the terrain with curriculum and check that caching works."""
        # create terrain generator with cache enabled
        cfg: TerrainGeneratorCfg = ROUGH_TERRAINS_CFG.copy()
        cfg.use_cache = True
        cfg.seed = 0
        cfg.cache_dir = self.output_dir
        cfg.curriculum = True
        terrain_generator = TerrainGenerator(cfg=cfg.copy())
        # keep a copy of the generated terrain mesh
        old_terrain_mesh = terrain_generator.terrain_mesh.copy()

        # check cache exists and is equal to the number of terrains
        # with curriculum, all sub-terrains are uniquely generated
        all_hash_ids = set(os.listdir(self.output_dir))
        self.assertTrue(os.listdir(self.output_dir))

        # create terrain generator with cache enabled
        terrain_generator = TerrainGenerator(cfg=cfg.copy())

        # check no new terrain is generated
        # print what is not common
        new_hash_ids = set(os.listdir(self.output_dir))
        print(new_hash_ids - all_hash_ids)
        self.assertEqual(len(os.listdir(self.output_dir)), len(all_hash_ids))

        # check if the mesh is the same
        # check they don't point to the same object
        self.assertIsNot(old_terrain_mesh, terrain_generator.terrain_mesh)
        # check if the meshes are equal
        np.testing.assert_allclose(
            old_terrain_mesh.vertices, terrain_generator.terrain_mesh.vertices, rtol=1e-5, err_msg="Vertices are not equal"
        )
        np.testing.assert_allclose(
            old_terrain_mesh.faces, terrain_generator.terrain_mesh.faces, rtol=1e-5, err_msg="Faces are not equal"
        )


if __name__ == "__main__":
    run_tests()
