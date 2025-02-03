# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import os
import shutil
import torch
import unittest

import isaacsim.core.utils.torch as torch_utils

from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


class TestTerrainGenerator(unittest.TestCase):
    """Test the procedural terrain generator."""

    def setUp(self):
        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(test_dir, "output", "generator")

    def test_generation(self):
        """Generates assorted terrains and tests that the resulting mesh has the expected size."""
        # create terrain generator
        cfg = ROUGH_TERRAINS_CFG.copy()
        terrain_generator = TerrainGenerator(cfg=cfg)

        # print terrain generator info
        print(terrain_generator)

        # get size from mesh bounds
        bounds = terrain_generator.terrain_mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])
        # compute the expected size
        expectedSizeX = cfg.size[0] * cfg.num_rows + 2 * cfg.border_width
        expectedSizeY = cfg.size[1] * cfg.num_cols + 2 * cfg.border_width

        # check if the size is as expected
        self.assertAlmostEqual(actualSize[0], expectedSizeX)
        self.assertAlmostEqual(actualSize[1], expectedSizeY)

    def test_generation_reproducibility(self):
        """Generates assorted terrains and tests that the resulting mesh is reproducible.

        We check both scenarios where the seed is set globally only and when it is set both globally and locally.
        Setting only locally is not tested as it is not supported.
        """
        for use_global_seed in [True, False]:
            for seed in [20, 40, 80]:
                with self.subTest(seed=seed):
                    # set initial seed
                    torch_utils.set_seed(seed)

                    # create terrain generator
                    cfg = ROUGH_TERRAINS_CFG.copy()
                    cfg.use_cache = False
                    cfg.seed = seed if use_global_seed else None
                    terrain_generator = TerrainGenerator(cfg=cfg)

                    # keep a copy of the generated terrain mesh
                    terrain_mesh_1 = terrain_generator.terrain_mesh.copy()

                    # set seed again
                    torch_utils.set_seed(seed)

                    # create terrain generator
                    terrain_generator = TerrainGenerator(cfg=cfg)

                    # keep a copy of the generated terrain mesh
                    terrain_mesh_2 = terrain_generator.terrain_mesh.copy()

                    # check if the meshes are equal
                    np.testing.assert_allclose(
                        terrain_mesh_1.vertices, terrain_mesh_2.vertices, atol=1e-5, err_msg="Vertices are not equal"
                    )
                    np.testing.assert_allclose(
                        terrain_mesh_1.faces, terrain_mesh_2.faces, atol=1e-5, err_msg="Faces are not equal"
                    )

    def test_generation_cache(self):
        """Generate the terrain and check that caching works.

        When caching is enabled, the terrain should be generated only once and the same terrain should be returned
        when the terrain generator is created again.
        """
        # try out with and without curriculum
        for curriculum in [True, False]:
            with self.subTest(curriculum=curriculum):
                # clear output directory
                if os.path.exists(self.output_dir):
                    shutil.rmtree(self.output_dir)
                # create terrain generator with cache enabled
                cfg: TerrainGeneratorCfg = ROUGH_TERRAINS_CFG.copy()
                cfg.use_cache = True
                cfg.seed = 0
                cfg.cache_dir = self.output_dir
                cfg.curriculum = curriculum
                terrain_generator = TerrainGenerator(cfg=cfg)
                # keep a copy of the generated terrain mesh
                terrain_mesh_1 = terrain_generator.terrain_mesh.copy()

                # check cache exists and is equal to the number of terrains
                # with curriculum, all sub-terrains are uniquely generated
                hash_ids_1 = set(os.listdir(cfg.cache_dir))
                self.assertTrue(os.listdir(cfg.cache_dir))

                # set a random seed to disturb the process
                # this is to ensure that the seed inside the terrain generator makes deterministic results
                torch_utils.set_seed(12456)

                # create terrain generator with cache enabled
                terrain_generator = TerrainGenerator(cfg=cfg)
                # keep a copy of the generated terrain mesh
                terrain_mesh_2 = terrain_generator.terrain_mesh.copy()

                # check no new terrain is generated
                hash_ids_2 = set(os.listdir(cfg.cache_dir))
                self.assertEqual(len(hash_ids_1), len(hash_ids_2))
                self.assertSetEqual(hash_ids_1, hash_ids_2)

                # check if the mesh is the same
                # check they don't point to the same object
                self.assertIsNot(terrain_mesh_1, terrain_mesh_2)

                # check if the meshes are equal
                np.testing.assert_allclose(
                    terrain_mesh_1.vertices, terrain_mesh_2.vertices, atol=1e-5, err_msg="Vertices are not equal"
                )
                np.testing.assert_allclose(
                    terrain_mesh_1.faces, terrain_mesh_2.faces, atol=1e-5, err_msg="Faces are not equal"
                )

    def test_terrain_flat_patches(self):
        """Test the flat patches generation."""
        # create terrain generator
        cfg = ROUGH_TERRAINS_CFG.copy()
        # add flat patch configuration
        for _, sub_terrain_cfg in cfg.sub_terrains.items():
            sub_terrain_cfg.flat_patch_sampling = {
                "root_spawn": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.5, max_height_diff=0.05),
                "target_spawn": FlatPatchSamplingCfg(num_patches=5, patch_radius=0.35, max_height_diff=0.05),
            }
        # generate terrain
        terrain_generator = TerrainGenerator(cfg=cfg)

        # check if flat patches are generated
        self.assertTrue(terrain_generator.flat_patches)
        # check the size of the flat patches
        self.assertTupleEqual(terrain_generator.flat_patches["root_spawn"].shape, (cfg.num_rows, cfg.num_cols, 8, 3))
        self.assertTupleEqual(terrain_generator.flat_patches["target_spawn"].shape, (cfg.num_rows, cfg.num_cols, 5, 3))
        # check that no flat patches are zero
        for _, flat_patches in terrain_generator.flat_patches.items():
            self.assertFalse(torch.allclose(flat_patches, torch.zeros_like(flat_patches)))


if __name__ == "__main__":
    run_tests()
