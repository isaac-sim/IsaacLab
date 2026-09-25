# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil

import numpy as np
import pytest
import torch

from isaaclab.terrains import (
    FlatPatchSamplingCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshStarTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.height_field import HfInvertedPyramidSlopedTerrainCfg
from isaaclab.utils.seed import configure_seed

pytestmark = pytest.mark.integration


@pytest.fixture
def output_dir():
    """Create directory to dump results."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "generator")
    yield output_dir
    # Cleanup
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def test_generation_star_terrain():
    """Generates a star sub-terrain and tests that the resulting mesh has the expected size.

    The star sub-terrain is not part of :obj:`ROUGH_TERRAINS_CFG`, so it needs its own coverage.
    """
    # create terrain generator with only the star sub-terrain
    cfg = TerrainGeneratorCfg(
        seed=0,
        size=(8.0, 8.0),
        num_rows=1,
        num_cols=1,
        use_cache=False,
        sub_terrains={
            # the number of bars is chosen so that all the branches of the bar-length computation are covered
            "star": MeshStarTerrainCfg(
                proportion=1.0,
                platform_width=1.5,
                num_bars=5,
                bar_width_range=(0.5, 1.0),
                bar_height_range=(0.05, 0.2),
            )
        },
    )
    terrain_generator = TerrainGenerator(cfg=cfg)

    # get size from mesh bounds
    bounds = terrain_generator.terrain_mesh.bounds
    actual_size = abs(bounds[1] - bounds[0])

    # check if the size is as expected
    assert actual_size[0] == pytest.approx(cfg.size[0] * cfg.num_rows + 2 * cfg.border_width)
    assert actual_size[1] == pytest.approx(cfg.size[1] * cfg.num_cols + 2 * cfg.border_width)
    # check the sub-terrain origin is at the center of the terrain
    assert terrain_generator.terrain_origins.shape == (cfg.num_rows, cfg.num_cols, 3)


def test_repeated_objects_default_object_type():
    """The default resolvable ``object_type`` of the repeated-object configs is called, not looked up by name."""
    object_cfg = MeshRepeatedBoxesTerrainCfg.ObjectCfg(num_objects=3, height=0.2, size=(0.3, 0.3))
    cfg = MeshRepeatedBoxesTerrainCfg(
        size=(4.0, 4.0), platform_width=1.0, object_params_start=object_cfg, object_params_end=object_cfg
    )
    np.random.seed(0)
    meshes, origin = cfg.function(0.5, cfg)
    # three objects, ground plane and platform
    assert len(meshes) == 5
    assert origin.shape == (3,)


@pytest.mark.parametrize("platform_width,border_width", [(0.5, 0.0), (1.0, 0.0), (1.5, 0.2)])
def test_inverted_pyramid_origin_matches_platform(platform_width: float, border_width: float):
    cfg = HfInvertedPyramidSlopedTerrainCfg(
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_width=border_width,
        slope_range=(0.4, 0.4),
        platform_width=platform_width,
    )
    meshes, origin = cfg.function(1.0, cfg)
    center_vertices = meshes[0].vertices[
        np.isclose(meshes[0].vertices[:, 0], 4.0) & np.isclose(meshes[0].vertices[:, 1], 4.0)
    ]

    np.testing.assert_allclose(origin[:2], (4.0, 4.0))
    assert len(center_vertices) == 1
    assert origin[2] == pytest.approx(center_vertices[0, 2])


@pytest.mark.parametrize("use_global_seed", [True, False])
def test_generation_reproducibility(use_global_seed):
    """Generates assorted terrains and tests that the resulting mesh is reproducible.

    We check both scenarios where the seed is set globally only and when it is set both globally and locally.
    Setting only locally is not tested as it is not supported.
    """
    seed = 20
    # set initial seed
    configure_seed(seed)

    # create terrain generator
    cfg = ROUGH_TERRAINS_CFG
    cfg.use_cache = False
    cfg.seed = seed if use_global_seed else None
    terrain_generator = TerrainGenerator(cfg=cfg)

    # keep a copy of the generated terrain mesh
    terrain_mesh_1 = terrain_generator.terrain_mesh.copy()

    # set seed again
    configure_seed(seed)

    # create terrain generator
    terrain_generator = TerrainGenerator(cfg=cfg)

    # keep a copy of the generated terrain mesh
    terrain_mesh_2 = terrain_generator.terrain_mesh.copy()

    # check if the meshes are equal
    np.testing.assert_allclose(
        terrain_mesh_1.vertices, terrain_mesh_2.vertices, atol=1e-5, err_msg="Vertices are not equal"
    )
    np.testing.assert_allclose(terrain_mesh_1.faces, terrain_mesh_2.faces, atol=1e-5, err_msg="Faces are not equal")


@pytest.mark.parametrize("curriculum", [True, False])
def test_generation_cache(output_dir, curriculum):
    """Generate the terrain and check that caching works.

    When caching is enabled, the terrain should be generated only once and the same terrain should be returned
    when the terrain generator is created again.
    """
    # create terrain generator with cache enabled
    cfg: TerrainGeneratorCfg = ROUGH_TERRAINS_CFG
    cfg.use_cache = True
    cfg.seed = 0
    cfg.cache_dir = output_dir
    cfg.curriculum = curriculum
    terrain_generator = TerrainGenerator(cfg=cfg)
    # keep a copy of the generated terrain mesh
    terrain_mesh_1 = terrain_generator.terrain_mesh.copy()

    # check cache exists and is equal to the number of terrains
    # with curriculum, all sub-terrains are uniquely generated
    hash_ids_1 = set(os.listdir(cfg.cache_dir))
    assert os.listdir(cfg.cache_dir)

    # set a random seed to disturb the process
    # this is to ensure that the seed inside the terrain generator makes deterministic results
    configure_seed(12456)

    # create terrain generator with cache enabled
    terrain_generator = TerrainGenerator(cfg=cfg)
    # keep a copy of the generated terrain mesh
    terrain_mesh_2 = terrain_generator.terrain_mesh.copy()

    # check no new terrain is generated
    hash_ids_2 = set(os.listdir(cfg.cache_dir))
    assert len(hash_ids_1) == len(hash_ids_2)
    assert hash_ids_1 == hash_ids_2

    # check if the mesh is the same
    # check they don't point to the same object
    assert terrain_mesh_1 is not terrain_mesh_2

    # check if the meshes are equal
    np.testing.assert_allclose(
        terrain_mesh_1.vertices, terrain_mesh_2.vertices, atol=1e-5, err_msg="Vertices are not equal"
    )
    np.testing.assert_allclose(terrain_mesh_1.faces, terrain_mesh_2.faces, atol=1e-5, err_msg="Faces are not equal")


def test_terrain_flat_patches():
    """Test the terrain size and the flat patches generation."""
    # create terrain generator
    cfg = ROUGH_TERRAINS_CFG
    # add flat patch configuration
    for _, sub_terrain_cfg in cfg.sub_terrains.items():
        sub_terrain_cfg.flat_patch_sampling = {
            "root_spawn": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.5, max_height_diff=0.05),
            "target_spawn": FlatPatchSamplingCfg(num_patches=5, patch_radius=0.35, max_height_diff=0.05),
        }
    # generate terrain
    terrain_generator = TerrainGenerator(cfg=cfg)

    # check the mesh spans every sub-terrain plus the border
    actual_size = abs(terrain_generator.terrain_mesh.bounds[1] - terrain_generator.terrain_mesh.bounds[0])
    assert actual_size[0] == pytest.approx(cfg.size[0] * cfg.num_rows + 2 * cfg.border_width)
    assert actual_size[1] == pytest.approx(cfg.size[1] * cfg.num_cols + 2 * cfg.border_width)

    # check if flat patches are generated
    assert terrain_generator.flat_patches
    # check the size of the flat patches
    assert terrain_generator.flat_patches["root_spawn"].shape == (cfg.num_rows, cfg.num_cols, 8, 3)
    assert terrain_generator.flat_patches["target_spawn"].shape == (cfg.num_rows, cfg.num_cols, 5, 3)
    # check that no flat patches are zero
    for _, flat_patches in terrain_generator.flat_patches.items():
        assert not torch.allclose(flat_patches, torch.zeros_like(flat_patches))
