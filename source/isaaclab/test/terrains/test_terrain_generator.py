# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import pytest
import torch

from isaaclab.terrains import (
    FlatPatchSamplingCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
    MeshRepeatedPyramidsTerrainCfg,
    MeshStarTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.seed import configure_seed

pytestmark = pytest.mark.unit


def _rough_cfg(**kwargs) -> TerrainGeneratorCfg:
    """Copy of the rough terrains config shrunk to one row per sub-terrain type so generation stays fast."""
    cfg = ROUGH_TERRAINS_CFG.copy()
    cfg.num_rows = 1
    cfg.num_cols = len(cfg.sub_terrains)
    cfg.border_width = 2.0
    cfg.use_cache = False
    cfg.seed = 0
    for name, value in kwargs.items():
        setattr(cfg, name, value)
    return cfg


def _assert_terrain_bounds(terrain_generator: TerrainGenerator, cfg: TerrainGeneratorCfg):
    bounds = terrain_generator.terrain_mesh.bounds
    actual_size = abs(bounds[1] - bounds[0])
    assert actual_size[0] == pytest.approx(cfg.size[0] * cfg.num_rows + 2 * cfg.border_width)
    assert actual_size[1] == pytest.approx(cfg.size[1] * cfg.num_cols + 2 * cfg.border_width)
    assert terrain_generator.terrain_origins.shape == (cfg.num_rows, cfg.num_cols, 3)


@pytest.mark.parametrize("curriculum", [True, False])
def test_generation(curriculum):
    """Every rough sub-terrain type generates and the combined mesh has the configured footprint."""
    cfg = _rough_cfg(curriculum=curriculum)
    terrain_generator = TerrainGenerator(cfg=cfg)
    assert str(terrain_generator).startswith("Terrain Generator:")
    _assert_terrain_bounds(terrain_generator, cfg)


def test_generation_star_terrain():
    """The star sub-terrain is not part of the rough config; the bar count covers every bar-length branch."""
    cfg = TerrainGeneratorCfg(
        seed=0,
        size=(8.0, 8.0),
        num_rows=1,
        num_cols=1,
        use_cache=False,
        sub_terrains={
            "star": MeshStarTerrainCfg(
                platform_width=1.5, num_bars=5, bar_width_range=(0.5, 1.0), bar_height_range=(0.05, 0.2)
            )
        },
    )
    _assert_terrain_bounds(TerrainGenerator(cfg=cfg), cfg)


@pytest.mark.parametrize(
    "cfg_type, object_kwargs",
    [
        (MeshRepeatedBoxesTerrainCfg, {"size": (0.3, 0.3)}),
        (MeshRepeatedCylindersTerrainCfg, {"radius": 0.2}),
        (MeshRepeatedPyramidsTerrainCfg, {"radius": 0.2}),
    ],
)
def test_repeated_objects_default_object_type(cfg_type, object_kwargs):
    """The default ``object_type`` of the repeated-object configs resolves to the matching mesh primitive."""
    cfg = cfg_type(
        size=(4.0, 4.0),
        platform_width=1.0,
        object_params_start=cfg_type.ObjectCfg(num_objects=3, height=0.2, **object_kwargs),
        object_params_end=cfg_type.ObjectCfg(num_objects=3, height=0.4, **object_kwargs),
    )
    np.random.seed(0)
    meshes, origin = cfg.function(0.5, cfg)
    # three objects, ground plane and platform
    assert len(meshes) == 5
    assert origin.shape == (3,)


@pytest.mark.parametrize("use_global_seed", [True, False])
def test_generation_reproducibility(use_global_seed):
    """Meshes are reproducible whether the seed comes from the config or only from the global RNG state."""
    seed = 20
    cfg = _rough_cfg(seed=seed if use_global_seed else None)

    configure_seed(seed)
    terrain_mesh_1 = TerrainGenerator(cfg=cfg).terrain_mesh
    configure_seed(seed)
    terrain_mesh_2 = TerrainGenerator(cfg=cfg).terrain_mesh

    np.testing.assert_allclose(terrain_mesh_1.vertices, terrain_mesh_2.vertices, atol=1e-5)
    np.testing.assert_array_equal(terrain_mesh_1.faces, terrain_mesh_2.faces)


@pytest.mark.parametrize("curriculum", [True, False])
def test_generation_cache(tmp_path, curriculum):
    """Cached sub-terrains are reused on the second generation regardless of the global RNG state."""
    cfg = _rough_cfg(use_cache=True, cache_dir=str(tmp_path), curriculum=curriculum)
    terrain_mesh_1 = TerrainGenerator(cfg=cfg).terrain_mesh
    hash_ids_1 = set(os.listdir(cfg.cache_dir))
    assert hash_ids_1

    # disturb the global RNG: cached terrains must still be picked up
    configure_seed(12456)
    terrain_mesh_2 = TerrainGenerator(cfg=cfg).terrain_mesh

    assert set(os.listdir(cfg.cache_dir)) == hash_ids_1
    assert terrain_mesh_1 is not terrain_mesh_2
    np.testing.assert_allclose(terrain_mesh_1.vertices, terrain_mesh_2.vertices, atol=1e-5)
    np.testing.assert_array_equal(terrain_mesh_1.faces, terrain_mesh_2.faces)


def test_terrain_flat_patches():
    """Flat patches are sampled per sub-terrain with the configured shapes and non-trivial values."""
    cfg = _rough_cfg()
    for sub_terrain_cfg in cfg.sub_terrains.values():
        sub_terrain_cfg.flat_patch_sampling = {
            "root_spawn": FlatPatchSamplingCfg(num_patches=8, patch_radius=0.5, max_height_diff=0.05),
            "target_spawn": FlatPatchSamplingCfg(num_patches=5, patch_radius=[0.35, 0.5], max_height_diff=0.05),
        }
    terrain_generator = TerrainGenerator(cfg=cfg)

    assert set(terrain_generator.flat_patches) == {"root_spawn", "target_spawn"}
    assert terrain_generator.flat_patches["root_spawn"].shape == (cfg.num_rows, cfg.num_cols, 8, 3)
    assert terrain_generator.flat_patches["target_spawn"].shape == (cfg.num_rows, cfg.num_cols, 5, 3)
    for flat_patches in terrain_generator.flat_patches.values():
        assert not torch.allclose(flat_patches, torch.zeros_like(flat_patches))
