# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from isaaclab.terrains.trimesh import MeshStarTerrainCfg

pytestmark = pytest.mark.unit


def test_star_terrain_generation():
    """Generate every bar of a star terrain with finite geometry."""
    cfg = MeshStarTerrainCfg(
        size=(8.0, 8.0),
        platform_width=1.5,
        num_bars=5,
        bar_width_range=(0.5, 1.0),
        bar_height_range=(0.05, 0.2),
    )

    meshes, origin = cfg.function(difficulty=0.5, cfg=cfg)

    assert meshes
    assert all(np.isfinite(mesh.vertices).all() for mesh in meshes)
    np.testing.assert_allclose(origin, (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))
