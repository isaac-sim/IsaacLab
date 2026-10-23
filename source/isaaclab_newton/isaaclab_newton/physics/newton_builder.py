# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD construction helpers for Newton models."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import warp as wp
from newton import Heightfield, ModelBuilder

from pxr import Usd, UsdGeom

logger = logging.getLogger(__name__)


def inject_terrain_heightfields(
    stage: Usd.Stage, builder: ModelBuilder, *, root_paths: Sequence[str], device: str = "cpu"
) -> list[str]:
    """Replace height-field-tagged terrain colliders with Newton heightfields.

    Scans the stage for prims carrying the ``newton:heightfield:resolution`` attribute authored by
    :class:`~isaaclab.terrains.TerrainImporter`. Each collision mesh is rasterized into a
    :class:`newton.Heightfield` and added to ``builder`` as a static shape. Heightfields compile on MuJoCo roughly
    two orders of magnitude faster than the equivalent terrain mesh while colliding identically at the same
    horizontal resolution.

    Args:
        stage: The USD stage being imported.
        builder: The Newton model builder receiving the heightfield shapes.
        root_paths: Concrete subtree roots to scan.

    Returns:
        Prim paths of converted colliders, which the caller excludes from the USD import.
    """
    ignore_paths: list[str] = []
    xform_cache = UsdGeom.XformCache()
    for prim in (prim for root_path in root_paths for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))):
        attr = prim.GetAttribute("newton:heightfield:resolution")
        if not attr or not attr.HasAuthoredValue():
            continue
        resolution = float(attr.Get())
        mesh_prim = (
            prim if prim.IsA(UsdGeom.Mesh) else next((p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)), None)
        )
        if mesh_prim is None:
            continue
        mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        # Transform vertices into world frame (USD uses row-vector convention).
        mat = np.array(xform_cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64).reshape(4, 4)
        world = (points @ mat[:3, :3] + mat[3, :3]).astype(np.float32)
        wp_mesh = wp.Mesh(
            points=wp.array(world, dtype=wp.vec3, device=device),
            indices=wp.array(faces, dtype=wp.int32, device=device),
        )
        heightfield, xform = Heightfield.create_from_mesh(wp_mesh, resolution)
        builder.add_shape_heightfield(heightfield=heightfield, xform=xform)
        logger.info(
            "Converted terrain collider %s (%d faces) to a %dx%d heightfield.",
            prim.GetPath().pathString,
            faces.shape[0] // 3,
            heightfield.nrow,
            heightfield.ncol,
        )
        ignore_paths.append(prim.GetPath().pathString)
    return ignore_paths
