# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The imported coffee props must retain cavities that can carry MPM material."""

from pathlib import Path

import newton
import numpy as np
import trimesh
import warp as wp

from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import build_coffee_assets


def test_imported_coffee_props_have_open_cavities_and_solid_bottoms(tmp_path: Path):
    """Detect sealed mouths, inverted shells and accidental convexification in USD."""
    paths = build_coffee_assets(tmp_path)
    # Distinct physical fixtures: a curved bowl, a straight cup and an elliptical
    # spoon. Probe heights are inside the cavities, independently of tessellation.
    fixtures = (("bowl", 0.020, 0.004), ("cup", 0.040, 0.004), ("spoon", 0.005, 0.0))
    with wp.ScopedDevice("cpu"):
        for name, interior_height, floor_height in fixtures:
            builder = newton.ModelBuilder()
            builder.add_usd(paths[name], hide_collision_shapes=True)
            colliders = [
                trimesh.Trimesh(source.vertices, np.asarray(source.indices).reshape(-1, 3), process=False)
                for source, flags in zip(builder.shape_source, builder.shape_flags, strict=True)
                if flags & newton.ShapeFlags.COLLIDE_PARTICLES
            ]
            assert colliders, f"{name} has no particle collider"
            for mesh in colliders:
                assert mesh.is_volume, f"{name} contains a non-watertight or inverted wall"
            geometry = trimesh.util.concatenate(colliders)
            origin = np.array([[0.0, 0.0, interior_height]])
            upward = geometry.ray.intersects_any(origin, np.array([[0.0, 0.0, 1.0]]))
            assert not upward[0], f"{name} mouth was capped or convexified"
            hits, _, _ = geometry.ray.intersects_location(origin, np.array([[0.0, 0.0, -1.0]]))
            assert len(hits) >= 2, f"{name} has no solid bottom thickness"
            np.testing.assert_allclose(hits[:, 2].max(), floor_height, atol=1.0e-6)
            sideways = geometry.ray.intersects_any(origin, np.array([[1.0, 0.0, 0.0]]))
            assert sideways[0], f"{name} has no retaining side wall"
            # Isaac Lab hides colliders on bodies that have visual-only shapes.
            # A decorative handle must not hide the vessel's visible shell.
            visuals = [
                trimesh.Trimesh(source.vertices, np.asarray(source.indices).reshape(-1, 3), process=False)
                for source, flags in zip(builder.shape_source, builder.shape_flags, strict=True)
                if flags & newton.ShapeFlags.VISIBLE
            ]
            assert visuals, f"{name} has no visible geometry"
            visible_bottom = trimesh.util.concatenate(visuals).ray.intersects_any(origin, np.array([[0.0, 0.0, -1.0]]))
            assert visible_bottom[0], f"{name} decoration hides the vessel's visible shell"
            if name == "cup":
                handle = next(
                    trimesh.Trimesh(source.vertices, np.asarray(source.indices).reshape(-1, 3), process=False)
                    for label, source in zip(builder.shape_label, builder.shape_source, strict=True)
                    if label.endswith("/Handle")
                )
                assert handle.is_volume, "Cup handle has an open or inverted tube"
                attached = geometry.contains(handle.vertices)
                assert attached[handle.vertices[:, 2] > 0.058].any(), "Cup handle has no upper wall attachment"
                assert attached[handle.vertices[:, 2] < 0.030].any(), "Cup handle has no lower wall attachment"
                assert np.linalg.norm(handle.vertices[:, :2], axis=1).min() > 0.039, "Handle enters the cup opening"
