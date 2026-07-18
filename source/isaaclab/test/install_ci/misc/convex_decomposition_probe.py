# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe: USD-authored ``convexDecomposition`` must decompose under Newton.

Run inside an installed environment. Authors an in-memory concave L-prism whose
collision mesh carries ``physics:approximation = "convexDecomposition"`` and
verifies it decomposes into at least two convex collision shapes, without a
remeshing-fallback warning, through both import paths:

1. Newton's USD importer (plain-scene loading). Fails when the ``coacd``
   package is missing: Newton warns and falls back to a single convex hull,
   which fills the L's concavity.
2. Isaac Lab's Newton cloner (:func:`build_source_builders`, the path RL tasks
   use). Fails when the cloner discards authored approximations
   (``skip_mesh_approximation=True`` + unconditional convex-hull simplify).
"""

from __future__ import annotations

import sys
import warnings

import newton
from newton import GeoType, ShapeFlags

from pxr import Usd, UsdGeom, UsdPhysics


def make_stage() -> Usd.Stage:
    """Author a rigid concave L-prism with convexDecomposition collision."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/World/Asset")
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/World/Asset/geom")
    # Concave hexagonal cross-section (an "L"), extruded along z.
    base = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    points = [(x, y, 0.0) for x, y in base] + [(x, y, 1.0) for x, y in base]
    tris = []
    # Bottom and top caps: a fan from vertex 0 (which sees the whole L polygon).
    for a, b in ((1, 2), (2, 3), (3, 4), (4, 5)):
        tris.append((0, b, a))
        tris.append((6, 6 + a, 6 + b))
    # Side walls.
    for a in range(6):
        b = (a + 1) % 6
        tris.append((a, b, b + 6))
        tris.append((a, b + 6, a + 6))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr([i for t in tris for i in t])

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("convexDecomposition")
    return stage


def _collision_shape_count(builder) -> int:
    """Number of colliding mesh/convex-mesh shapes in the builder."""
    return sum(
        1
        for i, shape_type in enumerate(builder.shape_type)
        if shape_type in (GeoType.MESH, GeoType.CONVEX_MESH)
        and builder.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES
        and builder.shape_source[i] is not None
    )


def _check(label: str, build) -> bool:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builder = build()

    fallback_warnings = [str(w.message) for w in caught if "falling back" in str(w.message).lower()]
    shape_count = _collision_shape_count(builder)
    print(f"{label}: collision shapes = {shape_count}")
    for message in fallback_warnings:
        print(f"{label}: fallback warning: {message}")

    if fallback_warnings:
        print(f"PROBE FAIL ({label}): Newton fell back instead of honoring convexDecomposition (is coacd installed?).")
        return False
    if shape_count < 2:
        print(f"PROBE FAIL ({label}): authored convexDecomposition produced a single collision shape.")
        return False
    return True


def _newton_native() -> object:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_usd(make_stage(), root_path="/World/Asset", skip_mesh_approximation=False)
    return builder


def _isaaclab_cloner() -> object:
    from isaaclab_newton.cloner.newton_clone_utils import build_source_builders

    builders = build_source_builders(
        make_stage(),
        ["/World/Asset"],
        create_builder=lambda: newton.ModelBuilder(up_axis=newton.Axis.Z),
        schema_resolvers=[],
    )
    return builders["/World/Asset"]


def main() -> int:
    ok = _check("newton native import", _newton_native)
    ok = _check("isaaclab cloner", _isaaclab_cloner) and ok
    if not ok:
        return 1
    print("PROBE PASS: convexDecomposition produced a multi-hull collision on both import paths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
