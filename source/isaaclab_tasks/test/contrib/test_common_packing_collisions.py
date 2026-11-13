# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry-preserving common collision preparation, independent of a physics engine."""

from types import SimpleNamespace

import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.packing_table import configure_packing_table_collisions


@pytest.mark.parametrize("physx_offsets", [False, True])
def test_existing_mesh_transforms_and_bin_colliders_are_preserved(physx_offsets):
    stage = Usd.Stage.CreateInMemory()
    expected = {}

    def spawn(path, cfg):
        root = stage.DefinePrim(path, "Xform")
        root.SetInstanceable(True)
        parent = stage.DefinePrim(path + "/Table", "Xform")
        parent.SetInstanceable(True)
        UsdGeom.Xformable(parent).AddTranslateOp().Set((0.3, -0.2, 0.4))
        UsdPhysics.CollisionAPI.Apply(parent)
        UsdPhysics.MeshCollisionAPI.Apply(parent).CreateApproximationAttr().Set("boundingCube")
        parent.CreateAttribute("horde:filterOwner", Sdf.ValueTypeNames.String).Set("table")
        for index in range(4):
            mesh = UsdGeom.Mesh.Define(stage, path + f"/Table/Mesh_{index}")
            mesh.CreatePointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            mesh.CreateFaceVertexCountsAttr().Set([3])
            mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2])
            UsdGeom.Xformable(mesh).AddTranslateOp().Set((index, index / 2, index / 4))
        stage.DefinePrim(path + "/Bin", "Xform").SetInstanceable(True)
        for index in range(5):
            cube = UsdGeom.Cube.Define(stage, path + f"/Bin/Cube_{index}")
            cube.CreateSizeAttr().Set(0.1 + index)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cache = UsdGeom.XformCache()
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Gprim):
                expected[str(prim.GetPath())] = cache.GetLocalToWorldTransform(prim)
        return root

    cfg = SimpleNamespace(spawn=SimpleNamespace(func=spawn))
    configure_packing_table_collisions(cfg, physx_offsets=physx_offsets)
    prepared = cfg.spawn.func
    configure_packing_table_collisions(cfg, physx_offsets=physx_offsets)
    assert cfg.spawn.func is prepared
    root = prepared("/World/PackingTable", cfg.spawn)
    shapes = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(shapes) == 9
    assert {str(p.GetPath()) for p in shapes} == set(expected)
    assert sum(p.IsA(UsdGeom.Mesh) for p in shapes) == 4
    assert sum(p.IsA(UsdGeom.Cube) for p in shapes) == 5
    assert not stage.GetPrimAtPath("/World/PackingTable/Table").HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath("/World/PackingTable/Table").GetAttribute("horde:filterOwner").Get() == "table"
    assert not any(prim.IsInstance() or prim.IsInstanceProxy() for prim in Usd.PrimRange(root))
    cache = UsdGeom.XformCache()
    for shape in shapes:
        assert cache.GetLocalToWorldTransform(shape) == expected[str(shape.GetPath())]
        assert UsdPhysics.CollisionAPI(shape).GetCollisionEnabledAttr().Get()
        assert UsdGeom.Imageable(shape).GetPurposeAttr().Get() == UsdGeom.Tokens.default_
        if shape.IsA(UsdGeom.Mesh):
            assert UsdPhysics.MeshCollisionAPI(shape).GetApproximationAttr().Get() == "boundingCube"
            assert list(UsdGeom.Mesh(shape).GetFaceVertexIndicesAttr().Get()) == [0, 1, 2]
        if physx_offsets:
            assert shape.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.001)
            assert shape.GetAttribute("physxCollision:restOffset").Get() == 0.0
