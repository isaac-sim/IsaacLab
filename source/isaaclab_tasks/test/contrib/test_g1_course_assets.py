"""Validate the standalone fixed-course assets and their sensor-safe appearance switch."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

ASSET_DIR = Path(importlib.util.find_spec("isaaclab_tasks").origin).parent / "contrib" / "g1_terrain" / "assets"


@pytest.mark.parametrize("layout", ("platforms", "ramps", "granular"))
def test_standalone_course(layout):
    """Assets resolve alone and switching spectator appearance preserves static contacts."""
    asset_path = ASSET_DIR / f"{layout}.usdc"
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(asset_path))
    assert len(layers) == 1
    assert not assets
    assert not unresolved
    stage = Usd.Stage.CreateInMemory()
    course = stage.DefinePrim("/World/ground/terrain", "Xform")
    course.GetReferences().AddReference(str(asset_path))
    assert UsdGeom.GetStageUpAxis(Usd.Stage.Open(str(asset_path))) == UsdGeom.Tokens.z
    colliders = {
        prim.GetPath(): (
            np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()).copy(),
            np.asarray(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()).copy(),
        )
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    assert colliders
    for path in colliders:
        mesh = stage.GetPrimAtPath(path)
        assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
        if layout != "granular" or path.GetParentPath().name != "granular_container":
            material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial("physics")
            assert UsdPhysics.MaterialAPI(material).GetStaticFrictionAttr().Get() == 1.0
            assert mesh.GetAttribute("mjc:priority").Get() == 1
    if layout == "granular":
        return
    original = UsdGeom.Mesh(stage.GetPrimAtPath(f"{course.GetPath()}/wood_base/mesh"))
    visual_path = f"{course.GetPath()}/wood_base/wood_base_visual"
    assert original.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert not stage.GetPrimAtPath(visual_path).IsActive()
    assert course.GetVariantSet("appearance").SetVariantSelection("visual")
    assert original.ComputeVisibility() == UsdGeom.Tokens.invisible
    visual = UsdGeom.Mesh(stage.GetPrimAtPath(visual_path))
    assert visual.GetPrim().IsActive()
    assert visual.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert not visual.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert {prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)} == set(colliders)
    for path, (points, indices) in colliders.items():
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        np.testing.assert_array_equal(mesh.GetPointsAttr().Get(), points)
        np.testing.assert_array_equal(mesh.GetFaceVertexIndicesAttr().Get(), indices)
