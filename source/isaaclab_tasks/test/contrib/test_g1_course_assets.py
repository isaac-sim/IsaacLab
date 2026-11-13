"""Validate the standalone fixed-course assets and their sensor-safe appearance switch."""

import numpy as np
import pytest

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Contrib/G1Terrain"


@pytest.mark.parametrize("layout", ("platforms", "ramps", "granular"))
def test_standalone_course(layout):
    """Assets resolve alone and switching spectator appearance preserves static contacts."""
    asset_path = retrieve_file_path(f"{ASSET_DIR}/{layout}.usdc")
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


def test_physx_particle_asset_references_remap_without_duplicate_tray_contacts():
    """The saved PBD scene stays dynamic and self-contained when imported below the terrain root."""
    path = retrieve_file_path(f"{ASSET_DIR}/granular_physx.usdc")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    assert len(layers) == 1 and not assets and not unresolved
    stage = Usd.Stage.CreateInMemory()
    root = "/World/ground/terrain"
    stage.DefinePrim(root, "Xform").GetReferences().AddReference(str(path))
    assert not stage.GetPrimAtPath(f"{root}/granular_container/mesh").HasAPI(UsdPhysics.CollisionAPI)
    assert sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse()) == 8
    for name, count in (("gravel", 116222), ("sand", 930666)):
        points = UsdGeom.Points(stage.GetPrimAtPath(f"{root}/{name}_particles"))
        assert len(points.GetPointsAttr().Get()) == count
        assert len(points.GetWidthsAttr().Get()) == count
        assert len(points.GetDisplayColorAttr().Get()) == count
        assert points.GetPointsAttr().GetNumTimeSamples() == 0
        assert not np.any(np.asarray(points.GetVelocitiesAttr().Get()))
        assert UsdPhysics.MassAPI(points).GetMassAttr().Get() > 0
        system_path = f"{root}/{name}_ParticleSystem"
        targets = points.GetPrim().GetRelationship("physxParticle:particleSystem").GetTargets()
        assert [str(target) for target in targets] == [system_path]
        system = stage.GetPrimAtPath(system_path)
        assert system.GetTypeName() == "PhysxParticleSystem"
        material, _ = UsdShade.MaterialBindingAPI(system).ComputeBoundMaterial("physics")
        assert str(material.GetPath()) == f"{root}/PBDMaterial"
