# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ground-plane spawner contract against different asset layouts (no Kit required).

The ground-plane spawner historically assumed the internal layout of the default grid asset from
Isaac Sim (a ``Plane``-typed collision prim and the ``Looks/theGrid/Shader`` material). These tests
verify that the spawner works with any ground-plane USD file and only applies the asset-specific
overrides when the corresponding prims exist. See: https://github.com/isaac-sim/IsaacLab/issues/6326
"""

from __future__ import annotations

import logging

import pytest

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import create_new_stage, get_current_stage

pytestmark = pytest.mark.unit


"""
Fixtures.
"""


@pytest.fixture
def stage() -> Usd.Stage:
    """Create a blank new stage for each test."""
    create_new_stage()
    return get_current_stage()


@pytest.fixture
def legacy_grid_asset(tmp_path) -> str:
    """Create a ground-plane USD file that mimics the layout of the default grid asset from Isaac Sim."""
    asset_path = str(tmp_path / "legacy_grid_ground.usda")
    stage = Usd.Stage.CreateNew(asset_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    # grid material and shader (targeted by the color override)
    UsdShade.Material.Define(stage, "/World/Looks/theGrid")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/theGrid/Shader")
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    # collision plane
    plane = UsdGeom.Plane.Define(stage, "/World/GroundPlane/CollisionPlane")
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    # environment mesh (targeted by the size override)
    environment = UsdGeom.Xform.Define(stage, "/World/Environment")
    environment.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    # sphere light (hidden by the spawner)
    UsdLux.SphereLight.Define(stage, "/World/SphereLight")
    stage.Save()
    return asset_path


@pytest.fixture
def mesh_ground_asset(tmp_path) -> str:
    """Create a ground-plane USD file with a mesh-based collider and no legacy grid prims."""
    asset_path = str(tmp_path / "mesh_ground.usda")
    stage = Usd.Stage.CreateNew(asset_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    # collision-enabled ground mesh
    mesh = UsdGeom.Mesh.Define(stage, "/World/Geometry/GroundMesh")
    mesh.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    # preview-surface material bound to the mesh
    material = UsdShade.Material.Define(stage, "/World/Looks/GroundMat")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/GroundMat/PreviewShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.Save()
    return asset_path


@pytest.fixture
def no_collision_asset(tmp_path) -> str:
    """Create a ground-plane USD file without any collision-enabled prim."""
    asset_path = str(tmp_path / "no_collision_ground.usda")
    stage = Usd.Stage.CreateNew(asset_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Geometry/GroundMesh")
    mesh.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    stage.Save()
    return asset_path


"""
Tests - spawn_ground_plane.
"""


def test_physics_material_binds_to_collision_enabled_prims(stage, mesh_ground_asset):
    """Physics material binds to prims with a collision API, independent of the prim type."""
    cfg = sim_utils.GroundPlaneCfg(
        usd_path=mesh_ground_asset,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        color=None,
    )
    prim = cfg.func("/World/ground", cfg)

    assert prim.IsValid()
    mesh_prim = stage.GetPrimAtPath("/World/ground/Geometry/GroundMesh")
    binding_rel = mesh_prim.GetRelationship("material:binding:physics")
    assert binding_rel.IsValid()
    assert binding_rel.GetTargets() == [Sdf.Path("/World/ground/physicsMaterial")]


def test_missing_collision_prim_warns_and_skips_binding(stage, no_collision_asset, caplog):
    """Physics material binding is skipped with a warning when the asset has no collision prim."""
    cfg = sim_utils.GroundPlaneCfg(
        usd_path=no_collision_asset,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        color=None,
    )
    with caplog.at_level(logging.WARNING):
        prim = cfg.func("/World/ground", cfg)

    assert prim.IsValid()
    assert "Skipping physics material binding" in caplog.text


def test_color_override_skipped_when_grid_shader_missing(stage, mesh_ground_asset, caplog):
    """Color override is skipped with a warning when the asset has no legacy grid shader."""
    cfg = sim_utils.GroundPlaneCfg(usd_path=mesh_ground_asset, physics_material=None, color=(1.0, 0.0, 0.0))
    with caplog.at_level(logging.WARNING):
        prim = cfg.func("/World/ground", cfg)

    assert prim.IsValid()
    assert "Skipping color override" in caplog.text
    assert not stage.GetPrimAtPath("/World/ground/Looks/theGrid/Shader").IsValid()


def test_spawn_without_overrides_succeeds_on_any_asset(stage, mesh_ground_asset):
    """Spawning with no material or color overrides works with any ground-plane asset."""
    cfg = sim_utils.GroundPlaneCfg(usd_path=mesh_ground_asset, physics_material=None, color=None)
    prim = cfg.func("/World/ground", cfg)

    assert prim.IsValid()


def test_legacy_grid_asset_keeps_full_behavior(stage, legacy_grid_asset):
    """All overrides still apply to assets that follow the default grid asset's layout."""
    cfg = sim_utils.GroundPlaneCfg(
        usd_path=legacy_grid_asset,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        color=(0.2, 0.3, 0.4),
        size=(200.0, 300.0),
    )
    prim = cfg.func("/World/ground", cfg)
    assert prim.IsValid()

    # physics material is bound to the collision plane
    plane_prim = stage.GetPrimAtPath("/World/ground/GroundPlane/CollisionPlane")
    binding_rel = plane_prim.GetRelationship("material:binding:physics")
    assert binding_rel.IsValid()
    assert binding_rel.GetTargets() == [Sdf.Path("/World/ground/physicsMaterial")]
    # color is written to the grid shader
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/ground/Looks/theGrid/Shader"))
    assert shader.GetInput("diffuse_tint").Get() == Gf.Vec3f(0.2, 0.3, 0.4)
    # size is applied to the environment prim
    environment_prim = stage.GetPrimAtPath("/World/ground/Environment")
    assert environment_prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3f(2.0, 3.0, 1.0)
    # light is hidden
    light_prim = stage.GetPrimAtPath("/World/ground/SphereLight")
    assert UsdGeom.Imageable(light_prim).ComputeVisibility() == UsdGeom.Tokens.invisible


"""
Tests - TerrainImporter ground-plane color contract.
"""


@pytest.mark.parametrize(
    ("visual_material", "expected_color"),
    [
        pytest.param(None, None, id="no_visual_material"),
        pytest.param(sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.3)), (0.1, 0.2, 0.3), id="diffuse_color"),
        pytest.param(sim_utils.GlassMdlCfg(), None, id="material_without_diffuse_color"),
    ],
)
def test_ground_plane_color_contract(visual_material, expected_color, caplog):
    """The terrain importer only overrides the ground-plane color when a diffuse color is configured."""
    from isaaclab.terrains.terrain_importer import _ground_plane_color

    with caplog.at_level(logging.WARNING):
        assert _ground_plane_color(visual_material) == expected_color

    if visual_material is not None and expected_color is None:
        assert "Skipping the color override" in caplog.text
