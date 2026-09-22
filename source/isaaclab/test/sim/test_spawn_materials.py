# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect

import pytest

from pxr import Sdf, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import visual_materials
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

pytestmark = [pytest.mark.unit, pytest.mark.isaacsim_ci]


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


def test_spawn_preview_surface(stage):
    """Preview surfaces use renderer-agnostic USD shader inputs and outputs."""
    cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    prim = cfg.func("/Looks/PreviewSurface", cfg)

    shader = UsdShade.Shader(prim)
    material = UsdShade.Material(stage.GetPrimAtPath("/Looks/PreviewSurface"))
    assert prim.GetTypeName() == "Shader"
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("diffuseColor").Get() == cfg.diffuse_color
    assert shader.GetInput("diffuseColor").GetTypeName() == Sdf.ValueTypeNames.Color3f
    assert shader.GetInput("emissiveColor").GetTypeName() == Sdf.ValueTypeNames.Color3f
    assert shader.GetInput("roughness").GetTypeName() == Sdf.ValueTypeNames.Float
    assert shader.GetOutput("surface").GetTypeName() == Sdf.ValueTypeNames.Token
    assert shader.GetOutput("displacement").GetTypeName() == Sdf.ValueTypeNames.Token
    assert material.GetSurfaceOutput().HasConnectedSource()
    assert material.GetDisplacementOutput().HasConnectedSource()
    with pytest.raises(ValueError, match="already exists"):
        cfg.func("/Looks/PreviewSurface", cfg)


@pytest.mark.parametrize(
    ("cfg", "sub_identifier", "inputs"),
    [
        (
            sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
                project_uvw=True,
                albedo_brightness=0.5,
            ),
            "Aluminum_Anodized",
            {"project_uvw": True, "albedo_brightness": 0.5},
        ),
        (
            sim_utils.GlassMdlCfg(thin_walled=False, glass_ior=1.0, glass_color=(0.0, 1.0, 0.0)),
            "OmniGlass",
            {"thin_walled": False, "glass_ior": 1.0, "glass_color": (0.0, 1.0, 0.0)},
        ),
    ],
    ids=["mdl_file", "glass"],
)
def test_spawn_mdl_material(stage, cfg, sub_identifier, inputs):
    prim = cfg.func("/Looks/MdlMaterial", cfg)

    shader = UsdShade.Shader(prim)
    material = UsdShade.Material(stage.GetPrimAtPath("/Looks/MdlMaterial"))
    assert prim.GetTypeName() == "Shader"
    # the nucleus placeholder is expanded in the authored asset path
    assert shader.GetSourceAsset("mdl").path == cfg.mdl_path.format(NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR)
    assert shader.GetSourceAssetSubIdentifier("mdl") == sub_identifier
    assert shader.GetOutput("out").GetRenderType() == "material"
    assert material.GetSurfaceOutput("mdl").HasConnectedSource()
    assert material.GetDisplacementOutput("mdl").HasConnectedSource()
    assert material.GetVolumeOutput("mdl").HasConnectedSource()
    for name, value in inputs.items():
        assert prim.GetAttribute(f"inputs:{name}").Get() == value


def test_visual_material_spawners_do_not_depend_on_kit_commands():
    """Visual material ownership stays in the OpenUSD layer."""
    source = inspect.getsource(visual_materials)
    assert "has_kit" not in source
    assert "omni.usd.commands" not in source


def test_spawn_rigid_body_material(stage):
    cfg = sim_utils.RigidBodyMaterialCfg(
        dynamic_friction=1.5,
        restitution=1.5,
        static_friction=0.5,
        restitution_combine_mode="max",
        friction_combine_mode="max",
    )
    expected = {
        "physics:staticFriction": 0.5,
        "physics:dynamicFriction": 1.5,
        "physics:restitution": 1.5,
        "physxMaterial:restitutionCombineMode": "max",
        "physxMaterial:frictionCombineMode": "max",
    }
    prim = cfg.func("/Looks/RigidBodyMaterial", cfg)
    assert prim.IsA(UsdShade.Material)
    for name, value in expected.items():
        assert prim.GetAttribute(name).Get() == value

    # a physics material can be layered onto an existing visual material prim
    glass_cfg = sim_utils.GlassMdlCfg()
    glass_cfg.func("/Looks/Material", glass_cfg)
    prim = cfg.func("/Looks/Material", cfg)
    for name, value in expected.items():
        assert prim.GetAttribute(name).Get() == value
    # but not onto a non-material prim
    sim_utils.create_prim("/World/Xform", "Xform")
    with pytest.raises(ValueError, match="not a material"):
        cfg.func("/World/Xform", cfg)


def test_bind_materials(stage):
    object_prim = sim_utils.create_prim("/World/Geometry/box", "Cube")
    UsdPhysics.CollisionAPI.Apply(object_prim)
    plain_prim = sim_utils.create_prim("/World/Geometry/plain", "Xform")
    visual_cfg = sim_utils.GlassMdlCfg(glass_ior=1.0, thin_walled=True)
    visual_cfg.func("/World/Looks/glassMaterial", visual_cfg)
    physics_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=1.5, restitution=1.5)
    physics_cfg.func("/World/Physics/rubberMaterial", physics_cfg)

    sim_utils.bind_visual_material("/World/Geometry/box", "/World/Looks/glassMaterial")
    sim_utils.bind_physics_material("/World/Geometry/box", "/World/Physics/rubberMaterial")
    # physics materials only bind to physics-enabled prims
    sim_utils.bind_physics_material("/World/Geometry/plain", "/World/Physics/rubberMaterial")

    binding_api = UsdShade.MaterialBindingAPI(object_prim)
    visual_binding = binding_api.GetDirectBinding()
    assert visual_binding.GetMaterialPath() == "/World/Looks/glassMaterial"
    assert visual_binding.GetMaterialPurpose() == ""
    physics_binding = binding_api.GetDirectBinding("physics")
    assert physics_binding.GetMaterialPath() == "/World/Physics/rubberMaterial"
    assert physics_binding.GetMaterialPurpose() == "physics"
    assert not plain_prim.HasAPI(UsdShade.MaterialBindingAPI)

    with pytest.raises(ValueError, match="not valid"):
        sim_utils.bind_visual_material("/World/Missing", "/World/Looks/glassMaterial")
    with pytest.raises(ValueError, match="Visual material '/World/Looks/missing' does not exist"):
        sim_utils.bind_visual_material("/World/Geometry/box", "/World/Looks/missing")
    with pytest.raises(ValueError, match="Physics material '/World/Physics/missing' does not exist"):
        sim_utils.bind_physics_material("/World/Geometry/box", "/World/Physics/missing")
