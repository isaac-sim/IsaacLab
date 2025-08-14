# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import pytest
from isaacsim.core.api.simulation_context import SimulationContext
from pxr import UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR


@pytest.fixture
def sim():
    """Create a simulation context."""
    stage_utils.create_new_stage()
    dt = 0.1
    sim = SimulationContext(physics_dt=dt, rendering_dt=dt, backend="numpy")
    stage_utils.update_stage()
    yield sim
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


def test_spawn_preview_surface(sim):
    """Test spawning preview surface."""
    cfg = sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    prim = cfg.func("/Looks/PreviewSurface", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/Looks/PreviewSurface")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Shader"
    # Check properties
    assert prim.GetAttribute("inputs:diffuseColor").Get() == cfg.diffuse_color


def test_spawn_mdl_material(sim):
    """Test spawning mdl material."""
    cfg = sim_utils.materials.MdlFileCfg(
        mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
        project_uvw=True,
        albedo_brightness=0.5,
    )
    prim = cfg.func("/Looks/MdlMaterial", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/Looks/MdlMaterial")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Shader"
    # Check properties
    assert prim.GetAttribute("inputs:project_uvw").Get() == cfg.project_uvw
    assert prim.GetAttribute("inputs:albedo_brightness").Get() == cfg.albedo_brightness


def test_spawn_glass_mdl_material(sim):
    """Test spawning a glass mdl material."""
    cfg = sim_utils.materials.GlassMdlCfg(thin_walled=False, glass_ior=1.0, glass_color=(0.0, 1.0, 0.0))
    prim = cfg.func("/Looks/GlassMaterial", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/Looks/GlassMaterial")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Shader"
    # Check properties
    assert prim.GetAttribute("inputs:thin_walled").Get() == cfg.thin_walled
    assert prim.GetAttribute("inputs:glass_ior").Get() == cfg.glass_ior
    assert prim.GetAttribute("inputs:glass_color").Get() == cfg.glass_color


def test_spawn_rigid_body_material(sim):
    """Test spawning a rigid body material."""
    cfg = sim_utils.materials.RigidBodyMaterialCfg(
        dynamic_friction=1.5,
        restitution=1.5,
        static_friction=0.5,
        restitution_combine_mode="max",
        friction_combine_mode="max",
    )
    prim = cfg.func("/Looks/RigidBodyMaterial", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/Looks/RigidBodyMaterial")
    # Check properties
    assert prim.GetAttribute("physics:staticFriction").Get() == cfg.static_friction
    assert prim.GetAttribute("physics:dynamicFriction").Get() == cfg.dynamic_friction
    assert prim.GetAttribute("physics:restitution").Get() == cfg.restitution
    assert prim.GetAttribute("physxMaterial:restitutionCombineMode").Get() == cfg.restitution_combine_mode
    assert prim.GetAttribute("physxMaterial:frictionCombineMode").Get() == cfg.friction_combine_mode


def test_spawn_deformable_body_material(sim):
    """Test spawning a deformable body material."""
    cfg = sim_utils.materials.DeformableBodyMaterialCfg(
        density=1.0,
        dynamic_friction=0.25,
        youngs_modulus=50000000.0,
        poissons_ratio=0.5,
        elasticity_damping=0.005,
        damping_scale=1.0,
    )
    prim = cfg.func("/Looks/DeformableBodyMaterial", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/Looks/DeformableBodyMaterial")
    # Check properties
    assert prim.GetAttribute("physxDeformableBodyMaterial:density").Get() == cfg.density
    assert prim.GetAttribute("physxDeformableBodyMaterial:dynamicFriction").Get() == cfg.dynamic_friction
    assert prim.GetAttribute("physxDeformableBodyMaterial:youngsModulus").Get() == cfg.youngs_modulus
    assert prim.GetAttribute("physxDeformableBodyMaterial:poissonsRatio").Get() == cfg.poissons_ratio
    assert prim.GetAttribute("physxDeformableBodyMaterial:elasticityDamping").Get() == pytest.approx(
        cfg.elasticity_damping
    )
    assert prim.GetAttribute("physxDeformableBodyMaterial:dampingScale").Get() == cfg.damping_scale


def test_apply_rigid_body_material_on_visual_material(sim):
    """Test applying a rigid body material on a visual material."""
    cfg = sim_utils.materials.GlassMdlCfg(thin_walled=False, glass_ior=1.0, glass_color=(0.0, 1.0, 0.0))
    prim = cfg.func("/Looks/Material", cfg)
    cfg = sim_utils.materials.RigidBodyMaterialCfg(
        dynamic_friction=1.5,
        restitution=1.5,
        static_friction=0.5,
        restitution_combine_mode="max",
        friction_combine_mode="max",
    )
    prim = cfg.func("/Looks/Material", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/Looks/Material")
    # Check properties
    assert prim.GetAttribute("physics:staticFriction").Get() == cfg.static_friction
    assert prim.GetAttribute("physics:dynamicFriction").Get() == cfg.dynamic_friction
    assert prim.GetAttribute("physics:restitution").Get() == cfg.restitution
    assert prim.GetAttribute("physxMaterial:restitutionCombineMode").Get() == cfg.restitution_combine_mode
    assert prim.GetAttribute("physxMaterial:frictionCombineMode").Get() == cfg.friction_combine_mode


def test_bind_prim_to_material(sim):
    """Test binding a rigid body material on a mesh prim."""

    # create a mesh prim
    object_prim = prim_utils.create_prim("/World/Geometry/box", "Cube")
    UsdPhysics.CollisionAPI.Apply(object_prim)

    # create a visual material
    visual_material_cfg = sim_utils.GlassMdlCfg(glass_ior=1.0, thin_walled=True)
    visual_material_cfg.func("/World/Looks/glassMaterial", visual_material_cfg)
    # create a physics material
    physics_material_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=1.5, restitution=1.5)
    physics_material_cfg.func("/World/Physics/rubberMaterial", physics_material_cfg)
    sim_utils.bind_visual_material("/World/Geometry/box", "/World/Looks/glassMaterial")
    sim_utils.bind_physics_material("/World/Geometry/box", "/World/Physics/rubberMaterial")

    # check the material binding
    material_binding_api = UsdShade.MaterialBindingAPI(object_prim)
    # -- visual material
    material_direct_binding = material_binding_api.GetDirectBinding()
    assert material_direct_binding.GetMaterialPath() == "/World/Looks/glassMaterial"
    assert material_direct_binding.GetMaterialPurpose() == ""
    # -- physics material
    material_direct_binding = material_binding_api.GetDirectBinding("physics")
    assert material_direct_binding.GetMaterialPath() == "/World/Physics/rubberMaterial"
    assert material_direct_binding.GetMaterialPurpose() == "physics"
