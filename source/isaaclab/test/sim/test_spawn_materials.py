# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api.simulation_context import SimulationContext
from pxr import UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR


class TestSpawningMaterials(unittest.TestCase):
    """Test fixture for checking spawning of materials."""

    def setUp(self) -> None:
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.1
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="numpy")
        # Wait for spawning
        stage_utils.update_stage()

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    def test_spawn_preview_surface(self):
        """Test spawning preview surface."""
        # Spawn preview surface
        cfg = sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        prim = cfg.func("/Looks/PreviewSurface", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/Looks/PreviewSurface"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Shader")
        # Check properties
        self.assertEqual(prim.GetAttribute("inputs:diffuseColor").Get(), cfg.diffuse_color)

    def test_spawn_mdl_material(self):
        """Test spawning mdl material."""
        # Spawn mdl material
        cfg = sim_utils.materials.MdlFileCfg(
            mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
            project_uvw=True,
            albedo_brightness=0.5,
        )
        prim = cfg.func("/Looks/MdlMaterial", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/Looks/MdlMaterial"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Shader")
        # Check properties
        self.assertEqual(prim.GetAttribute("inputs:project_uvw").Get(), cfg.project_uvw)
        self.assertEqual(prim.GetAttribute("inputs:albedo_brightness").Get(), cfg.albedo_brightness)

    def test_spawn_glass_mdl_material(self):
        """Test spawning a glass mdl material."""
        # Spawn mdl material
        cfg = sim_utils.materials.GlassMdlCfg(thin_walled=False, glass_ior=1.0, glass_color=(0.0, 1.0, 0.0))
        prim = cfg.func("/Looks/GlassMaterial", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/Looks/GlassMaterial"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Shader")
        # Check properties
        self.assertEqual(prim.GetAttribute("inputs:thin_walled").Get(), cfg.thin_walled)
        self.assertEqual(prim.GetAttribute("inputs:glass_ior").Get(), cfg.glass_ior)
        self.assertEqual(prim.GetAttribute("inputs:glass_color").Get(), cfg.glass_color)

    def test_spawn_rigid_body_material(self):
        """Test spawning a rigid body material."""
        # spawn physics material
        cfg = sim_utils.materials.RigidBodyMaterialCfg(
            dynamic_friction=1.5,
            restitution=1.5,
            static_friction=0.5,
            restitution_combine_mode="max",
            friction_combine_mode="max",
            improve_patch_friction=True,
        )
        prim = cfg.func("/Looks/RigidBodyMaterial", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/Looks/RigidBodyMaterial"))
        # Check properties
        self.assertEqual(prim.GetAttribute("physics:staticFriction").Get(), cfg.static_friction)
        self.assertEqual(prim.GetAttribute("physics:dynamicFriction").Get(), cfg.dynamic_friction)
        self.assertEqual(prim.GetAttribute("physics:restitution").Get(), cfg.restitution)
        self.assertEqual(prim.GetAttribute("physxMaterial:improvePatchFriction").Get(), cfg.improve_patch_friction)
        self.assertEqual(prim.GetAttribute("physxMaterial:restitutionCombineMode").Get(), cfg.restitution_combine_mode)
        self.assertEqual(prim.GetAttribute("physxMaterial:frictionCombineMode").Get(), cfg.friction_combine_mode)

    def test_spawn_deformable_body_material(self):
        """Test spawning a deformable body material."""
        # spawn deformable body material
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
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/Looks/DeformableBodyMaterial"))
        # Check properties
        self.assertEqual(prim.GetAttribute("physxDeformableBodyMaterial:density").Get(), cfg.density)
        self.assertEqual(prim.GetAttribute("physxDeformableBodyMaterial:dynamicFriction").Get(), cfg.dynamic_friction)
        self.assertEqual(prim.GetAttribute("physxDeformableBodyMaterial:youngsModulus").Get(), cfg.youngs_modulus)
        self.assertEqual(prim.GetAttribute("physxDeformableBodyMaterial:poissonsRatio").Get(), cfg.poissons_ratio)
        self.assertAlmostEqual(
            prim.GetAttribute("physxDeformableBodyMaterial:elasticityDamping").Get(), cfg.elasticity_damping
        )
        self.assertEqual(prim.GetAttribute("physxDeformableBodyMaterial:dampingScale").Get(), cfg.damping_scale)

    def test_apply_rigid_body_material_on_visual_material(self):
        """Test applying a rigid body material on a visual material."""
        # Spawn mdl material
        cfg = sim_utils.materials.GlassMdlCfg(thin_walled=False, glass_ior=1.0, glass_color=(0.0, 1.0, 0.0))
        prim = cfg.func("/Looks/Material", cfg)
        # spawn physics material
        cfg = sim_utils.materials.RigidBodyMaterialCfg(
            dynamic_friction=1.5,
            restitution=1.5,
            static_friction=0.5,
            restitution_combine_mode="max",
            friction_combine_mode="max",
            improve_patch_friction=True,
        )
        prim = cfg.func("/Looks/Material", cfg)
        # Check validity
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/Looks/Material"))
        # Check properties
        self.assertEqual(prim.GetAttribute("physics:staticFriction").Get(), cfg.static_friction)
        self.assertEqual(prim.GetAttribute("physics:dynamicFriction").Get(), cfg.dynamic_friction)
        self.assertEqual(prim.GetAttribute("physics:restitution").Get(), cfg.restitution)
        self.assertEqual(prim.GetAttribute("physxMaterial:improvePatchFriction").Get(), cfg.improve_patch_friction)
        self.assertEqual(prim.GetAttribute("physxMaterial:restitutionCombineMode").Get(), cfg.restitution_combine_mode)
        self.assertEqual(prim.GetAttribute("physxMaterial:frictionCombineMode").Get(), cfg.friction_combine_mode)

    def test_bind_prim_to_material(self):
        """Test binding a rigid body material on a mesh prim."""

        # create a mesh prim
        object_prim = prim_utils.create_prim("/World/Geometry/box", "Cube")
        UsdPhysics.CollisionAPI.Apply(object_prim)

        # create a visual material
        visual_material_cfg = sim_utils.GlassMdlCfg(glass_ior=1.0, thin_walled=True)
        visual_material_cfg.func("/World/Looks/glassMaterial", visual_material_cfg)
        # create a physics material
        physics_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=1.5, restitution=1.5
        )
        physics_material_cfg.func("/World/Physics/rubberMaterial", physics_material_cfg)
        # bind the visual material to the mesh prim
        sim_utils.bind_visual_material("/World/Geometry/box", "/World/Looks/glassMaterial")
        sim_utils.bind_physics_material("/World/Geometry/box", "/World/Physics/rubberMaterial")

        # check the main material binding
        material_binding_api = UsdShade.MaterialBindingAPI(object_prim)
        # -- visual
        material_direct_binding = material_binding_api.GetDirectBinding()
        self.assertEqual(material_direct_binding.GetMaterialPath(), "/World/Looks/glassMaterial")
        self.assertEqual(material_direct_binding.GetMaterialPurpose(), "")
        # -- physics
        material_direct_binding = material_binding_api.GetDirectBinding("physics")
        self.assertEqual(material_direct_binding.GetMaterialPath(), "/World/Physics/rubberMaterial")
        self.assertEqual(material_direct_binding.GetMaterialPurpose(), "physics")


if __name__ == "__main__":
    run_tests()
