# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.simulation_context import SimulationContext
from pxr import UsdPhysics, UsdShade

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.utils.assets import NVIDIA_NUCLEUS_DIR


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
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
