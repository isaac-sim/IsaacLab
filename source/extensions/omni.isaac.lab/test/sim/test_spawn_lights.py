# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import unittest

import omni.usd
from pxr import UsdLux

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import build_simulation_context
from omni.isaac.lab.utils.string import to_camel_case


class TestSpawningLights(unittest.TestCase):
    """Test fixture for checking spawning of USD lights with different settings."""

    """
    Basic spawning.
    """

    def test_spawn_disk_light(self):
        """Test spawning a disk light source."""
        with build_simulation_context():
            cfg = sim_utils.DiskLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
            )
            prim = cfg.func("/World/disk_light", cfg)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            # check if the light is spawned
            self.assertTrue(prim.IsValid())
            self.assertTrue(stage.GetPrimAtPath("/World/disk_light").IsValid())
            self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "DiskLight")
            # validate properties on the prim
            self._validate_properties_on_prim("/World/disk_light", cfg)

    def test_spawn_distant_light(self):
        """Test spawning a distant light."""
        with build_simulation_context():
            cfg = sim_utils.DistantLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, angle=20
            )
            prim = cfg.func("/World/distant_light", cfg)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            # check if the light is spawned
            self.assertTrue(prim.IsValid())
            self.assertTrue(stage.GetPrimAtPath("/World/distant_light").IsValid())
            self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "DistantLight")
            # validate properties on the prim
            self._validate_properties_on_prim("/World/distant_light", cfg)

    def test_spawn_dome_light(self):
        """Test spawning a dome light source."""
        with build_simulation_context():
            cfg = sim_utils.DomeLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100
            )
            prim = cfg.func("/World/dome_light", cfg)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            # check if the light is spawned
            self.assertTrue(prim.IsValid())
            self.assertTrue(stage.GetPrimAtPath("/World/dome_light").IsValid())
            self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "DomeLight")
            # validate properties on the prim
            self._validate_properties_on_prim("/World/dome_light", cfg)

    def test_spawn_cylinder_light(self):
        """Test spawning a cylinder light source."""
        with build_simulation_context():
            cfg = sim_utils.CylinderLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
            )
            prim = cfg.func("/World/cylinder_light", cfg)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            # check if the light is spawned
            self.assertTrue(prim.IsValid())
            self.assertTrue(stage.GetPrimAtPath("/World/cylinder_light").IsValid())
            self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "CylinderLight")
            # validate properties on the prim
            self._validate_properties_on_prim("/World/cylinder_light", cfg)

    def test_spawn_sphere_light(self):
        """Test spawning a sphere light source."""
        with build_simulation_context():
            cfg = sim_utils.SphereLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
            )
            prim = cfg.func("/World/sphere_light", cfg)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            # check if the light is spawned
            self.assertTrue(prim.IsValid())
            self.assertTrue(stage.GetPrimAtPath("/World/sphere_light").IsValid())
            self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "SphereLight")
            # validate properties on the prim
            self._validate_properties_on_prim("/World/sphere_light", cfg)

    """
    Helper functions.
    """

    def _validate_properties_on_prim(self, prim_path: str, cfg: sim_utils.LightCfg):
        """Validate the properties on the prim.

        Args:
            prim_path: The prim name.
            cfg: The configuration for the light source.
        """
        # get current stage
        stage = omni.usd.get_context().get_stage()
        # default list of params to skip
        non_usd_params = ["func", "prim_type", "visible", "semantic_tags", "copy_from_source"]
        # obtain prim
        prim = stage.GetPrimAtPath(prim_path)
        for attr_name, attr_value in cfg.__dict__.items():
            # skip names we know are not present
            if attr_name in non_usd_params or attr_value is None:
                continue
            # deal with texture input names
            if "texture" in attr_name:
                light_prim = UsdLux.DomeLight(prim)
                if attr_name == "texture_file":
                    configured_value = light_prim.GetTextureFileAttr().Get()
                elif attr_name == "texture_format":
                    configured_value = light_prim.GetTextureFormatAttr().Get()
                else:
                    raise ValueError(f"Unknown texture attribute: '{attr_name}'")
            else:
                # convert attribute name in prim to cfg name
                if attr_name == "visible_in_primary_ray":
                    prim_prop_name = f"{to_camel_case(attr_name, to='cC')}"
                else:
                    prim_prop_name = f"inputs:{to_camel_case(attr_name, to='cC')}"
                # configured value
                configured_value = prim.GetAttribute(prim_prop_name).Get()
            # validate the values
            self.assertEqual(configured_value, attr_value, msg=f"Failed for attribute: '{attr_name}'")


if __name__ == "__main__":
    run_tests()
