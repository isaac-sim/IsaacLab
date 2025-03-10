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
from pxr import UsdLux

import isaaclab.sim as sim_utils
from isaaclab.utils.string import to_camel_case


class TestSpawningLights(unittest.TestCase):
    """Test fixture for checking spawning of USD lights with different settings."""

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

    """
    Basic spawning.
    """

    def test_spawn_disk_light(self):
        """Test spawning a disk light source."""
        cfg = sim_utils.DiskLightCfg(
            color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
        )
        prim = cfg.func("/World/disk_light", cfg)

        # check if the light is spawned
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/disk_light"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "DiskLight")
        # validate properties on the prim
        self._validate_properties_on_prim("/World/disk_light", cfg)

    def test_spawn_distant_light(self):
        """Test spawning a distant light."""
        cfg = sim_utils.DistantLightCfg(
            color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, angle=20
        )
        prim = cfg.func("/World/distant_light", cfg)

        # check if the light is spawned
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/distant_light"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "DistantLight")
        # validate properties on the prim
        self._validate_properties_on_prim("/World/distant_light", cfg)

    def test_spawn_dome_light(self):
        """Test spawning a dome light source."""
        cfg = sim_utils.DomeLightCfg(
            color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100
        )
        prim = cfg.func("/World/dome_light", cfg)

        # check if the light is spawned
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/dome_light"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "DomeLight")
        # validate properties on the prim
        self._validate_properties_on_prim("/World/dome_light", cfg)

    def test_spawn_cylinder_light(self):
        """Test spawning a cylinder light source."""
        cfg = sim_utils.CylinderLightCfg(
            color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
        )
        prim = cfg.func("/World/cylinder_light", cfg)

        # check if the light is spawned
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/cylinder_light"))
        self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "CylinderLight")
        # validate properties on the prim
        self._validate_properties_on_prim("/World/cylinder_light", cfg)

    def test_spawn_sphere_light(self):
        """Test spawning a sphere light source."""
        cfg = sim_utils.SphereLightCfg(
            color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
        )
        prim = cfg.func("/World/sphere_light", cfg)

        # check if the light is spawned
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim_utils.is_prim_path_valid("/World/sphere_light"))
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
        # default list of params to skip
        non_usd_params = ["func", "prim_type", "visible", "semantic_tags", "copy_from_source"]
        # obtain prim
        prim = prim_utils.get_prim_at_path(prim_path)
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
