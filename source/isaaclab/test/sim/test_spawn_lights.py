# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""


import pytest

from pxr import Usd, UsdLux

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.string import to_camel_case

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.fixture(autouse=True)
def sim():
    """Setup and teardown for each test."""
    # Setup: Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.1
    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=dt))
    # Wait for spawning
    sim_utils.update_stage()

    # Yield the simulation context for the test
    yield sim

    # Teardown: Stop simulation
    sim.stop()
    sim.clear_instance()


def test_spawn_lights(sim):
    """Test spawning each light type through the shared light spawner."""
    common = dict(color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100)
    cases = [
        ("/World/disk_light", sim_utils.DiskLightCfg(**common, radius=20.0), "DiskLight"),
        ("/World/distant_light", sim_utils.DistantLightCfg(**common, angle=20), "DistantLight"),
        ("/World/dome_light", sim_utils.DomeLightCfg(**common), "DomeLight"),
        ("/World/cylinder_light", sim_utils.CylinderLightCfg(**common, radius=20.0), "CylinderLight"),
        ("/World/sphere_light", sim_utils.SphereLightCfg(**common, radius=20.0), "SphereLight"),
    ]
    for prim_path, cfg, type_name in cases:
        prim = cfg.func(prim_path, cfg)

        # check if the light is spawned
        assert prim.IsValid()
        assert sim.stage.GetPrimAtPath(prim_path).IsValid()
        assert prim.GetPrimTypeInfo().GetTypeName() == type_name
        # validate properties on the prim
        _validate_properties_on_prim(prim, cfg)


"""
Helper functions.
"""


def _validate_properties_on_prim(prim: Usd.Prim, cfg: sim_utils.LightCfg):
    """Validate the properties on the prim.

    Args:
        prim: The prim.
        cfg: The configuration for the light source.
    """
    # default list of params to skip
    non_usd_params = ["func", "prim_type", "visible", "semantic_tags", "copy_from_source"]
    # validate the properties
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
        assert configured_value == attr_value, f"Failed for attribute: '{attr_name}'"
