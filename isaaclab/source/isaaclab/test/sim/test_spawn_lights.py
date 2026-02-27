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
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


def test_spawn_disk_light(sim):
    """Test spawning a disk light source."""
    cfg = sim_utils.DiskLightCfg(
        color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
    )
    prim = cfg.func("/World/disk_light", cfg)

    # check if the light is spawned
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/disk_light").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "DiskLight"
    # validate properties on the prim
    _validate_properties_on_prim(prim, cfg)


def test_spawn_distant_light(sim):
    """Test spawning a distant light."""
    cfg = sim_utils.DistantLightCfg(
        color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, angle=20
    )
    prim = cfg.func("/World/distant_light", cfg)

    # check if the light is spawned
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/distant_light").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "DistantLight"
    # validate properties on the prim
    _validate_properties_on_prim(prim, cfg)


def test_spawn_dome_light(sim):
    """Test spawning a dome light source."""
    cfg = sim_utils.DomeLightCfg(
        color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100
    )
    prim = cfg.func("/World/dome_light", cfg)

    # check if the light is spawned
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/dome_light").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "DomeLight"
    # validate properties on the prim
    _validate_properties_on_prim(prim, cfg)


def test_spawn_cylinder_light(sim):
    """Test spawning a cylinder light source."""
    cfg = sim_utils.CylinderLightCfg(
        color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
    )
    prim = cfg.func("/World/cylinder_light", cfg)

    # check if the light is spawned
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/cylinder_light").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "CylinderLight"
    # validate properties on the prim
    _validate_properties_on_prim(prim, cfg)


def test_spawn_sphere_light(sim):
    """Test spawning a sphere light source."""
    cfg = sim_utils.SphereLightCfg(
        color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=100, radius=20.0
    )
    prim = cfg.func("/World/sphere_light", cfg)

    # check if the light is spawned
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/sphere_light").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "SphereLight"
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
