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

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.sensors.sensors import CUSTOM_FISHEYE_CAMERA_ATTRIBUTES, CUSTOM_PINHOLE_CAMERA_ATTRIBUTES
from isaaclab.utils.string import to_camel_case


@pytest.fixture
def sim():
    """Create a simulation context."""
    sim_utils.create_new_stage()
    dt = 0.1
    sim = SimulationContext(SimulationCfg(dt=dt))
    sim_utils.update_stage()
    yield sim
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


"""
Basic spawning.
"""


def test_spawn_pinhole_camera(sim):
    """Test spawning a pinhole camera."""
    cfg = sim_utils.PinholeCameraCfg(
        focal_length=5.0, f_stop=10.0, clipping_range=(0.1, 1000.0), horizontal_aperture=10.0
    )
    prim = cfg.func("/World/pinhole_camera", cfg)
    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/pinhole_camera").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Camera"
    # Check properties
    _validate_properties_on_prim(prim, cfg, CUSTOM_PINHOLE_CAMERA_ATTRIBUTES)


def test_spawn_fisheye_camera(sim):
    """Test spawning a fisheye camera."""
    cfg = sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        focal_length=5.0,
        f_stop=10.0,
        clipping_range=(0.1, 1000.0),
        horizontal_aperture=10.0,
    )
    # FIXME: This throws a warning. Check with Replicator team if this is expected/known.
    #   [omni.hydra] Camera '/World/fisheye_camera': Unknown projection type, defaulting to pinhole
    prim = cfg.func("/World/fisheye_camera", cfg)
    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/fisheye_camera").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Camera"
    # Check properties
    _validate_properties_on_prim(prim, cfg, CUSTOM_FISHEYE_CAMERA_ATTRIBUTES)


"""
Helper functions.
"""


def _validate_properties_on_prim(prim: Usd.Prim, cfg: object, custom_attr: dict):
    """Validate the properties on the prim.

    Args:
        prim: The prim.
        cfg: The configuration object.
        custom_attr: The custom attributes for sensor.
    """
    # delete custom attributes in the config that are not USD parameters
    non_usd_cfg_param_names = [
        "func",
        "copy_from_source",
        "lock_camera",
        "visible",
        "semantic_tags",
        "from_intrinsic_matrix",
    ]
    # validate the properties
    for attr_name, attr_value in cfg.__dict__.items():
        # skip names we know are not present
        if attr_name in non_usd_cfg_param_names or attr_value is None:
            continue
        # obtain prim property name
        if attr_name in custom_attr:
            # check custom attributes
            prim_prop_name = custom_attr[attr_name][0]
        else:
            # convert attribute name in prim to cfg name
            prim_prop_name = to_camel_case(attr_name, to="cC")
        # validate the values
        assert prim.GetAttribute(prim_prop_name).Get() == pytest.approx(attr_value, rel=1e-5)
