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

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import build_simulation_context
from omni.isaac.lab.sim.spawners.sensors.sensors import (
    CUSTOM_FISHEYE_CAMERA_ATTRIBUTES,
    CUSTOM_PINHOLE_CAMERA_ATTRIBUTES,
)
from omni.isaac.lab.utils.string import to_camel_case


class TestSpawningSensors(unittest.TestCase):
    """Test fixture for checking spawning of USD sensors with different settings."""

    """
    Basic spawning.
    """

    def test_spawn_pinhole_camera(self):
        """Test spawning a pinhole camera."""
        with build_simulation_context():
            cfg = sim_utils.PinholeCameraCfg(
                focal_length=5.0, f_stop=10.0, clipping_range=(0.1, 1000.0), horizontal_aperture=10.0
            )
            prim = cfg.func("/World/pinhole_camera", cfg)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            # check if the light is spawned
            self.assertTrue(prim.IsValid())
            self.assertTrue(stage.GetPrimAtPath("/World/pinhole_camera").IsValid())
            self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Camera")
            # validate properties on the prim
            self._validate_properties_on_prim("/World/pinhole_camera", cfg, CUSTOM_PINHOLE_CAMERA_ATTRIBUTES)

    def test_spawn_fisheye_camera(self):
        """Test spawning a fisheye camera."""
        with build_simulation_context():
            cfg = sim_utils.FisheyeCameraCfg(
                projection_type="fisheye_equidistant",
                focal_length=5.0,
                f_stop=10.0,
                clipping_range=(0.1, 1000.0),
                horizontal_aperture=10.0,
            )
            # FIXME: This throws a warning. Check with Replicator team if this is expected/known.
            #   [omni.hydra] Camera '/World/fisheye_camera': Unknown projection type, defaulting to pinhole
            prim = cfg.func("/World/fisheye_camera", cfg)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            # check if the light is spawned
            self.assertTrue(prim.IsValid())
            self.assertTrue(stage.GetPrimAtPath("/World/fisheye_camera").IsValid())
            self.assertEqual(prim.GetPrimTypeInfo().GetTypeName(), "Camera")
            # validate properties on the prim
            self._validate_properties_on_prim("/World/fisheye_camera", cfg, CUSTOM_FISHEYE_CAMERA_ATTRIBUTES)

    """
    Helper functions.
    """

    def _validate_properties_on_prim(self, prim_path: str, cfg: object, custom_attr: dict):
        """Validate the properties on the prim.

        Args:
            prim_path: The prim name.
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
        # get prim
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
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
            self.assertAlmostEqual(prim.GetAttribute(prim_prop_name).Get(), attr_value, places=5)


if __name__ == "__main__":
    run_tests()
