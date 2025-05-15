# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore private usage of variables warning.
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.app import AppLauncher

# Can set this to False to see the GUI for debugging.
HEADLESS = True

# Launch omniverse app.
app_launcher = AppLauncher(headless=HEADLESS, kit_args="--enable isaacsim.xr.openxr")
simulation_app = app_launcher.app

import numpy as np
import unittest

import carb
import omni.usd
from isaacsim.core.prims import XFormPrim

from isaaclab.devices import OpenXRDevice
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class EmptyManagerCfg:
    """Empty manager."""

    pass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


@configclass
class EmptyEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the empty test environment."""

    scene: EmptySceneCfg = EmptySceneCfg(num_envs=1, env_spacing=1.0)
    actions: EmptyManagerCfg = EmptyManagerCfg()
    observations: EmptyManagerCfg = EmptyManagerCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 5
        self.episode_length_s = 30.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2


class TestOpenXRDevice(unittest.TestCase):
    """Test for OpenXRDevice"""

    def test_xr_anchor(self):
        env_cfg = EmptyEnvCfg()
        env_cfg.xr = XrCfg(anchor_pos=(1, 2, 3), anchor_rot=(0, 1, 0, 0))

        # Create a new stage.
        omni.usd.get_context().new_stage()
        # Create environment.
        env = ManagerBasedEnv(cfg=env_cfg)

        device = OpenXRDevice(env_cfg.xr)

        # Check that the xr anchor prim is created with the correct pose.
        xr_anchor_prim = XFormPrim("/XRAnchor")
        self.assertTrue(xr_anchor_prim.is_valid())
        position, orientation = xr_anchor_prim.get_world_poses()
        np.testing.assert_almost_equal(position.tolist(), [[1, 2, 3]])
        np.testing.assert_almost_equal(orientation.tolist(), [[0, 1, 0, 0]])

        # Check that xr anchor mode and custom anchor are set correctly.
        self.assertEqual(carb.settings.get_settings().get("/persistent/xr/profile/ar/anchorMode"), "custom anchor")
        self.assertEqual(carb.settings.get_settings().get("/xrstage/profile/ar/customAnchor"), "/XRAnchor")

        device.reset()
        env.close()

    def test_xr_anchor_default(self):
        env_cfg = EmptyEnvCfg()

        # Create a new stage.
        omni.usd.get_context().new_stage()
        # Create environment.
        env = ManagerBasedEnv(cfg=env_cfg)

        device = OpenXRDevice(None)

        # Check that the xr anchor prim is created with the correct default pose.
        xr_anchor_prim = XFormPrim("/XRAnchor")
        self.assertTrue(xr_anchor_prim.is_valid())
        position, orientation = xr_anchor_prim.get_world_poses()
        np.testing.assert_almost_equal(position.tolist(), [[0, 0, 0]])
        np.testing.assert_almost_equal(orientation.tolist(), [[1, 0, 0, 0]])

        # Check that xr anchor mode and custom anchor are set correctly.
        self.assertEqual(carb.settings.get_settings().get("/persistent/xr/profile/ar/anchorMode"), "custom anchor")
        self.assertEqual(carb.settings.get_settings().get("/xrstage/profile/ar/customAnchor"), "/XRAnchor")

        device.reset()
        env.close()

    def test_xr_anchor_multiple_devices(self):
        env_cfg = EmptyEnvCfg()

        # Create a new stage.
        omni.usd.get_context().new_stage()
        # Create environment.
        env = ManagerBasedEnv(cfg=env_cfg)

        device_1 = OpenXRDevice(None)
        device_2 = OpenXRDevice(None)

        # Check that the xr anchor prim is created with the correct default pose.
        xr_anchor_prim = XFormPrim("/XRAnchor")
        self.assertTrue(xr_anchor_prim.is_valid())
        position, orientation = xr_anchor_prim.get_world_poses()
        np.testing.assert_almost_equal(position.tolist(), [[0, 0, 0]])
        np.testing.assert_almost_equal(orientation.tolist(), [[1, 0, 0, 0]])

        # Check that xr anchor mode and custom anchor are set correctly.
        self.assertEqual(carb.settings.get_settings().get("/persistent/xr/profile/ar/anchorMode"), "custom anchor")
        self.assertEqual(carb.settings.get_settings().get("/xrstage/profile/ar/customAnchor"), "/XRAnchor")

        device_1.reset()
        device_2.reset()
        env.close()
