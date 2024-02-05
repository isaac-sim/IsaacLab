from __future__ import annotations

"""Launch Isaac Sim Simulator firtst"""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app 
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Reset everything follows"""

import ctypes
import torch
import traceback
import unittest

import carb
import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets import Articulation, ArticulationCfg
from omni.isaac.orbit_assets import ORBIT_ASSETS_DATA_DIR

##
# Pre-defined confgis
##
from omni.isaac.orbit_assets import ANDROID_CFG, CASSIE_CFG  # isort: skip


class TestArticulation(unittest.TestCase):
    """Test class for Articulation."""

    def setUp(self) -> None:
        """Create a blank new stage for each test"""
        # Create a new stage
        stage_utils.create_new_stage()
        #Simulation time-step
        self.dt = 0.005
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device="cuda:0")
        self.sim = sim_utils.SimulationContext(sim_cfg)

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # clear the stage
        self.sim.clear_instance()

    """
    Tests
    """

    def test_initialization_floating_base_non_root(self):