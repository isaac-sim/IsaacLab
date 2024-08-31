# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import unittest

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.cloner import GridCloner

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.controllers import OperationSpaceController, OperationSpaceControllerCfg
from omni.isaac.lab.utils.math import compute_pose_error, subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

class TestOperationSpaceController(unittest.TestCase):
    """Test fixture for checking that differential IK controller tracks commands properly."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Wait for spawning
        stage_utils.create_new_stage()
        # Constants
        self.num_envs = 128
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=0.01)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # TODO: Remove this once we have a better way to handle this.
        self.sim._app_control_on_stop_handle = None

        # Create a ground plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/GroundPlane", cfg)

        # Create interface to clone the scene
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        self.env_prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        # create source prim
        prim_utils.define_prim(self.env_prim_paths[0], "Xform")
        # clone the env xform
        self.env_origins = cloner.clone(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            replicate_physics=True,
        )

        # Define goals for the arm [xyz, quat_wxyz]
        ee_goals_set = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
        self.ee_pose_b_des_set = torch.tensor(ee_goals_set, device=self.sim.device)

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Test fixtures.
    """

    def test_franka_position_abs(self):
        """Test absolute position control."""
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        robot = Articulation(cfg=robot_cfg)

        ctrl_cfg = OperationSpaceControllerCfg()
        controller = OperationSpaceController(ctrl_cfg,....)

    def test_franka_pose_abs(self):
        """Test absolute pose control."""

    def test_franka_force_abs(self):
        """test absolute force control."""

    def test_franka_fixed(self):
        """Tests operational space controller for franka using fixed impedance."""
        
    def test_franka_variable_kp(self):
        """Tests operational space controller for franka using variable stiffness impedance."""

    def test_franka_variable(self):
        """Tests operational space controller for franka using variable stiffness and damping ratio impedance."""


    def test_franka_gravity_compensation(self):
        """Tests operational space control with gravity compensation."""
    
    def test_franka_inertial_compensation(self):
        """Tests operational space control with inertial compensation."""
    
    def test_franka_control_hybrid_motion_force(self):
        """Tests operational space control for hybrid motion and force control."""

    """
    Helper functions
    """

    def _run_op_space_controller(self):
