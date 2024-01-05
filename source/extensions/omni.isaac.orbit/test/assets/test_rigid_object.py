# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes
import torch
import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR


class TestRigidObject(unittest.TestCase):
    """Test for rigid object class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device="cuda:0")
        self.sim = sim_utils.SimulationContext(sim_cfg)

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # clear the stage
        self.sim.clear_instance()

    """
    Tests
    """

    def test_initialization(self):
        """Test initialization for with rigid body API at the provided prim path."""
        # Create rigid object
        cube_object_cfg = RigidObjectCfg(
            prim_path="/World/Object",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        cube_object = RigidObject(cfg=cube_object_cfg)

        # Check that boundedness of articulation is correct
        self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

        # Play sim
        self.sim.reset()
        # Check if object is initialized
        self.assertTrue(cube_object._is_initialized)
        self.assertEqual(len(cube_object.body_names), 1)
        # Check buffers that exists and have correct shapes
        self.assertTrue(cube_object.data.root_pos_w.shape == (1, 3))
        self.assertTrue(cube_object.data.root_quat_w.shape == (1, 4))

        # Simulate physics
        for _ in range(20):
            # perform rendering
            self.sim.step()
            # update object
            cube_object.update(self.dt)

    def test_external_force_on_single_body(self):
        """Test application of external force on the base of the object.

        In this test, we apply a force equal to the weight of the object on the base of
        one of the objects. We check that the object does not move. For the other object,
        we do not apply any force and check that it falls down.
        """
        # Create parent prims
        prim_utils.create_prim("/World/Table_1", "Xform", translation=(0.0, -1.0, 0.0))
        prim_utils.create_prim("/World/Table_2", "Xform", translation=(0.0, 1.0, 0.0))

        # Create rigid object
        cube_object_cfg = RigidObjectCfg(
            prim_path="/World/Table_.*/Object",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        # create handles for the objects
        cube_object = RigidObject(cfg=cube_object_cfg)

        # Play the simulator
        self.sim.reset()

        # Find bodies to apply the force
        body_ids, _ = cube_object.find_bodies(".*")

        # Sample a large force
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=self.sim.device)
        external_wrench_b[0, 0, 2] = 9.81 * cube_object.root_physx_view.get_masses()[0]

        # Now we are ready!
        for _ in range(5):
            # reset root state
            root_state = cube_object.data.default_root_state.clone()
            root_state[0, :2] = torch.tensor([0.0, -1.0], device=self.sim.device)
            root_state[1, :2] = torch.tensor([0.0, 1.0], device=self.sim.device)
            cube_object.write_root_state_to_sim(root_state)
            # reset object
            cube_object.reset()
            # apply force
            cube_object.set_external_force_and_torque(
                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
            )
            # perform simulation
            for _ in range(50):
                # apply action to the object
                cube_object.write_data_to_sim()
                # perform step
                self.sim.step()
                # update buffers
                cube_object.update(self.dt)
            # check condition that the objects have fallen down
            self.assertLess(abs(cube_object.data.root_pos_w[0, 2].item()), 1e-3)
            self.assertLess(cube_object.data.root_pos_w[1, 2].item(), -1.0)


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
