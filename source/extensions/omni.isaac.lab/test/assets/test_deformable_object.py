# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# Can set this to False to see the GUI for debugging
# This will also add lights to the scene
HEADLESS = True

# launch omniverse app
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes
import torch
import unittest

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg
from omni.isaac.lab.sim import build_simulation_context


def generate_cubes_scene(
    num_cubes: int = 1, height=1.0, has_api: bool = True, kinematic_enabled: bool = False, device: str = "cuda:0"
) -> DeformableObject:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        has_api: Whether the cubes have a deformable body API on them.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        The deformable object representing the cubes.

    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    # Resolve spawn configuration
    if has_api:
        spawn_cfg = sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        # since no deformable body properties defined, this is just a static collider
        spawn_cfg = sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    # Create deformable object
    cube_object_cfg = DeformableObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn_cfg,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = DeformableObject(cfg=cube_object_cfg)

    return cube_object


class TestDeformableObject(unittest.TestCase):
    """Test for deformable object class."""

    """
    Tests
    """

    def test_initialization(self):
        """Test initialization for prim with deformable body API at the provided prim path."""
        for num_cubes in (1, 2):
            with self.subTest(num_cubes=num_cubes):
                with build_simulation_context(auto_add_lighting=True) as sim:
                    # Generate cubes scene
                    cube_object = generate_cubes_scene(num_cubes=num_cubes)

                    # Check that boundedness of deformable object is correct
                    self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                    # Play sim
                    sim.reset()

                    # Check that boundedness of deformable object is correct
                    self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                    # Check if object is initialized
                    self.assertTrue(cube_object.is_initialized)

                    # Check buffers that exists and have correct shapes
                    self.assertEqual(
                        cube_object.data.nodal_state_w.shape, (num_cubes, cube_object.max_sim_mesh_vertices_per_body, 6)
                    )
                    self.assertEqual(cube_object.data.root_pos_w.shape, (num_cubes, 3))
                    self.assertEqual(cube_object.data.root_vel_w.shape, (num_cubes, 3))

                    # Simulate physics
                    for _ in range(2):
                        # perform rendering
                        sim.step()
                        # update object
                        cube_object.update(sim.cfg.dt)

    def test_initialization_on_device_cpu(self):
        """Test that initialization fails with deformable body API on the CPU."""
        with build_simulation_context(device="cpu", auto_add_lighting=True) as sim:
            # Generate cubes scene
            cube_object = generate_cubes_scene(num_cubes=5, device="cpu")

            # Check that boundedness of deformable object is correct
            self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

            # Play sim
            sim.reset()

            # Check if object is initialized
            self.assertFalse(cube_object.is_initialized)

    def test_initialization_with_kinematic_enabled(self):
        """Test that initialization for prim with kinematic flag enabled."""
        for num_cubes in (1, 2):
            with self.subTest(num_cubes=num_cubes):
                with build_simulation_context(auto_add_lighting=True) as sim:
                    # Generate cubes scene
                    cube_object = generate_cubes_scene(num_cubes=num_cubes, kinematic_enabled=True)

                    # Check that boundedness of deformable object is correct
                    self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                    # Play sim
                    sim.reset()

                    # Check if object is initialized
                    self.assertTrue(cube_object.is_initialized)

                    # Check buffers that exists and have correct shapes
                    self.assertEqual(cube_object.data.root_pos_w.shape, (num_cubes, 3))
                    self.assertEqual(cube_object.data.root_vel_w.shape, (num_cubes, 3))

                    # Simulate physics
                    for _ in range(2):
                        # perform rendering
                        sim.step()
                        # update object
                        cube_object.update(sim.cfg.dt)
                        # check that the object is kinematic
                        default_nodal_state_w = cube_object.data.default_nodal_state_w.clone()
                        torch.testing.assert_close(cube_object.data.nodal_state_w, default_nodal_state_w)

    def test_initialization_with_no_deformable_body(self):
        """Test that initialization fails when no deformable body is found at the provided prim path."""
        for num_cubes in (1, 2):
            with self.subTest(num_cubes=num_cubes):
                with build_simulation_context(auto_add_lighting=True) as sim:
                    # Generate cubes scene
                    cube_object = generate_cubes_scene(num_cubes=num_cubes, has_api=False)

                    # Check that boundedness of deformable object is correct
                    self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                    # Play sim
                    sim.reset()

                    # Check if object is initialized
                    self.assertFalse(cube_object.is_initialized)

    def test_set_deformable_object_state(self):
        """Test setting the state of the deformable object.

        In this test, we set the state of the deformable object to a random state and check
        that the object is in that state after simulation. We set gravity to zero as
        we don't want any external forces acting on the object to ensure state remains static.
        """
        for num_cubes in (1, 2):
            with self.subTest(num_cubes=num_cubes):
                # Turn off gravity for this test as we don't want any external forces acting on the object
                # to ensure state remains static
                with build_simulation_context(gravity_enabled=False, auto_add_lighting=True) as sim:
                    # Generate cubes scene
                    cube_object = generate_cubes_scene(num_cubes=num_cubes)

                    # Play the simulator
                    sim.reset()

                    state_types = ["nodal_pos_w", "nodal_vel_w"]

                    # Set each state type individually as they are dependent on each other
                    for state_type_to_randomize in state_types:
                        state_dict = {
                            "nodal_pos_w": torch.zeros_like(cube_object.data.nodal_pos_w),
                            "nodal_vel_w": torch.zeros_like(cube_object.data.nodal_vel_w),
                        }

                        # Now we are ready!
                        for _ in range(5):
                            # reset object
                            cube_object.reset()

                            # Set random state
                            state_dict[state_type_to_randomize] = torch.randn(
                                num_cubes, cube_object.max_sim_mesh_vertices_per_body, 3, device=sim.device
                            )

                            # perform simulation
                            for _ in range(5):
                                nodal_state = torch.cat(
                                    [
                                        state_dict["nodal_pos_w"],
                                        state_dict["nodal_vel_w"],
                                    ],
                                    dim=-1,
                                )
                                # reset nodal state
                                cube_object.write_nodal_state_to_sim(nodal_state)

                                # perform step
                                sim.step()

                                # assert that set node quantities are equal to the ones set in the state_dict
                                for key, expected_value in state_dict.items():
                                    value = getattr(cube_object.data, key)
                                    torch.testing.assert_close(value, expected_value, rtol=1e-5, atol=1e-5)

                                cube_object.update(sim.cfg.dt)


if __name__ == "__main__":
    run_tests()
