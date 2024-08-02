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
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import build_simulation_context
from omni.isaac.lab.sim.spawners import materials
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import default_orientation, random_orientation


def generate_cubes_scene(
    num_cubes: int = 1, height=1.0, has_api: bool = True, kinematic_enabled: bool = False, device: str = "cuda:0"
) -> tuple[RigidObject, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        has_api: Whether the cubes have a rigid body API on them.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        A tuple containing the rigid object representing the cubes and the origins of the cubes.

    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    # Resolve spawn configuration
    if has_api:
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    # Create rigid object
    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = RigidObject(cfg=cube_object_cfg)

    return cube_object, origins


class TestRigidObject(unittest.TestCase):
    """Test for rigid object class."""

    """
    Tests
    """

    def test_initialization(self):
        """Test initialization for prim with rigid body API at the provided prim path."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                        # Generate cubes scene
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

                        # Check that boundedness of rigid object is correct
                        self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                        # Play sim
                        sim.reset()

                        # Check if object is initialized
                        self.assertTrue(cube_object.is_initialized)
                        self.assertEqual(len(cube_object.body_names), 1)

                        # Check buffers that exists and have correct shapes
                        self.assertEqual(cube_object.data.root_pos_w.shape, (num_cubes, 3))
                        self.assertEqual(cube_object.data.root_quat_w.shape, (num_cubes, 4))

                        # Simulate physics
                        for _ in range(2):
                            # perform rendering
                            sim.step()
                            # update object
                            cube_object.update(sim.cfg.dt)

    def test_initialization_with_kinematic_enabled(self):
        """Test that initialization for prim with kinematic flag enabled."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                        # Generate cubes scene
                        cube_object, origins = generate_cubes_scene(
                            num_cubes=num_cubes, kinematic_enabled=True, device=device
                        )

                        # Check that boundedness of rigid object is correct
                        self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                        # Play sim
                        sim.reset()

                        # Check if object is initialized
                        self.assertTrue(cube_object.is_initialized)
                        self.assertEqual(len(cube_object.body_names), 1)

                        # Check buffers that exists and have correct shapes
                        self.assertEqual(cube_object.data.root_pos_w.shape, (num_cubes, 3))
                        self.assertEqual(cube_object.data.root_quat_w.shape, (num_cubes, 4))

                        # Simulate physics
                        for _ in range(2):
                            # perform rendering
                            sim.step()
                            # update object
                            cube_object.update(sim.cfg.dt)
                            # check that the object is kinematic
                            default_root_state = cube_object.data.default_root_state.clone()
                            default_root_state[:, :3] += origins
                            torch.testing.assert_close(cube_object.data.root_state_w, default_root_state)

    def test_initialization_with_no_rigid_body(self):
        """Test that initialization fails when no rigid body is found at the provided prim path."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                        # Generate cubes scene
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, has_api=False, device=device)

                        # Check that boundedness of rigid object is correct
                        self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                        # Play sim
                        sim.reset()

                        # Check if object is initialized
                        self.assertFalse(cube_object.is_initialized)

    def test_external_force_on_single_body(self):
        """Test application of external force on the base of the object.

        In this test, we apply a force equal to the weight of an object on the base of
        one of the objects. We check that the object does not move. For the other object,
        we do not apply any force and check that it falls down.
        """
        for num_cubes in (2, 4):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    # Generate cubes scene
                    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)

                        # Play the simulator
                        sim.reset()

                        # Find bodies to apply the force
                        body_ids, body_names = cube_object.find_bodies(".*")

                        # Sample a force equal to the weight of the object
                        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
                        # Every 2nd cube should have a force applied to it
                        external_wrench_b[0::2, :, 2] = 9.81 * cube_object.root_physx_view.get_masses()[0]

                        # Now we are ready!
                        for _ in range(5):
                            # reset root state
                            root_state = cube_object.data.default_root_state.clone()

                            # need to shift the position of the cubes otherwise they will be on top of each other
                            root_state[:, :3] = origins
                            cube_object.write_root_state_to_sim(root_state)

                            # reset object
                            cube_object.reset()

                            # apply force
                            cube_object.set_external_force_and_torque(
                                external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
                            )
                            # perform simulation
                            for _ in range(5):
                                # apply action to the object
                                cube_object.write_data_to_sim()

                                # perform step
                                sim.step()

                                # update buffers
                                cube_object.update(sim.cfg.dt)

                            # First object should still be at the same Z position (1.0)
                            torch.testing.assert_close(
                                cube_object.data.root_pos_w[0::2, 2], torch.ones(num_cubes // 2, device=sim.device)
                            )
                            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
                            self.assertTrue(torch.all(cube_object.data.root_pos_w[1::2, 2] < 1.0))

    def test_set_rigid_object_state(self):
        """Test setting the state of the rigid object.

        In this test, we set the state of the rigid object to a random state and check
        that the object is in that state after simulation. We set gravity to zero as
        we don't want any external forces acting on the object to ensure state remains static.
        """
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    # Turn off gravity for this test as we don't want any external forces acting on the object
                    # to ensure state remains static
                    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
                        # Generate cubes scene
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

                        # Play the simulator
                        sim.reset()

                        state_types = ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"]

                        # Set each state type individually as they are dependent on each other
                        for state_type_to_randomize in state_types:
                            state_dict = {
                                "root_pos_w": torch.zeros_like(cube_object.data.root_pos_w, device=sim.device),
                                "root_quat_w": default_orientation(num=num_cubes, device=sim.device),
                                "root_lin_vel_w": torch.zeros_like(cube_object.data.root_lin_vel_w, device=sim.device),
                                "root_ang_vel_w": torch.zeros_like(cube_object.data.root_ang_vel_w, device=sim.device),
                            }

                            # Now we are ready!
                            for _ in range(5):
                                # reset object
                                cube_object.reset()

                                # Set random state
                                if state_type_to_randomize == "root_quat_w":
                                    state_dict[state_type_to_randomize] = random_orientation(
                                        num=num_cubes, device=sim.device
                                    )
                                else:
                                    state_dict[state_type_to_randomize] = torch.randn(num_cubes, 3, device=sim.device)

                                # perform simulation
                                for _ in range(5):
                                    root_state = torch.cat(
                                        [
                                            state_dict["root_pos_w"],
                                            state_dict["root_quat_w"],
                                            state_dict["root_lin_vel_w"],
                                            state_dict["root_ang_vel_w"],
                                        ],
                                        dim=-1,
                                    )
                                    # reset root state
                                    cube_object.write_root_state_to_sim(root_state=root_state)

                                    sim.step()

                                    # assert that set root quantities are equal to the ones set in the state_dict
                                    for key, expected_value in state_dict.items():
                                        value = getattr(cube_object.data, key)
                                        torch.testing.assert_close(value, expected_value, rtol=1e-5, atol=1e-5)

                                    cube_object.update(sim.cfg.dt)

    def test_reset_rigid_object(self):
        """Test resetting the state of the rigid object."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(device=device, gravity_enabled=True, auto_add_lighting=True) as sim:
                        # Generate cubes scene
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

                        # Play the simulator
                        sim.reset()

                        for i in range(5):
                            # perform rendering
                            sim.step()

                            # update object
                            cube_object.update(sim.cfg.dt)

                            # Move the object to a random position
                            root_state = cube_object.data.default_root_state.clone()
                            root_state[:, :3] = torch.randn(num_cubes, 3, device=sim.device)

                            # Random orientation
                            root_state[:, 3:7] = random_orientation(num=num_cubes, device=sim.device)
                            cube_object.write_root_state_to_sim(root_state)

                            if i % 2 == 0:
                                # reset object
                                cube_object.reset()

                                # Reset should zero external forces and torques
                                self.assertFalse(cube_object.has_external_wrench)
                                self.assertEqual(torch.count_nonzero(cube_object._external_force_b), 0)
                                self.assertEqual(torch.count_nonzero(cube_object._external_torque_b), 0)

    def test_rigid_body_set_material_properties(self):
        """Test getting and setting material properties of rigid object."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(
                        device=device, gravity_enabled=True, add_ground_plane=True, auto_add_lighting=True
                    ) as sim:
                        # Generate cubes scene
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

                        # Play sim
                        sim.reset()

                        # Set material properties
                        static_friction = torch.FloatTensor(num_cubes, 1).uniform_(0.4, 0.8)
                        dynamic_friction = torch.FloatTensor(num_cubes, 1).uniform_(0.4, 0.8)
                        restitution = torch.FloatTensor(num_cubes, 1).uniform_(0.0, 0.2)

                        materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

                        indices = torch.tensor(range(num_cubes), dtype=torch.int)
                        # Add friction to cube
                        cube_object.root_physx_view.set_material_properties(materials, indices)

                        # Simulate physics
                        # perform rendering
                        sim.step()
                        # update object
                        cube_object.update(sim.cfg.dt)

                        # Get material properties
                        materials_to_check = cube_object.root_physx_view.get_material_properties()

                        # Check if material properties are set correctly
                        torch.testing.assert_close(materials_to_check.reshape(num_cubes, 3), materials)

    def test_rigid_body_no_friction(self):
        """Test that a rigid object with no friction will maintain it's velocity when sliding across a plane."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                        # Generate cubes scene
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)

                        # Create ground plane with no friction
                        cfg = sim_utils.GroundPlaneCfg(
                            physics_material=materials.RigidBodyMaterialCfg(
                                static_friction=0.0,
                                dynamic_friction=0.0,
                                restitution=0.0,
                            )
                        )
                        cfg.func("/World/GroundPlane", cfg)

                        # Play sim
                        sim.reset()

                        # Set material friction properties to be all zero
                        static_friction = torch.zeros(num_cubes, 1)
                        dynamic_friction = torch.zeros(num_cubes, 1)
                        restitution = torch.FloatTensor(num_cubes, 1).uniform_(0.0, 0.2)

                        cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)
                        indices = torch.tensor(range(num_cubes), dtype=torch.int)

                        cube_object.root_physx_view.set_material_properties(cube_object_materials, indices)

                        # Set initial velocity
                        # Initial velocity in X to get the block moving
                        initial_velocity = torch.zeros((num_cubes, 6), device=sim.cfg.device)
                        initial_velocity[:, 0] = 0.1

                        cube_object.write_root_velocity_to_sim(initial_velocity)

                        # Simulate physics
                        for _ in range(5):
                            # perform rendering
                            sim.step()
                            # update object
                            cube_object.update(sim.cfg.dt)

                            # Non-deterministic when on GPU, so we use different tolerances
                            if device == "cuda:0":
                                tolerance = 1e-2
                            else:
                                tolerance = 1e-5

                            torch.testing.assert_close(
                                cube_object.data.root_lin_vel_w, initial_velocity[:, :3], rtol=1e-5, atol=tolerance
                            )

    # def test_rigid_body_with_static_friction(self):
    #     """Test that static friction applied to rigid object works as expected.

    #     This test works by applying a force to the object and checking if the object moves or not based on the
    #     mu (coefficient of static friction) value set for the object. We set the static friction to be non-zero and
    #     apply a force to the object. When the force applied is below mu, the object should not move. When the force
    #     applied is above mu, the object should move.
    #     """
    #     for num_cubes in (1, 2):
    #         for device in ("cuda:0", "cpu"):
    #             with self.subTest(num_cubes=num_cubes, device=device):
    #                 with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
    #                     cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.03125, device=device)

    #                     # Create ground plane with no friction
    #                     cfg = sim_utils.GroundPlaneCfg(
    #                         physics_material=materials.RigidBodyMaterialCfg(
    #                             static_friction=0.0,
    #                             dynamic_friction=0.0,
    #                         )
    #                     )
    #                     cfg.func("/World/GroundPlane", cfg)

    #                     # Play sim
    #                     sim.reset()

    #                     # Set static friction to be non-zero
    #                     static_friction_coefficient = 0.5
    #                     static_friction = torch.Tensor([[static_friction_coefficient]] * num_cubes)
    #                     dynamic_friction = torch.zeros(num_cubes, 1)
    #                     restitution = torch.FloatTensor(num_cubes, 1).uniform_(0.0, 0.2)

    #                     cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

    #                     indices = torch.tensor(range(num_cubes), dtype=torch.int)

    #                     # Add friction to cube
    #                     cube_object.root_physx_view.set_material_properties(cube_object_materials, indices)

    #                     # 2 cases: force applied is below and above mu
    #                     # below mu: block should not move as the force applied is <= mu
    #                     # above mu: block should move as the force applied is > mu
    #                     for force in "below_mu", "above_mu":
    #                         with self.subTest(force=force):
    #                             external_wrench_b = torch.zeros((num_cubes, 1, 6), device=sim.device)

    #                             if force == "below_mu":
    #                                 external_wrench_b[:, 0, 0] = static_friction_coefficient * 0.999
    #                             else:
    #                                 external_wrench_b[:, 0, 0] = static_friction_coefficient * 1.001

    #                             cube_object.set_external_force_and_torque(
    #                                 external_wrench_b[..., :3],
    #                                 external_wrench_b[..., 3:],
    #                             )

    #                             # Get root state
    #                             initial_root_state = cube_object.data.root_state_w

    #                             # Simulate physics
    #                             for _ in range(10):
    #                                 # perform rendering
    #                                 sim.step()
    #                                 # update object
    #                                 cube_object.update(sim.cfg.dt)

    #                                 if force == "below_mu":
    #                                     # Assert that the block has not moved
    #                                     torch.testing.assert_close(
    #                                         cube_object.data.root_state_w, initial_root_state, rtol=1e-5, atol=1e-5
    #                                     )
    #                                 else:
    #                                     torch.testing.assert_close(
    #                                         cube_object.data.root_state_w, initial_root_state, rtol=1e-5, atol=1e-5
    #                                     )

    # def test_rigid_body_with_restitution(self):
    #     """Test that restitution when applied to rigid object works as expected.

    #     This test works by dropping a block from a height and checking if the block bounces or not based on the
    #     restitution value set for the object. We set the restitution to be non-zero and drop the block from a height.
    #     When the restitution is 0, the block should not bounce. When the restitution is 1, the block should bounce
    #     with the same energy. When the restitution is between 0 and 1, the block should bounce with less energy.

    #     """
    #     for num_cubes in (1, 2):
    #         for device in ("cuda:0", "cpu"):
    #             with self.subTest(num_cubes=num_cubes, device=device):
    #                 with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
    #                     cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)

    #                     # Create ground plane such that has a restitution of 1.0 (perfectly elastic collision)
    #                     cfg = sim_utils.GroundPlaneCfg(
    #                         physics_material=materials.RigidBodyMaterialCfg(
    #                             restitution=1.0,
    #                         )
    #                     )
    #                     cfg.func("/World/GroundPlane", cfg)

    #                     indices = torch.tensor(range(num_cubes), dtype=torch.int)

    #                     # Play sim
    #                     sim.reset()

    #                     # 3 cases: inelastic, partially elastic, elastic
    #                     # inelastic: resitution = 0, block should not bounce
    #                     # partially elastic: 0 <= restitution <= 1, block should bounce with less energy
    #                     # elastic: restitution = 1, block should bounce with same energy
    #                     for expected_collision_type in "inelastic", "partially_elastic", "elastic":
    #                         root_state = torch.zeros(1, 13, device=sim.device)
    #                         root_state[0, 3] = 1.0  # To make orientation a quaternion
    #                         root_state[0, 2] = 0.1  # Set an initial drop height
    #                         root_state[0, 9] = -1.0  # Set an initial downward velocity

    #                         cube_object.write_root_state_to_sim(root_state=root_state)

    #                         prev_z_velocity = 0.0
    #                         curr_z_velocity = 0.0

    #                         with self.subTest(expected_collision_type=expected_collision_type):
    #                             # cube_object.reset()
    #                             # Set static friction to be non-zero
    #                             if expected_collision_type == "inelastic":
    #                                 restitution_coefficient = 0.0
    #                             elif expected_collision_type == "partially_elastic":
    #                                 restitution_coefficient = 0.5
    #                             else:
    #                                 restitution_coefficient = 1.0

    #                             restitution = 0.5
    #                             static_friction = torch.zeros(num_cubes, 1)
    #                             dynamic_friction = torch.zeros(num_cubes, 1)
    #                             restitution = torch.Tensor([[restitution_coefficient]] * num_cubes)

    #                             cube_object_materials = torch.cat(
    #                                 [static_friction, dynamic_friction, restitution], dim=-1
    #                             )

    #                             # Add friction to cube
    #                             cube_object.root_physx_view.set_material_properties(cube_object_materials, indices)

    #                             curr_z_velocity = cube_object.data.root_lin_vel_w[:, 2]

    #                             while torch.all(curr_z_velocity <= 0.0):
    #                                 # Simulate physics
    #                                 curr_z_velocity = cube_object.data.root_lin_vel_w[:, 2]

    #                                 # perform rendering
    #                                 sim.step()

    #                                 # update object
    #                                 cube_object.update(sim.cfg.dt)
    #                                 if torch.all(curr_z_velocity <= 0.0):
    #                                     # Still in the air
    #                                     prev_z_velocity = curr_z_velocity

    #                             # We have made contact with the ground and can verify expected collision type
    #                             # based on how velocity has changed after the collision
    #                             if expected_collision_type == "inelastic":
    #                                 # Assert that the block has lost most energy by checking that the z velocity is < 1/2 previous
    #                                 # velocity. This is because the floor's resitution means it will bounce back an object that itself
    #                                 # has restitution set to 0.0
    #                                 self.assertTrue(torch.all(torch.le(curr_z_velocity / 2, abs(prev_z_velocity))))
    #                             elif expected_collision_type == "partially_elastic":
    #                                 # Assert that the block has lost some energy by checking that the z velocity is less
    #                                 self.assertTrue(torch.all(torch.le(abs(curr_z_velocity), abs(prev_z_velocity))))
    #                             elif expected_collision_type == "elastic":
    #                                 # Assert that the block has not lost any energy by checking that the z velocity is the same
    #                                 torch.testing.assert_close(abs(curr_z_velocity), abs(prev_z_velocity))

    def test_rigid_body_set_mass(self):
        """Test getting and setting mass of rigid object."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(
                        device=device, gravity_enabled=False, add_ground_plane=True, auto_add_lighting=True
                    ) as sim:
                        # Create a scene with random cubes
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)

                        # Play sim
                        sim.reset()

                        # Get masses before increasing
                        original_masses = cube_object.root_physx_view.get_masses()

                        self.assertEqual(original_masses.shape, (num_cubes, 1))

                        # Randomize mass of the object
                        masses = original_masses + torch.FloatTensor(num_cubes, 1).uniform_(4, 8)

                        indices = torch.tensor(range(num_cubes), dtype=torch.int)

                        # Add friction to cube
                        cube_object.root_physx_view.set_masses(masses, indices)

                        torch.testing.assert_close(cube_object.root_physx_view.get_masses(), masses)

                        # Simulate physics
                        # perform rendering
                        sim.step()
                        # update object
                        cube_object.update(sim.cfg.dt)

                        masses_to_check = cube_object.root_physx_view.get_masses()

                        # Check if mass is set correctly
                        torch.testing.assert_close(masses, masses_to_check)

    def test_gravity_vec_w(self):
        """Test that gravity vector direction is set correctly for the rigid object."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                for gravity_enabled in [True, False]:
                    with self.subTest(num_cubes=num_cubes, device=device, gravity_enabled=gravity_enabled):
                        with build_simulation_context(device=device, gravity_enabled=gravity_enabled) as sim:
                            # Create a scene with random cubes
                            cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

                            # Obtain gravity direction
                            if gravity_enabled:
                                gravity_dir = (0.0, 0.0, -1.0)
                            else:
                                gravity_dir = (0.0, 0.0, 0.0)

                            # Play sim
                            sim.reset()

                            # Check that gravity is set correctly
                            self.assertEqual(cube_object.data.GRAVITY_VEC_W[0, 0], gravity_dir[0])
                            self.assertEqual(cube_object.data.GRAVITY_VEC_W[0, 1], gravity_dir[1])
                            self.assertEqual(cube_object.data.GRAVITY_VEC_W[0, 2], gravity_dir[2])

                            # Simulate physics
                            for _ in range(2):
                                # perform rendering
                                sim.step()
                                # update object
                                cube_object.update(sim.cfg.dt)

                                # Expected gravity value is the acceleration of the body
                                gravity = torch.zeros(num_cubes, 1, 6, device=device)
                                if gravity_enabled:
                                    gravity[:, :, 2] = -9.81
                                # Check the body accelerations are correct
                                torch.testing.assert_close(cube_object.data.body_acc_w, gravity)


if __name__ == "__main__":
    run_tests()
