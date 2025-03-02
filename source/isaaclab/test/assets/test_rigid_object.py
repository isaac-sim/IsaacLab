# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

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
from typing import Literal

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim.spawners import materials
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import default_orientation, quat_mul, quat_rotate_inverse, random_orientation


def generate_cubes_scene(
    num_cubes: int = 1,
    height=1.0,
    api: Literal["none", "rigid_body", "articulation_root"] = "rigid_body",
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> tuple[RigidObject, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        api: The type of API that the cubes should have.
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
    if api == "none":
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    elif api == "rigid_body":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    elif api == "articulation_root":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Tests/RigidObject/Cube/dex_cube_instanceable_with_articulation_root.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        raise ValueError(f"Unknown api: {api}")

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
                        sim._app_control_on_stop_handle = None
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
                        self.assertEqual(cube_object.data.default_mass.shape, (num_cubes, 1))
                        self.assertEqual(cube_object.data.default_inertia.shape, (num_cubes, 9))

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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
                        # Generate cubes scene
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, api="none", device=device)

                        # Check that boundedness of rigid object is correct
                        self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                        # Play sim
                        sim.reset()

                        # Check if object is initialized
                        self.assertFalse(cube_object.is_initialized)

    def test_initialization_with_articulation_root(self):
        """Test that initialization fails when an articulation root is found at the provided prim path."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                        sim._app_control_on_stop_handle = None
                        # Generate cubes scene
                        cube_object, _ = generate_cubes_scene(
                            num_cubes=num_cubes, api="articulation_root", device=device
                        )

                        # Check that boundedness of rigid object is correct
                        self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                        # Play sim
                        sim.reset()

                        # Check if object is initialized
                        self.assertFalse(cube_object.is_initialized)

    def test_external_force_buffer(self):
        """Test if external force buffer correctly updates in the force value is zero case.

        In this test, we apply a non-zero force, then a zero force, then finally a non-zero force
        to an object. We check if the force buffer is properly updated at each step.
        """

        for device in ("cuda:0", "cpu"):
            with self.subTest(num_cubes=1, device=device):
                # Generate cubes scene
                with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                    sim._app_control_on_stop_handle = None
                    cube_object, origins = generate_cubes_scene(num_cubes=1, device=device)

                    # play the simulator
                    sim.reset()

                    # find bodies to apply the force
                    body_ids, body_names = cube_object.find_bodies(".*")

                    # reset object
                    cube_object.reset()

                    # perform simulation
                    for step in range(5):

                        # initiate force tensor
                        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)

                        if step == 0 or step == 3:
                            # set a non-zero force
                            force = 1
                        else:
                            # set a zero force
                            force = 0

                        # set force value
                        external_wrench_b[:, :, 0] = force
                        external_wrench_b[:, :, 3] = force

                        # apply force
                        cube_object.set_external_force_and_torque(
                            external_wrench_b[..., :3], external_wrench_b[..., 3:], body_ids=body_ids
                        )

                        # check if the cube's force and torque buffers are correctly updated
                        self.assertTrue(cube_object._external_force_b[0, 0, 0].item() == force)
                        self.assertTrue(cube_object._external_torque_b[0, 0, 0].item() == force)

                        # apply action to the object
                        cube_object.write_data_to_sim()

                        # perform step
                        sim.step()

                        # update buffers
                        cube_object.update(sim.cfg.dt)

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
                        sim._app_control_on_stop_handle = None
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
                            cube_object.write_root_pose_to_sim(root_state[:, :7])
                            cube_object.write_root_velocity_to_sim(root_state[:, 7:])

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
                        sim._app_control_on_stop_handle = None
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
                                    cube_object.write_root_pose_to_sim(root_state[:, :7])
                                    cube_object.write_root_velocity_to_sim(root_state[:, 7:])

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
                        sim._app_control_on_stop_handle = None
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
                            cube_object.write_root_pose_to_sim(root_state[:, :7])
                            cube_object.write_root_velocity_to_sim(root_state[:, 7:])

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
                        sim._app_control_on_stop_handle = None
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
                        sim._app_control_on_stop_handle = None
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

    def test_rigid_body_with_static_friction(self):
        """Test that static friction applied to rigid object works as expected.

        This test works by applying a force to the object and checking if the object moves or not based on the
        mu (coefficient of static friction) value set for the object. We set the static friction to be non-zero and
        apply a force to the object. When the force applied is below mu, the object should not move. When the force
        applied is above mu, the object should move.
        """
        for num_cubes in (1, 2):
            for device in ("cuda", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(
                        device=device, dt=0.01, add_ground_plane=False, auto_add_lighting=True
                    ) as sim:
                        sim._app_control_on_stop_handle = None
                        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.03125, device=device)

                        # Create ground plane
                        static_friction_coefficient = 0.5
                        cfg = sim_utils.GroundPlaneCfg(
                            physics_material=materials.RigidBodyMaterialCfg(
                                static_friction=static_friction_coefficient,
                                dynamic_friction=static_friction_coefficient,  # This shouldn't be required but is due to a bug in PhysX
                            )
                        )
                        cfg.func("/World/GroundPlane", cfg)

                        # Play sim
                        sim.reset()

                        # Set static friction to be non-zero
                        # Dynamic friction also needs to be zero due to a bug in PhysX
                        static_friction = torch.Tensor([[static_friction_coefficient]] * num_cubes)
                        dynamic_friction = torch.Tensor([[static_friction_coefficient]] * num_cubes)
                        restitution = torch.zeros(num_cubes, 1)

                        cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

                        indices = torch.tensor(range(num_cubes), dtype=torch.int)

                        # Add friction to cube
                        cube_object.root_physx_view.set_material_properties(cube_object_materials, indices)

                        # let everything settle
                        for _ in range(100):
                            sim.step()
                            cube_object.update(sim.cfg.dt)
                        cube_object.write_root_velocity_to_sim(torch.zeros((num_cubes, 6), device=sim.device))
                        cube_mass = cube_object.root_physx_view.get_masses()
                        gravity_magnitude = abs(sim.cfg.gravity[2])
                        # 2 cases: force applied is below and above mu
                        # below mu: block should not move as the force applied is <= mu
                        # above mu: block should move as the force applied is > mu
                        for force in "below_mu", "above_mu":
                            with self.subTest(force=force):
                                # set initial velocity to zero
                                cube_object.write_root_velocity_to_sim(torch.zeros((num_cubes, 6), device=sim.device))

                                external_wrench_b = torch.zeros((num_cubes, 1, 6), device=sim.device)
                                if force == "below_mu":
                                    external_wrench_b[..., 0] = (
                                        static_friction_coefficient * cube_mass * gravity_magnitude * 0.99
                                    )
                                else:
                                    external_wrench_b[..., 0] = (
                                        static_friction_coefficient * cube_mass * gravity_magnitude * 1.01
                                    )

                                cube_object.set_external_force_and_torque(
                                    external_wrench_b[..., :3],
                                    external_wrench_b[..., 3:],
                                )

                                # Get root state
                                initial_root_pos = cube_object.data.root_pos_w.clone()
                                # Simulate physics
                                for _ in range(200):
                                    # apply the wrench
                                    cube_object.write_data_to_sim()
                                    sim.step()
                                    # update object
                                    cube_object.update(sim.cfg.dt)
                                    if force == "below_mu":
                                        # Assert that the block has not moved
                                        torch.testing.assert_close(
                                            cube_object.data.root_pos_w, initial_root_pos, rtol=1e-3, atol=1e-3
                                        )
                                if force == "above_mu":
                                    self.assertTrue(
                                        (cube_object.data.root_state_w[..., 0] - initial_root_pos[..., 0] > 0.02).all()
                                    )

    def test_rigid_body_with_restitution(self):
        """Test that restitution when applied to rigid object works as expected.

        This test works by dropping a block from a height and checking if the block bounces or not based on the
        restitution value set for the object. We set the restitution to be non-zero and drop the block from a height.
        When the restitution is 0, the block should not bounce. When the restitution is between 0 and 1, the block
        should bounce with less energy.
        """
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                for expected_collision_type in "partially_elastic", "inelastic":
                    with self.subTest(
                        expected_collision_type=expected_collision_type, num_cubes=num_cubes, device=device
                    ):
                        with build_simulation_context(
                            device=device, add_ground_plane=False, auto_add_lighting=True
                        ) as sim:
                            sim._app_control_on_stop_handle = None
                            cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)

                            # Set static friction to be non-zero
                            if expected_collision_type == "inelastic":
                                restitution_coefficient = 0.0
                            elif expected_collision_type == "partially_elastic":
                                restitution_coefficient = 0.5

                            # Create ground plane such that has a restitution of 1.0 (perfectly elastic collision)
                            cfg = sim_utils.GroundPlaneCfg(
                                physics_material=materials.RigidBodyMaterialCfg(
                                    restitution=restitution_coefficient,
                                )
                            )
                            cfg.func("/World/GroundPlane", cfg)

                            indices = torch.tensor(range(num_cubes), dtype=torch.int)

                            # Play sim
                            sim.reset()

                            root_state = torch.zeros(num_cubes, 13, device=sim.device)
                            root_state[:, 3] = 1.0  # To make orientation a quaternion
                            for i in range(num_cubes):
                                root_state[i, 1] = 1.0 * i
                            root_state[:, 2] = 1.0  # Set an initial drop height
                            root_state[:, 9] = -1.0  # Set an initial downward velocity

                            cube_object.write_root_pose_to_sim(root_state[:, :7])
                            cube_object.write_root_velocity_to_sim(root_state[:, 7:])

                            static_friction = torch.zeros(num_cubes, 1)
                            dynamic_friction = torch.zeros(num_cubes, 1)
                            restitution = torch.Tensor([[restitution_coefficient]] * num_cubes)

                            cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

                            # Add restitution to cube
                            cube_object.root_physx_view.set_material_properties(cube_object_materials, indices)

                            curr_z_velocity = cube_object.data.root_lin_vel_w[:, 2].clone()

                            for _ in range(100):
                                sim.step()

                                # update object
                                cube_object.update(sim.cfg.dt)
                                curr_z_velocity = cube_object.data.root_lin_vel_w[:, 2].clone()

                                if expected_collision_type == "inelastic":
                                    # assert that the block has not bounced by checking that the z velocity is less than or equal to 0
                                    self.assertTrue((curr_z_velocity <= 0.0).all())

                                if torch.all(curr_z_velocity <= 0.0):
                                    # Still in the air
                                    prev_z_velocity = curr_z_velocity
                                else:
                                    # collision has happened, exit the for loop
                                    break

                            if expected_collision_type == "partially_elastic":
                                # Assert that the block has lost some energy by checking that the z velocity is less
                                self.assertTrue(torch.all(torch.le(abs(curr_z_velocity), abs(prev_z_velocity))))
                                self.assertTrue((curr_z_velocity > 0.0).all())

    def test_rigid_body_set_mass(self):
        """Test getting and setting mass of rigid object."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(
                        device=device, gravity_enabled=False, add_ground_plane=True, auto_add_lighting=True
                    ) as sim:
                        sim._app_control_on_stop_handle = None
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
                            sim._app_control_on_stop_handle = None
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

    def test_body_root_state_properties(self):
        """Test the root_com_state_w, root_link_state_w, body_com_state_w, and body_link_state_w properties."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                for with_offset in [True, False]:
                    with self.subTest(num_cubes=num_cubes, device=device, with_offset=with_offset):
                        with build_simulation_context(
                            device=device, gravity_enabled=False, auto_add_lighting=True
                        ) as sim:
                            sim._app_control_on_stop_handle = None
                            # Create a scene with random cubes
                            cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
                            env_idx = torch.tensor([x for x in range(num_cubes)])

                            # Play sim
                            sim.reset()

                            # Check if cube_object is initialized
                            self.assertTrue(cube_object.is_initialized)

                            # change center of mass offset from link frame
                            if with_offset:
                                offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
                            else:
                                offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

                            com = cube_object.root_physx_view.get_coms()
                            com[..., :3] = offset.to("cpu")
                            cube_object.root_physx_view.set_coms(com, env_idx)

                            # check ceter of mass has been set
                            torch.testing.assert_close(cube_object.root_physx_view.get_coms(), com)

                            # random z spin velocity
                            spin_twist = torch.zeros(6, device=device)
                            spin_twist[5] = torch.randn(1, device=device)

                            # Simulate physics
                            for _ in range(100):
                                # spin the object around Z axis (com)
                                cube_object.write_root_velocity_to_sim(spin_twist.repeat(num_cubes, 1))
                                # perform rendering
                                sim.step()
                                # update object
                                cube_object.update(sim.cfg.dt)

                                # get state properties
                                root_state_w = cube_object.data.root_state_w
                                root_link_state_w = cube_object.data.root_link_state_w
                                root_com_state_w = cube_object.data.root_com_state_w
                                body_state_w = cube_object.data.body_state_w
                                body_link_state_w = cube_object.data.body_link_state_w
                                body_com_state_w = cube_object.data.body_com_state_w

                                # if offset is [0,0,0] all root_state_%_w will match and all body_%_w will match
                                if not with_offset:
                                    torch.testing.assert_close(root_state_w, root_com_state_w)
                                    torch.testing.assert_close(root_state_w, root_link_state_w)
                                    torch.testing.assert_close(body_state_w, body_com_state_w)
                                    torch.testing.assert_close(body_state_w, body_link_state_w)
                                else:
                                    # cubes are spinning around center of mass
                                    # position will not match
                                    # center of mass position will be constant (i.e. spining around com)
                                    torch.testing.assert_close(env_pos + offset, root_com_state_w[..., :3])
                                    torch.testing.assert_close(env_pos + offset, body_com_state_w[..., :3].squeeze(-2))
                                    # link position will be moving but should stay constant away from center of mass
                                    root_link_state_pos_rel_com = quat_rotate_inverse(
                                        root_link_state_w[..., 3:7],
                                        root_link_state_w[..., :3] - root_com_state_w[..., :3],
                                    )
                                    torch.testing.assert_close(-offset, root_link_state_pos_rel_com)
                                    body_link_state_pos_rel_com = quat_rotate_inverse(
                                        body_link_state_w[..., 3:7],
                                        body_link_state_w[..., :3] - body_com_state_w[..., :3],
                                    )
                                    torch.testing.assert_close(-offset, body_link_state_pos_rel_com.squeeze(-2))

                                    # orientation of com will be a constant rotation from link orientation
                                    com_quat_b = cube_object.data.com_quat_b
                                    com_quat_w = quat_mul(body_link_state_w[..., 3:7], com_quat_b)
                                    torch.testing.assert_close(com_quat_w, body_com_state_w[..., 3:7])
                                    torch.testing.assert_close(com_quat_w.squeeze(-2), root_com_state_w[..., 3:7])

                                    # orientation of link will match root state will always match
                                    torch.testing.assert_close(root_state_w[..., 3:7], root_link_state_w[..., 3:7])
                                    torch.testing.assert_close(body_state_w[..., 3:7], body_link_state_w[..., 3:7])

                                    # lin_vel will not match
                                    # center of mass vel will be constant (i.e. spining around com)
                                    torch.testing.assert_close(
                                        torch.zeros_like(root_com_state_w[..., 7:10]), root_com_state_w[..., 7:10]
                                    )
                                    torch.testing.assert_close(
                                        torch.zeros_like(body_com_state_w[..., 7:10]), body_com_state_w[..., 7:10]
                                    )
                                    # link frame will be moving, and should be equal to input angular velocity cross offset
                                    lin_vel_rel_root_gt = quat_rotate_inverse(
                                        root_link_state_w[..., 3:7], root_link_state_w[..., 7:10]
                                    )
                                    lin_vel_rel_body_gt = quat_rotate_inverse(
                                        body_link_state_w[..., 3:7], body_link_state_w[..., 7:10]
                                    )
                                    lin_vel_rel_gt = torch.linalg.cross(
                                        spin_twist.repeat(num_cubes, 1)[..., 3:], -offset
                                    )
                                    torch.testing.assert_close(
                                        lin_vel_rel_gt, lin_vel_rel_root_gt, atol=1e-4, rtol=1e-4
                                    )
                                    torch.testing.assert_close(
                                        lin_vel_rel_gt, lin_vel_rel_body_gt.squeeze(-2), atol=1e-4, rtol=1e-4
                                    )

                                    # ang_vel will always match
                                    torch.testing.assert_close(root_state_w[..., 10:], root_com_state_w[..., 10:])
                                    torch.testing.assert_close(root_state_w[..., 10:], root_link_state_w[..., 10:])
                                    torch.testing.assert_close(body_state_w[..., 10:], body_com_state_w[..., 10:])
                                    torch.testing.assert_close(body_state_w[..., 10:], body_link_state_w[..., 10:])

    def test_write_root_state(self):
        """Test the setters for root_state using both the link frame and center of mass as reference frame."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                for with_offset in [True, False]:
                    for state_location in ("com", "link"):
                        with self.subTest(num_cubes=num_cubes, device=device, with_offset=with_offset):
                            with build_simulation_context(
                                device=device, gravity_enabled=False, auto_add_lighting=True
                            ) as sim:
                                sim._app_control_on_stop_handle = None
                                # Create a scene with random cubes
                                cube_object, env_pos = generate_cubes_scene(
                                    num_cubes=num_cubes, height=0.0, device=device
                                )
                                env_idx = torch.tensor([x for x in range(num_cubes)])

                                # Play sim
                                sim.reset()

                                # Check if cube_object is initialized
                                self.assertTrue(cube_object.is_initialized)

                                # change center of mass offset from link frame
                                if with_offset:
                                    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
                                else:
                                    offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

                                com = cube_object.root_physx_view.get_coms()
                                com[..., :3] = offset.to("cpu")
                                cube_object.root_physx_view.set_coms(com, env_idx)

                                # check ceter of mass has been set
                                torch.testing.assert_close(cube_object.root_physx_view.get_coms(), com)

                                rand_state = torch.zeros_like(cube_object.data.root_state_w)
                                rand_state[..., :7] = cube_object.data.default_root_state[..., :7]
                                rand_state[..., :3] += env_pos
                                # make quaternion a unit vector
                                rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

                                env_idx = env_idx.to(device)
                                for i in range(10):

                                    # perform step
                                    sim.step()
                                    # update buffers
                                    cube_object.update(sim.cfg.dt)

                                    if state_location == "com":
                                        if i % 2 == 0:
                                            cube_object.write_root_com_state_to_sim(rand_state)
                                        else:
                                            cube_object.write_root_com_state_to_sim(rand_state, env_ids=env_idx)
                                    elif state_location == "link":
                                        if i % 2 == 0:
                                            cube_object.write_root_link_state_to_sim(rand_state)
                                        else:
                                            cube_object.write_root_link_state_to_sim(rand_state, env_ids=env_idx)

                                    if state_location == "com":
                                        torch.testing.assert_close(rand_state, cube_object.data.root_com_state_w)
                                    elif state_location == "link":
                                        torch.testing.assert_close(rand_state, cube_object.data.root_link_state_w)


if __name__ == "__main__":
    run_tests()
