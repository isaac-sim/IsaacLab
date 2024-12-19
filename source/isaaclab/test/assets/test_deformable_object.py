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

import carb
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import build_simulation_context


def generate_cubes_scene(
    num_cubes: int = 1,
    height: float = 1.0,
    initial_rot: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    has_api: bool = True,
    material_path: str | None = "material",
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> DeformableObject:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes. Default is 1.0.
        initial_rot: Initial rotation of the cubes. Default is (1.0, 0.0, 0.0, 0.0).
        has_api: Whether the cubes have a deformable body API on them.
        material_path: Path to the material file. If None, no material is added. Default is "material",
            which is path relative to the spawned object prim path.
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
        # Add physics material if provided
        if material_path is not None:
            spawn_cfg.physics_material = sim_utils.DeformableBodyMaterialCfg()
            spawn_cfg.physics_material_path = material_path
        else:
            spawn_cfg.physics_material = None
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
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height), rot=initial_rot),
    )
    cube_object = DeformableObject(cfg=cube_object_cfg)

    return cube_object


class TestDeformableObject(unittest.TestCase):
    """Test for deformable object class."""

    """
    Tests
    """

    def test_initialization(self):
        """Test initialization for prim with deformable body API at the provided prim path.

        This test checks that the deformable object is correctly initialized with deformable material at
        different paths.
        """
        for material_path in [None, "/World/SoftMaterial", "material"]:
            for num_cubes in (1, 2):
                with self.subTest(num_cubes=num_cubes, material_path=material_path):
                    with build_simulation_context(auto_add_lighting=True) as sim:
                        sim._app_control_on_stop_handle = None
                        # Generate cubes scene
                        cube_object = generate_cubes_scene(num_cubes=num_cubes, material_path=material_path)

                        # Check that boundedness of deformable object is correct
                        self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                        # Play sim
                        sim.reset()

                        # Check that boundedness of deformable object is correct
                        self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                        # Check if object is initialized
                        self.assertTrue(cube_object.is_initialized)

                        # Check correct number of cubes
                        self.assertEqual(cube_object.num_instances, num_cubes)
                        self.assertEqual(cube_object.root_physx_view.count, num_cubes)

                        # Check correct number of materials in the view
                        if material_path:
                            if material_path.startswith("/"):
                                self.assertEqual(cube_object.material_physx_view.count, 1)
                            else:
                                self.assertEqual(cube_object.material_physx_view.count, num_cubes)
                        else:
                            self.assertIsNone(cube_object.material_physx_view)

                        # Check buffers that exists and have correct shapes
                        self.assertEqual(
                            cube_object.data.nodal_state_w.shape,
                            (num_cubes, cube_object.max_sim_vertices_per_body, 6),
                        )
                        self.assertEqual(
                            cube_object.data.nodal_kinematic_target.shape,
                            (num_cubes, cube_object.max_sim_vertices_per_body, 4),
                        )
                        self.assertEqual(cube_object.data.root_pos_w.shape, (num_cubes, 3))
                        self.assertEqual(cube_object.data.root_vel_w.shape, (num_cubes, 3))

                        # Simulate physics
                        for _ in range(2):
                            # perform rendering
                            sim.step()
                            # update object
                            cube_object.update(sim.cfg.dt)

                        # check we can get all the sim data from the object
                        self.assertEqual(
                            cube_object.data.sim_element_quat_w.shape,
                            (num_cubes, cube_object.max_sim_elements_per_body, 4),
                        )
                        self.assertEqual(
                            cube_object.data.sim_element_deform_gradient_w.shape,
                            (num_cubes, cube_object.max_sim_elements_per_body, 3, 3),
                        )
                        self.assertEqual(
                            cube_object.data.sim_element_stress_w.shape,
                            (num_cubes, cube_object.max_sim_elements_per_body, 3, 3),
                        )
                        self.assertEqual(
                            cube_object.data.collision_element_quat_w.shape,
                            (num_cubes, cube_object.max_collision_elements_per_body, 4),
                        )
                        self.assertEqual(
                            cube_object.data.collision_element_deform_gradient_w.shape,
                            (num_cubes, cube_object.max_collision_elements_per_body, 3, 3),
                        )
                        self.assertEqual(
                            cube_object.data.collision_element_stress_w.shape,
                            (num_cubes, cube_object.max_collision_elements_per_body, 3, 3),
                        )

    def test_initialization_on_device_cpu(self):
        """Test that initialization fails with deformable body API on the CPU."""
        with build_simulation_context(device="cpu", auto_add_lighting=True) as sim:
            sim._app_control_on_stop_handle = None
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
                    sim._app_control_on_stop_handle = None
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
                    sim._app_control_on_stop_handle = None
                    # Generate cubes scene
                    cube_object = generate_cubes_scene(num_cubes=num_cubes, has_api=False)

                    # Check that boundedness of deformable object is correct
                    self.assertEqual(ctypes.c_long.from_address(id(cube_object)).value, 1)

                    # Play sim
                    sim.reset()

                    # Check if object is initialized
                    self.assertFalse(cube_object.is_initialized)

    def test_set_nodal_state(self):
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
                    sim._app_control_on_stop_handle = None
                    # Generate cubes scene
                    cube_object = generate_cubes_scene(num_cubes=num_cubes)

                    # Play the simulator
                    sim.reset()

                    # Set each state type individually as they are dependent on each other
                    for state_type_to_randomize in ["nodal_pos_w", "nodal_vel_w"]:
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
                                num_cubes, cube_object.max_sim_vertices_per_body, 3, device=sim.device
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

                                # assert that set node quantities are equal to the ones set in the state_dict
                                torch.testing.assert_close(
                                    cube_object.data.nodal_state_w, nodal_state, rtol=1e-5, atol=1e-5
                                )

                                # perform step
                                sim.step()
                                # update object
                                cube_object.update(sim.cfg.dt)

    def test_set_nodal_state_with_applied_transform(self):
        """Test setting the state of the deformable object with applied transform.

        In this test, we apply a random pose to the object and check that the mean of the nodal positions
        is equal to the applied pose after simulation. We set gravity to zero as we don't want any external
        forces acting on the object to ensure state remains static.
        """

        # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
        # test on many environments.
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

        for num_cubes in (1, 2):
            with self.subTest(num_cubes=num_cubes):
                # Turn off gravity for this test as we don't want any external forces acting on the object
                # to ensure state remains static
                with build_simulation_context(gravity_enabled=False, auto_add_lighting=True) as sim:
                    sim._app_control_on_stop_handle = None
                    # Generate cubes scene
                    cube_object = generate_cubes_scene(num_cubes=num_cubes)

                    # Play the simulator
                    sim.reset()

                    for randomize_pos in [True, False]:
                        for randomize_rot in [True, False]:
                            # Now we are ready!
                            for _ in range(5):
                                # reset the nodal state of the object
                                nodal_state = cube_object.data.default_nodal_state_w.clone()
                                mean_nodal_pos_default = nodal_state[..., :3].mean(dim=1)
                                # sample randomize position and rotation
                                if randomize_pos:
                                    pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device)
                                    pos_w[:, 2] += 0.5
                                else:
                                    pos_w = None
                                if randomize_rot:
                                    quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
                                else:
                                    quat_w = None
                                # apply random pose to the object
                                nodal_state[..., :3] = cube_object.transform_nodal_pos(
                                    nodal_state[..., :3], pos_w, quat_w
                                )
                                # compute mean of initial nodal positions
                                mean_nodal_pos_init = nodal_state[..., :3].mean(dim=1)

                                # check computation is correct
                                if pos_w is None:
                                    torch.testing.assert_close(
                                        mean_nodal_pos_init, mean_nodal_pos_default, rtol=1e-5, atol=1e-5
                                    )
                                else:
                                    torch.testing.assert_close(
                                        mean_nodal_pos_init, mean_nodal_pos_default + pos_w, rtol=1e-5, atol=1e-5
                                    )

                                # write nodal state to simulation
                                cube_object.write_nodal_state_to_sim(nodal_state)
                                # reset object
                                cube_object.reset()

                                # perform simulation
                                for _ in range(50):
                                    # perform step
                                    sim.step()
                                    # update object
                                    cube_object.update(sim.cfg.dt)

                                # check that the mean of the nodal positions is equal to the applied pose
                                torch.testing.assert_close(
                                    cube_object.data.root_pos_w, mean_nodal_pos_init, rtol=1e-5, atol=1e-5
                                )

    def test_set_kinematic_targets(self):
        """Test setting kinematic targets for the deformable object.

        In this test, we set one of the cubes with only kinematic targets for its nodal positions and check
        that the object is in that state after simulation.
        """
        for num_cubes in (2, 4):
            with self.subTest(num_cubes=num_cubes):
                # Turn off gravity for this test as we don't want any external forces acting on the object
                # to ensure state remains static
                with build_simulation_context(auto_add_lighting=True) as sim:
                    sim._app_control_on_stop_handle = None
                    # Generate cubes scene
                    cube_object = generate_cubes_scene(num_cubes=num_cubes, height=1.0)

                    # Play the simulator
                    sim.reset()

                    # Get sim kinematic targets
                    nodal_kinematic_targets = cube_object.root_physx_view.get_sim_kinematic_targets().clone()

                    # Now we are ready!
                    for _ in range(5):
                        # reset nodal state
                        cube_object.write_nodal_state_to_sim(cube_object.data.default_nodal_state_w)

                        default_root_pos = cube_object.data.default_nodal_state_w.mean(dim=1)

                        # reset object
                        cube_object.reset()

                        # write kinematic targets
                        # -- enable kinematic targets for the first cube
                        nodal_kinematic_targets[1:, :, 3] = 1.0
                        nodal_kinematic_targets[0, :, 3] = 0.0
                        # -- set kinematic targets for the first cube
                        nodal_kinematic_targets[0, :, :3] = cube_object.data.default_nodal_state_w[0, :, :3]
                        # -- write kinematic targets to simulation
                        cube_object.write_nodal_kinematic_target_to_sim(
                            nodal_kinematic_targets[0], env_ids=torch.tensor([0], device=sim.device)
                        )

                        # perform simulation
                        for _ in range(20):
                            # perform step
                            sim.step()
                            # update object
                            cube_object.update(sim.cfg.dt)

                            # assert that set node quantities are equal to the ones set in the state_dict
                            torch.testing.assert_close(
                                cube_object.data.nodal_pos_w[0], nodal_kinematic_targets[0, :, :3], rtol=1e-5, atol=1e-5
                            )
                            # see other cubes are dropping
                            root_pos_w = cube_object.data.root_pos_w
                            self.assertTrue(torch.all(root_pos_w[1:, 2] < default_root_pos[1:, 2]))


if __name__ == "__main__":
    run_tests()
