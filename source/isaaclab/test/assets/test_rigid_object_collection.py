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

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import default_orientation, quat_mul, quat_rotate_inverse, random_orientation


def generate_cubes_scene(
    num_envs: int = 1,
    num_cubes: int = 1,
    height=1.0,
    has_api: bool = True,
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> tuple[RigidObjectCollection, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_envs: Number of envs to generate.
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        has_api: Whether the cubes have a rigid body API on them.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        A tuple containing the rigid object representing the cubes and the origins of the cubes.

    """
    origins = torch.tensor([(i * 3.0, 0, height) for i in range(num_envs)]).to(device)
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

    # create the rigid object configs
    cube_config_dict = {}
    for i in range(num_cubes):
        cube_object_cfg = RigidObjectCfg(
            prim_path=f"/World/Table_.*/Object_{i}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3 * i, height)),
        )
        cube_config_dict[f"cube_{i}"] = cube_object_cfg
    # create the rigid object collection
    cube_object_collection_cfg = RigidObjectCollectionCfg(rigid_objects=cube_config_dict)
    cube_object_colection = RigidObjectCollection(cfg=cube_object_collection_cfg)

    return cube_object_colection, origins


class TestRigidObjectCollection(unittest.TestCase):
    """Test for rigid object collection class."""

    """
    Tests
    """

    def test_initialization(self):
        """Test initialization for prim with rigid body API at the provided prim path."""
        for num_envs in (1, 2):
            for num_cubes in (1, 3):
                for device in ("cuda:0", "cpu"):
                    with self.subTest(num_envs=num_envs, num_cubes=num_cubes, device=device):
                        with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                            sim._app_control_on_stop_handle = None
                            # Generate cubes scene
                            object_collection, _ = generate_cubes_scene(
                                num_envs=num_envs, num_cubes=num_cubes, device=device
                            )

                            # Check that boundedness of rigid object is correct
                            self.assertEqual(ctypes.c_long.from_address(id(object_collection)).value, 1)

                            # Play sim
                            sim.reset()

                            # Check if object is initialized
                            self.assertTrue(object_collection.is_initialized)
                            self.assertEqual(len(object_collection.object_names), num_cubes)

                            # Check buffers that exists and have correct shapes
                            self.assertEqual(object_collection.data.object_link_pos_w.shape, (num_envs, num_cubes, 3))
                            self.assertEqual(object_collection.data.object_link_quat_w.shape, (num_envs, num_cubes, 4))
                            self.assertEqual(object_collection.data.default_mass.shape, (num_envs, num_cubes, 1))
                            self.assertEqual(object_collection.data.default_inertia.shape, (num_envs, num_cubes, 9))

                            # Simulate physics
                            for _ in range(2):
                                # perform rendering
                                sim.step()
                                # update object
                                object_collection.update(sim.cfg.dt)

    def test_id_conversion(self):
        """Test environment and object index conversion to physics view indices."""
        for device in ("cuda:0", "cpu"):
            with self.subTest(num_envs=2, num_cubes=3, device=device):
                with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                    sim._app_control_on_stop_handle = None
                    # Generate cubes scene
                    object_collection, _ = generate_cubes_scene(num_envs=2, num_cubes=3, device=device)

                    # Play sim
                    sim.reset()

                    expected = [
                        torch.tensor([4, 5], device=device, dtype=torch.long),
                        torch.tensor([4], device=device, dtype=torch.long),
                        torch.tensor([0, 2, 4], device=device, dtype=torch.long),
                        torch.tensor([1, 3, 5], device=device, dtype=torch.long),
                    ]

                    view_ids = object_collection._env_obj_ids_to_view_ids(
                        object_collection._ALL_ENV_INDICES, object_collection._ALL_OBJ_INDICES[None, 2]
                    )
                    self.assertTrue((view_ids == expected[0]).all())
                    view_ids = object_collection._env_obj_ids_to_view_ids(
                        object_collection._ALL_ENV_INDICES[None, 0], object_collection._ALL_OBJ_INDICES[None, 2]
                    )
                    self.assertTrue((view_ids == expected[1]).all())
                    view_ids = object_collection._env_obj_ids_to_view_ids(
                        object_collection._ALL_ENV_INDICES[None, 0], object_collection._ALL_OBJ_INDICES
                    )
                    self.assertTrue((view_ids == expected[2]).all())
                    view_ids = object_collection._env_obj_ids_to_view_ids(
                        object_collection._ALL_ENV_INDICES[None, 1], object_collection._ALL_OBJ_INDICES
                    )
                    self.assertTrue((view_ids == expected[3]).all())

    def test_initialization_with_kinematic_enabled(self):
        """Test that initialization for prim with kinematic flag enabled."""
        for num_envs in (1, 2):
            for num_cubes in (1, 3):
                for device in ("cuda:0", "cpu"):
                    with self.subTest(num_envs=num_envs, num_cubes=num_cubes, device=device):
                        with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                            sim._app_control_on_stop_handle = None
                            # Generate cubes scene
                            object_collection, origins = generate_cubes_scene(
                                num_envs=num_envs, num_cubes=num_cubes, kinematic_enabled=True, device=device
                            )

                            # Check that boundedness of rigid object is correct
                            self.assertEqual(ctypes.c_long.from_address(id(object_collection)).value, 1)

                            # Play sim
                            sim.reset()

                            # Check if object is initialized
                            self.assertTrue(object_collection.is_initialized)
                            self.assertEqual(len(object_collection.object_names), num_cubes)

                            # Check buffers that exists and have correct shapes
                            self.assertEqual(object_collection.data.object_link_pos_w.shape, (num_envs, num_cubes, 3))
                            self.assertEqual(object_collection.data.object_link_quat_w.shape, (num_envs, num_cubes, 4))

                            # Simulate physics
                            for _ in range(2):
                                # perform rendering
                                sim.step()
                                # update object
                                object_collection.update(sim.cfg.dt)
                                # check that the object is kinematic
                                default_object_state = object_collection.data.default_object_state.clone()
                                default_object_state[..., :3] += origins.unsqueeze(1)
                                torch.testing.assert_close(
                                    object_collection.data.object_link_state_w, default_object_state
                                )

    def test_initialization_with_no_rigid_body(self):
        """Test that initialization fails when no rigid body is found at the provided prim path."""
        for num_cubes in (1, 2):
            for device in ("cuda:0", "cpu"):
                with self.subTest(num_cubes=num_cubes, device=device):
                    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                        sim._app_control_on_stop_handle = None
                        # Generate cubes scene
                        object_collection, _ = generate_cubes_scene(num_cubes=num_cubes, has_api=False, device=device)

                        # Check that boundedness of rigid object is correct
                        self.assertEqual(ctypes.c_long.from_address(id(object_collection)).value, 1)

                        # Play sim
                        sim.reset()

                        # Check if object is initialized
                        self.assertFalse(object_collection.is_initialized)

    def test_external_force_buffer(self):
        """Test if external force buffer correctly updates in the force value is zero case.

        In this test, we apply a non-zero force, then a zero force, then finally a non-zero force
        to an object collection. We check if the force buffer is properly updated at each step.
        """

        num_envs = 2
        num_cubes = 1
        for device in ("cuda:0", "cpu"):
            with self.subTest(num_cubes=1, device=device):
                # Generate cubes scene
                with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
                    sim._app_control_on_stop_handle = None
                    object_collection, origins = generate_cubes_scene(
                        num_envs=num_envs, num_cubes=num_cubes, device=device
                    )
                    # play the simulator
                    sim.reset()

                    # find objects to apply the force
                    object_ids, object_names = object_collection.find_objects(".*")

                    # reset object
                    object_collection.reset()

                    # perform simulation
                    for step in range(5):

                        # initiate force tensor
                        external_wrench_b = torch.zeros(
                            object_collection.num_instances, len(object_ids), 6, device=sim.device
                        )

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
                        object_collection.set_external_force_and_torque(
                            external_wrench_b[..., :3], external_wrench_b[..., 3:], object_ids=object_ids
                        )

                        # check if the object collection's force and torque buffers are correctly updated
                        for i in range(num_envs):
                            self.assertTrue(object_collection._external_force_b[i, 0, 0].item() == force)
                            self.assertTrue(object_collection._external_torque_b[i, 0, 0].item() == force)

                        # apply action to the object collection
                        object_collection.write_data_to_sim()

                        # perform step
                        sim.step()

                        # update buffers
                        object_collection.update(sim.cfg.dt)

    def test_external_force_on_single_body(self):
        """Test application of external force on the base of the object.

        In this test, we apply a force equal to the weight of an object on the base of
        one of the objects. We check that the object does not move. For the other object,
        we do not apply any force and check that it falls down.
        """
        for num_envs in (1, 2):
            for num_cubes in (1, 4):
                for device in ("cuda:0", "cpu"):
                    with self.subTest(num_envs=num_envs, num_cubes=num_cubes, device=device):
                        # Generate cubes scene
                        with build_simulation_context(
                            device=device, add_ground_plane=True, auto_add_lighting=True
                        ) as sim:
                            sim._app_control_on_stop_handle = None
                            object_collection, origins = generate_cubes_scene(
                                num_envs=num_envs, num_cubes=num_cubes, device=device
                            )

                            # Play the simulator
                            sim.reset()

                            # Find objects to apply the force
                            object_ids, object_names = object_collection.find_objects(".*")

                            # Sample a force equal to the weight of the object
                            external_wrench_b = torch.zeros(
                                object_collection.num_instances, len(object_ids), 6, device=sim.device
                            )
                            # Every 2nd cube should have a force applied to it
                            external_wrench_b[:, 0::2, 2] = 9.81 * object_collection.data.default_mass[:, 0::2, 0]

                            # Now we are ready!
                            for _ in range(5):
                                # reset object state
                                object_state = object_collection.data.default_object_state.clone()

                                # need to shift the position of the cubes otherwise they will be on top of each other
                                object_state[..., :2] += origins.unsqueeze(1)[..., :2]
                                object_collection.write_object_state_to_sim(object_state)

                                # reset object
                                object_collection.reset()

                                # apply force
                                object_collection.set_external_force_and_torque(
                                    external_wrench_b[..., :3], external_wrench_b[..., 3:], object_ids=object_ids
                                )
                                # perform simulation
                                for _ in range(10):
                                    # apply action to the object
                                    object_collection.write_data_to_sim()

                                    # perform step
                                    sim.step()

                                    # update buffers
                                    object_collection.update(sim.cfg.dt)

                                # First object should still be at the same Z position (1.0)
                                torch.testing.assert_close(
                                    object_collection.data.object_link_pos_w[:, 0::2, 2],
                                    torch.ones_like(object_collection.data.object_pos_w[:, 0::2, 2]),
                                )
                                # Second object should have fallen, so it's Z height should be less than initial height of 1.0
                                self.assertTrue(torch.all(object_collection.data.object_link_pos_w[:, 1::2, 2] < 1.0))

    def test_set_object_state(self):
        """Test setting the state of the object.

        In this test, we set the state of the object to a random state and check
        that the object is in that state after simulation. We set gravity to zero as
        we don't want any external forces acting on the object to ensure state remains static.
        """
        for num_envs in (1, 3):
            for num_cubes in (1, 2):
                for device in ("cuda:0", "cpu"):
                    with self.subTest(num_envs=num_envs, num_cubes=num_cubes, device=device):
                        # Turn off gravity for this test as we don't want any external forces acting on the object
                        # to ensure state remains static
                        with build_simulation_context(
                            device=device, gravity_enabled=False, auto_add_lighting=True
                        ) as sim:
                            sim._app_control_on_stop_handle = None
                            # Generate cubes scene
                            object_collection, origins = generate_cubes_scene(
                                num_envs=num_envs, num_cubes=num_cubes, device=device
                            )

                            # Play the simulator
                            sim.reset()

                            state_types = ["object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"]

                            # Set each state type individually as they are dependent on each other
                            for state_type_to_randomize in state_types:
                                state_dict = {
                                    "object_pos_w": torch.zeros_like(
                                        object_collection.data.object_pos_w, device=sim.device
                                    ),
                                    "object_quat_w": default_orientation(
                                        num=num_cubes * num_envs, device=sim.device
                                    ).view(num_envs, num_cubes, 4),
                                    "object_lin_vel_w": torch.zeros_like(
                                        object_collection.data.object_lin_vel_w, device=sim.device
                                    ),
                                    "object_ang_vel_w": torch.zeros_like(
                                        object_collection.data.object_ang_vel_w, device=sim.device
                                    ),
                                }

                                # Now we are ready!
                                for _ in range(5):
                                    # reset object
                                    object_collection.reset()

                                    # Set random state
                                    if state_type_to_randomize == "object_quat_w":
                                        state_dict[state_type_to_randomize] = random_orientation(
                                            num=num_cubes * num_envs, device=sim.device
                                        ).view(num_envs, num_cubes, 4)
                                    else:
                                        state_dict[state_type_to_randomize] = torch.randn(
                                            num_envs, num_cubes, 3, device=sim.device
                                        )
                                        # make sure objects do not overlap
                                        if state_type_to_randomize == "object_pos_w":
                                            state_dict[state_type_to_randomize][..., :2] += origins.unsqueeze(1)[
                                                ..., :2
                                            ]

                                    # perform simulation
                                    for _ in range(5):
                                        object_state = torch.cat(
                                            [
                                                state_dict["object_pos_w"],
                                                state_dict["object_quat_w"],
                                                state_dict["object_lin_vel_w"],
                                                state_dict["object_ang_vel_w"],
                                            ],
                                            dim=-1,
                                        )
                                        # reset object state
                                        object_collection.write_object_state_to_sim(object_state=object_state)

                                        sim.step()

                                        # assert that set object quantities are equal to the ones set in the state_dict
                                        for key, expected_value in state_dict.items():
                                            value = getattr(object_collection.data, key)
                                            torch.testing.assert_close(value, expected_value, rtol=1e-5, atol=1e-5)

                                        object_collection.update(sim.cfg.dt)

    def test_object_state_properties(self):
        """Test the object_com_state_w and object_link_state_w properties."""
        for num_envs in (1, 4):
            for num_cubes in (1, 2):
                for device in ("cuda:0", "cpu"):
                    for with_offset in [True, False]:
                        with self.subTest(
                            num_envs=num_envs, num_cubes=num_cubes, device=device, with_offset=with_offset
                        ):
                            with build_simulation_context(
                                device=device, gravity_enabled=False, auto_add_lighting=True
                            ) as sim:
                                sim._app_control_on_stop_handle = None
                                # Create a scene with random cubes
                                cube_object, env_pos = generate_cubes_scene(
                                    num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device
                                )
                                view_ids = torch.tensor([x for x in range(num_cubes * num_envs)])

                                # Play sim
                                sim.reset()

                                # Check if cube_object is initialized
                                self.assertTrue(cube_object.is_initialized)

                                # change center of mass offset from link frame
                                if with_offset:
                                    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
                                else:
                                    offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)

                                com = cube_object.reshape_view_to_data(cube_object.root_physx_view.get_coms())
                                com[..., :3] = offset.to("cpu")
                                cube_object.root_physx_view.set_coms(
                                    cube_object.reshape_data_to_view(com.clone()), view_ids
                                )

                                # check center of mass has been set
                                torch.testing.assert_close(
                                    cube_object.reshape_view_to_data(cube_object.root_physx_view.get_coms()), com
                                )

                                # random z spin velocity
                                spin_twist = torch.zeros(6, device=device)
                                spin_twist[5] = torch.randn(1, device=device)

                                # initial spawn point
                                init_com = cube_object.data.object_com_state_w[..., :3]

                                # Simulate physics
                                for i in range(10):
                                    # spin the object around Z axis (com)
                                    cube_object.write_object_com_velocity_to_sim(
                                        spin_twist.repeat(num_envs, num_cubes, 1)
                                    )
                                    # perform rendering
                                    sim.step()
                                    # update object
                                    cube_object.update(sim.cfg.dt)

                                    # get state properties
                                    object_state_w = cube_object.data.object_state_w
                                    object_link_state_w = cube_object.data.object_link_state_w
                                    object_com_state_w = cube_object.data.object_com_state_w

                                    # if offset is [0,0,0] all object_state_%_w will match and all body_%_w will match
                                    if not with_offset:
                                        torch.testing.assert_close(object_state_w, object_com_state_w)
                                        torch.testing.assert_close(object_state_w, object_link_state_w)
                                    else:
                                        # cubes are spinning around center of mass
                                        # position will not match
                                        # center of mass position will be constant (i.e. spinning around com)
                                        torch.testing.assert_close(init_com, object_com_state_w[..., :3])

                                        # link position will be moving but should stay constant away from center of mass
                                        object_link_state_pos_rel_com = quat_rotate_inverse(
                                            object_link_state_w[..., 3:7],
                                            object_link_state_w[..., :3] - object_com_state_w[..., :3],
                                        )

                                        torch.testing.assert_close(-offset, object_link_state_pos_rel_com)

                                        # orientation of com will be a constant rotation from link orientation
                                        com_quat_b = cube_object.data.com_quat_b
                                        com_quat_w = quat_mul(object_link_state_w[..., 3:7], com_quat_b)
                                        torch.testing.assert_close(com_quat_w, object_com_state_w[..., 3:7])

                                        # orientation of link will match object state will always match
                                        torch.testing.assert_close(
                                            object_state_w[..., 3:7], object_link_state_w[..., 3:7]
                                        )

                                        # lin_vel will not match
                                        # center of mass vel will be constant (i.e. spining around com)
                                        torch.testing.assert_close(
                                            torch.zeros_like(object_com_state_w[..., 7:10]),
                                            object_com_state_w[..., 7:10],
                                        )

                                        # link frame will be moving, and should be equal to input angular velocity cross offset
                                        lin_vel_rel_object_gt = quat_rotate_inverse(
                                            object_link_state_w[..., 3:7], object_link_state_w[..., 7:10]
                                        )
                                        lin_vel_rel_gt = torch.linalg.cross(
                                            spin_twist.repeat(num_envs, num_cubes, 1)[..., 3:], -offset
                                        )
                                        torch.testing.assert_close(
                                            lin_vel_rel_gt, lin_vel_rel_object_gt, atol=1e-4, rtol=1e-3
                                        )

                                        # ang_vel will always match
                                        torch.testing.assert_close(
                                            object_state_w[..., 10:], object_com_state_w[..., 10:]
                                        )
                                        torch.testing.assert_close(
                                            object_state_w[..., 10:], object_link_state_w[..., 10:]
                                        )

    def test_write_object_state(self):
        """Test the setters for object_state using both the link frame and center of mass as reference frame."""
        for num_envs in (1, 3):
            for num_cubes in (1, 2):
                for device in ("cuda:0", "cpu"):
                    for with_offset in [True, False]:
                        for state_location in ("com", "link"):
                            with self.subTest(
                                num_envs=num_envs, num_cubes=num_cubes, device=device, with_offset=with_offset
                            ):
                                with build_simulation_context(
                                    device=device, gravity_enabled=False, auto_add_lighting=True
                                ) as sim:
                                    sim._app_control_on_stop_handle = None
                                    # Create a scene with random cubes
                                    cube_object, env_pos = generate_cubes_scene(
                                        num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device
                                    )
                                    view_ids = torch.tensor([x for x in range(num_cubes * num_cubes)])
                                    env_ids = torch.tensor([x for x in range(num_envs)])
                                    object_ids = torch.tensor([x for x in range(num_cubes)])

                                    # Play sim
                                    sim.reset()

                                    # Check if cube_object is initialized
                                    self.assertTrue(cube_object.is_initialized)

                                    # change center of mass offset from link frame
                                    if with_offset:
                                        offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(
                                            num_envs, num_cubes, 1
                                        )
                                    else:
                                        offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(
                                            num_envs, num_cubes, 1
                                        )

                                    com = cube_object.reshape_view_to_data(cube_object.root_physx_view.get_coms())
                                    com[..., :3] = offset.to("cpu")
                                    cube_object.root_physx_view.set_coms(
                                        cube_object.reshape_data_to_view(com.clone()), view_ids
                                    )

                                    # check center of mass has been set
                                    torch.testing.assert_close(
                                        cube_object.reshape_view_to_data(cube_object.root_physx_view.get_coms()), com
                                    )

                                    rand_state = torch.zeros_like(cube_object.data.object_link_state_w)
                                    rand_state[..., :7] = cube_object.data.default_object_state[..., :7]
                                    rand_state[..., :3] += cube_object.data.object_link_pos_w
                                    # make quaternion a unit vector
                                    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

                                    env_ids = env_ids.to(device)
                                    object_ids = object_ids.to(device)
                                    for i in range(10):

                                        # perform step
                                        sim.step()
                                        # update buffers
                                        cube_object.update(sim.cfg.dt)

                                        if state_location == "com":
                                            if i % 2 == 0:
                                                cube_object.write_object_com_state_to_sim(rand_state)
                                            else:
                                                cube_object.write_object_com_state_to_sim(
                                                    rand_state, env_ids=env_ids, object_ids=object_ids
                                                )
                                        elif state_location == "link":
                                            if i % 2 == 0:
                                                cube_object.write_object_link_state_to_sim(rand_state)
                                            else:
                                                cube_object.write_object_link_state_to_sim(
                                                    rand_state, env_ids=env_ids, object_ids=object_ids
                                                )

                                        if state_location == "com":
                                            torch.testing.assert_close(rand_state, cube_object.data.object_com_state_w)
                                        elif state_location == "link":
                                            torch.testing.assert_close(rand_state, cube_object.data.object_link_state_w)

    def test_reset_object_collection(self):
        """Test resetting the state of the rigid object."""
        for num_envs in (1, 3):
            for num_cubes in (1, 2):
                for device in ("cuda:0", "cpu"):
                    with self.subTest(num_envs=num_envs, num_cubes=num_cubes, device=device):
                        with build_simulation_context(
                            device=device, gravity_enabled=True, auto_add_lighting=True
                        ) as sim:
                            sim._app_control_on_stop_handle = None
                            # Generate cubes scene
                            object_collection, _ = generate_cubes_scene(
                                num_envs=num_envs, num_cubes=num_cubes, device=device
                            )

                            # Play the simulator
                            sim.reset()

                            for i in range(5):
                                # perform rendering
                                sim.step()

                                # update object
                                object_collection.update(sim.cfg.dt)

                                # Move the object to a random position
                                object_state = object_collection.data.default_object_state.clone()
                                object_state[..., :3] = torch.randn(num_envs, num_cubes, 3, device=sim.device)

                                # Random orientation
                                object_state[..., 3:7] = random_orientation(num=num_cubes, device=sim.device)
                                object_collection.write_object_state_to_sim(object_state)

                                if i % 2 == 0:
                                    # reset object
                                    object_collection.reset()

                                    # Reset should zero external forces and torques
                                    self.assertFalse(object_collection.has_external_wrench)
                                    self.assertEqual(torch.count_nonzero(object_collection._external_force_b), 0)
                                    self.assertEqual(torch.count_nonzero(object_collection._external_torque_b), 0)

    def test_set_material_properties(self):
        """Test getting and setting material properties of rigid object."""
        for num_envs in (1, 3):
            for num_cubes in (1, 2):
                for device in ("cuda:0", "cpu"):
                    with self.subTest(num_envs=num_envs, num_cubes=num_cubes, device=device):
                        with build_simulation_context(
                            device=device, gravity_enabled=True, add_ground_plane=True, auto_add_lighting=True
                        ) as sim:
                            sim._app_control_on_stop_handle = None
                            # Generate cubes scene
                            object_collection, _ = generate_cubes_scene(
                                num_envs=num_envs, num_cubes=num_cubes, device=device
                            )

                            # Play sim
                            sim.reset()

                            # Set material properties
                            static_friction = torch.FloatTensor(num_envs, num_cubes, 1).uniform_(0.4, 0.8)
                            dynamic_friction = torch.FloatTensor(num_envs, num_cubes, 1).uniform_(0.4, 0.8)
                            restitution = torch.FloatTensor(num_envs, num_cubes, 1).uniform_(0.0, 0.2)

                            materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

                            indices = torch.tensor(range(num_cubes * num_envs), dtype=torch.int)
                            # Add friction to cube
                            object_collection.root_physx_view.set_material_properties(
                                object_collection.reshape_data_to_view(materials), indices
                            )

                            # Simulate physics
                            sim.step()
                            # update object
                            object_collection.update(sim.cfg.dt)

                            # Get material properties
                            materials_to_check = object_collection.root_physx_view.get_material_properties()

                            # Check if material properties are set correctly
                            torch.testing.assert_close(
                                object_collection.reshape_view_to_data(materials_to_check), materials
                            )

    def test_gravity_vec_w(self):
        """Test that gravity vector direction is set correctly for the rigid object."""
        for num_envs in (1, 3):
            for num_cubes in (1, 2):
                for device in ("cuda:0", "cpu"):
                    for gravity_enabled in [True, False]:
                        with self.subTest(
                            num_envs=num_envs, num_cubes=num_cubes, device=device, gravity_enabled=gravity_enabled
                        ):
                            with build_simulation_context(device=device, gravity_enabled=gravity_enabled) as sim:
                                sim._app_control_on_stop_handle = None
                                # Create a scene with random cubes
                                object_collection, _ = generate_cubes_scene(
                                    num_envs=num_envs, num_cubes=num_cubes, device=device
                                )

                                # Obtain gravity direction
                                if gravity_enabled:
                                    gravity_dir = (0.0, 0.0, -1.0)
                                else:
                                    gravity_dir = (0.0, 0.0, 0.0)

                                # Play sim
                                sim.reset()

                                # Check that gravity is set correctly
                                self.assertEqual(object_collection.data.GRAVITY_VEC_W[0, 0, 0], gravity_dir[0])
                                self.assertEqual(object_collection.data.GRAVITY_VEC_W[0, 0, 1], gravity_dir[1])
                                self.assertEqual(object_collection.data.GRAVITY_VEC_W[0, 0, 2], gravity_dir[2])

                                # Simulate physics
                                for _ in range(2):
                                    sim.step()
                                    # update object
                                    object_collection.update(sim.cfg.dt)

                                    # Expected gravity value is the acceleration of the body
                                    gravity = torch.zeros(num_envs, num_cubes, 6, device=device)
                                    if gravity_enabled:
                                        gravity[..., 2] = -9.81
                                    # Check the body accelerations are correct
                                    torch.testing.assert_close(object_collection.data.object_acc_w, gravity)


if __name__ == "__main__":
    run_tests()
